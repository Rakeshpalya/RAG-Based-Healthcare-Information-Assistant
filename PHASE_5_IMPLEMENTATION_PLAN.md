# Phase 5 — RAG Evaluation, Observability, Quality Metrics & Production Readiness Implementation Plan

## 1. Current Architecture Overview

The `AI-Healthcare-Agent` system is an enterprise-grade Clinical RAG application built around a multi-stage deterministic pipeline:

```
[ User Input ]
      │
      ▼
1. Safety Classifier (Phase 4.1) ─────────────► [ Emergency / Overdose / Self-Harm Advisory ]
      │ (If Normal / Allowed Query)             (Immediate bypass: 0 ms LLM, empty sources)
      ▼
2. Query Expansion & Disambiguation (Phase 2E/2F)
      │
      ▼
3. FAISS Dense Vector Retrieval (384-d, Inner Product Cosine) + Multi-Tenant Isolation
      │
      ▼
4. Pre-LLM Grounding & Relevance Gate (Strict Cosine Threshold + Scoped Document Check)
      │
      ▼
5. Grounded Prompt & Context Assembler (Phase 3.2: [Source X] Formatting + Anti-Confabulation)
      │
      ▼
6. Gemini LLM Generation (Deterministic 0.2 temperature, streaming or sync)
      │
      ▼
7. Citation Enforcement & Sentence Splitting (Phase 3.3: CitationValidator)
      │
      ▼
8. Hallucination Protection (Phase 3.4: HallucinationGuard - Entity & Contradiction Filter)
      │
      ▼
9. Medical Safety Guard (Phase 4.3: Clinical boundary enforcement & Disclaimer injection)
      │
      ▼
[ Final Verified Answer ]
```

### Key Components:
- **Embedding & Vector Storage**: `EmbeddingService` (`all-MiniLM-L6-v2`, 384 dimensions) and `VectorStoreService` (`FAISS IndexFlatIP` with 744 vectors and 744 metadata records).
- **RAG Orchestration**: `RAGService.generate_rag_answer()` in `backend/rag/rag_service.py` coordinates query resolution, vector search, relevance gates, generation, validation, and timings.
- **Citation Enforcement**: `CitationValidator` in `backend/evaluation/citation_validator.py` ensures 100% claim-level citations with bracket parsing (`[Source X]`) and cosine verification.
- **Hallucination Protection**: `HallucinationGuard` in `backend/evaluation/hallucination_guard.py` detects medication hallucinations, dosage distortions, negation contradictions, and directional inversions.
- **Clinical Safety Layer**: `SafetyClassifier` and `MedicalSafetyGuard` in `backend/safety/` enforce 15 safety categories, intercepting acute crises and post-screening boundaries.

---

## 2. Existing Evaluation Capabilities

Currently, evaluation routines are scattered across development scripts and targeted unit tests:
1. **Retrieval Metrics** (`backend/evaluation/retrieval_evaluator.py`):
   - Computes `Precision@K`, `Recall@K`, and `MRR@K` for arbitrary lists of chunk IDs.
   - Evaluates queries through `RetrievalEvaluator.evaluate_query()` against `RAGService.query()`.
2. **Ad-hoc Benchmark Datasets** (`tests/evaluation/`):
   - `retrieval_eval_dataset.py` contains 40+ medical queries across categories A–I.
   - `phase2f_adversarial_dataset.py` contains distractor and adversarial queries.
   - Scripts like `run_retrieval_evaluation.py` run ad-hoc benchmarking from CLI.
3. **Grounding Evaluation** (`backend/evaluation/grounding_evaluator.py`):
   - `GroundingEvaluator.evaluate_grounding()` verifies basic lexical overlap, presence of context, and citation validity for development tests.
4. **Citation & Claim Checks**:
   - `CitationValidator` extracts claims and calculates `citation_coverage`, `claims_checked`, `claims_supported`, `claims_unsupported`.
5. **Hallucination Detection**:
   - `HallucinationGuard` computes `hallucination_rate`, identifies `hallucination_types`, and flags contradiction flags.

---

## 3. Existing Logging & Metrics

1. **RAG Diagnostic Audit Logging**:
   - `RAGService` logs a text-based ASCII block (`=== RAG DIAGNOSTIC AUDIT ===`) to Python standard logging (`logger.info`).
   - Logs query, retrieved chunk IDs, document names, scores, threshold, relevance flag, LLM called flag, number of sources, and latency breakdown (embedding, search, context prep, Gemini gen, total).
2. **Response Timings Dictionary**:
   - Returned under the `"timings"` key in `generate_rag_answer()` with granular millisecond fields (`embedding_time_ms`, `faiss_retrieval_time_ms`, `deduplication_time_ms`, `context_construction_time_ms`, `llm_generation_time_ms`, `total_time_ms`).
   - Includes citation and safety metadata (`citation_coverage`, `safety_category`, `risk_level`, `safety_post_check_passed`).
3. **Health Check Endpoint**:
   - `GET /health` in `backend/main.py` returns minimal static dictionary:
     `{"status": "healthy", "service": "AI-Healthcare-Agent", "environment": "development"}`.

---

## 4. Missing Evaluation & Observability Capabilities

Despite existing components, several enterprise production requirements are missing:
1. **Unified Deterministic RAG Quality Evaluator**:
   - No single evaluator coordinates retrieval, answer quality, citation compliance, hallucination checks, and safety classification into a single unified JSON evaluation contract.
   - Missing explicit metrics: `Hit Rate@K`, `Context Relevance`, `Context Sufficiency`, `Citation Completeness` vs `Citation Correctness`.
2. **Version-Controlled Golden Evaluation Datasets (`tests/evaluation_data/`)**:
   - Datasets currently reside as Python script arrays rather than structured, version-controlled JSON files.
   - Missing standardized JSON files: `retrieval_cases.json`, `citation_cases.json`, `hallucination_cases.json`, `safety_cases.json`.
3. **Automated Safety Evaluator**:
   - Missing a quantitative safety benchmark engine measuring Precision, Recall, F1, False Positives, and critical False Negatives across all 15 Phase 4 safety categories.
4. **End-to-End Evaluation Runner CLI**:
   - Missing a single CLI entrypoint (`python -m backend.evaluation.evaluation_runner`) that loads golden datasets, executes all 5 evaluation tracks, displays tabular CLI output, and persists non-destructive timestamped JSON reports.
5. **Structured JSON Observability & Tracing**:
   - Audit logging in `RAGService` is unstructured multiline text; lacks machine-readable JSON logging, request IDs (`trace_id` / `request_id`), and PHI/token sanitization guarantees.
6. **Aggregated Latency Profiler**:
   - No statistical analyzer computing Average, Median, P95, and P99 latency percentiles across historical RAG runs.
7. **Comprehensive System Health Readiness**:
   - `/health` does not probe whether the vector store (744 vectors), embedding model, and safety classifier are actually operational in memory.

---

## 5. Proposed Phase 5 Architecture

```
                                  [ Golden Datasets ]
                                (tests/evaluation_data/)
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │      EvaluationRunner (CLI / SDK)       │
                      │  (backend/evaluation/evaluation_runner) │
                      └────┬───────────┬───────────┬──────────┬─┘
                           │           │           │          │
         ┌─────────────────┘           │           │          └────────────────┐
         ▼                             ▼           ▼                           ▼
┌──────────────────┐          ┌────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│RetrievalEvaluator│          │AnswerEvaluator │ │ SafetyEvaluator  │ │LatencyEvaluator │
│ - Precision@K    │          │ - Claim Support│ │ - 15 Categories  │ │ - Avg, Median   │
│ - Recall@K       │          │ - Cit Correct  │ │ - Confusion Mtx  │ │ - P95, P99      │
│ - HitRate@K      │          │ - Groundedness │ │ - Precision, Rec │ │ - Breakdown     │
│ - MRR@K          │          │ - Halluc Rate  │ │ - False Negative │ └─────────────────┘
│ - Sufficiency    │          │ - Contradiction│ └──────────────────┘
└──────────────────┘          └────────────────┘
         │                             │                   │                   │
         └─────────────────────────────┴───────────────────┴───────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │        Unified Evaluation Report        │
                      │   - evaluation_reports/*.json           │
                      │   - Formatted Console Table Output      │
                      └─────────────────────────────────────────┘
```

### Observability Architecture:
- `backend/evaluation/observability.py`:
  - `StructuredRAGLogger`: Emits structured JSON event logs with `request_id`, component latencies, tokens, query, scores, safety tags, and status.
  - Automatic redaction of sensitive credentials, API keys, and PHI patterns.
  - Pluggable into `RAGService` without breaking existing text logs.

---

## 6. Files That Will Be Modified

1. **`backend/evaluation/retrieval_evaluator.py`**:
   - Extend existing file to implement `compute_hit_rate_at_k()`, `evaluate_retrieval()`, and `evaluate_context_sufficiency()`.
   - Preserve all existing functions and classes (`compute_precision_at_k`, `compute_recall_at_k`, `compute_mrr_at_k`, `RetrievalEvaluator`, etc.).
2. **`backend/rag/rag_service.py`**:
   - Integrate structured observability logging alongside existing diagnostic audit logs.
   - Attach `request_id` / `trace_id` support to `generate_rag_answer()` timings.
3. **`backend/config.py`**:
   - Add Phase 5 configuration fields to `Settings`: `EVALUATION_DATA_DIR`, `EVALUATION_REPORTS_DIR`, `LOG_FORMAT` (`json` vs `text`), `RAG_LATENCY_P95_BUDGET_MS`.
4. **`backend/main.py`**:
   - Enhance `GET /health` to perform non-destructive readiness checks for `vector_store`, `embedding_service`, and `safety_engine`.
5. **`backend/evaluation/__init__.py`**:
   - Export new evaluators and runner classes.

---

## 7. New Files That Will Be Created

1. **Golden Evaluation Datasets**:
   - `tests/evaluation_data/retrieval_cases.json`: Gold standard queries with expected chunks, relevant documents, and sufficiency labels.
   - `tests/evaluation_data/citation_cases.json`: Answer-source pairs testing citation correctness, index out-of-bounds, and ungrounded statements.
   - `tests/evaluation_data/hallucination_cases.json`: Test pairs covering medication substitution, dosage fabrication, negation reversals, and directional contradictions.
   - `tests/evaluation_data/safety_cases.json`: Queries testing all 15 Phase 4 safety categories with expected classification, risk level, and emergency guidance.
2. **Evaluators & Observability Engine**:
   - `backend/evaluation/answer_evaluator.py`: Deterministic evaluator for claim support, citation correctness/completeness, groundedness, and hallucination rate.
   - `backend/evaluation/safety_evaluator.py`: Quantitative evaluator for Phase 4 safety classifier calculating classification accuracy, precision, recall, F1, and false negatives.
   - `backend/evaluation/latency_evaluator.py`: Statistical performance profiler calculating average, median, P95, and P99 latencies.
   - `backend/evaluation/observability.py`: Structured JSON logger, request ID generator, and data sanitization filter.
   - `backend/evaluation/evaluation_runner.py`: Orchestrator running all evaluations, generating `EvaluationReport`, and providing CLI execution via `python -m backend.evaluation.evaluation_runner`.

---

## 8. Tests That Will Be Added

1. **`tests/test_phase5_retrieval_evaluator.py`**:
   - Test Precision@K, Recall@K, HitRate@K, MRR@K on perfect, partial, and zero retrieval.
   - Test configurable K (1, 3, 5, 10).
   - Test context sufficiency detection.
2. **`tests/test_phase5_answer_evaluator.py`**:
   - Test claim support scoring.
   - Test citation correctness vs completeness.
   - Test groundedness calculation on fully supported, partially supported, and unsupported text.
   - Test contradiction detection integration.
3. **`tests/test_phase5_safety_evaluator.py`**:
   - Test safety classification evaluation against golden safety dataset.
   - Test metrics calculations: accuracy, precision, recall, F1, false positive, and false negative counts.
   - Verify explicit alerting on false negatives for critical emergency categories.
4. **`tests/test_phase5_observability.py`**:
   - Test structured JSON log format.
   - Test request ID generation and propagation.
   - Test sensitive data redaction (API keys, passwords, bearer tokens).
   - Test latency metrics calculation (average, median, P95, P99).
5. **`tests/test_phase5_evaluation.py`**:
   - End-to-end integration test of `EvaluationRunner`.
   - Test loading golden datasets from JSON.
   - Test running complete evaluation report generation.
   - Test non-destructive report persistence in `evaluation_reports/`.
   - Test enhanced `/health` endpoint response structure.

---

## 9. Backward-Compatibility Strategy

- **Zero Breaking Changes**:
  - `retrieval_evaluator.py` retains all original function signatures (`compute_precision_at_k`, etc.) and classes (`RetrievalEvaluator`, `QueryEvaluationResult`).
  - `RAGService.generate_rag_answer()` retains its exact return signature and dictionary structure; new observability fields are additive.
  - `/health` endpoint retains existing keys (`status`, `service`, `environment`) while adding optional health probe sub-keys.
  - `VectorStoreService` and FAISS index remain completely read-only.
- **Existing 473 Tests**:
  - Full test suite will be verified at every step; all existing tests must pass (100%).

---

## 10. Risks & Rollback Strategy

| Risk | Mitigation | Rollback Plan |
| :--- | :--- | :--- |
| **Vector Store Mutation** during evaluation | Evaluator uses in-memory mocked chunks or read-only queries against existing store. Golden datasets reference read-only chunks. | Automated test invariant assertion immediately restores 744 vector snapshot if altered. |
| **Performance Overhead** from Observability | Structured logging uses lightweight JSON serialization only when enabled; avoids costly duplicate embedding calls. | Default log level set to INFO; formatting is inline without extra disk I/O. |
| **False Negative Masking** in Safety Eval | Metric computation explicitly reports `false_negatives` per category and raises safety alerts for critical categories. | Safety evaluator enforces zero-tolerance policy for emergency false negatives. |
| **Test Interference** with existing fixtures | New test modules will use isolated temporary directories for any file writes (`tmp_path`). | Zero shared state between test modules. |

---

## 11. Implementation Steps Checklist

- [ ] Step 1: Create structured golden datasets in `tests/evaluation_data/` (`retrieval_cases.json`, `citation_cases.json`, `hallucination_cases.json`, `safety_cases.json`).
- [ ] Step 2: Extend `backend/evaluation/retrieval_evaluator.py` with `Hit Rate@K`, `Context Sufficiency`, and `evaluate_retrieval()`.
- [ ] Step 3: Implement `backend/evaluation/answer_evaluator.py` integrating `CitationValidator` and `HallucinationGuard`.
- [ ] Step 4: Implement `backend/evaluation/safety_evaluator.py` benchmarking Phase 4 `SafetyClassifier`.
- [ ] Step 5: Implement `backend/evaluation/latency_evaluator.py` for statistical latency percentiles.
- [ ] Step 6: Implement `backend/evaluation/observability.py` for structured JSON logging and sanitization.
- [ ] Step 7: Implement `backend/evaluation/evaluation_runner.py` with CLI support (`python -m backend.evaluation.evaluation_runner`).
- [ ] Step 8: Update `backend/config.py` with evaluation and observability settings.
- [ ] Step 9: Update `backend/main.py` `/health` endpoint with component readiness probes.
- [ ] Step 10: Integrate structured observability in `backend/rag/rag_service.py`.
- [ ] Step 11: Write comprehensive test suites (`test_phase5_*.py`).
- [ ] Step 12: Execute targeted Phase 5 tests and verify 100% pass rate.
- [ ] Step 13: Execute full regression suite (473+ tests) and verify vector store read-only invariant (744 vectors).
- [ ] Step 14: Generate `PHASE_5_EVALUATION_REPORT.md` and complete certification.
