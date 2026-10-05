# Phase 6.2 — Clinical Query Planning & Adaptive Retrieval Strategies Report

## Executive Summary & Objective

**Milestone 6.2** introduces a deterministic, lightweight, explainable, and production-hardened **Clinical Query Planning and Adaptive Retrieval** layer into the **AI-Healthcare-Agent** medical RAG application.

Consuming the structured `IntentClassificationResult` produced in Milestone 6.1, the query planner deterministically configures retrieval execution parameters—such as dynamic Top-K, similarity thresholds, multi-document evidence selection, chunk weighting algorithms, and bounded clinical query expansion—**before** vector search is invoked.

The entire query planning layer executes in **sub-millisecond latency (p95: 0.319 ms)**, adds zero external LLM dependencies, enforces absolute medical safety precedence, and preserves all critical production invariants.

---

## 1. Architectural Pipeline & Execution Order

The query planning engine integrates directly between Milestone 6.1 (Clinical Intent Classification) and retrieval/cache resolution:

```
1. Client HTTP Request (/rag/query or /rag/stream)
     ↓
2. Authentication (Bearer JWT multi-tenant user resolution)
     ↓
3. Rate Limiter (Distributed Redis token bucket / sliding window)
     ↓
4. Trace ID Assignment (X-Request-ID propagation)
     ↓
5. Medical Safety Pre-Screen (MedicalSafetyGuard.pre_screen_inquiry)
     - Emergency symptoms (chest pain, stroke, dyspnea)
     - Self-harm / suicide ideation (988 Lifeline)
     - Poisoning / acute toxic ingestion
     → If triggered: immediate safety refusal (0 retrieval, 0 LLM calls).
     ↓
6. Clinical Intent Detection (ClinicalIntentClassifier.classify)
     - Fast deterministic intent mapping (< 0.5 ms)
     - Safety priority assignment
     ↓
7. Clinical Query Planning (NEW - Milestone 6.2: ClinicalQueryPlanner.plan)
     - Retrieval requirement determination
     - Strategy-specific Top-K (5 to 8) and similarity threshold (0.20 to 0.30)
     - Chunk weighting strategy selection
     - Multi-document aggregation strategy selection
     - Bounded, deterministic query expansion (max 4, pure Python)
     - High-confidence evidence requirement gating
     - Scoped document identification ([Doc: <name>])
     - Multi-tenant document scoping (USER_DOCUMENTS_ONLY vs ALL_AVAILABLE)
     ↓
8. Cache Lookup (L1 LRU / L2 Redis, isolated by query_plan_strategy)
     ↓
9. Adaptive Evidence Retrieval (RAGService.query)
     - Query expansion union retrieval
     - Candidate chunk weighting and clinical score boost
     - Multi-document evidence distribution (Single, Multi, Map-Reduce, Balanced)
     ↓
10. Relevance & High-Confidence Sufficiency Gate (RAGService.verify_relevance_and_sufficiency)
     - Minimum similarity cutoff check
     - Strict numerical dosage & unit verification for DOSAGE_QUERY
     - Diagnostic guidelines & criteria verification for DIAGNOSIS_QUERY
     ↓
11. Generative LLM Synthesis (Gemini 2.5 Flash Lite)
     ↓
12. Validation Pipeline:
     - Citation bracket validation
     - Grounding evaluation & hallucination guard
     - Medical safety post-screen (prohibited prescriptive/diagnostic scrubbing)
     ↓
13. Cache Storage (Safe, validated responses only)
     ↓
14. Structured Response Delivery (JSON payload or SSE token stream with query_plan metadata)
```

---

## 2. Strongly Typed QueryPlan Data Schema

Implemented in `backend/intelligence/query_plan.py`:

```python
@dataclass
class QueryPlan:
    intent: ClinicalIntent
    retrieval_required: bool
    retrieval_strategy: ClinicalRoutingStrategy
    top_k: int = 5
    similarity_threshold: float = 0.25
    max_chunks: int = 8
    chunk_weighting_strategy: ChunkWeightingStrategy = ChunkWeightingStrategy.STANDARD
    document_filter_strategy: DocumentFilterStrategy = DocumentFilterStrategy.ALL_AVAILABLE
    query_expansion_enabled: bool = False
    query_expansions: List[str] = field(default_factory=list)
    multi_document_strategy: MultiDocumentStrategy = MultiDocumentStrategy.MULTI_DOCUMENT
    context_budget: int = 3500
    requires_high_confidence_evidence: bool = False
    safety_priority: SafetyPriority = SafetyPriority.NORMAL
    generation_strategy: GenerationStrategy = GenerationStrategy.STANDARD_GROUNDED
    scoped_document_name: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Enumerated Strategies
- **`ChunkWeightingStrategy`**: `STANDARD`, `MEDICATION_PRIORITY`, `DOSAGE_PRIORITY`, `LAB_PRIORITY`, `DIAGNOSTIC_PRIORITY`, `SYMPTOM_PRIORITY`, `TREATMENT_PRIORITY`, `PREVENTION_PRIORITY`, `BROAD_COVERAGE`, `BALANCED_MULTI_DOCUMENT`.
- **`DocumentFilterStrategy`**: `ALL_AVAILABLE`, `USER_DOCUMENTS_ONLY`, `SCOPED_DOCUMENT`, `EXCLUDE_NON_CLINICAL`.
- **`MultiDocumentStrategy`**: `SINGLE_DOCUMENT`, `MULTI_DOCUMENT`, `MAP_REDUCE`, `BALANCED_DOCUMENT_RETRIEVAL`.
- **`GenerationStrategy`**: `STANDARD_GROUNDED`, `HIGH_CONFIDENCE_GROUNDED`, `MULTI_SOURCE_SYNTHESIS`, `DOCUMENT_SUMMARY_SYNTHESIS`, `COMPARATIVE_SYNTHESIS`, `SAFETY_REFUSAL`, `OUT_OF_SCOPE_REFUSAL`.

---

## 3. Clinical Intent-to-Strategy Mapping Matrix

| Clinical Intent | Retrieval Required | Top-K | Sim. Thresh. | Max Chunks | Chunk Weighting | Multi-Doc Strategy | High Evidence Gate | Safety Priority |
|---|---|---|---|---|---|---|---|---|
| `EMERGENCY` | **False** | 0 | 1.00 | 0 | `STANDARD` | `SINGLE_DOCUMENT` | False | `CRITICAL` |
| `SELF_HARM` | **False** | 0 | 1.00 | 0 | `STANDARD` | `SINGLE_DOCUMENT` | False | `CRITICAL` |
| `POISONING` | **False** | 0 | 1.00 | 0 | `STANDARD` | `SINGLE_DOCUMENT` | False | `CRITICAL` |
| `OUT_OF_SCOPE` | **False** | 0 | 1.00 | 0 | `STANDARD` | `SINGLE_DOCUMENT` | False | `NONE` |
| `DOSAGE_QUERY` | **True** | 6 | 0.30 | 8 | `DOSAGE_PRIORITY` | `MULTI_DOCUMENT` | **True** | `HIGH` |
| `DIAGNOSIS_QUERY` | **True** | 7 | 0.28 | 8 | `DIAGNOSTIC_PRIORITY` | `MULTI_DOCUMENT` | **True** | `HIGH` |
| `MEDICATION_QUERY` | **True** | 5 | 0.25 | 7 | `MEDICATION_PRIORITY` | `MULTI_DOCUMENT` | False | `NORMAL` |
| `LAB_RESULT_QUERY` | **True** | 6 | 0.25 | 8 | `LAB_PRIORITY` | `MULTI_DOCUMENT` | False | `NORMAL` |
| `DOCUMENT_SUMMARY` | **True** | 8 | 0.20 | 10 | `BROAD_COVERAGE` | `MAP_REDUCE` | False | `NORMAL` |
| `DOCUMENT_COMPARISON` | **True** | 8 | 0.20 | 10 | `BALANCED_MULTI_DOCUMENT` | `BALANCED_DOCUMENT_RETRIEVAL` | False | `NORMAL` |
| `TREATMENT_QUERY` | **True** | 6 | 0.25 | 7 | `TREATMENT_PRIORITY` | `MULTI_DOCUMENT` | False | `NORMAL` |
| `PREVENTION_QUERY` | **True** | 5 | 0.25 | 6 | `PREVENTION_PRIORITY` | `MULTI_DOCUMENT` | False | `NORMAL` |
| `SYMPTOM_QUERY` | **True** | 5 | 0.25 | 6 | `SYMPTOM_PRIORITY` | `MULTI_DOCUMENT` | False | `NORMAL` |
| `GENERAL_HEALTH` | **True** | 5 | 0.25 | 6 | `STANDARD` | `MULTI_DOCUMENT` | False | `NORMAL` |
| `UNCERTAIN` | **True** | 5 | 0.25 | 5 | `STANDARD` | `MULTI_DOCUMENT` | **True** | `NONE` |

---

## 4. Adaptive Retrieval & Gating Mechanisms

### 4.1 Lightweight Deterministic Chunk Weighting (`apply_chunk_weighting`)
Adjusts candidate chunk relevance scores without external LLM calls:
- **`MEDICATION_PRIORITY`**: +0.06 boost for chunks referencing drug pharmacology, adverse effects, side effects, contraindications, and mechanism of action.
- **`DOSAGE_PRIORITY`**: +0.08 boost for chunks containing numerical dosage amounts, frequencies, titration schedules, or mg/mcg units.
- **`LAB_PRIORITY`**: +0.08 boost for reference ranges, assays, normal/abnormal biomarker levels, or mg/dL cutoffs.
- **`DIAGNOSTIC_PRIORITY`**: +0.07 boost for diagnostic criteria, differential diagnosis, and workup guidelines.
- **`TREATMENT_PRIORITY`**: +0.06 boost for first-line therapies, protocols, and clinical management.
- **`BROAD_COVERAGE`**: +0.03 boost preserving wide document section dispersion.
- **`BALANCED_MULTI_DOCUMENT`**: +0.02 boost prioritizing comparative evidence.

### 4.2 Multi-Document Evidence Selection (`select_multi_document_evidence`)
- **`SINGLE_DOCUMENT`**: Returns top-ranked chunks from the active/scoped document.
- **`MULTI_DOCUMENT`**: Standard multi-document selection with duplicate suppression.
- **`MAP_REDUCE`**: Ensures broad representation across sections and pages for document summaries.
- **`BALANCED_DOCUMENT_RETRIEVAL`**: Implements deterministic round-robin extraction across distinct document IDs, preventing any single document from monopolizing evidence slots simply because it contains more total chunks.

### 4.3 High-Confidence Sufficiency Gate
When `requires_high_confidence_evidence` is enabled:
- **`DOSAGE_QUERY`**: Retrieved chunks must contain quantified numerical dosages with explicit units (e.g., `10 mg`, `500 mcg`, `tablets once daily`). General mentions of drug names without dosage amounts fail sufficiency with `insufficient_dosage_evidence_in_context`.
- **`DIAGNOSIS_QUERY`**: Retrieved chunks must contain diagnostic criteria, clinical evaluation guidelines, or differential diagnosis. General condition mentions fail sufficiency with `insufficient_diagnostic_evidence_in_context`.
- **`UNCERTAIN`**: Enforces strict conservative validation to prevent hallucinated answers for ambiguous questions.

### 4.4 Bounded Clinical Query Expansion (`generate_expansions`)
- Extracts primary clinical entity (e.g., drug name or medical condition) via pure Python regex.
- Formulates up to 4 high-signal clinical expansions (e.g., `"{entity} adverse effects"`, `"{entity} contraindications"`).
- Bound: Strictly capped at **<= 4 expansions**.
- Latency: Pure Python execution taking **< 0.25 ms**.
- Safety Bound: Strictly returns `[]` (zero expansions) for `EMERGENCY`, `SELF_HARM`, `POISONING`, and `OUT_OF_SCOPE`.

---

## 5. Security, Tenancy & Cache Isolation

1. **Multi-Tenant Document Filtering**:
   - When `user_id` is present, the plan assigns `document_filter_strategy=DocumentFilterStrategy.USER_DOCUMENTS_ONLY`, restricting vector search to user-owned chunks.
   - When `user_id` is `None`, `DocumentFilterStrategy.ALL_AVAILABLE` is assigned.
   - When an explicit document reference (`[Doc: lab_report.pdf]`) is detected, `DocumentFilterStrategy.SCOPED_DOCUMENT` isolates search to the requested document.
2. **Cache Key Isolation**:
   - Updated `generate_cache_key(normalized_query, ..., query_plan_strategy=...)` incorporates the active routing strategy into the SHA-256 hash.
   - Queries with identical text but different planning strategies (e.g., `medication_rag` vs `dosage_rag`) generate completely distinct cache keys, eliminating cache cross-contamination.
   - Fully backward-compatible: existing calls without `query_plan_strategy` generate identical legacy keys.
3. **Observability Privacy Invariant**:
   - Zero user queries, user IDs, document IDs, patient names, or clinical details are included in Prometheus metric labels. Only low-cardinality enum values are exported.

---

## 6. Performance Benchmark Results

Executed via `scripts/benchmark_phase6_2.py` (1,200 invocations across representative clinical queries):

| Benchmark Metric | Target | Measured Result | Status |
|---|---|---|---|
| Query Planning Latency (p50) | < 2.0 ms | **0.205 ms** | **PASSED [OK]** |
| Query Planning Latency (p95) | < 5.0 ms | **0.319 ms** | **PASSED [OK]** |
| Query Planning Latency (p99) | < 10.0 ms | **0.731 ms** | **PASSED [OK]** |
| Query Planning Latency (Avg) | < 2.0 ms | **0.203 ms** | **PASSED [OK]** |
| Query Expansion Latency (p50) | < 1.0 ms | **0.146 ms** | **PASSED [OK]** |
| Query Expansion Latency (p95) | < 2.0 ms | **0.241 ms** | **PASSED [OK]** |
| Query Expansion Count Bound | <= 4 | **Max 4** | **PASSED [OK]** |
| Chunk Weighting Overhead (p95) | < 1.0 ms | **0.2651 ms** | **PASSED [OK]** |
| Multi-Doc Selection Overhead (p95) | < 0.5 ms | **0.0659 ms** | **PASSED [OK]** |
| **All Criteria Passed** | — | **True** | **PASSED [OK]** |

Benchmark artifact saved at: `evaluation_reports/benchmark_phase6_2_results.json`.

---

## 7. Test Suite Verification & Invariants Check

### 7.1 Dedicated Phase 6.2 Test Suite (`tests/test_phase6_2_query_planner.py`)
Covers all 20 required specifications:
1. Valid plan generated for every intent type (12 clinical intents + 3 safety categories) — **PASSED**
2. Emergency, self-harm, and poisoning queries result in retrieval being completely disabled — **PASSED**
3. Dosage and diagnosis queries require high-confidence evidence — **PASSED**
4. Medication queries use `MEDICATION_PRIORITY` weighting — **PASSED**
5. Lab queries use `LAB_PRIORITY` weighting — **PASSED**
6. Document summary queries use `MAP_REDUCE` multi-document strategy — **PASSED**
7. Document comparison queries use `BALANCED_DOCUMENT_RETRIEVAL` strategy — **PASSED**
8. Out-of-scope queries bypass retrieval — **PASSED**
9. Low-confidence/uncertain intents produce conservative plans — **PASSED**
10. Query expansion produces bounded (<= 4), deterministic output — **PASSED**
11. Query expansion handles edge cases (empty strings, special chars, numbers, null bytes) — **PASSED**
12. Multi-tenant isolation: different users receive independent, isolated plans — **PASSED**
13. Cache isolation: plans with different strategies don't collide in cache — **PASSED**
14. End-to-end integration: query planning works with RAGService (non-streaming and streaming) — **PASSED**
15. Retrieval parameters in plan are respected by RAGService — **PASSED**
16. Relevance verification uses plan-specified thresholds and high-confidence gating — **PASSED**
17. Chunk weighting affects chunk ordering deterministically — **PASSED**
18. Multi-document evidence selection respects document limits and balanced distribution — **PASSED**
19. Performance: query planning completes in < 5ms (measured p95 = 0.319 ms) — **PASSED**
20. Observability: query planning records low-cardinality Prometheus metrics without error — **PASSED**

**Result: 52 passed in 66.28s (100% pass rate).**

### 7.2 Regression Test Suite Verification
- `tests/test_phase6_1_intent_classifier.py`: **29 passed** (100%)
- `tests/test_phase4_medical_safety.py`: **8 passed** (100%)
- `tests/test_phase3_5_redis_cache.py` & `tests/test_phase3_4_auth_isolation.py`: **20 passed** (100%)

### 7.3 Production FAISS Vector Store Invariants
Verified live via `VectorStoreService`:
- **Vector Count**: **744** (exact match)
- **Metadata Records**: **744** (exact match)
- **Embedding Dimension**: **384** (exact match)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (read-only, unmodified)

---

## 8. Summary & Transition to Phase 6.3

Milestone 6.2 is **100% COMPLETE**. The clinical query planning and adaptive retrieval strategies layer is fully integrated, thoroughly validated against 20 critical criteria, backed by extensive benchmark data, and verified to have zero regressions across all existing production phases.

The system is fully prepared for **Phase 6.3 — Multi-Hop Medical Reasoning & Agentic Tool Invocation**.
