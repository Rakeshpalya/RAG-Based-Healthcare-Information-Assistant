# Phase 6.8 Requirements Specification
## Clinical Intelligence Orchestration & Auditability

> **AUTHORITATIVE NOTICE**:<br>
> The original Phase 6.8 specification was exhaustively searched for across the repository, parent folders, Downloads, Desktop, Documents, Git commit history, tags, branches, reflogs, the GitHub remote, and conversation transcripts, and was confirmed to NOT exist.<br>
> Following explicit user authorization, Phase 6.8 is an **agent-designed continuation** extending the completed Phase 6.1–6.7 architecture without replacing any existing functionality.

---

### 1. Milestone Overview
- **Title**: Phase 6.8 — Clinical Intelligence Orchestration & Auditability
- **Module**: `backend/intelligence/orchestration_models.py`, `backend/intelligence/clinical_orchestrator.py`
- **Scope**: Deterministic end-to-end clinical reasoning pipeline orchestration, structured provenance generation, cryptographic audit checksums, and zero-PHI operational observability.

---

### 2. Design Goals
1. **End-to-End Orchestration**: Provide a unified, deterministic orchestrator that sequences and verifies Phase 6.1 (Intent Classification), Phase 6.2 (Query Planning), Phase 6.3 (Evidence Fusion), Phase 6.4 (Answer Synthesis), Phase 6.5 (Citation Attribution), Phase 6.6 (Clinical Grounding Verification), and Phase 6.7 (Clinical Decision Support).
2. **Deterministic Provenance Tracking**: Capture a structured execution record detailing the status, execution latency, and safety invariants of each clinical pipeline stage.
3. **Cryptographic Audit Checksum**: Generate a deterministic SHA256 checksum of stage inputs, outputs, citation validations, and risk assessments to ensure auditability and tamper-evidence for medical-legal compliance.
4. **Multi-Tenant & User Isolation**: Verify that provenance records, accessed chunk identifiers, and document metadata remain strictly bound to the requesting user/tenant without cross-tenant contamination.
5. **Medical Safety & Non-Prescriptive Guardianship**: Guarantee that existing emergency, self-harm, and poisoning intercepts remain authoritative, and ensure decision-support provenance cannot be converted into autonomous diagnosis or prescribing authority.
6. **Zero-PHI Observability**: Record low-cardinality Prometheus metrics without leaking patient identifiers, medical record numbers, raw clinical text, or query queries.
7. **Strict Invariant Preservation**: Maintain FAISS vectors = 744, metadata records = 744, embedding dimension = 384, and model = `sentence-transformers/all-MiniLM-L6-v2`.
8. **100% Backward Compatibility**: Ensure existing RAG query endpoints (`/rag/query`) and SSE streaming (`/rag/stream`) function seamlessly without breaking legacy consumers.

---

### 3. Explicit Non-Goals
- **NO** EHR / FHIR interoperability or external hospital database integrations.
- **NO** multi-turn conversation state machines or session persistence alterations.
- **NO** autonomous medical diagnosis or prescription calculation.
- **NO** external automated clinical actions or autonomous alert dispatching.
- **NO** additional LLM providers or external API dependencies.
- **NO** reindexing, index rebuilding, or vector-store modification.

---

### 4. Components Reused from Phases 6.1–6.7
- `ClinicalIntentClassifier` & `IntentClassificationResult` (Phase 6.1)
- `ClinicalQueryPlanner` & `QueryPlan` (Phase 6.2)
- `ClinicalEvidenceFusionEngine` & `FusedContextResult` (Phase 6.3)
- `ClinicalAnswerSynthesisEngine` & `AnswerSynthesisResult` (Phase 6.4)
- `ClinicalCitationAttributionEngine` & `CitationAttributionReport` (Phase 6.5)
- `ClinicalVerificationEngine` & `ClinicalVerificationResult` (Phase 6.6)
- `ClinicalDecisionSupportEngine` & `ClinicalDecisionSupportResult` (Phase 6.7)
- `MedicalSafetyClassifier` & `SafetyAssessment` (Phase 4 / Core Safety)

---

### 5. New Phase 6.8 Components
1. **`backend/intelligence/orchestration_models.py`**:
   - `PipelineStage`: Enum of stages (Intent, Planning, Fusion, Synthesis, Citation, Verification, Decision Support, Orchestration).
   - `StageExecutionStatus`: Enum (`SUCCESS`, `SKIPPED`, `DEGRADED`, `INTERCEPTED`, `FAILED`).
   - `StageAuditRecord`: Dataclass capturing stage name, status, latency_ms, flags, and sanitized summary.
   - `ClinicalSafetyProvenance`: Dataclass consolidating safety assessment, red flags, risk tier, uncertainty, and citation/grounding validation flags.
   - `DataIsolationProvenance`: Dataclass capturing tenant ID, user ID, accessed document/chunk counts, and isolation verification.
   - `ClinicalIntelligenceOrchestrationResult`: Root strongly typed model containing stage records, safety provenance, data isolation provenance, audit checksum, and concordance indicator.
2. **`backend/intelligence/clinical_orchestrator.py`**:
   - `ClinicalIntelligenceOrchestrator`: Pure Python deterministic orchestration and audit engine.
3. **`backend/evaluation/observability.py` additions**:
   - Metric counters: `phase6_8_orchestration_total`, `phase6_8_orchestration_concordant_total`, `phase6_8_orchestration_intercepted_total`.
   - Latency histogram: `latencies_orchestration`.
   - Helper function: `record_orchestration_event(...)`.
4. **Integration**:
   - Synchronous path: `RAGService.generate_rag_answer` output payload contains `orchestration` and `clinical_intelligence_orchestration`.
   - Streaming path: `RAGService.generate_rag_stream` yields `clinical_intelligence_orchestration` SSE event.
   - Router schemas: `backend/api/rag_router.py` exposes optional `orchestration` field in `RAGQueryResponse`.

---

### 6. Performance Targets
- **Deterministic Orchestration Overhead**: Mean latency < 1.0 ms, p95 < 2.5 ms over 200 benchmark iterations.
- **Resource Footprint**: Zero external network requests, zero additional memory allocations exceeding 1MB.

---

### 7. Acceptance Criteria Checklist
- [ ] `PHASE_6_8_REQUIREMENTS.md` created.
- [ ] `PHASE_6_8_ARCHITECTURE.md` created.
- [ ] Strongly typed models created in `backend/intelligence/orchestration_models.py`.
- [ ] Core orchestrator engine created in `backend/intelligence/clinical_orchestrator.py`.
- [ ] Models and engine exported in `backend/intelligence/__init__.py`.
- [ ] Observability metrics added in `backend/evaluation/observability.py` with zero PHI.
- [ ] Integrated into `RAGService.generate_rag_answer` and `RAGService.generate_rag_stream`.
- [ ] Dedicated test suite `tests/test_phase6_8_orchestration_auditability.py` covering ≥ 20 scenarios with 100% pass rate.
- [ ] Benchmark script `scripts/benchmark_phase6_8.py` executed for 200 iterations with p95 meeting target.
- [ ] Full regression suite passing 100% across all milestones (Phases 3.4, 3.5, 4, 6.1–6.8).
- [ ] Production invariants confirmed: FAISS = 744, metadata = 744, dimension = 384, model = `all-MiniLM-L6-v2`.
- [ ] `PHASE_6_8_REPORT.md` generated documenting all verification results.
- [ ] Review git diff, commit, and push to GitHub remote.
