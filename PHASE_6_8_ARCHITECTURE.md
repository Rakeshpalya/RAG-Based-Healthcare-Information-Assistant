# Phase 6.8 Architecture Mapping
## Clinical Intelligence Orchestration & Auditability

> **NOTICE**: Agent-designed architecture continuation authorized due to absence of original specification in repository or history.

---

### 1. End-to-End Pipeline Data Flow

The complete Clinical Intelligence architecture establishes a deterministic, multi-stage reasoning pipeline:

```
                      +---------------------------------------+
                      |         Incoming User Query           |
                      +-------------------+-------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 0: Medical Safety Classifier (Phase 4 / Core Safety)                        |
| - Intercepts: Emergency, Self-Harm, Poisoning                                      |
| - Precedence: Fast-fail before RAG/LLM invocation                                  |
+-----------------------------------------+-----------------------------------------+
                                          | Non-intercepted queries
                                          v
+-----------------------------------------------------------------------------------+
| Stage 1: Clinical Intent Detection & Routing (Phase 6.1)                          |
| - Output: IntentClassificationResult (Emergency, Diagnostic, Medication, etc.)    |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 2: Clinical Query Planning & Adaptive Retrieval (Phase 6.2)                 |
| - Output: QueryPlan (Chunk weighting, multi-doc strategy, top_k, threshold)       |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 3: Clinical Evidence Fusion & Multi-Document Reasoning (Phase 6.3)          |
| - Output: FusedContextResult (Deduplication, conflict detection, coverage status)  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 4: Clinical Answer Synthesis & Evidence Grounding (Phase 6.4)               |
| - Output: AnswerSynthesisResult (Structured sections, clinical confidence)        |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 5: Clinical Citation & Attribution Auditing (Phase 6.5)                     |
| - Output: CitationAttributionReport (Span-level support, precision, recall)        |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 6: Clinical Grounding Verification & Hallucination Guardrails (Phase 6.6)   |
| - Output: ClinicalVerificationResult (Trajectory, polarity, dosage validation)    |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Stage 7: Clinical Decision Support & Care Pathways (Phase 6.7)                    |
| - Output: ClinicalDecisionSupportResult (Uncertainty, risk tier, red flags, SBAR) |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+===================================================================================+
| STAGE 8: CLINICAL INTELLIGENCE ORCHESTRATION & AUDITABILITY (PHASE 6.8)           |
|                                                                                   |
| Engine: ClinicalIntelligenceOrchestrator                                          |
| Tasks:                                                                            |
| 1. Collects structured execution artifacts from Stages 1 through 7.               |
| 2. Validates cross-stage invariants (safety, citation grounding, risk tiering).   |
| 3. Enforces data isolation and verifies zero cross-tenant contamination.          |
| 4. Generates cryptographic SHA256 audit checksum for medical-legal provenance.     |
| 5. Emits zero-PHI Prometheus observability metrics.                               |
| 6. Returns ClinicalIntelligenceOrchestrationResult.                              |
+===================================================================================+
                                          |
                                          +-----------------------------------------+
                                          |                                         |
                                          v                                         v
                      +-----------------------+                 +-----------------------+
                      | POST /rag/query       |                 | POST /rag/stream      |
                      | JSON Response Payload |                 | Server-Sent Events    |
                      | - orchestration field |                 | - orchestration event |
                      +-----------------------+                 +-----------------------+
```

---

### 2. Module & File Mapping

| Milestone / Concern | Existing / Reused Module | New Phase 6.8 Module | Key Abstractions |
|:---|:---|:---|:---|
| **6.1 Intent** | `backend.intelligence.intent_classifier` | — | `ClinicalIntentClassifier`, `IntentClassificationResult` |
| **6.2 Query Plan** | `backend.intelligence.query_planner` | — | `ClinicalQueryPlanner`, `QueryPlan` |
| **6.3 Evidence Fusion** | `backend.intelligence.evidence_fusion` | — | `ClinicalEvidenceFusionEngine`, `FusedContextResult` |
| **6.4 Synthesis** | `backend.intelligence.answer_synthesis` | — | `ClinicalAnswerSynthesisEngine`, `AnswerSynthesisResult` |
| **6.5 Citation** | `backend.intelligence.citation_attribution` | — | `ClinicalCitationAttributionEngine`, `CitationAttributionReport` |
| **6.6 Verification** | `backend.intelligence.clinical_verification` | — | `ClinicalVerificationEngine`, `ClinicalVerificationResult` |
| **6.7 Decision Support** | `backend.intelligence.clinical_decision_support` | — | `ClinicalDecisionSupportEngine`, `ClinicalDecisionSupportResult` |
| **6.8 Orchestration & Audit** | — | `backend.intelligence.orchestration_models` | `PipelineStage`, `StageAuditRecord`, `ClinicalSafetyProvenance`, `DataIsolationProvenance`, `ClinicalIntelligenceOrchestrationResult` |
| **6.8 Orchestrator Engine** | — | `backend.intelligence.clinical_orchestrator` | `ClinicalIntelligenceOrchestrator` |
| **Observability** | `backend.evaluation.observability` | Extended | `record_orchestration_event`, `ProductionMetricsCollector` counters/histograms |
| **Pipeline Service** | `backend.rag.rag_service` | Modified | Integration into `generate_rag_answer` and `generate_rag_stream` |
| **API Router** | `backend.api.rag_router` | Modified | Optional `orchestration` field in `RAGQueryResponse` |
| **Dedicated Tests** | — | `tests/test_phase6_8_orchestration_auditability.py` | 20+ comprehensive tests |
| **Performance Benchmark** | — | `scripts/benchmark_phase6_8.py` | 200 iterations latency benchmark |

---

### 3. Pipeline Invariants & Safety Verification

1. **Non-Prescriptive Constraint**: The orchestrator verifies that Phase 6.7 decision support recommendations and Phase 6.6 verification results have neutralized directive language. It cannot convert guidance into autonomous diagnosis.
2. **Deterministic Tamper-Evidence**:
   $$\text{Audit Checksum} = \text{SHA256}(\text{trace\_id} \parallel \text{intent} \parallel \text{risk\_tier} \parallel \text{sanitized\_answer} \parallel \text{valid\_citations})$$
   This guarantees that post-synthesis clinical auditing is tamper-evident.
3. **Strict Zero-PHI Observability**: Metrics emitted to Prometheus contain ONLY:
   - Request counts (`phase6_8_orchestration_total`)
   - Concordance counts (`phase6_8_orchestration_concordant_total`)
   - Intercept counts (`phase6_8_orchestration_intercepted_total`)
   - Latency samples (`latencies_orchestration`)
   - Zero patient identifiers, zero document text, zero queries.
