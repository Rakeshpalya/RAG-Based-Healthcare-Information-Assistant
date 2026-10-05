# Phase 6.1 — Clinical Intent Detection & Intelligent Query Routing Report

## Executive Summary & Objective

**Milestone 6.1** establishes a deterministic, sub-millisecond clinical intent classification and query routing layer for the **AI-Healthcare-Agent** medical RAG application. By evaluating user inquiries *before* vector search or generative processing, the system accurately maps clinical queries to specialized retrieval and processing strategies while guaranteeing absolute precedence to emergency and medical safety guardrails.

---

## 1. Existing Pipeline Inspection

Prior to introducing intent detection, the RAG execution flow was inspected across `backend/api/rag_router.py`, `backend/rag/rag_service.py`, `backend/safety/`, `backend/evaluation/`, and `backend/security/`:

### Standard Execution Pipeline Order
```
1. Client HTTP POST (/rag/query or /rag/stream)
     ↓
2. Authentication (Bearer JWT resolution via FastAPI dependency)
     ↓
3. Rate Limiter (Redis-backed token bucket / sliding window per user_id & IP)
     ↓
4. Request ID Assignment (Propagation of X-Request-ID or UUID trace generation)
     ↓
5. Medical Safety Pre-Screen (MedicalSafetyGuard.pre_screen_inquiry)
     - Emergency symptoms (chest pain, stroke, dyspnea, cardiac arrest)
     - Self-harm & suicide ideation (988 Lifeline referral)
     - Poisoning & acute toxic overdose
     → If triggered: immediate interception without LLM call or vector search.
     ↓
6. Clinical Intent Detection & Query Routing (NEW - Milestone 6.1)
     - Deterministic intent classification (< 0.5 ms)
     - Safety precedence alignment
     - Structured routing strategy assignment
     - Observability telemetry emission
     ↓
7. Cache Lookup (L1 Memory / L2 Redis, user & document scoped)
     ↓
8. Query Resolution & FAISS Vector Retrieval (Top-K semantic chunks)
     ↓
9. Relevance & Sufficiency Gate (Threshold check; halts if unsupported)
     ↓
10. Gemini LLM Generation (System prompts & grounded evidence)
     ↓
11. Citation Validation (Deterministic bracket [N] verification against sources)
     ↓
12. Grounding Validation & Hallucination Guard (Claim support scoring)
     ↓
13. Medical Safety Post-Screen (Prohibited prescriptive/diagnostic scrubbing)
     ↓
14. Cache Store (Validated safe responses only)
     ↓
15. Structured Response / SSE Stream Delivery
```

### Safest Integration Point
The safest integration point is immediately following `MedicalSafetyGuard.pre_screen_inquiry(question)`. If the pre-screen flags an emergency or safety violation, the intent classifier consumes that assessment and locks the intent to `EMERGENCY`, `SELF_HARM`, or `POISONING` with `safety_priority="critical"` and `requires_retrieval=False`. If benign, deterministic signal evaluation proceeds to route the query appropriately.

---

## 2. Clinical Intent Taxonomy

Defined in `backend/intelligence/intent_models.py`, the standardized taxonomy encompasses:

| # | Clinical Intent | Description | Safety Priority | Default Routing Strategy |
|---|---|---|---|---|
| 1 | `EMERGENCY` | Acute life-threatening symptoms (chest pain, stroke, dyspnea) | `CRITICAL` | `emergency_safety` |
| 2 | `SELF_HARM` | Suicide or self-harm ideation/methods | `CRITICAL` | `self_harm_safety` |
| 3 | `POISONING` | Accidental/intentional toxic ingestion or overdose | `CRITICAL` | `poisoning_safety` |
| 4 | `DOSAGE_QUERY` | Inquiries regarding drug doses, schedules, or mg amounts | `HIGH` | `dosage_rag` |
| 5 | `DIAGNOSIS_QUERY` | Direct inquiries about illness diagnosis or symptoms analysis | `HIGH` | `diagnosis_rag` |
| 6 | `MEDICATION_QUERY` | Drug mechanisms, side effects, interactions, or pharmacology | `NORMAL` | `medication_rag` |
| 7 | `LAB_RESULT_QUERY` | Blood panels, biomarkers (HbA1c, CBC, lipids), and test ranges | `NORMAL` | `lab_rag` |
| 8 | `TREATMENT_QUERY` | First-line therapies, clinical management, and interventions | `NORMAL` | `treatment_rag` |
| 9 | `PREVENTION_QUERY` | Prophylaxis, lifestyle changes, risk reduction, vaccines | `NORMAL` | `prevention_rag` |
| 10 | `DOCUMENT_SUMMARY` | Requests to summarize attached reports, studies, or records | `NORMAL` | `document_summary_rag` |
| 11 | `DOCUMENT_COMPARISON` | Requests to compare or contrast multiple studies or documents | `NORMAL` | `document_comparison_rag` |
| 12 | `GENERAL_HEALTH` | Pathophysiology, disease definitions, or biological mechanisms | `NORMAL` | `general_health_rag` |
| 13 | `OUT_OF_SCOPE` | Non-medical inquiries, programming, trivia, prompt overrides | `NONE` | `out_of_scope_response` |
| 14 | `UNCERTAIN` | Vague, empty, punctuation-only, or unclassifiable queries | `NONE` | `standard_rag` |

---

## 3. Classifier Architecture

Implemented in `backend/intelligence/intent_classifier.py`:

- **Zero LLM Dependency**: Operates completely deterministically without calling external APIs or loading large transformer neural nets for classification.
- **Compiled Pattern Matching**: Precompiled regular expression groups and keyword dictionaries organized by clinical domain.
- **Sanitization & Boundary Enclosure**:
  - Strips null bytes (`\x00`), HTML/script tags (`<script>`, `<iframe>`), and unescapes entities.
  - Bounded input slicing (`query[:4000]`) to protect against ReDoS or memory exhaustion on large inputs.
- **Clinical Disambiguation Logic**:
  - When a query contains both medication keywords and dosage inquiry patterns (e.g., *"What dose of metformin is usually prescribed?"*), `DOSAGE_QUERY` strictly takes precedence over `MEDICATION_QUERY` due to its elevated clinical boundary requirements.
  - Document comparison takes precedence over general summarization when multiple comparative document references are identified.
- **Confidence Scoring**: Computed based on match score weights and signal separation margins (ranges from 0.0 for empty queries to 0.99 for exact clinical matches).

---

## 4. Safety Precedence Invariants

- **Medical Safety Absolute Supremacy**: Intent classification never overrides or relaxes an existing safety block.
- **Evaluation Order**:
  ```python
  allow_rag, safety_assessment, immediate_msg = MedicalSafetyGuard.pre_screen_inquiry(question)
  intent_result = ClinicalIntentClassifier.classify(question, safety_assessment=safety_assessment)
  ```
- **Guaranteed Lock**:
  - If `safety_assessment` indicates emergency, self-harm, or poisoning, `intent_result` is locked to that safety category with `requires_retrieval=False`, `safety_priority=CRITICAL`, and `routing_strategy=*_safety`.
  - Immediate refusal and safety advisories (911 / 988 Lifeline) take precedence over any subsequent pipeline step.

---

## 5. Routing Architecture & Phase 6.2 Readiness

The classification output returns a structured `IntentClassificationResult` object:

```json
{
  "intent": "MEDICATION_QUERY",
  "confidence": 0.96,
  "matched_signals": ["MED_SIDE_EFFECTS", "COMMON_DRUG_NAMES"],
  "requires_retrieval": true,
  "requires_document_context": false,
  "safety_priority": "normal",
  "routing_strategy": "medication_rag",
  "latency_ms": 0.18,
  "recommended_top_k": 5,
  "recommended_similarity_threshold": 0.25,
  "metadata": {
    "top_score": 3.6,
    "second_intent": null,
    "second_score": 0.0
  }
}
```

This structured result provides the foundation for **Phase 6.2 Query Planning**:
- `requires_document_context=True` instructs query planners to require user document scopes (`DOCUMENT_SUMMARY_RAG`, `DOCUMENT_COMPARISON_RAG`).
- `recommended_top_k` and `recommended_similarity_threshold` allow dynamic retrieval tuning per clinical domain.
- `requires_retrieval=False` enables immediate rejection of `OUT_OF_SCOPE` or emergency queries without database load.

---

## 6. Cache Compatibility & Security

- **Multi-Tenant Cache Isolation**: The intent result is embedded in the response payload after retrieval and generation without altering cache key isolation (`user_scope`, `document_signature`, `prompt_version`).
- **Zero Cache Collisions**: Query routing does not introduce ambiguous cache collisions; responses are cached only when validated by grounding and safety gates.
- **Low-Cardinality Telemetry**:
  - Intent metric labels strictly use bounded enum strings (e.g. `intent="MEDICATION_QUERY"`).
  - Raw query text, user IDs, tokens, and PHI are strictly excluded from all metric counters and logs.

---

## 7. Performance & Latency Measurements

Benchmarked using 1,000 synthetic clinical queries across all taxonomy categories:

| Metric | Measured Value | Requirement / Target | Status |
|---|---|---|---|
| **Classification Median (p50)** | **0.107 ms** | < 5.0 ms | **PASS** |
| **Classification 95th Percentile (p95)** | **0.273 ms** | < 10.0 ms | **PASS (36x faster than target)** |
| **Classification 99th Percentile (p99)** | **0.547 ms** | < 20.0 ms | **PASS** |
| **Memory Footprint** | **0 MB added** | Lightweight | **PASS** |

---

## 8. Test Verification & Regressions

### Phase 6.1 Test Suite (`tests/test_phase6_1_intent_classifier.py`)
- **Total Tests**: **29 / 29 passed (100%)**
- Category Coverage:
  - Clinical Taxonomy Classification (symptom, diagnosis, medication, dosage, lab result, treatment, prevention, summary, comparison, general health, out-of-scope): **11 / 11 passed**
  - Safety Precedence (emergency, self-harm, poisoning, external safety override): **4 / 4 passed**
  - Ambiguous & Edge Cases (empty, whitespace, punctuation, 10,000 char input, mixed intent): **5 / 5 passed**
  - Adversarial Defense (prompt injection, jailbreak, null bytes, HTML/XSS): **4 / 4 passed**
  - Performance & Observability (latency target, metric counters, Prometheus format): **2 / 2 passed**
  - Pipeline Integration (RAGService method, safety interception, grounded answer): **3 / 3 passed**

### Regression Test Matrix
All historical regression suites were executed and verified:

| Test Suite | Focus Area | Tests Passed | Status |
|---|---|---|---|
| `test_phase6_1_intent_classifier.py` | Clinical Intent Detection & Routing | **29 / 29** | **PASS** |
| `test_phase5_*.py` | Production Application & Security | **38 / 38** | **PASS** |
| `test_phase4_*.py` | Medical AI Evaluation & Safety | **50 / 50** | **PASS** |
| `test_phase3_6_*.py` | Production Release & Resilience | **80 / 80** | **PASS** |
| `test_phase3_5_*.py` | Distributed Rate Limiter & Redis | **58 / 58** | **PASS** |
| `test_phase3_4_*.py` | Production Hardening & Observability | **82 / 82** | **PASS** |
| **Total Regressions** | **Complete System Regression** | **337 / 337** | **PASS (100%)** |

### Vector Store Invariants
Verified via FAISS index and metadata introspection:
- FAISS Vectors: **744** (Exactly preserved)
- Metadata Records: **744** (Exactly preserved)
- Embedding Dimension: **384** (`all-MiniLM-L6-v2`)

---

## 9. Observability & Prometheus Metrics Added

Integrated into `backend/evaluation/observability.py`:
- `rag_intent_classifications_total{intent="..."}`: Counter tracking classifications per intent category.
- `rag_intent_routing_total{strategy="..."}`: Counter tracking queries routed to each clinical retrieval strategy.
- `rag_intent_uncertain_total`: Counter tracking ambiguous or unclassified queries.
- `rag_intent_latency_seconds`: Histogram tracking classification latency percentiles.

---

## 10. Known Limitations & Phase 6.2 Integration Plan

### Known Limitations
1. **Deterministic Rule Coverage**: While the regex and keyword patterns cover standard clinical nomenclature, slang or heavily misspelled queries (e.g., *"metforman sideefects"*) may fall back to `GENERAL_HEALTH` or `UNCERTAIN`. Phase 6.2 can incorporate clinical query expansion before intent classification.
2. **Compound Multi-Domain Queries**: Highly complex clinical queries spanning multiple distinct questions (e.g., *"What is the dose of lisinopril and can you summarize this attached trial?"*) are resolved to the highest priority single intent.

### Phase 6.2 Plan (Query Planning & Adaptive Retrieval)
- Leverage `IntentClassificationResult` to invoke dynamic query expansion algorithms tailored to the detected intent.
- Route `DOCUMENT_SUMMARY` and `DOCUMENT_COMPARISON` through dedicated multi-chunk map-reduce aggregators.
- Apply domain-specific relevance thresholds based on `recommended_similarity_threshold`.
