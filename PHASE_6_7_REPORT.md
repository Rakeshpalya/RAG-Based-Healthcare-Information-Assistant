# Phase 6.7 — Clinical Decision Support, Uncertainty Calibration & Care Pathways Report

## Executive Summary & Objectives

**Milestone 6.7** delivers an enterprise-grade, deterministic **Clinical Decision Support, Uncertainty Calibration & Care Pathways Engine** for the **AI-Healthcare-Agent** medical RAG platform.

Building directly on Phase 6.1 (Clinical Intent Classification), Phase 6.2 (Clinical Query Planning), Phase 6.3 (Clinical Evidence Fusion), Phase 6.4 (Clinical Answer Synthesis & Evidence Grounding), Phase 6.5 (Clinical Citation & Claim Attribution), and Phase 6.6 (Clinical Grounding Verification & Hallucination Guardrails), Phase 6.7 elevates the platform from an accurate retrieval engine into a guideline-concordant, physician-assisting **Clinical Decision Support (CDS)** system:

1. **Multi-Factor Composite Uncertainty Calibration**: Synthesizes algorithmic signals across the intelligence pipeline (intent classification confidence, vector similarity score, multi-document evidence coverage, citation precision, and post-synthesis clinical grounding verification). Explicitly categorizes uncertainty into **LOW**, **MODERATE**, **HIGH**, or **INDETERMINATE**, and pinpoints the primary source of uncertainty:
   - **ALEATORIC**: Inherent medical ambiguity, clinical guideline discrepancies, or conflicting inter-document trial evidence.
   - **EPISTEMIC**: Knowledge retrieval deficit, low vector similarity scores (< 0.35), or partial coverage of inquiry aspects.
   - **VERIFICATION_DEFICIT**: Grounding discrepancies, ungrounded claims pruned during post-screening, or detected directional contradictions.
   - **NONE**: High-confidence, fully grounded, guideline-concordant synthesis.
2. **Clinical Risk Tier Stratification**: Implements a calibrated 5-tier clinical risk classification system (**CRITICAL**, **HIGH**, **MODERATE**, **LOW**, **MINIMAL**) dynamically evaluated from query intent, clinical entities, and safety signals. Medication dosage, prescription regimens, and drug-drug interactions stratify as HIGH risk; emergency symptoms, self-harm, and acute toxicity stratify as CRITICAL.
3. **Evidence-Based Clinical Red-Flag Rule Triggers**: Detects acute medical red flags across three clinical domains:
   - **Cardiovascular**: Hypertensive crisis (BP ≥ 180/120 mmHg, headache, visual disturbances), acute coronary syndrome (chest pain radiating to arm/jaw, diaphoresis), and acute stroke (sudden facial droop, arm weakness, slurred speech).
   - **Metabolic**: Severe hypoglycemia (glucose < 54 mg/dL, confusion, tremors) and diabetic ketoacidosis (hyperglycemia, fruity breath, Kussmaul breathing).
   - **Pharmacological / Toxicological**: Anaphylaxis (stridor, wheezing, diffuse urticaria) and severe cutaneous adverse reactions (SCAR, SJS/TEN, progressive blistering rash with fever).
4. **Guideline-Concordant Actionable Care Pathways**: Generates structured, non-prescriptive actionable recommendations aligned with AHA, ACC, ESC, and ADA clinical practice guidelines. All recommendations strictly adhere to non-prescriptive phrasing standards (`"Maintain a structured home BP log"`, `"Consult with prescribing clinician"`, `"Monitor serum creatinine"`) and reject paternalistic directives.
5. **Agentic Context-Aware Follow-Up Inquiries**: Proactively constructs 2 to 4 clinically relevant, diagnostic and symptom-clarifying questions to deepen clinical intake (e.g., asking about recent blood pressure trends, adherence, or concomitant OTC medications).
6. **Physician SBAR Clinical Handoff Summaries**: Assembles standardized clinical handoff summaries conforming to the healthcare industry standard **SBAR** (Situation, Background, Assessment, Recommendation) framework for seamless escalation and electronic health record (EHR) transmission.
7. **Clinical Deferral & Escalation Triggering**: Automatically flags inquiries requiring immediate provider escalation when critical risk tiers, red-flag triggers, verification fallbacks, or low calibrated confidence (< 0.35) thresholds are encountered.
8. **End-to-End Pipeline & Real-Time Streaming Integration**: Fully integrated into `ClinicalAnswerSynthesisEngine` and `RAGService` across both synchronous JSON query endpoints (`/rag/query`) and Server-Sent Events (SSE) streaming (`/rag/stream`), broadcasting dedicated `clinical_decision_support` SSE events.
9. **Sub-Millisecond Deterministic Latency**: Achieves an exceptional engine evaluate p95 latency of **0.136 ms** (mean **0.067 ms**, p50 **0.065 ms**) over 200 benchmark iterations, outperforming the < 5.0 ms threshold requirement by 36x.
10. **Zero-PHI Observability & Production Invariants Preserved**: Emits low-cardinality Prometheus metrics without PHI or sensitive query text, maintains multi-tenant isolation, preserves emergency safety precedence, and strictly preserves production vector store invariants (744 FAISS vectors, 744 metadata records, 384 dimensions).

---

## 1. Architectural Pipeline & Integration Flow

The Clinical Decision Support Engine operates at the conclusion of answer synthesis, ingesting outputs from all preceding intelligence modules:

```
1. Client HTTP Request (/rag/query or /rag/stream)
     ↓
2. Authentication & Multi-Tenant Context Resolution (user_id tenant isolation)
     ↓
3. Rate Limiter (Distributed Redis token bucket)
     ↓
4. Medical Safety Pre-Screen (MedicalSafetyGuard.pre_screen_inquiry)
     - EMERGENCY_SYMPTOMS (chest pain, stroke, severe hemorrhage)
     - SELF_HARM_OR_SUICIDE (988 lifeline intervention)
     - POISONING_OR_OVERDOSE (Poison Control hotline)
     → If triggered: immediate safety refusal (0 retrieval, 0 LLM calls, 0 CDS overhead)
     ↓
5. Clinical Intent Classification (ClinicalIntentClassifier.classify)
     - Classifies inquiry into 1 of 12 clinical intents
     ↓
6. Clinical Query Planning (ClinicalQueryPlanner.plan)
     - Selects Top-K, similarity threshold, weighting, and multi-doc strategy
     ↓
7. Cache Lookup (L1 Memory / L2 Redis, isolated by strategy + user + document signature)
     ↓
8. Vector Search Retrieval (RAGService.query)
     - Tenant and document-scoped filtering
     ↓
9. Clinical Evidence Fusion Engine (ClinicalEvidenceFusionEngine.fuse_evidence)
     - Multi-document deduplication, diversity, conflict detection, coverage analysis
     ↓
10. Clinical Answer Synthesis Engine (ClinicalAnswerSynthesisEngine.synthesize)
     ├─ Step 1: Pre-synthesis coverage evaluation & prompt construction
     ├─ Step 2: LLM generation via GeminiService (or stream)
     ├─ Step 3: Markdown cleaning & bracket parsing
     ├─ Step 4: Citation & Attribution Auditing (Phase 6.5)
     ├─ Step 5: Clinical Grounding Verification & Hallucination Guardrails (Phase 6.6)
     ├─ Step 6: CLINICAL DECISION SUPPORT EVALUATION (Phase 6.7)
     │    ├─ Multi-Factor Uncertainty Calibration (composite score + source classification)
     │    ├─ Clinical Risk Stratification (CRITICAL, HIGH, MODERATE, LOW, MINIMAL)
     │    ├─ Clinical Red-Flag Trigger Screening (Cardiovascular, Metabolic, Pharmacological)
     │    ├─ Actionable Recommendation Assembly (Guideline-concordant care pathways)
     │    ├─ Agentic Follow-Up Inquiry Generation (2-4 relevant clinical questions)
     │    ├─ SBAR Provider Handoff Summary Construction (Situation, Background, Assessment, Recommendation)
     │    └─ Clinical Escalation Flagging (low confidence, critical risk, or red flags)
     ├─ Step 7: Attach ClinicalDecisionSupportResult to AnswerSynthesisResult.metadata
     └─ Step 8: Record Zero-PHI Prometheus Metrics
     ↓
11. Response Delivery (JSON payload or SSE stream with clinical_decision_support event)
```

---

## 2. Strongly Typed Data Models

Located at `backend/intelligence/decision_support_models.py`, all models are strongly typed, JSON-serializable dataclasses and string enums:

```python
class ClinicalUncertaintyLevel(str, Enum):
    """Calibrated confidence tier representing clinical certainty of the response."""
    LOW = "LOW"                      # Calibrated confidence >= 0.80: highly substantiated
    MODERATE = "MODERATE"            # Calibrated confidence 0.60 - 0.79: minor nuance or evidence limits
    HIGH = "HIGH"                    # Calibrated confidence 0.30 - 0.59: significant gaps or conflicts
    INDETERMINATE = "INDETERMINATE"  # Calibrated confidence < 0.30: insufficient evidence or fallback

class ClinicalRiskTier(str, Enum):
    """Clinical risk classification reflecting potential for patient harm."""
    CRITICAL = "CRITICAL"            # Acute emergency, self-harm, poisoning, severe toxicity
    HIGH = "HIGH"                    # Prescription medications, dosing changes, contraindications
    MODERATE = "MODERATE"            # Symptom queries, diagnostic stages, lab evaluations
    LOW = "LOW"                      # Preventive care, lifestyle modifications, guideline summaries
    MINIMAL = "MINIMAL"              # General health education, terminology explanations

class UncertaintySourceType(str, Enum):
    """Categorization of clinical uncertainty origin."""
    NONE = "NONE"                    # High certainty, no significant deficit
    ALEATORIC = "ALEATORIC"          # Inherent clinical variability, conflicting trial guidelines
    EPISTEMIC = "EPISTEMIC"          # Knowledge deficit, low similarity, missing aspects
    VERIFICATION_DEFICIT = "VERIFICATION_DEFICIT" # Hallucinated or pruned claims

class ActionRecommendationType(str, Enum):
    """Category of guideline-concordant care pathway recommendation."""
    DIAGNOSTIC_MONITORING = "DIAGNOSTIC_MONITORING"     # e.g., Daily home BP log, glucose log
    LABORATORY_TESTING = "LABORATORY_TESTING"           # e.g., Serum creatinine, eGFR, electrolytes
    MEDICATION_REVIEW = "MEDICATION_REVIEW"             # e.g., Prescriber reconciliation
    LIFESTYLE_MODIFICATION = "LIFESTYLE_MODIFICATION"   # e.g., DASH diet, sodium restriction
    PREVENTIVE_MEASURE = "PREVENTIVE_MEASURE"           # e.g., Age-appropriate screening
    URGENT_EVALUATION = "URGENT_EVALUATION"             # e.g., Immediate ED or urgent care visit

@dataclass
class ActionableRecommendation:
    recommendation_id: str
    category: ActionRecommendationType
    text: str
    urgency: str  # 'ROUTINE', 'PROMPT', 'EMERGENT'
    evidence_source: Optional[str] = None

@dataclass
class RedFlagTrigger:
    flag_id: str
    category: str
    trigger_name: str
    clinical_criteria: str
    recommended_action: str
    is_critical: bool = True

@dataclass
class ClinicalHandoffSummary:
    situation: str
    background: str
    assessment: str
    recommendation: str
    calibrated_confidence: float
    risk_tier: str
    uncertainty_level: str
    escalation_required: bool

@dataclass
class ClinicalDecisionSupportResult:
    calibrated_confidence: float
    uncertainty_level: ClinicalUncertaintyLevel
    primary_uncertainty_source: UncertaintySourceType
    risk_tier: ClinicalRiskTier
    escalation_required: bool
    escalation_reason: Optional[str]
    actionable_recommendations: List[ActionableRecommendation]
    red_flag_triggers: List[RedFlagTrigger]
    suggested_follow_up_inquiries: List[str]
    clinical_handoff: Optional[ClinicalHandoffSummary]
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 3. Decision Support Engine Capabilities

### 3.1 Composite Uncertainty Calibration Formula
Rather than relying on uncalibrated LLM confidence, the engine fuses multi-stage mathematical signals:
$$\text{Raw Confidence} = 0.15 \times C_{\text{intent}} + 0.20 \times S_{\text{max}} + 0.20 \times F_{\text{coverage}} + 0.20 \times P_{\text{citation}} + 0.25 \times G_{\text{verification}}$$
- $C_{\text{intent}}$: Intent classification baseline confidence (default 0.85).
- $S_{\text{max}}$: Maximum cosine similarity of retrieved evidence chunks.
- $F_{\text{coverage}}$: Evidence coverage factor (1.0 for FULL, 0.65 for PARTIAL, 0.30 for INSUFFICIENT).
- $P_{\text{citation}}$: Citation attribution precision from Phase 6.5.
- $G_{\text{verification}}$: Grounding verification score from Phase 6.6.
- **Penalty for Discrepancies**: If inter-document conflicts are detected by the evidence fusion engine, a direct penalty of $-0.22$ is applied.

### 3.2 Clinical Red-Flag Rule Triggers
The engine screens both user query and generated response text for high-risk red-flag presentations:
- **Cardiovascular**:
  - `RF_HTN_CRISIS`: Systolic BP $\ge$ 180 mmHg or diastolic BP $\ge$ 120 mmHg.
  - `RF_ACUTE_CORONARY`: Crushing chest pain, arm/jaw radiation, diaphoresis.
  - `RF_ACUTE_STROKE`: Sudden facial droop, unilateral arm drift, slurred speech (FAST criteria).
- **Metabolic**:
  - `RF_SEVERE_HYPOGLYCEMIA`: Blood glucose < 54 mg/dL with altered mental status.
  - `RF_DIABETIC_KETOACIDOSIS`: Severe hyperglycemia with Kussmaul respirations and fruity breath.
- **Pharmacological**:
  - `RF_ANAPHYLAXIS`: Acute stridor, wheezing, diffuse urticaria following allergen/drug exposure.
  - `RF_SCAR_REACTION`: Progressive mucocutaneous blistering rash with systemic fever.

### 3.3 Guideline-Concordant Care Pathways
For every identified clinical context, the engine generates concrete, non-prescriptive next steps:
- **Hypertension Pathways**: Initiating home blood pressure logging and dietary sodium restriction (< 2,300 mg/day).
- **Medication Pathways**: Pharmacist or prescribing clinician consultation prior to dosage adjustment; monitoring renal function (eGFR, BUN/Cr) and electrolytes (potassium for ACEi/ARBs).
- **Non-Prescriptive Language**: Replaces paternalistic assertions (`"Take 10 mg"`) with clinical recommendations (`"Clinical practice guidelines suggest discussing with your physician whether..."`).

### 3.4 SBAR Clinical Provider Handoff
Generates structured text facilitating seamless transitions of care:
- **Situation**: Primary clinical inquiry and patient-reported situation.
- **Background**: Referenced authoritative guidelines and evidence base.
- **Assessment**: Synthesized findings, calibrated confidence, risk tier, and detected conflicts.
- **Recommendation**: Suggested care pathway steps and physician escalation advisories.

---

## 4. Zero-PHI Observability & Prometheus Exposition

Observability additions in `backend/evaluation/observability.py` adhere strictly to HIPAA guidelines: zero patient identifiers, user IDs, or medical narratives are included in metrics.

### Counters & Gauges Added
- `rag_decision_support_evaluations_total`: Total CDS evaluations executed.
- `rag_decision_support_uncertainty_total`: Evaluations partitioned by `uncertainty_level` (`LOW`, `MODERATE`, `HIGH`, `INDETERMINATE`).
- `rag_decision_support_uncertainty_source_total`: Evaluations partitioned by `source_type` (`NONE`, `ALEATORIC`, `EPISTEMIC`, `VERIFICATION_DEFICIT`).
- `rag_decision_support_risk_tier_total`: Evaluations partitioned by `risk_tier` (`CRITICAL`, `HIGH`, `MODERATE`, `LOW`, `MINIMAL`).
- `rag_decision_support_red_flags_total`: Red flags detected partitioned by domain category (`cardiovascular`, `metabolic`, `pharmacological`).
- `rag_decision_support_escalations_total`: Total clinician escalation recommendations triggered.
- `rag_decision_support_latency_ms`: Rolling latency histogram tracking engine evaluation duration.

---

## 5. Performance Benchmark Analysis

Executed via `scripts/benchmark_phase6_7.py` across **200 benchmark iterations** with a representative multi-source clinical scenario:

| Component / Stage | Mean Latency | p50 Latency | p95 Latency | p99 Latency | Max Latency | Target Compliance |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Uncertainty Calibration** | 0.015 ms | 0.013 ms | 0.026 ms | 0.037 ms | 0.187 ms | **PASS** (< 1.0 ms) |
| **Risk Stratification** | 0.004 ms | 0.004 ms | 0.008 ms | 0.014 ms | 0.016 ms | **PASS** (< 0.5 ms) |
| **Red-Flag Screening** | 0.022 ms | 0.021 ms | 0.037 ms | 0.055 ms | 0.140 ms | **PASS** (< 1.0 ms) |
| **Actionable Recommendations** | 0.011 ms | 0.010 ms | 0.022 ms | 0.032 ms | 0.082 ms | **PASS** (< 1.0 ms) |
| **Agentic Follow-Up Inquiries** | 0.002 ms | 0.002 ms | 0.004 ms | 0.007 ms | 0.009 ms | **PASS** (< 0.5 ms) |
| **SBAR Clinical Handoff** | 0.019 ms | 0.016 ms | 0.036 ms | 0.076 ms | 0.187 ms | **PASS** (< 1.0 ms) |
| **End-to-End Decision Support evaluate()** | **0.067 ms** | **0.065 ms** | **0.136 ms** | **0.156 ms** | **0.197 ms** | **PASS (< 5.0 ms threshold)** |

### Key Benchmark Takeaways
- **Deterministic Local Overhead**: Total end-to-end evaluation overhead is **0.136 ms at p95**, which is **36x faster** than the strict 5.0 ms requirement.
- **Zero Heavy Computations**: All reasoning, calibration, rule-matching, and text formatting are vectorized and compiled without external API dependencies.
- **Artifact Output**: Full results stored in `evaluation_reports/benchmark_phase6_7_results.json`.

---

## 6. Comprehensive Regression Matrix (264 Tests Passing)

The platform was subjected to the complete regression matrix spanning Phase 3, Phase 4, and all Phase 6 modules. **264 of 264 tests passed (100% pass rate)**:

| Test Suite File | Tests | Status | Passing % | Key Capabilities Validated |
|:---|:---:|:---:|:---:|:---|
| `test_phase6_7_clinical_decision_support.py` | 28 | **PASSED** | 100% | Uncertainty calibration, risk tiers, red flags, recommendations, SBAR, follow-ups, streaming |
| `test_phase6_6_clinical_verification.py` | 28 | **PASSED** | 100% | Directional contradiction, negation conflict, entity grounding, claim pruning, fallback |
| `test_phase6_5_citation_attribution.py` | 27 | **PASSED** | 100% | Claim-level citation mapping, spoofing defense, precision & coverage tracking |
| `test_phase6_4_answer_synthesis.py` | 32 | **PASSED** | 100% | Grounded answer generation, section blueprints, conflict notes, prompt injection defense |
| `test_phase6_3_evidence_fusion.py` | 24 | **PASSED** | 100% | Multi-document fusion, cross-guideline conflict detection, balanced chunk ranking |
| `test_phase6_2_query_planner.py` | 53 | **PASSED** | 100% | Intent-to-plan mapping, query expansion, tenant filtering, adaptive retrieval |
| `test_phase6_1_intent_classifier.py` | 32 | **PASSED** | 100% | 12 clinical intents, safety precedence, adversarial prompt defense, latency |
| `test_phase4_medical_safety.py` | 8 | **PASSED** | 100% | Emergency symptom bypass, self-harm bypass, poisoning bypass, prescreen execution order |
| `test_phase3_5_redis_cache.py` | 8 | **PASSED** | 100% | Layered L1/L2 caching, connection resilience, multi-tenant cache isolation |
| `test_phase3_4_auth_isolation.py` | 12 | **PASSED** | 100% | Cross-tenant document isolation, user spoofing prevention, context authorization |
| `test_phase3_4_cache_invalidation.py` | 12 | **PASSED** | 100% | Multi-factor cache invalidation across document updates, deletions, and metadata |
| **Total Test Suite** | **264** | **PASSED** | **100%** | **Zero regressions across all completed project milestones** |

---

## 7. Production Vector Store Invariant Verification

As mandated by project constraints, production FAISS vector store invariants were strictly verified across multiple test suites and via direct runtime inspection:

```
Runtime Query:
vs = get_vector_store_service()
faiss_vectors = vs.index.ntotal      -> 744 (EXACT)
metadata_records = len(vs.metadata_store) -> 744 (EXACT)
embedding_dimension = vs.index.d     -> 384 (EXACT)
```

- **FAISS Vectors Count**: **744** (Invariant strictly preserved; no reindexing or index corruption)
- **Metadata Records Count**: **744** (1:1 alignment with index vectors verified)
- **Embedding Dimensions**: **384** (`all-MiniLM-L6-v2` dense vector space preserved)

---

## 8. Summary of Completed Files

1. [backend/intelligence/decision_support_models.py](file:///c:/Users/rakes/OneDrive/New%20folder/AI-Healthcare-Agent/backend/intelligence/decision_support_models.py): Strongly typed enums (`ClinicalUncertaintyLevel`, `ClinicalRiskTier`, `UncertaintySourceType`, `ActionRecommendationType`) and dataclasses (`ActionableRecommendation`, `RedFlagTrigger`, `ClinicalHandoffSummary`, `ClinicalDecisionSupportResult`).
2. [backend/intelligence/clinical_decision_support.py](file:///c:/Users/rakes/OneDrive/New%20folder/AI-Healthcare-Agent/backend/intelligence/clinical_decision_support.py): Core `ClinicalDecisionSupportEngine` implementing composite uncertainty calibration, risk tiering, cardiovascular/metabolic/pharmacological red-flag screening, care pathways, follow-up inquiries, and SBAR clinical handoffs.
3. [backend/intelligence/__init__.py](file:///c:/Users/rakes/OneDrive/New%20folder/AI-Healthcare-Agent/backend/intelligence/__init__.py): Exported all Phase 6.7 models and `ClinicalDecisionSupportEngine`.
4. [backend/evaluation/observability.py](file:///c:/Users/rakes/OneDrive/New%20folder/AI-Healthcare-Agent/backend/evaluation/observability.py): Added Prometheus counters, latency histogram, snapshot exposition, and `record_decision_support_event` helper.
5. [backend/intelligence/answer_synthesis.py](file:///c:/Users/rakes/OneDrive/New%20folder/AI-Healthcare-Agent/backend/intelligence/answer_synthesis.py): Integrated Phase 6.7 evaluation in `synthesize` step 8c and attached decision support metadata.
6. [backend/rag/rag_service.py](file:///c:/Users/rakes/OneDrive/New%20folder/AI-Healthcare-Agent/backend/rag/rag_service.py): Integrated Phase 6.7 decision support into `generate_rag_answer` and `generate_rag_stream` (yielding `clinical_decision_support` SSE event).
7. [tests/test_phase6_7_clinical_decision_support.py](file:///c:/Users/rakes/OneDrive/New%20folder/AI-Healthcare-Agent/tests/test_phase6_7_clinical_decision_support.py): 28 dedicated unit, integration, and streaming tests (100% pass rate).
8. [scripts/benchmark_phase6_7.py](file:///c:/Users/rakes/OneDrive/New%20folder/AI-Healthcare-Agent/scripts/benchmark_phase6_7.py): Automated benchmark script measuring percentiles over 200 iterations and validating invariants.
9. [evaluation_reports/benchmark_phase6_7_results.json](file:///c:/Users/rakes/OneDrive/New%20folder/AI-Healthcare-Agent/evaluation_reports/benchmark_phase6_7_results.json): Structured benchmark outputs (p95 = 0.136 ms, target compliance = PASS).
10. [PHASE_6_7_REPORT.md](file:///c:/Users/rakes/OneDrive/New%20folder/AI-Healthcare-Agent/PHASE_6_7_REPORT.md): This comprehensive report.
