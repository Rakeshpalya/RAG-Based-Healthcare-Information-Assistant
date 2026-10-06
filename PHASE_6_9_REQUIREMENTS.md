# USER-AUTHORIZED NEW SPECIFICATION
# Phase 6.9 Requirements: Longitudinal Clinical Context & Multi-Turn Interaction Memory

> **Specification Type**: USER-AUTHORIZED NEW SPECIFICATION
> **Status**: DRAFT PROPOSAL — PENDING USER APPROVAL
> **Milestone**: Phase 6.9
> **Project**: `AI-Healthcare-Agent` (`RAG-Based-Healthcare-Information-Assistant`)
> **Base Commit**: `e33ec0b5e36827977548ed84218dbbeddb6c78ff` (Phase 6.8 Complete)
> **Production Invariants**: 744 FAISS Vectors | 744 Metadata Records | 384 Dimensions | `sentence-transformers/all-MiniLM-L6-v2`

---

## 1. Phase Title

**Phase 6.9: Longitudinal Clinical Context & Multi-Turn Interaction Memory**
*Clinical Dialogue State Tracking, Cumulative Contraindication Propagation & Contextual Disambiguation*

---

## 2. Problem Statement

In real-world clinical consultations and patient interactions, medical inquiries are rarely self-contained in a single prompt. Dialogue is inherently iterative, cumulative, and longitudinal:
1. **Clinical Context Amnesia**: In Phases 6.1 through 6.8, each request is evaluated on an isolated turn-by-turn basis. The system relies only on a 10-line primitive regex pronoun replacer (`it`, `this`, `that`) in `rag_service.py` that fails on complex medical follow-ups.
2. **Severe Safety Blindspot (Contraindication Leakage)**: If a user discloses chronic kidney disease (CKD), pregnancy, or a severe medication allergy (e.g., penicillin or ACE-inhibitor angioedema) in Turn 1, and asks about symptom management in Turn 2 or Turn 3 without restating their underlying condition, downstream verification (Phase 6.6) and clinical decision support (Phase 6.7) evaluate the subsequent recommendation in a vacuum. A contraindicated medication (e.g., an NSAID in renal failure) could pass unflagged because Turn 2 did not mention the patient's renal status.
3. **Intent and Query Drift**: Follow-up inquiries such as *"What are the first-line alternatives?"* or *"How often should that be monitored?"* lack semantic grounding in earlier dialogue turns, causing Phase 6.1 (Intent Classification) and Phase 6.2 (Query Planning) to construct degraded, generic search plans that miss the primary clinical topic.
4. **Lack of Auditable Dialogue State**: Phase 6.8 introduced rigorous cryptographic execution auditing and provenance checksums for single-turn execution, but does not capture dialogue turn index, accumulated clinical constraints, or context resolution confidence in the audit trail.

---

## 3. Objective

Implement a production-grade, pure-Python, deterministic **Longitudinal Clinical Context Engine** (`ClinicalContextEngine`) within `backend/intelligence/` that:
1. Ingests bounded conversation history from `RAGQueryRequest.conversation_history` (or session cache) and extracts a structured, multi-turn clinical dialogue state.
2. Builds and maintains a deterministic **Cumulative Clinical Profile** tracking active conditions, symptoms, medications, allergies, and demographic constraints across conversational turns.
3. Propagates accumulated clinical constraints directly into **Phase 6.6 (Grounding & Verification)** and **Phase 6.7 (Decision Support & Risk Stratification)** so contraindication and risk checks evaluate the complete patient picture across turns.
4. Performs clinical-aware query reformulation and follow-up disambiguation before **Phase 6.1 (Intent)** and **Phase 6.2 (Query Planning)** to optimize FAISS vector retrieval.
5. Emits strongly typed dialogue state audit records integrated into **Phase 6.8 Orchestration** and the cryptographic SHA-256 provenance digest.
6. Operates under strict production constraints: sub-millisecond execution latency (< 3.0 ms p95 overhead), zero-PHI telemetry, strict tenant isolation, 100% backward compatibility, and zero vector store mutations.

---

## 4. Scope

### In-Scope:
1. **Core Context Engine**: `backend/intelligence/longitudinal_context.py` containing `ClinicalContextEngine`.
2. **Strongly Typed Context Models**: `backend/intelligence/context_models.py` defining dialogue states, entity profiles, and audit records.
3. **Clinical Entity & Constraint Extraction**: Deterministic, rule-based clinical entity extraction (conditions, symptoms, medications, allergies, negation, temporal qualifiers).
4. **Cumulative Profile Accumulation**: Merging multi-turn entities into an active clinical constraint profile with negation and resolution handling.
5. **Contextual Query Reformulation**: Generating an expanded `effective_query` for Phase 6.1 classification, Phase 6.2 planning, and FAISS retrieval.
6. **Downstream Safety Propagation**: Passing `CumulativeClinicalProfile` to Phase 6.6 (`ClinicalVerificationEngine`) and Phase 6.7 (`ClinicalDecisionSupportEngine`).
7. **Phase 6.8 Orchestration Integration**: Capturing `StageDialogueContext` in `ClinicalIntelligenceOrchestrator` and binding state digests into SHA-256 checksums.
8. **Observability & Metrics**: Exposing zero-PHI Prometheus counters (`rag_context_resolution_total`, `rag_context_entities_tracked_total`) and latency histograms.
9. **Dedicated Test Suite & Benchmark**: 25+ comprehensive unit/integration tests and a 250-iteration latency benchmark script.

### Explicit Non-Goals:
- **NO External Chat Database**: Conversation history is provided in-band via API payload (`conversation_history`) or optional ephemeral tenant-isolated session cache; no persistent chat logging to disk that could leak PHI.
- **NO LLM-Based Summarization**: Relies on deterministic clinical NLP algorithms to ensure zero hallucination risk, predictable execution, and sub-millisecond latency.
- **NO Autonomous Diagnostic or Prescriptive Claims**: Maintains purely advisory, guideline-concordant informational status.
- **NO Vector Store Reindexing or Mutations**: The 744 FAISS vectors and 744 metadata records remain strictly read-only and unmutated.
- **NO Replacement or Weakening of Prior Phases**: Completely preserves Phases 6.1 through 6.8.

---

## 5. Explicit Non-Goals

1. **Do not introduce persistent raw conversation databases**: We do not store raw clinical dialogue in an external unencrypted database.
2. **Do not make external network or LLM calls for context resolution**: Context extraction must be deterministic, local, and sub-millisecond.
3. **Do not alter or reindex FAISS**: Vector count (744) and metadata count (744) must remain strictly constant.
4. **Do not break single-turn requests**: Single-turn queries without history must execute with zero regression or alteration in behavior.
5. **Do not leak cross-tenant dialogue**: Sessions from Tenant A must never access or contaminate dialogue state from Tenant B.

---

## 6. Functional Requirements

### FR-1: Multi-Turn Conversation Ingestion
- Accept `conversation_history: Optional[List[Dict[str, str]]]` from `RAGQueryRequest`.
- Support standard role designations (`user`, `assistant`, `system`).
- Enforce bounded context windowing (default: last 6 turns / 3 dialogue cycles) to prevent memory ballooning and context drift.

### FR-2: Deterministic Clinical Entity & Attribute Extraction
- Detect clinical entities categorized into:
  - `CONDITION`: Diagnosed pathologies, chronic illnesses (e.g., T2D, CKD, HTN, asthma).
  - `SYMPTOM`: Acute and subacute clinical complaints (e.g., cough, dyspnea, fever, chest pain).
  - `MEDICATION`: Pharmaceutical agents, drug classes (e.g., lisinopril, metformin, ibuprofen).
  - `ALLERGY`: Documented drug or substance hypersensitivities.
  - `DEMOGRAPHIC`: Age group, sex, pregnancy status, renal/hepatic impairment qualifiers.
- Recognize negation patterns (e.g., *"denies chest pain"*, *"no history of diabetes"*, *"not pregnant"*).
- Distinguish between active patient conditions vs. hypothetical inquiries.

### FR-3: Cumulative Clinical Profile Construction
- Accumulate extracted entities across all conversation turns up to the current query.
- Maintain a deduplicated, normalized set of:
  - `active_conditions`: Set of confirmed patient conditions.
  - `active_medications`: Set of currently active medications.
  - `confirmed_allergies`: Set of known allergies.
  - `risk_factors`: Set of demographic or physiological risk modifiers.
- Update profile dynamically when a user revokes or negates an earlier statement.

### FR-4: Contextual Clinical Query Reformulation
- For follow-up queries containing demonstrative pronouns, elliptical questions, or incomplete noun phrases (e.g., *"What is the dosage?"*, *"Any alternative to that?"*):
  - Resolve the focal entity from immediate prior turns.
  - Construct an `effective_query` incorporating the resolved clinical entity while preserving the user's specific clinical intent.
- Preserve original `question` verbatim for downstream presentation and audit logging.

### FR-5: Cumulative Safety & Contraindication Propagation
- Pass the `CumulativeClinicalProfile` to Phase 6.6 (`ClinicalVerificationEngine`).
  - Verify that medications proposed in the synthesized answer do not conflict with conditions or allergies declared in earlier turns.
- Pass the `CumulativeClinicalProfile` to Phase 6.7 (`ClinicalDecisionSupportEngine`).
  - Factor prior-turn comorbidities and age factors into risk stratification calculation.

### FR-6: Orchestration & Cryptographic Audit Integration
- Add `StageDialogueContext` as an explicit stage in `backend/intelligence/orchestration_models.py`.
- Include dialogue turn count, resolved topic, and cumulative profile hash in the Phase 6.8 SHA-256 provenance checksum.

---

## 7. Non-Functional Requirements

| Requirement | Target | Verification Method |
| :--- | :--- | :--- |
| **Latency Overhead** | p50 < 0.5 ms, p95 < 2.5 ms | Measured via 250-iteration benchmark |
| **Memory Footprint** | < 100 KB per active dialogue session | In-memory footprint validation |
| **Determinism** | 100% deterministic (same input produces identical state) | Unit test assertions with zero stochastic variance |
| **Fault Tolerance** | Graceful degradation to single-turn baseline on malformed history | Malformed payload unit tests |
| **Zero PHI Exposure** | Zero patient names, identifiers, or free-text stored in metrics | Regex PHI scanning across telemetry and audit records |

---

## 8. Architecture

```
                    [Client Request: question + conversation_history]
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        FastAPI /rag/query or /rag/stream                               │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 0: Medical Safety Pre-Screening (Phase 4.1)                                      │
│          (Emergency, Self-Harm, Poisoning Fast-Path Intercepts)                        │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ [Passed Safe]
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 0.5: Longitudinal Clinical Context Engine (Phase 6.9)                            │
│   ├── Ingestion & Bounded Windowing (last N turns)                                     │
│   ├── Clinical Entity Extraction (conditions, medications, allergies, negation)        │
│   ├── Cumulative Clinical Profile Assembly (active conditions, risk factors)           │
│   ├── Contextual Query Reformulation -> effective_query                                │
│   └── Emits: TurnContextResolution + DialogueStateAuditRecord                          │
└──────────────┬───────────────────────────┬─────────────────────────────────────────────┘
               │ effective_query           │ CumulativeClinicalProfile
               ▼                           │
┌───────────────────────────────┐          │
│ Stage 1: Intent (Phase 6.1)   │          │
└──────────────┬────────────────┘          │
               ▼                           │
┌───────────────────────────────┐          │
│ Stage 2: Planning (Phase 6.2) │          │
└──────────────┬────────────────┘          │
               ▼                           │
┌───────────────────────────────┐          │
│ FAISS Retrieval (744 vectors) │          │
└──────────────┬────────────────┘          │
               ▼                           │
┌───────────────────────────────┐          │
│ Stage 3: Fusion (Phase 6.3)   │          │
└──────────────┬────────────────┘          │
               ▼                           │
┌───────────────────────────────┐          │
│ Stage 4: Synthesis (Phase 6.4)│          │
└──────────────┬────────────────┘          │
               ▼                           │
┌───────────────────────────────┐          │
│ Stage 5: Citations (Phase 6.5)│          │
└──────────────┬────────────────┘          │
               ▼                           │
┌──────────────────────────────────────────┴─────────────────────────────────────────────┐
│ Stage 6: Clinical Verification & Guardrails (Phase 6.6)                                │
│          *Evaluates candidate answer against CumulativeClinicalProfile*                │
│          (e.g., flags NSAID if Turn 1 noted CKD or renal impairment)                   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 7: Clinical Decision Support & Care Pathways (Phase 6.7)                         │
│          *Evaluates risk stratification informed by multi-turn comorbidities*          │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 8: Clinical Intelligence Orchestration (Phase 6.8)                               │
│          *Binds StageDialogueContext into final audit record & SHA-256 checksum*       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Data Models (`backend/intelligence/context_models.py`)

```python
from enum import Enum
from typing import List, Dict, Set, Optional, Any
from pydantic import BaseModel, Field

class ClinicalEntityType(str, Enum):
    CONDITION = "condition"
    SYMPTOM = "symptom"
    MEDICATION = "medication"
    ALLERGY = "allergy"
    DEMOGRAPHIC = "demographic"
    RISK_FACTOR = "risk_factor"

class EntityTemporalState(str, Enum):
    ACTIVE = "active"
    HISTORICAL = "historical"
    NEGATED = "negated"
    SUSPECTED = "suspected"

class ClinicalEntity(BaseModel):
    name: str
    entity_type: ClinicalEntityType
    temporal_state: EntityTemporalState = EntityTemporalState.ACTIVE
    turn_index: int
    negated: bool = False
    source_text: str

class CumulativeClinicalProfile(BaseModel):
    active_conditions: List[str] = Field(default_factory=list)
    active_symptoms: List[str] = Field(default_factory=list)
    active_medications: List[str] = Field(default_factory=list)
    confirmed_allergies: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    demographics: Dict[str, Any] = Field(default_factory=dict)
    total_entities_extracted: int = 0
    profile_hash: str = ""

class TurnContextResolution(BaseModel):
    original_query: str
    effective_query: str
    is_follow_up: bool
    resolved_topic: Optional[str] = None
    prior_turn_count: int = 0
    cumulative_profile: CumulativeClinicalProfile
    inherited_contraindications: List[str] = Field(default_factory=list)
    resolution_confidence: float = 1.0

class DialogueStateAuditRecord(BaseModel):
    turn_count: int
    is_follow_up: bool
    resolved_topic: Optional[str]
    entities_count: int
    profile_hash: str
    resolution_latency_ms: float
```

---

## 10. API Changes

### Backward Compatibility Guarantees:
- `POST /rag/query` and `POST /rag/stream` already accept `conversation_history` in `RAGQueryRequest`.
- No breaking changes to existing request models.
- Response model `RAGQueryResponse` will expose an optional `dialogue_context` field:
  ```python
  class RAGQueryResponse(BaseModel):
      # ... all existing Phase 1-6.8 fields preserved ...
      dialogue_context: Optional[Dict[str, Any]] = None
  ```
- If caller provides no `conversation_history`, `dialogue_context` indicates single-turn execution, and pipeline behaves identically to Phase 6.8.

---

## 11. Pipeline Integration

In `backend/rag/rag_service.py`:
1. In `generate_rag_answer(...)` and `generate_rag_stream(...)`:
   - Pass `question` and `conversation_history` into `ClinicalContextEngine.resolve_context(...)`.
   - Use `resolution.effective_query` for intent classification (Phase 6.1), query planning (Phase 6.2), and retrieval.
   - Pass `resolution.cumulative_profile` into Phase 6.6 (`ClinicalVerificationEngine.verify(...)`) and Phase 6.7 (`ClinicalDecisionSupportEngine.analyze(...)`).
   - Pass `resolution.to_audit_record()` into Phase 6.8 (`ClinicalIntelligenceOrchestrator.orchestrate(...)`).

---

## 12. Safety Requirements

1. **Anti-Amnesia Safety Rule**: If a contraindication (e.g., anaphylaxis to penicillin, renal failure, pregnancy) is extracted in any prior turn within the session window, it CANNOT be dropped or overwritten by subsequent queries unless explicitly revoked by the user.
2. **Emergency Pre-Screen Supremacy**: Stage 0 safety pre-screening always executes before context resolution. An emergency symptom (e.g., *"Now he has sudden chest pain and shortness of breath"*) triggers immediate emergency triage regardless of dialogue context.
3. **No Autonomous Prescribing**: Longitudinal context informs safety checks and educational explanations; it never generates direct medication prescriptions or medical advice.

---

## 13. Tenant Isolation Requirements

1. **Session-Bound Partitioning**: Dialogue state is constructed strictly from the caller's request payload `conversation_history` or scoped to `tenant_id` + `session_id`.
2. **Cross-Tenant Barrier**: Dialogue state from Tenant A cannot be accessed, shared, or referenced by Tenant B.
3. **Validation**: Test suite will include explicit cross-tenant multi-turn tests verifying that Tenant B cannot inherit or trigger contraindications from Tenant A's conversation.

---

## 14. PHI / Privacy Requirements

1. **Zero-PHI Persistence**: The engine stores zero raw dialogue text to persistent disk.
2. **Normalized Entities Only**: Telemetry counters and Prometheus exposition only record counts (e.g., `entities_extracted: 3`, `is_follow_up: true`), never entity names or patient details.
3. **Audit Hash Integrity**: The `profile_hash` is an anonymous SHA-256 digest of normalized medical terms, containing no personal identifiers.

---

## 15. Observability Requirements

### Prometheus Metrics to Register in `backend/evaluation/observability.py`:
1. `rag_context_resolution_total`: Counter tracking total context resolutions (labeled by `is_follow_up`).
2. `rag_context_entities_tracked_total`: Counter tracking extracted clinical entities (labeled by `entity_type`).
3. `rag_context_contraindications_inherited_total`: Counter tracking inherited multi-turn contraindications flagged by Phase 6.6.
4. `rag_context_resolution_duration_seconds`: Histogram tracking context engine latency overhead.

---

## 16. Error Handling

1. **Malformed History**: If `conversation_history` contains invalid dictionaries, missing keys, or non-string values, gracefully log a warning and fall back to single-turn processing without raising an unhandled exception.
2. **Exceeded Window Length**: If history exceeds 20 turns, truncate to the most recent 6 turns while preserving the earliest demographic/allergy disclosures.
3. **Context Ambiguity**: If pronoun resolution confidence is below threshold (< 0.4), retain the original user query without distortion.

---

## 17. Backward Compatibility

- **100% Backward Compatible**:
  - Existing clients sending single-turn queries (`conversation_history=None` or `[]`) experience zero behavioral changes.
  - All existing Phase 6.1 through 6.8 tests (277 clinical regression tests, 23 tenant isolation tests) must continue to pass with 0 failures.
  - Production vector invariants (FAISS=744, metadata=744, dim=384) remain completely untouched.

---

## 18. Testing Strategy

Create dedicated test suite `tests/test_phase6_9_longitudinal_context.py` containing 25+ tests covering:
1. **Entity Extraction**: Correct extraction of conditions, medications, symptoms, allergies, and negations.
2. **Follow-Up Query Reformulation**: Demonstrative pronoun resolution (*"What are its side effects?"* -> *"What are metformin side effects?"*).
3. **Cumulative Contraindication Verification**: Verification that Turn 1 renal failure triggers Phase 6.6 warnings when Turn 2 inquires about NSAID dosages.
4. **Allergy Propagation**: Verification that Turn 1 penicillin allergy prevents amoxicillin suggestions in Turn 3.
5. **Cross-Tenant Isolation**: Ensuring Tenant B dialogue never accesses Tenant A's clinical profile.
6. **Graceful Fallbacks**: Malformed history, empty turns, non-ASCII characters, and adversarial inputs.
7. **Phase 6.8 Orchestration Binding**: Verifying `StageDialogueContext` is audited and included in the SHA-256 checksum.

---

## 19. Benchmark Strategy

Create benchmark script `scripts/benchmark_phase6_9.py`:
- Execute 250 iterations simulating multi-turn clinical interactions.
- Measure:
  - Context engine latency (p50, p90, p95, p99).
  - Pipeline overhead percentage (< 1.0% of total RAG latency).
  - Memory consumption.
- Persist results to `evaluation_reports/benchmark_phase6_9_results.json`.

---

## 20. Acceptance Criteria

| ID | Criterion | Target |
| :--- | :--- | :--- |
| **AC-1** | Dedicated Phase 6.9 test suite | 25+ passed / 0 failed |
| **AC-2** | Full regression test suite | 300+ passed / 0 failed |
| **AC-3** | Tenant isolation test suite | 23+ passed / 0 failed |
| **AC-4** | Context engine latency overhead | p95 < 2.5 ms |
| **AC-5** | Multi-turn contraindication interception | 100% pass on renal/allergy carryover |
| **AC-6** | Production FAISS vector count | Exactly 744 (unmodified) |
| **AC-7** | Production metadata records count | Exactly 744 (unmodified) |
| **AC-8** | Embedding model & dimension | `all-MiniLM-L6-v2` / 384 dimensions |
| **AC-9** | Zero PHI in logs / metrics | Verified via automated scan |
| **AC-10** | Comprehensive report delivered | `PHASE_6_9_REPORT.md` created |

---

## 21. Rollback Strategy

Since Phase 6.9 is implemented as modular components in `backend/intelligence/longitudinal_context.py` and connected non-destructively via `rag_service.py`:
1. If context resolution fails or degrades, setting an environment variable `ENABLE_LONGITUDINAL_CONTEXT=false` immediately bypasses the context engine and restores Phase 6.8 single-turn execution.
2. Git rollback to commit `e33ec0b5e36827977548ed84218dbbeddb6c78ff` instantly restores clean Phase 6.8 state with zero data loss or database migration issues.

---

## 22. Relationship to Phase 6.8

- Phase 6.8 established **Clinical Intelligence Orchestration & Auditability**, executing Stages 0 through 8 and calculating an end-to-end cryptographic SHA-256 audit digest.
- Phase 6.9 integrates seamlessly into Phase 6.8 by providing a `StageDialogueContext` record that feeds into the orchestrator, binding the accumulated clinical profile hash into the cryptographic provenance checksum.
- Phase 6.8 audit models and checksum calculations remain intact and are strictly extended.

---

## 23. Relationship to Future Phase 7

- Phase 7 implements the **React 18 + Vite Frontend Live RAG Chat UI** with conversation sessions and message history.
- The Phase 7 frontend contract (`RAGQueryRequest.conversation_history`) requires a backend capable of intelligent multi-turn comprehension.
- Phase 6.9 provides the exact backend intelligence layer required for Phase 7, ensuring that when clinicians or patients use the live chat UI, multi-turn medical context, contraindication tracking, and pronoun resolution work seamlessly and safely.
