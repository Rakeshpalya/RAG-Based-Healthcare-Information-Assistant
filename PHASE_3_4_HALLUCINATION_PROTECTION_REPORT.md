# PHASE 3.4 — HALLUCINATION PROTECTION REPORT
**AI-Healthcare-Agent Project**  
**Phase:** 3.4 — Grounded LLM Generation & Medical Safety (Hallucination Protection)  
**Status:** Certified & Verified  
**Date:** September 27, 2026  

---

## 1. Executive Summary & Objective

In medical AI applications, syntactic citation verification and basic embedding similarity alone are insufficient to guarantee patient safety. As demonstrated in Phase 3.3, embedding similarity scores between a statement asserting *"Beta-blockers are recommended as first-line therapy"* and a source stating *"Beta-blockers are not recommended as first-line therapy"* yield a deceptively high cosine similarity of **0.9876** because 95% of the lexical tokens are identical. Without dedicated clinical hallucination protection, critical medical contradictions, fabricated dosages, and phantom drug recommendations pass undetected.

**Phase 3.4 implements deterministic medical hallucination protection**, establishing a multi-layer clinical grounding defense operating between LLM generation and final response delivery:
1. **Medical Claim-Level Validation**: Extracts candidate claims and analyses clinical entities (medications, numerical dosages, diagnoses, recommendations, polarity, and directional trends).
2. **Negation & Contradiction Detection**: Identifies polarity mismatches where the model affirms what the source negates or contraindicates.
3. **Directional Contradiction Detection**: Catches opposing clinical directions (e.g. claim asserts an intervention increases blood pressure when the evidence proves it reduces blood pressure).
4. **Medication & Pharmacological Hallucination Defense**: Detects pharmaceutical entities in the generated answer that do not exist anywhere in the cited evidence.
5. **Dosage & Numerical Measurement Defense**: Verifies numerical dosages and units (e.g., 500 mg, 5 mg daily, blood pressure readings) against cited evidence, preventing dosage inflation and fabrication.
6. **Unsupported Diagnosis & Recommendation Defense**: Prevents speculative clinical diagnoses or ungrounded surgical/prescriptive recommendations.
7. **Wrong-Source Attribution Defense**: Distinguishes between claims citing the wrong document versus claims entirely fabricated.
8. **Partial/Mixed Support Sanitization**: Prunes hallucinated or contradictory claims while safely preserving legitimate, grounded factual assertions.
9. **Fail-Safe Insufficient Evidence Fallback**: When an answer contains only unsupported claims, falls back to the certified safe refusal message without inventing medical facts.

---

## 2. Upgraded Architecture

```
                       USER QUESTION
                             │
                             ▼
                       RAG RETRIEVAL
                             │
                             ▼
                      GROUNDED PROMPT
                             │
                             ▼
                          GEMINI
                             │
                             ▼
                    CITATION VALIDATION
                             │
                             ▼
                      CLAIM EXTRACTION
                             │
                             ▼
                    EVIDENCE VALIDATION
                             │
                             ▼
               NEGATION / CONTRADICTION CHECK
                             │
                             ▼
                    HALLUCINATION FILTER
                             │
                             ▼
                       SAFE RESPONSE
```

---

## 3. Files Inspected

Before implementing Phase 3.4, existing evaluation, generation, and retrieval components were inspected:
- `backend/evaluation/citation_validator.py`: Evaluated claim extraction and semantic similarity checks.
- `backend/evaluation/grounding_evaluator.py`: Evaluated deterministic lexical and citation grounding logic.
- `backend/rag/rag_service.py`: Evaluated post-generation citation filtering and fallback response assembly.
- `backend/services/embedding_service.py`: Verified `all-MiniLM-L6-v2` embedding generation.
- `tests/test_citation_validator.py`: Inspected 18 legacy citation validation tests.
- `tests/test_phase3_citation_enforcement.py`: Inspected 13 Phase 3.3 citation enforcement tests.
- `tests/test_gemini_service.py`: Inspected generation failure and fallback handling.

---

## 4. Baseline Tests

Before modifying any files:
- `tests/test_citation_validator.py`: **18 passed**
- `tests/test_phase3_citation_enforcement.py`: **13 passed**
- `tests/test_phase3_grounded_prompt.py`: **10 passed**
- `tests/test_safety_agent.py`: **13 passed**
- `tests/test_gemini_service.py`: **8 passed**
- Baseline Group: **62 passed**, 0 failures.

---

## 5. Defects & Vulnerabilities Identified

### Defect 1: Negation Blindness of Dense Embeddings
Because cosine similarity on transformer sentence embeddings (`all-MiniLM-L6-v2`) evaluates dense semantic topic vectors, a single negation token (`not`, `contraindicated`, `avoid`) only slightly alters the vector. A direct contradiction can score $\ge 0.90$ similarity, easily bypassing threshold-based similarity filters ($\tau = 0.65$).

### Defect 2: Dosage Fabrication Blindness
An LLM may correctly identify that Amlodipine is indicated for hypertension, but hallucinate a fatal dosage (e.g. *"Amlodipine 50 mg daily"* instead of *"5 mg"*). Dense vector models do not distinguish fine-grained numerical dosage differences in clinical contexts.

### Defect 3: Cross-Condition Drug Hallucination
An LLM may answer a hypertension question by recommending an unmentioned antidiabetic drug (e.g., Metformin) alongside lifestyle changes. If the prompt context discusses general cardiometabolic risk, the semantic similarity may remain moderately high, allowing unmentioned pharmaceutical agents to pass into the answer.

### Defect 4: Directional Contradiction
An LLM claiming an intervention *"elevates"* or *"increases"* a biological marker when the evidence explicitly states it *"lowers"* or *"reduces"* it shares high lexical overlap with the evidence, evading naive semantic similarity checks.

---

## 6. Implementation Details

### A. New Module: `backend/evaluation/hallucination_guard.py`
Created a comprehensive, deterministic medical hallucination protection engine:

1. **`HallucinationType` Constants**:
   - `NONE`: Valid, fully grounded claim.
   - `MEDICATION_HALLUCINATION`: Drug entity present in claim but absent from cited evidence.
   - `DOSAGE_HALLUCINATION`: Numerical dosage/unit present in claim but absent or mismatched in cited evidence.
   - `NEGATION_CONTRADICTION`: Polarity mismatch between claim and evidence (e.g. recommended vs contraindicated).
   - `DIRECTIONAL_CONTRADICTION`: Opposing clinical direction (e.g. increase vs decrease).
   - `UNSUPPORTED_DIAGNOSIS`: Diagnostic entity not supported by cited evidence.
   - `UNSUPPORTED_RECOMMENDATION`: Prescriptive clinical advice not substantiated.
   - `WRONG_SOURCE`: Evidence found in another retrieved source, but cited source does not contain it.
   - `SEMANTIC_MISMATCH`: Semantic similarity $< 0.65$.
   - `UNCITED_CLAIM`: Factual claim missing required source citation.
   - `INVALID_SOURCE`: Citation references non-existent source index.

2. **`ExtractedEntities` Dataclass**:
   - Tracks `medications`, `dosages`, `diagnoses`, `recommendations`, `has_negation`, `negation_terms`, `direction`, and `direction_terms`.

3. **`ClaimGuardDetail` & `HallucinationGuardResult` Dataclasses**:
   - Provides granular diagnostic auditing for every sentence in the response, capturing support scores, reasons, and entity breakdowns.

4. **Multi-Layer Validation Methods**:
   - `extract_entities(text)`: Extracts clinical medications (via curated 70+ generic/brand dictionary + 18 pharmacological suffixes like `-olol`, `-pril`, `-sartan`, `-statin`, `-dipine`), dosages (`\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|units?|iu)\b`), blood pressure readings (`\d{2,3}/\d{2,3}`), negations, and directional terms.
   - `find_best_source_segment(claim_text, source_text)`: Evaluates both the full passage and segmented sentences, returning maximum cosine similarity and the focal sentence.
   - `check_negation_contradiction(claim, entities, best_seg)`: Compares polarity of claim vs focal source segment.
   - `check_directional_contradiction(entities, best_seg)`: Compares directional clinical vectors (UP: `increase`, `elevate`, `raise`, `worsen` vs DOWN: `decrease`, `lower`, `reduce`, `attenuate`).
   - `check_medication_hallucinations(claim_meds, source_text)`: Verifies all cited drugs exist in the evidence.
   - `check_dosage_hallucinations(claim_dosages, source_text)`: Verifies all numerical dosages match the evidence.
   - `inspect_claim(claim, source_map)`: Orchestrates the audit for a single claim.
   - `guard_answer(answer_text, retrieved_sources)`: Executes the full pipeline, prunes unsupported sentences, and triggers safe fallback if necessary.

### B. Integration with `backend/evaluation/citation_validator.py`
Updated `CitationValidator.check_claim_support()` to delegate deep claim inspection to `HallucinationGuard.inspect_claim()`.
- Automatically inherits medication verification, dosage checking, negation contradiction detection, and directional analysis into `CitationValidator.validate_grounded_citations()`.
- Preserves full backwards compatibility for all legacy tests.

### C. Updates to `backend/evaluation/__init__.py`
Exported `HallucinationGuard`, `HallucinationGuardResult`, `HallucinationType`, `ClaimGuardDetail`, and `ExtractedEntities`.

---

## 7. Test Suite Summary (`tests/test_phase3_hallucination_protection.py`)

13 comprehensive tests validating every required clinical hallucination scenario:

| # | Test Name | Scenario Tested | Result |
|:---:|:---|:---|:---:|
| 1 | `test_1_medication_hallucination_detection` | Cites unmentioned drug (Metformin in HTN doc) $\to$ `MEDICATION_HALLUCINATION` | **PASSED** |
| 2 | `test_2_dosage_hallucination_detection` | Fabricated dosage (50 mg instead of 5 mg) $\to$ `DOSAGE_HALLUCINATION` | **PASSED** |
| 3 | `test_3_negation_contradiction_detection` | Beta-blockers recommended vs not recommended $\to$ `NEGATION_CONTRADICTION` | **PASSED** |
| 4 | `test_4_contraindication_contradiction_detection` | Drug recommended in severe renal disease vs contraindicated $\to$ `NEGATION_CONTRADICTION` | **PASSED** |
| 5 | `test_5_directional_contradiction_detection` | Exercise increases BP vs exercise reduces BP $\to$ `DIRECTIONAL_CONTRADICTION` | **PASSED** |
| 6 | `test_6_unsupported_diagnosis_detection` | Diagnoses pheochromocytoma when document only discusses general HTN $\to$ REJECTED | **PASSED** |
| 7 | `test_7_unsupported_medical_recommendation_detection` | Prescribes ungrounded renal denervation surgery $\to$ REJECTED | **PASSED** |
| 8 | `test_8_wrong_source_attribution_detection` | Cites Source 1 for diabetes monitoring found in Source 2 $\to$ `WRONG_SOURCE` | **PASSED** |
| 9 | `test_9_partial_mixed_support_pruning` | Multi-claim answer with 1 valid + 1 hallucinated drug $\to$ Hallucination pruned, valid preserved | **PASSED** |
| 10 | `test_10_complete_hallucination_fallback` | Answer entirely made up of hallucinated drugs/doses $\to$ Safe fallback triggered | **PASSED** |
| 11 | `test_11_valid_grounded_medical_statement` | Grounded claim with real drug and real dosage (Amlodipine 5 mg) $\to$ `is_safe=True` | **PASSED** |
| 12 | `test_12_rag_service_rejects_hallucination_with_mock_gemini` | End-to-end `RAGService` halts on mock Gemini hallucination | **PASSED** |
| 13 | `test_13_vector_store_read_only_invariance` | Verification that guard operations cause zero mutations to vector store | **PASSED** |

---

## 8. Full Regression Verification

```powershell
venv\Scripts\pytest.exe -q
```
**Result: 451 passed, 3 warnings in 78.13s (100% pass rate across entire repository)**

Component breakdown:
- `tests/test_phase3_hallucination_protection.py`: **13 passed**
- `tests/test_phase3_citation_enforcement.py`: **13 passed**
- `tests/test_phase3_grounded_prompt.py`: **10 passed**
- `tests/test_citation_validator.py`: **18 passed**
- `tests/test_safety_agent.py`: **13 passed**
- `tests/test_gemini_service.py`: **8 passed**
- `tests/test_phase2_retrieval_hardening.py`: **25 passed**
- All other test suites (unit, integration, E2E, auth, RAG): **351 passed**

---

## 9. Vector Store Read-Only Invariance

The vector store was verified before and after Phase 3.4 execution:
```powershell
venv\Scripts\python.exe -c "import faiss,json; idx=faiss.read_index('data/vector_store/index.faiss'); data=json.load(open('data/vector_store/metadata.json')); print('Vectors:',idx.ntotal); print('Metadata:',data['count'],len(data['records']))"
```
- **Vector Count Before Phase 3.4 Execution:** 744 vectors
- **Vector Count After Phase 3.4 Execution:** 744 vectors
- **Metadata Records Count:** 744 records (1:1 aligned)
- **Mutations During Phase 3.4:** **0 mutations (100% read-only)**

---

## 10. Remaining Limitations & Phase 3.5 Hand-off

1. **Embedding Similarity & Negation Heuristics**:
   - While the negation polarity and directional contradiction engines detect explicit contradictions (e.g. recommended vs contraindicated, increase vs decrease), complex multi-condition nested clauses (e.g. *"Drug X is indicated for condition A except when complicated by condition B"*) require formal natural language inference (NLI) or clinical LLM-as-a-judge verification.
2. **Novel / Rare Drug Synonyms**:
   - The dictionary covers 70+ major pharmaceutical entities and 18 generic suffixes. Investigational compounds or rare brand names not matching established patterns rely on token overlap and semantic similarity.
3. **Phase 3.5 Hand-off**:
   - Phase 3.5 will focus on **Insufficient-Evidence & Refusal Behavior Redesign**, standardizing graceful conversational refusals when documents lack supporting clinical context.

---

## 11. Certification

Phase 3.4 (Hallucination Protection) is hereby **CERTIFIED**.
- Multi-layer medical claim validation active.
- Medication and dosage hallucination defenses active.
- Negation and directional contradiction defenses active.
- Vector store invariant and intact.
- 451/451 tests passing.
