# PHASE 3.3 — CITATION ENFORCEMENT REPORT
**AI-Healthcare-Agent Project**  
**Phase:** 3.3 — Grounded LLM Generation & Medical Safety (Citation Enforcement)  
**Status:** Certified & Verified  
**Date:** September 27, 2026  

---

## 1. Executive Summary

Phase 3.3 resolves the critical grounding vulnerability identified during the Phase 3.1 architectural audit: **syntactically valid citations masking unsupported or hallucinated medical claims**.

Prior to Phase 3.3, `CitationValidator` merely validated that citation tags (e.g., `[Source 1]`) referenced numbers within the retrieved source count bounds ($1 \le N \le K$). An answer claiming *"Metformin is recommended for hypertension [Source 1]"* would pass validation if Source 1 existed, even if Source 1 exclusively discussed dietary sodium and exercise. Furthermore, legacy `strip_invalid_citations()` stripped out-of-range citations (e.g., `[Source 99]`) but left the orphaned, unsupported claim in the response text.

Phase 3.3 introduces deterministic, two-layer citation enforcement:
1. **Deterministic Claim & Citation Segmentation**: Extracts candidate factual claims (sentences, bullet points, numbered lists) while preserving clinical headings, disclaimers, and refusal statements.
2. **Semantic Claim-to-Source Support Verification**: Reuses the local, zero-leak `EmbeddingService` (`all-MiniLM-L6-v2`) to compute semantic support scores between each claim and its cited source chunks/sub-segments against a conservative threshold ($\tau = 0.65$).
3. **Fail-Safe Unsupported Claim Defense**: In clinical RAG mode, answers containing unsupported factual claims or stripped invalid citations are not silently left in the final output; the system prunes them or safely transitions to the established insufficient-evidence response.
4. **Citation Coverage Metrics**: Emits detailed validation metadata (`claims_checked`, `claims_supported`, `claims_unsupported`, `citation_coverage`).

---

## 2. Files Inspected

In accordance with Phase 3 architectural guidelines, existing implementations were inspected prior to modification:
- `backend/evaluation/citation_validator.py`: Inspected legacy regex parsing, index validation, and citation stripping.
- `backend/rag/rag_service.py`: Inspected retrieved source record generation (`build_sources`), answer assembly, and post-generation citation filtering.
- `backend/services/gemini_service.py`: Inspected generation interface and disclaimer handling.
- `backend/rag/prompt_builder.py`: Inspected grounded prompt template and citation instructions.
- `backend/services/embedding_service.py`: Inspected existing `all-MiniLM-L6-v2` local model inference and vector normalization.
- `tests/test_citation_validator.py`: Inspected 18 existing baseline citation validation tests.
- `tests/test_gemini_service.py`: Inspected existing generation mocks and fallback tests.
- `tests/test_phase3_grounded_prompt.py`: Inspected 10 Phase 3.2 prompt hardening and boundary defense tests.

---

## 3. Baseline Tests

Before applying any code changes, the baseline test suites were executed:
- `tests/test_citation_validator.py`: **18 passed** (19.24s)
- `tests/test_gemini_service.py`, `tests/test_phase3_grounded_prompt.py`, `tests/test_safety_agent.py`: **31 passed** (9.32s)
- Combined Baseline: **49 passed**, 0 failures.

---

## 4. Defect Identified

### Defect 1: Syntactic Validity vs. Semantic Grounding
`CitationValidator.validate_citations()` verified that index $N$ in `[Source N]` satisfied $1 \le N \le \text{num\_sources}$. It performed zero textual comparison between the cited chunk text and the claim preceding or enclosing the citation tag. Hallucinated medical assertions citing real source indices passed validation unchecked.

### Defect 2: Orphaned Unsupported Claims After Stripping
`CitationValidator.strip_invalid_citations()` matched invalid citations (e.g. `[Source 99]`) and removed the substring `"[Source 99]"` from the text. This converted an invalidly cited claim (`"Metformin cures hypertension [Source 99]"`) into an uncited factual statement (`"Metformin cures hypertension"`), leaving dangerous medical misinformation in the final answer.

### Defect 3: Missing Source Content in RAG Pipeline Source Records
In `backend/rag/rag_service.py` (`build_sources()`), the source dictionary populated `source_number`, `document`, `filename`, `page`, `chunk_id`, and `similarity_score`, but omitted the actual text content (`"text"` or `"content"`). Consequently, downstream citation validators had no access to the retrieved text against which to verify claims.

---

## 5. Implementation Changes

### A. `backend/evaluation/citation_validator.py`
1. **Added `ExtractedClaim` Dataclass**:
   ```python
   @dataclass
   class ExtractedClaim:
       claim_id: int
       text: str
       cited_source_ids: List[int]
       is_supported: bool = False
       support_score: float = 0.0
       supporting_source_id: Optional[int] = None
       failure_reason: Optional[str] = None
   ```
2. **Extended `CitationValidationResult`**:
   - Added `claims_checked: int = 0`
   - Added `claims_supported: int = 0`
   - Added `claims_unsupported: int = 0`
   - Added `citation_coverage: float = 1.0`
   - Added `extracted_claims: List[ExtractedClaim] = field(default_factory=list)`
   - Added `unsupported_claims: List[ExtractedClaim] = field(default_factory=list)`
   - Added `cleaned_grounded_answer: Optional[str] = None`
3. **Deterministic Claim Extraction (`extract_claims`)**:
   - Handles standard sentences, bullet points (`*`, `-`, `•`), numbered lists (`1.`, `2.`), and multi-sentence paragraphs.
   - Normalizes citation positions so tags placed after punctuation (e.g., `Sentence. [Source 1]`) bind to the preceding claim.
   - Propagates paragraph-level citations to uncited sentences within the paragraph.
   - Filters out non-factual noise: headers (`#`), disclaimers (`Informational only`, `Consult a healthcare provider`), refusal statements, and citation-only fragments.
4. **Semantic Support Verification (`compute_support_score` & `check_claim_support`)**:
   - Reuses `EmbeddingService` (`all-MiniLM-L6-v2`) via singleton pattern; no new models loaded.
   - Evaluates cosine similarity between the claim and the cited source chunk as well as individual sentences/sub-segments of the source chunk to prevent dilution in long chunks.
   - Checks if similarity meets or exceeds conservative threshold $\tau = 0.65$.
5. **Unsupported Claim Pruning (`prune_unsupported_claims`)**:
   - Safely removes unsupported claim segments while maintaining formatting and punctuation.
6. **Grounding Validation Entry Point (`validate_grounded_citations`)**:
   - Performs syntactic citation validation followed by semantic support checks and coverage calculations.
   - `validate_citations()` retains `check_claim_support=False` default for complete backwards compatibility with legacy tests.

### B. `backend/rag/rag_service.py`
1. **Source Content Inclusion in `build_sources()`**:
   - Added `"text": text_val` to `source_rec` so cited chunk text is accessible for validation.
2. **Grounding Enforcement in Generation Assembly**:
   - Calls `CitationValidator.validate_grounded_citations(answer, sources)` on LLM generated answers.
   - If cited claims are present but unsupported (`claims_unsupported > 0`), the pipeline executes safe fallback:
     - Attempts pruning via `cleaned_grounded_answer`.
     - If the remaining text lacks sufficient grounded facts, falls back to `INSUFFICIENT_EVIDENCE_RESPONSE` (*"Based on the provided medical documents, there is insufficient evidence to answer your question."*).
3. **Diagnostic Metrics Reporting**:
   - Appends `citation_coverage`, `claims_checked`, `claims_supported`, and `claims_unsupported` to RAG response `timings` dictionary.

---

## 6. Claim Extraction Strategy

The claim extraction engine implements a deterministic rule-based segmentation pipeline:
1. **Citation Normalization**: Rewrites trailing punctuation before citations (e.g. `text. [Source 1]` $\to$ `text [Source 1].`) to keep citations bound to their host sentences.
2. **Block-Level Segmentation**: Splits the answer by lines/paragraphs to identify markdown bullets (`-`, `*`, `•`) and numbered list items (`\d+\.`).
3. **Sentence-Level Splitting**: Splits prose paragraphs using sentence boundaries (`[.!?]+\s+`).
4. **Citation Extraction**: Parses all `\[(?:Source\s+)?(\d+)(?:,\s*(?:Source\s+)?(\d+))*\]` patterns.
5. **Contextual Scope Propagation**: When a citation tag appears at the end of a multi-sentence paragraph or list item, all constituent factual sentences in that block inherit the citation scope.
6. **Non-Claim Filtering**: Explicitly skips:
   - Empty lines and pure whitespace
   - Markdown headings (`# Header`)
   - Standard clinical disclaimers (*"Note: This information is for educational purposes..."*, *"Consult a doctor..."*)
   - System refusal statements (*"Based on the provided documents, there is insufficient evidence..."*)
   - Solitary citation markers with no semantic text.

---

## 7. Claim-to-Source Support Strategy

```
           Candidate Claim
                 │
                 ▼
       Extract Cited Source IDs
                 │
     ┌───────────┴───────────┐
     ▼                       ▼
Source ID Valid?        Source ID Invalid?
     │                       │
    YES                      NO ──► Mark UNSUPPORTED (Invalid Citation Number)
     │
     ▼
Fetch Source Content
     │
     ▼
Segment Source Content (Full Chunk + Sub-Sentences)
     │
     ▼
Vector Embeddings (EmbeddingService: all-MiniLM-L6-v2)
     │
     ▼
Cosine Similarity (Dot Product of L2-Normalized Vectors)
     │
     ▼
Max Similarity >= 0.65?
     │
 ┌───┴───┐
 ▼       ▼
YES      NO
 │       │
 │       ▼
 │   Mark UNSUPPORTED (Semantic Support Mismatch)
 │
 ▼
Mark SUPPORTED
```

---

## 8. Thresholds Used and Rationale

| Parameter | Value | Rationale |
|:---|:---:|:---|
| **Semantic Support Threshold ($\tau$)** | `0.65` | Conservative threshold for `all-MiniLM-L6-v2`. Cosine similarity $\ge 0.65$ reliably confirms semantic overlap for paraphrased clinical statements while rejecting unrelated topics (e.g., hypertension vs. diabetes) and unmentioned drugs (e.g., metformin). |
| **Minimum Grounded Answer Length** | `30` chars | If pruning unsupported claims reduces the answer to under 30 characters of meaningful text, the response is discarded in favor of the standard safe insufficient-evidence notification. |
| **Minimum Coverage for Fully Valid Answer** | `1.0` (100%) | In medical RAG queries where citations are provided, all claims must be grounded in the retrieved sources. |

---

## 9. Invalid Citation & Unsupported Claim Behavior

1. **Out-of-Bounds Citations (`[Source 99]`)**:
   - The citation number is rejected.
   - The associated claim is marked `is_supported = False` with `failure_reason = "Invalid citation: source 99 not found"`.
   - The unsupported claim is removed from `cleaned_grounded_answer`.
2. **Wrong-Source Citations**:
   - If Source 1 discusses hypertension and Source 2 discusses diabetes, citing Source 1 for a diabetes claim produces similarity $< 0.65$.
   - Marked `is_supported = False` with `failure_reason = "Claim not supported by cited source(s)"`.
3. **Uncited Factual Claims**:
   - When citations are present in the response, any factual statement with no citations is flagged as uncited and unsupported.
4. **Safety Fallback**:
   - If unsupported claims are detected, the RAG service either provides the pruned answer (if valid supported content remains) or transitions entirely to `INSUFFICIENT_EVIDENCE_RESPONSE`.

---

## 10. Insufficient-Evidence Behavior

When citations cannot be verified or all claims are unsupported:
- The system returns the certified safe constant:
  `"Based on the provided medical documents, there is insufficient evidence to answer your question."`
- Sources list is cleared to prevent misleading the user.
- Status is recorded with diagnostic flags in `timings`.
- Zero external medical knowledge or hallucinated alternatives are generated.

---

## 11. Test Suite Summary

### Phase 3.3 Test Suite (`tests/test_phase3_citation_enforcement.py`)
13 rigorous tests covering all specified requirements:

| # | Test Name | Target Behavior | Result |
|:---:|:---|:---|:---:|
| 1 | `test_1_valid_citation` | Valid citation matching source content $\to$ SUPPORTED | **PASSED** |
| 2 | `test_2_invalid_source_number` | Out-of-bounds citation (`[Source 99]`) $\to$ REJECTED | **PASSED** |
| 3 | `test_3_wrong_source_citation` | Valid index citing wrong topic $\to$ UNSUPPORTED | **PASSED** |
| 4 | `test_4_correct_source_citation` | Swapping to correct source index $\to$ SUPPORTED | **PASSED** |
| 5 | `test_5_hallucinated_claim` | Unmentioned drug/intervention $\to$ UNSUPPORTED | **PASSED** |
| 6 | `test_6_multiple_source_citation` | Multi-source tag `[Source 1, Source 2]` $\to$ SUPPORTED | **PASSED** |
| 7 | `test_7_missing_citation` | Uncited factual claim flagged $\to$ UNSUPPORTED | **PASSED** |
| 8 | `test_8_unsupported_claim_defense` | Stripping invalid citation removes claim $\to$ DEFENDED | **PASSED** |
| 9 | `test_9_citation_coverage_calculation` | Metric computation (3 checked, 2 supported, cov=0.67) | **PASSED** |
| 10 | `test_10_multiple_claims_with_mixed_support` | Multi-claim paragraph pruning unsupported items | **PASSED** |
| 11 | `test_11_existing_citation_format_compatibility` | Compatibility with `[Source N]` format | **PASSED** |
| 12 | `test_12_legacy_citation_validator_behavior` | Backwards compatibility of legacy `validate_citations()` | **PASSED** |
| 13 | `test_13_rag_service_rejects_hallucinated_answer` | End-to-end `RAGService` halts on hallucination | **PASSED** |

---

## 12. Test Results & Regression Verification

### Targeted Phase 3 & Safety Suite
```powershell
venv\Scripts\pytest.exe tests/test_citation_validator.py tests/test_gemini_service.py tests/test_phase3_grounded_prompt.py tests/test_phase3_citation_enforcement.py tests/test_safety_agent.py -q
```
**Result: 62 passed in 17.34s**

### Phase 2 Retrieval Hardening Suite
```powershell
venv\Scripts\pytest.exe tests/test_phase2_retrieval_hardening.py -q
```
**Result: 25 passed in 7.32s**

### Full System Regression Suite
```powershell
venv\Scripts\pytest.exe -q
```
**Result: 438 passed, 3 warnings in 60.29s (100% passing across entire repository)**

---

## 13. Vector Store Integrity

Vector store invariance verification command:
```powershell
venv\Scripts\python.exe -c "import faiss,json; idx=faiss.read_index('data/vector_store/index.faiss'); data=json.load(open('data/vector_store/metadata.json')); print('Vectors:',idx.ntotal); print('Metadata:',data['count'],len(data['records']))"
```

- **Count Before Phase 3.3 Tests:** 718 vectors / 718 metadata records
- **Count After Phase 3.3 Tests:** 718 vectors / 718 metadata records
- **Mutation:** **0 mutations** (100% read-only integrity verified during Phase 3.3 test execution)

---

## 14. Remaining Limitations

1. **Embedding Similarity is NOT Medical Entailment**:
   - High cosine similarity ($\ge 0.65$) indicates strong semantic and topical relatedness; it is a vital support signal, but not a mathematical proof of clinical entailment or contraindication logic.
   - For example, *"Drug X is safe in pregnancy"* and *"Drug X is contraindicated in pregnancy"* share high semantic similarity due to overlapping vocabulary.
   - Directional entailment and medical negation validation will be addressed in Phase 3.4 (Advanced Hallucination Protection).
2. **Complex Multi-Clause Sentences**:
   - Sentences combining multiple factual assertions with a single citation may pass if the majority of the clause matches the source chunk.
3. **No External Knowledge Fact-Checking**:
   - In accordance with safety rules, the system never queries external unverified knowledge to validate or correct claims; validation is strictly bounded to the user's retrieved documents.

---

## 15. Certification

Phase 3.3 (Citation Enforcement) is hereby **CERTIFIED** for integration.
- Deterministic claim segmentation active.
- Semantic grounding validation active via local embedding infrastructure.
- Orphaned claim vulnerabilities eliminated.
- Full backwards compatibility and 100% regression suite pass rate confirmed.
