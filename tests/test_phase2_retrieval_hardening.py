"""
Phase 2: Comprehensive Retrieval Pipeline Hardening Test Suite.

Validates the complete 20 requirements defined in Phase 19:
1. Query normalization
2. Synonym expansion
3. Medication-intent gating
4. Subject extraction
5. Unknown subject handling
6. Multi-aspect decomposition
7. Multi-query retrieval
8. Candidate merging
9. Deduplication
10. Precision filtering
11. Ambiguity rejection
12. Distractor rejection
13. Out-of-scope rejection
14. Citation grounding
15. User isolation
16. Top_k enforcement
17. Vector-count invariance
18. Latency sanity
19. Document comparison
20. Safety-critical false positives
"""

import time
import pytest
from unittest.mock import MagicMock

from backend.rag.query_expander import MedicalQueryExpander
from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import get_vector_store_service


# ==============================================================================
# 1. Query Normalization
# ==============================================================================
def test_1_query_normalization():
    """Verify whitespace, casing, and punctuation normalization without losing medical semantics."""
    raw_query = "   What IS the DEFINITION of...   Hypertension???   "
    clean = raw_query.strip().lower().rstrip("?!., ")
    assert clean == "what is the definition of...   hypertension"
    terms = MedicalQueryExpander.get_expanded_terms(raw_query)
    assert any("blood pressure" in t.lower() for t in terms)


# ==============================================================================
# 2. Synonym Expansion
# ==============================================================================
def test_2_synonym_expansion():
    """Verify medical clinical terminology maps to canonical concepts."""
    q_bp = "What causes elevated arterial blood pressure?"
    terms_bp = MedicalQueryExpander.get_expanded_terms(q_bp)
    assert any("hypertension" in t.lower() for t in terms_bp)

    q_dm = "What is the clinical management of elevated blood glucose?"
    terms_dm = MedicalQueryExpander.get_expanded_terms(q_dm)
    assert any("diabetes" in t.lower() for t in terms_dm)

    q_seq = "What are the common secondary sequelae of high blood pressure?"
    terms_seq = MedicalQueryExpander.get_expanded_terms(q_seq)
    assert any("complications" in t.lower() for t in terms_seq)


# ==============================================================================
# 3. Medication-Intent Gating
# ==============================================================================
def test_3_medication_intent_gating():
    """Verify medication names are NOT injected unless explicit medication intent is present."""
    q_no_med = "What are glycemic disorders and what is their definition?"
    terms_no_med = MedicalQueryExpander.get_expanded_terms(q_no_med)
    assert any("diabetes" in t.lower() for t in terms_no_med)
    # Metformin must NOT be injected
    assert not any("metformin" in t.lower() for t in terms_no_med)

    q_with_med = "What medications regulate blood sugar in glycemic disorders?"
    terms_with_med = MedicalQueryExpander.get_expanded_terms(q_with_med)
    assert any("diabetes" in t.lower() for t in terms_with_med)
    assert any("metformin" in t.lower() for t in terms_with_med)


# ==============================================================================
# 4. Subject Extraction
# ==============================================================================
@pytest.mark.parametrize(
    "query, expected_subject",
    [
        ("What causes elevated blood pressure?", "hypertension"),
        ("Explain diagnostic criteria for type 2 diabetes", "diabetes"),
        ("What triggers bronchial wheezing and shortness of breath in asthma?", "asthma"),
        ("What acute intervention is recommended for myocardial infarction?", "cardiovascular"),
        ("What are the clinical signs of community-acquired pneumonia?", "pneumonia"),
        ("How is rheumatoid arthritis diagnosed in clinical practice?", "arthritis"),
    ]
)
def test_4_subject_extraction(query, expected_subject):
    """Verify accurate extraction of primary condition."""
    subject = MedicalQueryExpander.extract_primary_subject(query)
    assert subject.lower() == expected_subject.lower()


# ==============================================================================
# 5. Unknown Subject Handling
# ==============================================================================
def test_5_unknown_subject_handling():
    """Verify queries with unknown or no subjects return empty string and NEVER default to hypertension."""
    queries = [
        "What is the weather forecast for tomorrow?",
        "How do you implement an asynchronous B-tree index in Rust?",
        "What treatment is recommended?",
        "What follow-up plan is established?",
    ]
    for q in queries:
        subj = MedicalQueryExpander.extract_primary_subject(q)
        assert subj != "hypertension"
        assert subj == ""


# ==============================================================================
# 6. Multi-Aspect Decomposition
# ==============================================================================
def test_6_multi_aspect_decomposition():
    """Verify compound queries are decomposed into 2-4 bounded sub-queries with the true subject."""
    q = "Explain hypertension definition, risk factors, lifestyle measures, and complications."
    assert MedicalQueryExpander.is_multi_aspect_query(q) is True
    sub_queries = MedicalQueryExpander.decompose_multi_aspect_query(q)
    assert 2 <= len(sub_queries) <= 4
    assert all("hypertension" in sq.lower() for sq in sub_queries)
    # Check facet representations
    joined = " ".join(sub_queries).lower()
    assert "definition" in joined or "criteria" in joined
    assert "risk factors" in joined
    assert "lifestyle" in joined
    assert "complications" in joined


# ==============================================================================
# 7. Multi-Query Retrieval
# ==============================================================================
def test_7_multi_query_retrieval():
    """Verify multi-aspect queries retrieve across all requested facets."""
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    q = "What are the diagnostic blood pressure criteria, medication choices, and long-term monitoring recommendations for hypertension?"
    res = rag.query(q, top_k=5)
    assert res.get("retrieval_status") == "success"
    sources = res.get("sources", [])
    assert len(sources) >= 2


# ==============================================================================
# 8. Candidate Merging
# ==============================================================================
def test_8_candidate_merging():
    """Verify candidates from multiple sub-query executions are combined without losing high-similarity chunks."""
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    # Both synthetic hypertension test and hypertension summary should be represented
    q = "Detail diagnostic blood pressure criteria and medication choices for hypertension."
    res = rag.query(q, top_k=5)
    fnames = [s.get("filename") for s in res.get("sources", [])]
    assert "synthetic_hypertension_test.pdf" in fnames or "hypertension_summary.pdf" in fnames


# ==============================================================================
# 9. Deduplication
# ==============================================================================
def test_9_deduplication():
    """Verify duplicate uploads are deduplicated while distinct documents sharing mock IDs are preserved."""
    # Subtest A: Exact duplicate content deduplication
    dup_chunks = [
        {"document_id": "1", "chunk_id": "c1", "similarity_score": 0.85, "text": "Same text here", "metadata": {"filename": "docA.pdf"}},
        {"document_id": "2", "chunk_id": "c1", "similarity_score": 0.89, "text": "Same text here", "metadata": {"filename": "docA.pdf"}},
    ]
    deduped_a = RAGService.deduplicate_chunks(dup_chunks)
    assert len(deduped_a) == 1
    assert deduped_a[0]["similarity_score"] == 0.89

    # Subtest B: Distinct documents sharing document_id: "1" must NOT be merged
    shared_id_chunks = [
        {"document_id": "1", "chunk_id": "chunk_0", "similarity_score": 0.88, "text": "Doc 1 text", "metadata": {"filename": "file1.pdf"}},
        {"document_id": "1", "chunk_id": "chunk_0", "similarity_score": 0.82, "text": "Doc 2 text", "metadata": {"filename": "file2.pdf"}},
    ]
    deduped_b = RAGService.deduplicate_chunks(shared_id_chunks)
    assert len(deduped_b) == 2


# ==============================================================================
# 10. Precision Filtering
# ==============================================================================
def test_10_precision_filtering():
    """Verify candidates with low relative similarity or irrelevant medical content are filtered."""
    chunks = [
        {"document_id": "1", "chunk_id": "c1", "similarity_score": 0.85, "text": "Hypertension is high blood pressure.", "metadata": {"filename": "htn.pdf"}},
        {"document_id": "2", "chunk_id": "c2", "similarity_score": 0.28, "text": "Dermatology skin lesions and rash treatment.", "metadata": {"filename": "derma.pdf"}},
    ]
    filtered = RAGService.filter_candidate_precision("What is hypertension?", chunks, threshold=0.25)
    assert len(filtered) == 1
    assert filtered[0]["document_id"] == "1"


# ==============================================================================
# 11. Ambiguity Rejection
# ==============================================================================
def test_11_ambiguity_rejection():
    """Verify completely unanchored ambiguous queries safely halt without guessing diseases."""
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    res = rag.query("What follow-up plan is established?", top_k=5)
    assert res.get("retrieval_status") == "no_relevant_context"
    assert len(res.get("sources", [])) == 0


# ==============================================================================
# 12. Distractor Rejection
# ==============================================================================
def test_12_distractor_rejection():
    """Verify false-framing distractor queries targeting wrong documents are rejected."""
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    q = "What does the dermatology pdf say regarding emergency cardioversion and defibrillation protocols?"
    res = rag.query(q, top_k=5)
    assert res.get("retrieval_status") == "no_relevant_context"
    assert len(res.get("sources", [])) == 0


# ==============================================================================
# 13. Out-of-Scope Rejection
# ==============================================================================
def test_13_out_of_scope_rejection():
    """Verify non-medical queries return no_relevant_context."""
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    q = "What is the recommended chemotherapy regimen for metastatic glioblastoma multiforme?"
    res = rag.query(q, top_k=5)
    assert res.get("retrieval_status") == "no_relevant_context"
    assert len(res.get("sources", [])) == 0


# ==============================================================================
# 14. Citation Grounding
# ==============================================================================
def test_14_citation_grounding():
    """Verify retrieved sources strictly correspond to returned citations and source numbering."""
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    q = "What causes elevated arterial blood pressure?"
    res = rag.query(q, top_k=3)
    assert res.get("retrieval_status") == "success"
    sources = res.get("sources", [])
    assert len(sources) > 0
    for idx, src in enumerate(sources, start=1):
        assert src.get("source_index") == idx
        assert src.get("source_label") == f"[Source {idx}]"
        assert src.get("filename") is not None


# ==============================================================================
# 15. User Isolation
# ==============================================================================
def test_15_user_isolation():
    """Verify user_id filter is strictly applied to FAISS retrieval."""
    mock_vs = MagicMock()
    mock_vs.count.return_value = 10
    mock_vs.search.return_value = [
        {"document_id": "1", "chunk_id": "c1", "similarity_score": 0.85, "text": "User 101 text", "metadata": {"filename": "u101.pdf", "user_id": 101}}
    ]

    rag = RAGService(vector_store=mock_vs)
    rag.query("What causes hypertension?", user_id=101)

    # user_id=101 must be passed to all search invocations
    for call_args in mock_vs.search.call_args_list:
        assert call_args.kwargs.get("user_id") == 101


# ==============================================================================
# 16. Top_k Enforcement
# ==============================================================================
def test_16_top_k_enforcement():
    """Verify retrieval respects top_k limit."""
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    for k in [1, 2, 4]:
        res = rag.query("What are the symptoms and lifestyle changes for hypertension?", top_k=k)
        assert len(res.get("sources", [])) <= k


# ==============================================================================
# 17. Vector-Count Invariance
# ==============================================================================
def test_17_vector_count_invariance():
    """Verify retrieval operations do not alter vector store count or mutate the store."""
    vs = get_vector_store_service()
    before = vs.count()
    rag = RAGService(vector_store=vs)
    rag.query("hypertension", top_k=5)
    rag.query("diabetes", top_k=5)
    after = vs.count()
    assert before == after
    assert before >= 637


# ==============================================================================
# 18. Latency Sanity
# ==============================================================================
def test_18_latency_sanity():
    """Verify deterministic query expansion and decomposition takes < 5 ms."""
    q = "Explain hypertension definition, risk factors, lifestyle measures, and complications."
    t0 = time.perf_counter()
    _ = MedicalQueryExpander.is_multi_aspect_query(q)
    _ = MedicalQueryExpander.decompose_multi_aspect_query(q)
    _ = MedicalQueryExpander.get_expanded_terms(q)
    _ = MedicalQueryExpander.expand_query(q)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 5.0, f"Query expansion took {elapsed_ms:.2f} ms"


# ==============================================================================
# 19. Document Comparison
# ==============================================================================
def test_19_document_comparison():
    """Verify comparative queries do not prune candidates down to a single document."""
    q = "Compare the general lifestyle guidance in the synthetic test document with patient John Doe's actual clinical vitals."
    scoped = RAGService.extract_scoped_document_name(q)
    # In filter_candidate_precision, is_comparison prevents single-document pruning
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    res = rag.query(q, top_k=5)
    assert res.get("retrieval_status") == "success"
    sources = res.get("sources", [])
    fnames = {s.get("filename") for s in sources}
    # Multiple documents should be retrieved
    assert len(fnames) >= 2


# ==============================================================================
# 20. Safety-Critical False Positives
# ==============================================================================
def test_20_safety_critical_false_positives():
    """Verify cross-specialty questions cannot retrieve unrelated clinical records."""
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    q = "What are the chemotherapy protocols for pancreatic adenocarcinoma in the general hypertension guideline?"
    res = rag.query(q, top_k=5)
    # Must NOT return hypertension documents as chemotherapy evidence
    assert res.get("retrieval_status") == "no_relevant_context"
    assert len(res.get("sources", [])) == 0
