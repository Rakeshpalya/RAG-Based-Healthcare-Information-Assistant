"""
Phase 2E: Retrieval Recall Improvement Unit & Regression Tests.

Validates the 12 key recall requirements:
1. Hypertension synonym expansion
2. Diabetes/glycemic synonym expansion
3. Complications/sequelae synonym expansion
4. Multi-aspect query decomposition
5. Multi-query candidate merging
6. Duplicate candidate removal
7. Irrelevant medical document rejection
8. Ambiguous query safety
9. Preservation of Phase 2C precision
10. Top_k enforcement
11. Existing grounded-boundary behavior
12. User-document isolation
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.rag.query_expander import MedicalQueryExpander
from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import VectorStoreService


# ==============================================================================
# 1. Hypertension Synonym Expansion
# ==============================================================================
def test_1_hypertension_synonym_expansion():
    """Validates that high blood pressure queries expand and match hypertension evidence."""
    query = "What causes elevated arterial blood pressure?"
    terms = MedicalQueryExpander.get_expanded_terms(query)
    assert any("hypertension" in t.lower() for t in terms)

    expanded_q = MedicalQueryExpander.expand_query(query)
    assert "hypertension" in expanded_q.lower()

    # Verify _is_chunk_relevant_to_query matches chunk discussing hypertension
    chunk_text = "Primary hypertension is associated with genetics, sodium intake, and vascular stiffness."
    assert RAGService._is_chunk_relevant_to_query(query, chunk_text) is True


# ==============================================================================
# 2. Diabetes / Glycemic Synonym Expansion
# ==============================================================================
def test_2_diabetes_glycemic_synonym_expansion():
    """Validates that glycemic and blood sugar queries match diabetes and metformin evidence."""
    query = "What medications help regulate blood sugar in glycemic disorders?"
    terms = MedicalQueryExpander.get_expanded_terms(query)
    assert any(term in ["diabetes", "type 2 diabetes", "glucose", "insulin", "metformin"] for term in terms)

    chunk_text = "Metformin is the first-line pharmacotherapy for Type 2 Diabetes to control hyperglycemia."
    assert RAGService._is_chunk_relevant_to_query(query, chunk_text) is True


# ==============================================================================
# 3. Complications / Sequelae Synonym Expansion
# ==============================================================================
def test_3_complications_sequelae_synonym_expansion():
    """Validates that sequelae queries expand to complications and target organ damage."""
    query = "What are the common secondary sequelae of uncontrolled blood pressure?"
    terms = MedicalQueryExpander.get_expanded_terms(query)
    assert any(term in ["complications", "organ damage", "heart disease", "stroke"] for term in terms)

    chunk_text = "Chronic high blood pressure leads to secondary complications including stroke, nephropathy, and heart failure."
    assert RAGService._is_chunk_relevant_to_query(query, chunk_text) is True


# ==============================================================================
# 4. Multi-Aspect Query Decomposition
# ==============================================================================
def test_4_multi_aspect_query_decomposition():
    """Validates that compound queries are recognized and decomposed into focused sub-queries."""
    query = "Provide an overview of hypertension definition, risk factors, lifestyle measures, and complications."
    assert MedicalQueryExpander.is_multi_aspect_query(query) is True

    sub_queries = MedicalQueryExpander.decompose_multi_aspect_query(query)
    assert len(sub_queries) >= 3
    combined = " ".join(sub_queries).lower()
    assert "definition" in combined
    assert "risk factor" in combined or "factors" in combined
    assert "lifestyle" in combined
    assert "complications" in combined


# ==============================================================================
# 5. Multi-Query Candidate Merging
# ==============================================================================
def test_5_multi_query_candidate_merging():
    """Validates that candidate chunks from distinct aspects are merged during multi-aspect retrieval."""
    mock_vs = MagicMock(spec=VectorStoreService)

    # Sub-query 1 returns definition chunk; Sub-query 2 returns complications chunk
    chunk_def = {
        "document_id": "doc_def",
        "chunk_id": "c0",
        "similarity_score": 0.85,
        "text": "Hypertension is defined as persistent systolic BP >= 130 mmHg.",
        "metadata": {"filename": "def.pdf"}
    }
    chunk_comp = {
        "document_id": "doc_comp",
        "chunk_id": "c1",
        "similarity_score": 0.80,
        "text": "Complications of hypertension include stroke and myocardial infarction.",
        "metadata": {"filename": "comp.pdf"}
    }

    call_count = [0]
    def mock_search(vec, top_k=20, user_id=None):
        call_count[0] += 1
        if call_count[0] <= 1:
            return [chunk_def]
        return [chunk_comp]

    mock_vs.search.side_effect = mock_search

    rag = RAGService(vector_store=mock_vs)
    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        results = rag.retrieve_context(
            "hypertension definition, risk factors, and complications",
            top_k=5
        )

    doc_ids = [c["document_id"] for c in results]
    assert "doc_def" in doc_ids
    assert "doc_comp" in doc_ids


# ==============================================================================
# 6. Duplicate Candidate Removal
# ==============================================================================
def test_6_duplicate_candidate_removal():
    """Validates that multiple identical chunk copies across sub-queries are deduplicated."""
    chunk_copy1 = {
        "document_id": "doc1",
        "chunk_id": "c0",
        "similarity_score": 0.75,
        "text": "Regular aerobic exercise reduces systolic blood pressure by 5-8 mmHg.",
        "metadata": {"filename": "exercise.pdf"}
    }
    chunk_copy2 = {
        "document_id": "doc1",
        "chunk_id": "c0",
        "similarity_score": 0.82,  # higher score
        "text": "Regular aerobic exercise reduces systolic blood pressure by 5-8 mmHg.",
        "metadata": {"filename": "exercise.pdf"}
    }

    deduped = RAGService.deduplicate_chunks([chunk_copy1, chunk_copy2])
    assert len(deduped) == 1
    assert deduped[0]["similarity_score"] == 0.82


# ==============================================================================
# 7. Irrelevant Medical Document Rejection
# ==============================================================================
def test_7_irrelevant_medical_document_rejection():
    """Validates that out-of-scope medical questions are safely rejected by pre-LLM safety gate."""
    query = "What are the chemotherapy guidelines for pancreatic carcinoma in the general hypertension guideline?"
    irrelevant_chunks = [
        {
            "document_id": "doc_htn",
            "chunk_id": "c0",
            "similarity_score": 0.35,
            "text": "General clinical guideline for hypertension management and target BP.",
            "metadata": {"filename": "hypertension_summary.pdf"}
        }
    ]

    is_relevant, reason = RAGService.verify_relevance_and_sufficiency(
        question=query,
        retrieved_chunks=irrelevant_chunks,
        similarity_threshold=0.25
    )
    assert is_relevant is False
    assert "query_targets_condition" in reason or "no_core_query_terms" in reason


# ==============================================================================
# 8. Ambiguous Query Safety
# ==============================================================================
def test_8_ambiguous_query_safety():
    """Validates that ambiguous queries without sufficient grounded evidence halt generation safely."""
    query = "What treatment was prescribed for the patient?"
    unrelated_chunks = [
        {
            "document_id": "doc_diag",
            "chunk_id": "c0",
            "similarity_score": 0.30,
            "text": "Patient vitals and demographic data. Blood pressure 145/95 mmHg. No prescription written.",
            "metadata": {"filename": "triage.pdf"}
        }
    ]

    is_relevant, reason = RAGService.verify_relevance_and_sufficiency(
        question=query,
        retrieved_chunks=unrelated_chunks,
        similarity_threshold=0.25
    )
    assert is_relevant is False


# ==============================================================================
# 9. Preservation of Phase 2C Precision
# ==============================================================================
def test_9_preservation_of_phase_2c_precision():
    """Validates that Phase 2C dynamic relative pruning prunes weakly-related chunks when a strong match exists."""
    query = "What are the diagnostic blood pressure thresholds for hypertension?"
    strong_chunk = {
        "document_id": "doc_htn_thresh",
        "chunk_id": "c0",
        "similarity_score": 0.85,
        "text": "Hypertension is diagnosed when blood pressure is consistently 130/80 mmHg or higher.",
        "metadata": {"filename": "htn.pdf"}
    }
    weak_irrelevant_chunk = {
        "document_id": "doc_skin",
        "chunk_id": "c1",
        "similarity_score": 0.45,
        "text": "Eczema and dermatitis are inflammatory skin conditions presenting with erythema.",
        "metadata": {"filename": "derma.pdf"}
    }

    filtered = RAGService.filter_candidate_precision(
        query=query,
        chunks=[strong_chunk, weak_irrelevant_chunk],
        threshold=0.25
    )
    assert len(filtered) == 1
    assert filtered[0]["document_id"] == "doc_htn_thresh"


# ==============================================================================
# 10. Top_k Enforcement
# ==============================================================================
def test_10_top_k_enforcement():
    """Validates that retrieval output strictly adheres to top_k even after multi-query merging."""
    mock_vs = MagicMock(spec=VectorStoreService)

    # 10 distinct chunks returned across sub-queries
    many_chunks = [
        {
            "document_id": f"doc_{i}",
            "chunk_id": f"c_{i}",
            "similarity_score": 0.70 + (i * 0.01),
            "text": f"Hypertension clinical guideline fact number {i}.",
            "metadata": {"filename": f"doc_{i}.pdf"}
        }
        for i in range(10)
    ]
    mock_vs.search.return_value = many_chunks

    rag = RAGService(vector_store=mock_vs)
    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        retrieved = rag.retrieve_context("hypertension overview", top_k=3)

    assert len(retrieved) <= 3


# ==============================================================================
# 11. Existing Grounded-Boundary Behavior
# ==============================================================================
def test_11_existing_grounded_boundary_behavior():
    """Validates that explicit document boundary statements are preserved through precision filtering."""
    query = "What does the uploaded test document state about tuberculosis treatments?"
    boundary_chunk = {
        "document_id": "doc_boundary",
        "chunk_id": "c_boundary",
        "similarity_score": 0.42,
        "text": "Scope Boundary: This document does not contain information about tuberculosis treatments or regimens.",
        "metadata": {"filename": "HealthAI_RAG_Test_Document.pdf"}
    }

    filtered = RAGService.filter_candidate_precision(
        query=query,
        chunks=[boundary_chunk],
        threshold=0.25
    )
    assert len(filtered) == 1
    assert filtered[0]["document_id"] == "doc_boundary"


# ==============================================================================
# 12. User-Document Isolation
# ==============================================================================
def test_12_user_document_isolation():
    """Validates that multi-query retrieval passes user_id to all vector store searches."""
    mock_vs = MagicMock(spec=VectorStoreService)
    mock_vs.search.return_value = []

    rag = RAGService(vector_store=mock_vs)
    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        rag.retrieve_context(
            "hypertension definition, risk factors, and complications",
            top_k=5,
            user_id=42
        )

    # All calls to search must have user_id=42
    assert mock_vs.search.call_count >= 1
    for call_args in mock_vs.search.call_args_list:
        assert call_args.kwargs.get("user_id") == 42
