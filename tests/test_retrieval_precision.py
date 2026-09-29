"""
Phase 2C: Retrieval Precision & Evidence Quality Unit and Regression Tests.

Verifies:
A. Highly relevant candidate survives.
B. Clearly irrelevant high-scoring candidate is removed.
C. Clearly irrelevant lower-scoring candidate is removed.
D. Legitimate 3-document evidence survives.
E. Multiple chunks from one legitimate document survive.
F. Identical PDFs are still deduplicated by Phase 2A.
G. Explicit grounded-boundary chunk survives.
H. Low top-score queries do not become overly aggressive.
I. Precision filtering is applied in both retrieve_context() and query().
J. Existing source numbering remains correct.
K. Existing source-card/citation alignment remains correct.
L. Candidate pool expansion remains max(k*4, 20).
"""

import pytest
from unittest.mock import MagicMock, patch
from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiService


def test_a_highly_relevant_candidate_survives():
    """Test A: Highly relevant candidate strongly matching the query survives."""
    query = "What are the symptoms of hypertension?"
    candidates = [
        {
            "document_id": "doc_A_htn",
            "chunk_id": "c0",
            "similarity_score": 0.82,
            "text": "Hypertension is often asymptomatic, but severe symptoms of hypertension include severe headaches, fatigue, dizziness, nausea, and chest pain.",
            "metadata": {"filename": "hypertension_guide.pdf"}
        }
    ]

    filtered = RAGService.filter_candidate_precision(query, candidates, threshold=0.25)

    assert len(filtered) == 1
    assert filtered[0]["document_id"] == "doc_A_htn"
    assert filtered[0]["similarity_score"] == 0.82


def test_b_clearly_irrelevant_high_scoring_candidate_removed():
    """
    Test B: Clearly irrelevant high-scoring candidate (0.58) is removed when top_score is 0.82.
    Under the old formula: min_relative was max(0.25, 0.82*0.65=0.533, 0.82-0.25=0.57) = 0.570,
    so 0.58 survived. Under Phase 2C, min_relative is max(0.25, 0.82*0.75=0.615, 0.82-0.20=0.62) = 0.620,
    and the candidate is also recognized as lexically/topically irrelevant.
    """
    query = "What are the symptoms of hypertension?"
    candidates = [
        {
            "document_id": "doc_A_htn",
            "chunk_id": "c0",
            "similarity_score": 0.82,
            "text": "Hypertension symptoms include severe headaches, fatigue, dizziness, and chest pain.",
            "metadata": {"filename": "hypertension_guide.pdf"}
        },
        {
            "document_id": "doc_B_clinic_admin",
            "chunk_id": "c0",
            "similarity_score": 0.58,
            "text": "Clinic Administrative Protocol: Patients presenting to outpatient clinic must have identification verified, insurance documented, and primary care physician recorded.",
            "metadata": {"filename": "clinic_administration_policy.pdf"}
        }
    ]

    filtered = RAGService.filter_candidate_precision(query, candidates, threshold=0.25)

    assert len(filtered) == 1
    assert filtered[0]["document_id"] == "doc_A_htn"
    assert not any(c["document_id"] == "doc_B_clinic_admin" for c in filtered)


def test_c_clearly_irrelevant_lower_scoring_candidate_removed():
    """Test C: Clearly irrelevant candidate (0.51) is removed."""
    query = "What are the symptoms of hypertension?"
    candidates = [
        {
            "document_id": "doc_A_htn",
            "chunk_id": "c0",
            "similarity_score": 0.82,
            "text": "Hypertension symptoms include severe headaches, fatigue, dizziness, and chest pain.",
            "metadata": {"filename": "hypertension_guide.pdf"}
        },
        {
            "document_id": "doc_C_orthopedic",
            "chunk_id": "c0",
            "similarity_score": 0.51,
            "text": "Orthopedic Fracture Protocol: Suspected tibial shaft fractures require anteroposterior and lateral radiographs of the lower extremity.",
            "metadata": {"filename": "orthopedic_fracture_management.pdf"}
        }
    ]

    filtered = RAGService.filter_candidate_precision(query, candidates, threshold=0.25)

    assert len(filtered) == 1
    assert filtered[0]["document_id"] == "doc_A_htn"
    assert not any(c["document_id"] == "doc_C_orthopedic" for c in filtered)


def test_d_legitimate_three_document_evidence_survives():
    """
    Test D: Legitimate multi-document evidence (cardiology 0.88, diet 0.78, exercise 0.77)
    must all survive Phase 2C precision filtering.
    """
    query = "What lifestyle interventions and treatments help with hypertension?"
    candidates = [
        {
            "document_id": "doc_cardio",
            "chunk_id": "c0",
            "similarity_score": 0.88,
            "text": "Cardiology Guidelines: Hypertension management requires pharmacotherapy when blood pressure exceeds 140/90 mmHg.",
            "metadata": {"filename": "cardio.pdf"}
        },
        {
            "document_id": "doc_diet",
            "chunk_id": "c0",
            "similarity_score": 0.78,
            "text": "DASH Diet Study: Dietary approaches to stop hypertension substantially lower systolic and diastolic blood pressure.",
            "metadata": {"filename": "diet.pdf"}
        },
        {
            "document_id": "doc_exercise",
            "chunk_id": "c0",
            "similarity_score": 0.77,
            "text": "Exercise Medicine: Regular aerobic physical activity and exercise interventions reduce blood pressure in hypertensive patients.",
            "metadata": {"filename": "exercise.pdf"}
        }
    ]

    filtered = RAGService.filter_candidate_precision(query, candidates, threshold=0.25)

    assert len(filtered) == 3
    doc_ids = [c["document_id"] for c in filtered]
    assert doc_ids == ["doc_cardio", "doc_diet", "doc_exercise"]

    # Verify that diverse evidence selection also keeps all 3
    selected = RAGService.select_diverse_evidence(filtered, top_k=5, max_per_doc=2)
    assert len(selected) == 3
    assert set(c["document_id"] for c in selected) == {"doc_cardio", "doc_diet", "doc_exercise"}


def test_e_multiple_chunks_from_one_legitimate_document_survive():
    """
    Test E: Multiple chunks from a single legitimate document (0.90, 0.85, 0.80, 0.75, 0.70)
    all survive Phase 2C filtering and are selected by Phase 2B backfill.
    """
    query = "What is hypertension, what are its causes, symptoms, and lifestyle measures?"
    candidates = [
        {"document_id": "doc_htn", "chunk_id": "c0", "similarity_score": 0.90, "text": "Hypertension is chronically elevated arterial blood pressure."},
        {"document_id": "doc_htn", "chunk_id": "c1", "similarity_score": 0.85, "text": "Causes of hypertension include genetic factors, obesity, and sodium intake."},
        {"document_id": "doc_htn", "chunk_id": "c2", "similarity_score": 0.80, "text": "Symptoms of severe hypertension include headaches, dizziness, and blurred vision."},
        {"document_id": "doc_htn", "chunk_id": "c3", "similarity_score": 0.75, "text": "Lifestyle measures for high blood pressure include regular aerobic physical exercise."},
        {"document_id": "doc_htn", "chunk_id": "c4", "similarity_score": 0.70, "text": "Complications of uncontrolled blood pressure involve heart failure and stroke."},
    ]

    filtered = RAGService.filter_candidate_precision(query, candidates, threshold=0.25)
    assert len(filtered) == 5

    selected = RAGService.select_diverse_evidence(filtered, top_k=5, max_per_doc=2)
    assert len(selected) == 5
    assert [c["chunk_id"] for c in selected] == ["c0", "c1", "c2", "c3", "c4"]


def test_f_identical_pdfs_still_deduplicated_by_phase2a():
    """
    Test F: 3 identical copies of the same PDF are collapsed by Phase 2A deduplication,
    and Phase 2C preserves the single representative.
    """
    text = "HealthAI RAG Test Document for testing document-grounded question answering. Hypertension symptoms."
    chunks = [
        {"document_id": "18", "chunk_id": "c0", "similarity_score": 0.85, "text": text, "metadata": {"filename": "test.pdf"}},
        {"document_id": "17", "chunk_id": "c0", "similarity_score": 0.84, "text": text, "metadata": {"filename": "test.pdf"}},
        {"document_id": "16", "chunk_id": "c0", "similarity_score": 0.83, "text": text, "metadata": {"filename": "test.pdf"}},
    ]

    deduped = RAGService.deduplicate_chunks(chunks)
    assert len(deduped) == 1

    filtered = RAGService.filter_candidate_precision("What are hypertension symptoms?", deduped, threshold=0.25)
    assert len(filtered) == 1
    assert filtered[0]["document_id"] == "18"
    assert filtered[0]["similarity_score"] == 0.85


def test_g_explicit_grounded_boundary_chunk_survives():
    """
    Test G: Explicit document-boundary chunk (e.g. malaria out of scope) survives
    Phase 2C precision filtering so downstream grounded boundary logic can process it.
    """
    query = "What are the symptoms of malaria according to my uploaded PDF?"
    boundary_chunk = {
        "document_id": "doc_boundary",
        "chunk_id": "c0",
        "similarity_score": 0.395,
        "text": (
            "HealthAI RAG Test Document. Testing Boundary: This document does not contain "
            "information about tuberculosis, malaria, cancer, or their treatments. "
            "Questions about those topics should be treated as unsupported by this document."
        ),
        "metadata": {"filename": "HealthAI_RAG_Test_Document.pdf"}
    }

    filtered = RAGService.filter_candidate_precision(query, [boundary_chunk], threshold=0.25)

    assert len(filtered) == 1
    assert filtered[0]["document_id"] == "doc_boundary"
    assert filtered[0]["similarity_score"] == 0.395


def test_h_low_top_score_queries_do_not_become_overly_aggressive():
    """
    Test H: When top_score < 0.60, relative filtering is not overly aggressive,
    allowing valid evidence above 0.25 to survive.
    """
    query = "What is the clinical protocol for outpatient clinic follow-up?"
    candidates = [
        {
            "document_id": "doc_clinic",
            "chunk_id": "c0",
            "similarity_score": 0.45,
            "text": "Outpatient clinic follow-up protocol: scheduled consultations must occur within 14 days.",
            "metadata": {"filename": "clinic.pdf"}
        },
        {
            "document_id": "doc_clinic",
            "chunk_id": "c1",
            "similarity_score": 0.38,
            "text": "Consultation guidelines and clinical follow-up for registered patients.",
            "metadata": {"filename": "clinic.pdf"}
        }
    ]

    filtered = RAGService.filter_candidate_precision(query, candidates, threshold=0.25)

    # Both chunks should survive since top_score (0.45) < 0.60 and both are above 0.25 and relevant
    assert len(filtered) == 2
    assert filtered[0]["similarity_score"] == 0.45
    assert filtered[1]["similarity_score"] == 0.38


def test_i_precision_filtering_applied_in_both_retrieve_context_and_query():
    """
    Test I: Verify precision filtering eliminates irrelevant candidate (0.58) in both
    retrieve_context() and query().
    """
    mock_vs = MagicMock()
    # Return Doc A (0.82, relevant) and Doc B (0.58, irrelevant)
    mock_vs.search.return_value = [
        {
            "document_id": "doc_A",
            "chunk_id": "c0",
            "similarity_score": 0.82,
            "text": "Hypertension symptoms include severe headaches and dizziness.",
            "metadata": {"filename": "docA.pdf"}
        },
        {
            "document_id": "doc_B",
            "chunk_id": "c0",
            "similarity_score": 0.58,
            "text": "Clinic Administrative Protocol: Identification must be verified upon arrival.",
            "metadata": {"filename": "docB.pdf"}
        }
    ]

    rag = RAGService(vector_store=mock_vs)

    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        # 1. Test retrieve_context()
        ctx_results = rag.retrieve_context("What are the symptoms of hypertension?", top_k=5)
        assert len(ctx_results) == 1
        assert ctx_results[0]["document_id"] == "doc_A"

        # 2. Test query()
        query_results = rag.query("What are the symptoms of hypertension?", top_k=5)
        assert len(query_results["retrieved_chunks"]) == 1
        assert query_results["retrieved_chunks"][0]["document_id"] == "doc_A"
        assert len(query_results["sources"]) == 1
        assert query_results["sources"][0]["source_label"] == "[Source 1]"


def test_j_existing_source_numbering_remains_correct():
    """Test J: Survived sources maintain sequential [Source 1], [Source 2] numbering."""
    rag = RAGService()
    chunks = [
        {"document_id": "doc_A", "chunk_id": "c0", "similarity_score": 0.85, "text": "Hypertension text", "metadata": {"filename": "docA.pdf"}},
        {"document_id": "doc_B", "chunk_id": "c0", "similarity_score": 0.78, "text": "Diet text", "metadata": {"filename": "docB.pdf"}}
    ]

    sources = rag.build_sources(chunks)
    assert len(sources) == 2
    assert sources[0]["source_label"] == "[Source 1]"
    assert sources[0]["source_index"] == 1
    assert sources[1]["source_label"] == "[Source 2]"
    assert sources[1]["source_index"] == 2


def test_k_existing_source_card_citation_alignment():
    """Test K: Citation alignment and evidence card pruning work with precision-filtered sources."""
    mock_vs = MagicMock()
    mock_vs.search.return_value = [
        {"document_id": "doc_A", "chunk_id": "c0", "similarity_score": 0.85, "text": "Hypertension symptoms include headaches.", "metadata": {"filename": "docA.pdf"}},
        {"document_id": "doc_B", "chunk_id": "c0", "similarity_score": 0.78, "text": "Blood pressure dietary sodium restrictions.", "metadata": {"filename": "docB.pdf"}},
    ]

    rag = RAGService(vector_store=mock_vs)
    mock_gemini = MagicMock(spec=GeminiService)
    # Gemini only cites Source 1
    mock_gemini.generate_answer.return_value = {
        "answer": "Hypertension symptoms include headaches [Source 1].",
        "generation_time_ms": 100.0,
        "api_request_time_ms": 100.0,
        "gemini_calls_count": 1,
        "status": "success",
        "model": "gemini-3.5-flash-lite"
    }

    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        res = rag.generate_rag_answer("What are the symptoms of hypertension?", gemini_service=mock_gemini)

    assert res["retrieval_status"] == "success"
    assert len(res["sources"]) == 1
    assert res["sources"][0]["source_label"] == "[Source 1]"
    assert res["sources"][0]["document_id"] == "doc_A"


def test_l_candidate_pool_expansion_remains_max_k4_20():
    """Test L: Candidate retrieval pool requests max(k * 4, 20) candidates."""
    mock_vs = MagicMock()
    mock_vs.search.return_value = []
    rag = RAGService(vector_store=mock_vs, default_top_k=5)

    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        rag.retrieve_context("Hypertension query", top_k=5)

    assert mock_vs.search.called
    call_args = mock_vs.search.call_args
    # k = 5, candidate_k = max(5 * 4, 20) = 20
    assert call_args.kwargs.get("top_k") == 20 or call_args[1].get("top_k") == 20
