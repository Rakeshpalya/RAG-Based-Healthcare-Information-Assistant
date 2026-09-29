"""
Phase 2B: Multi-Document Evidence Quality Regression Tests.

Verifies:
A. Three different documents: 3 different PDFs with relevant chunks -> all 3 represented.
B. Single document: One document with 5 relevant chunks -> up to 5 chunks remain.
C. Three identical PDFs: Phase 2A collapses them to 1 evidence source.
D. Two identical + one different: Exactly 2 unique evidence sources.
E. Uneven distribution: Document A = 5 chunks, Document B = 1 chunk -> B not starved.
F. Candidate pool expansion: Retrieval requests up to 20 candidates for k=5.
"""

import pytest
from unittest.mock import MagicMock, patch
from backend.rag.rag_service import RAGService


def test_three_different_documents_all_represented():
    """
    Test A: Three different documents with relevant chunks.
    All 3 documents must be represented in top-5 evidence.
    """
    chunks = [
        {"document_id": "doc_cardio", "chunk_id": "c0", "similarity_score": 0.88, "text": "Cardio guideline intro.", "metadata": {"filename": "cardio.pdf"}},
        {"document_id": "doc_cardio", "chunk_id": "c1", "similarity_score": 0.86, "text": "Cardio guideline details.", "metadata": {"filename": "cardio.pdf"}},
        {"document_id": "doc_cardio", "chunk_id": "c2", "similarity_score": 0.84, "text": "Cardio guideline stage 2.", "metadata": {"filename": "cardio.pdf"}},
        {"document_id": "doc_cardio", "chunk_id": "c3", "similarity_score": 0.82, "text": "Cardio guideline follow-up.", "metadata": {"filename": "cardio.pdf"}},
        {"document_id": "doc_cardio", "chunk_id": "c4", "similarity_score": 0.80, "text": "Cardio guideline summary.", "metadata": {"filename": "cardio.pdf"}},
        {"document_id": "doc_diet", "chunk_id": "c0", "similarity_score": 0.78, "text": "DASH diet reduces blood pressure.", "metadata": {"filename": "diet.pdf"}},
        {"document_id": "doc_exercise", "chunk_id": "c0", "similarity_score": 0.77, "text": "Aerobic physical activity recommendations.", "metadata": {"filename": "exercise.pdf"}},
    ]

    selected = RAGService.select_diverse_evidence(chunks, top_k=5, max_per_doc=2)

    assert len(selected) == 5
    unique_docs = set(c["document_id"] for c in selected)
    assert unique_docs == {"doc_cardio", "doc_diet", "doc_exercise"}, f"Expected all 3 docs represented, got {unique_docs}"

    # Verify doc_cardio has 3 chunks (2 from pass 1, 1 backfilled in pass 2)
    cardio_count = sum(1 for c in selected if c["document_id"] == "doc_cardio")
    assert cardio_count == 3
    # Verify diet and exercise have 1 chunk each
    assert sum(1 for c in selected if c["document_id"] == "doc_diet") == 1
    assert sum(1 for c in selected if c["document_id"] == "doc_exercise") == 1


def test_single_document_preserves_all_slots():
    """
    Test B: Single document with 5 relevant chunks.
    All 5 chunks must remain in the top-5 evidence set via backfill.
    """
    chunks = [
        {"document_id": "doc_single", "chunk_id": "c0", "similarity_score": 0.90, "text": "Chunk 0"},
        {"document_id": "doc_single", "chunk_id": "c1", "similarity_score": 0.85, "text": "Chunk 1"},
        {"document_id": "doc_single", "chunk_id": "c2", "similarity_score": 0.80, "text": "Chunk 2"},
        {"document_id": "doc_single", "chunk_id": "c3", "similarity_score": 0.75, "text": "Chunk 3"},
        {"document_id": "doc_single", "chunk_id": "c4", "similarity_score": 0.70, "text": "Chunk 4"},
    ]

    selected = RAGService.select_diverse_evidence(chunks, top_k=5, max_per_doc=2)

    assert len(selected) == 5
    assert [c["chunk_id"] for c in selected] == ["c0", "c1", "c2", "c3", "c4"]
    assert [c["similarity_score"] for c in selected] == [0.90, 0.85, 0.80, 0.75, 0.70]


def test_three_identical_pdfs_collapses_to_one_evidence_source():
    """
    Test C: Three identical copies of the same PDF.
    Phase 2A deduplication must collapse them to 1 unique evidence source,
    and diverse evidence selection must retain that 1 source.
    """
    shared_text = "HealthAI RAG Test Document for testing grounded question answering."
    chunks = [
        {"document_id": "18", "chunk_id": "c0", "similarity_score": 0.89, "text": shared_text, "metadata": {"filename": "test.pdf"}},
        {"document_id": "17", "chunk_id": "c0", "similarity_score": 0.88, "text": shared_text, "metadata": {"filename": "test.pdf"}},
        {"document_id": "16", "chunk_id": "c0", "similarity_score": 0.87, "text": shared_text, "metadata": {"filename": "test.pdf"}},
    ]

    deduped = RAGService.deduplicate_chunks(chunks)
    assert len(deduped) == 1

    selected = RAGService.select_diverse_evidence(deduped, top_k=5, max_per_doc=2)
    assert len(selected) == 1
    assert selected[0]["document_id"] == "18"
    assert selected[0]["similarity_score"] == 0.89

    rag = RAGService(vector_store=MagicMock())
    sources = rag.build_sources(selected)
    assert len(sources) == 1
    assert sources[0]["source_label"] == "[Source 1]"


def test_two_identical_plus_one_different_yields_two_sources():
    """
    Test D: Two identical PDFs + one different PDF.
    Must produce exactly 2 unique evidence sources.
    """
    shared_text = "Cardiovascular disease management guidelines."
    distinct_text = "Chronic kidney disease diagnostic staging."

    chunks = [
        {"document_id": "doc_cvd_1", "chunk_id": "c0", "similarity_score": 0.81, "text": shared_text, "metadata": {"filename": "cvd.pdf"}},
        {"document_id": "doc_cvd_2", "chunk_id": "c0", "similarity_score": 0.89, "text": shared_text, "metadata": {"filename": "cvd.pdf"}},
        {"document_id": "doc_ckd", "chunk_id": "c0", "similarity_score": 0.76, "text": distinct_text, "metadata": {"filename": "ckd.pdf"}},
    ]

    deduped = RAGService.deduplicate_chunks(chunks)
    assert len(deduped) == 2

    selected = RAGService.select_diverse_evidence(deduped, top_k=5, max_per_doc=2)
    assert len(selected) == 2
    assert selected[0]["similarity_score"] == 0.89
    assert selected[1]["similarity_score"] == 0.76

    rag = RAGService(vector_store=MagicMock())
    sources = rag.build_sources(selected)
    assert len(sources) == 2
    assert sources[0]["source_label"] == "[Source 1]"
    assert sources[1]["source_label"] == "[Source 2]"


def test_uneven_distribution_document_not_starved():
    """
    Test E: Document A = 5 chunks, Document B = 1 chunk.
    Document B must not be starved during diversity pass.
    """
    chunks = [
        {"document_id": "doc_A", "chunk_id": "c0", "similarity_score": 0.92, "text": "A0"},
        {"document_id": "doc_A", "chunk_id": "c1", "similarity_score": 0.90, "text": "A1"},
        {"document_id": "doc_A", "chunk_id": "c2", "similarity_score": 0.88, "text": "A2"},
        {"document_id": "doc_A", "chunk_id": "c3", "similarity_score": 0.86, "text": "A3"},
        {"document_id": "doc_A", "chunk_id": "c4", "similarity_score": 0.84, "text": "A4"},
        {"document_id": "doc_B", "chunk_id": "c0", "similarity_score": 0.82, "text": "B0"},
    ]

    selected = RAGService.select_diverse_evidence(chunks, top_k=5, max_per_doc=2)

    assert len(selected) == 5
    unique_docs = set(c["document_id"] for c in selected)
    assert "doc_B" in unique_docs, "Document B should not be starved by Document A"
    assert sum(1 for c in selected if c["document_id"] == "doc_A") == 4
    assert sum(1 for c in selected if c["document_id"] == "doc_B") == 1


def test_candidate_pool_expansion():
    """
    Test F: Candidate pool expansion.
    Verify retrieval requests up to 20 candidates for k=5.
    """
    mock_vs = MagicMock()
    mock_vs.search.return_value = []
    rag = RAGService(vector_store=mock_vs, default_top_k=5)

    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        rag.retrieve_context("Test query for candidate k expansion", top_k=5)

    # Verify vector_store.search was called with top_k = max(5 * 4, 20) = 20
    assert mock_vs.search.called
    call_args = mock_vs.search.call_args
    assert call_args[1].get("top_k") == 20 or call_args[0][1] == 20
