"""
Unit tests for RAG Chunk Deduplication, Source Numbering, and Granular Latency Metrics.
"""

import pytest
from unittest.mock import MagicMock
from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiService


def test_deduplicate_chunks_preserves_highest_similarity():
    """Verify that duplicate chunks for same (document_id, chunk_id) are deduplicated keeping max score."""
    chunks = [
        {
            "document_id": "DOC_HTN_01",
            "chunk_id": "CHUNK_0",
            "similarity_score": 0.65,
            "text": "First lower-score instance of chunk 0",
            "page_number": 1
        },
        {
            "document_id": "DOC_HTN_01",
            "chunk_id": "CHUNK_0",
            "similarity_score": 0.88,
            "text": "Second higher-score instance of chunk 0",
            "page_number": 1
        },
        {
            "document_id": "DOC_HTN_01",
            "chunk_id": "CHUNK_1",
            "similarity_score": 0.72,
            "text": "Distinct chunk 1 from same document",
            "page_number": 2
        }
    ]

    deduped = RAGService.deduplicate_chunks(chunks)

    # Should have exactly 2 unique chunks
    assert len(deduped) == 2
    # First chunk should be CHUNK_0 with the higher similarity score 0.88
    assert deduped[0]["chunk_id"] == "CHUNK_0"
    assert deduped[0]["similarity_score"] == 0.88
    assert deduped[0]["text"] == "Second higher-score instance of chunk 0"

    # Second chunk should be CHUNK_1
    assert deduped[1]["chunk_id"] == "CHUNK_1"
    assert deduped[1]["similarity_score"] == 0.72


def test_build_sources_correct_indexing_and_mapping():
    """Verify that build_sources assigns sequential source_index and [Source X] labels."""
    rag = RAGService()
    chunks = [
        {
            "document_id": "DOC_HTN_01",
            "chunk_id": "CHUNK_0",
            "similarity_score": 0.88,
            "text": "Hypertension definition chunk",
            "page_number": 1,
            "metadata": {"filename": "hypertension_guide.pdf"}
        },
        {
            "document_id": "DOC_HTN_01",
            "chunk_id": "CHUNK_1",
            "similarity_score": 0.75,
            "text": "Lifestyle changes chunk",
            "page_number": 2,
            "metadata": {"filename": "hypertension_guide.pdf"}
        }
    ]

    sources = rag.build_sources(chunks)
    assert len(sources) == 2

    assert sources[0]["source_index"] == 1
    assert sources[0]["source_label"] == "[Source 1]"
    assert sources[0]["filename"] == "hypertension_guide.pdf"
    assert sources[0]["page_number"] == 1
    assert sources[0]["chunk_id"] == "CHUNK_0"

    assert sources[1]["source_index"] == 2
    assert sources[1]["source_label"] == "[Source 2]"
    assert sources[1]["page_number"] == 2
    assert sources[1]["chunk_id"] == "CHUNK_1"


def test_generate_rag_answer_returns_complete_latency_breakdown():
    """Verify that generate_rag_answer returns granular latency breakdown for each stage."""
    mock_vector_store = MagicMock()
    # Return duplicate raw results to test end-to-end deduplication in query pipeline
    mock_vector_store.search.return_value = [
        {
            "document_id": "DOC_1",
            "chunk_id": "CHUNK_0",
            "similarity_score": 0.85,
            "text": "Clinical context for hypertension.",
            "page_number": 1
        },
        {
            "document_id": "DOC_1",
            "chunk_id": "CHUNK_0",
            "similarity_score": 0.70,
            "text": "Duplicate chunk with lower score.",
            "page_number": 1
        }
    ]

    rag = RAGService(vector_store=mock_vector_store)

    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.generate_answer.return_value = {
        "answer": "Hypertension is high blood pressure [Source 1].",
        "model": "gemini-3.5-flash-lite",
        "disclaimer": "Medical Disclaimer",
        "generation_time_ms": 120.5,
        "status": "success"
    }

    result = rag.generate_rag_answer(
        question="What is hypertension?",
        top_k=5,
        similarity_threshold=0.25,
        gemini_service=mock_gemini
    )

    assert result["retrieval_status"] == "success"
    # Verify deduplication reduced 2 chunks to 1 unique chunk
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_label"] == "[Source 1]"
    assert result["sources"][0]["similarity_score"] == 0.85

    # Verify separate latency breakdown metrics
    timings = result["timings"]
    assert "embedding_time_ms" in timings
    assert "faiss_retrieval_time_ms" in timings
    assert "deduplication_time_ms" in timings
    assert "context_construction_time_ms" in timings
    assert "llm_generation_time_ms" in timings
    assert "total_time_ms" in timings

    assert timings["llm_generation_time_ms"] == 120.5
    assert timings["total_time_ms"] >= timings["llm_generation_time_ms"]
