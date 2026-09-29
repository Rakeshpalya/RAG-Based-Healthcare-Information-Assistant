import json
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiServiceError
from backend.services.vector_store_service import VectorStoreService

client = TestClient(app)


def test_rag_empty_retrieval_returns_safe_fallback():
    """Verify that when no relevant chunks match, pipeline returns safe fallback without LLM hallucination."""
    service = RAGService()
    res = service.generate_rag_answer(
        question="NonsensicalQueryUnrelatedToMedicineX99QZ",
        similarity_threshold=0.99  # Unobtainable threshold to simulate empty retrieval
    )
    assert res["retrieval_status"] == "no_relevant_context"
    assert "could not be found" in res["answer"].lower() or "no relevant" in res["answer"].lower()
    assert "medical disclaimer" in res["disclaimer"].lower() or len(res["disclaimer"]) > 10
    assert len(res["sources"]) == 0


def test_rag_gemini_service_error_handling():
    """Verify that when Gemini raises GeminiServiceError, pipeline catches it cleanly."""
    with patch("backend.rag.rag_service.RAGService.generate_rag_answer") as mock_rag:
        mock_rag.side_effect = GeminiServiceError("Gemini API connection timed out after 30s")
        
        # Test endpoint
        res = client.post("/rag/query", json={"question": "What is hypertension?"})
        # rag_router gracefully returns 200 with service_unavailable status and safe fallback message
        assert res.status_code == 200
        data = res.json()
        assert data["retrieval_status"] == "service_unavailable"
        assert "temporarily unavailable" in data["answer"].lower()
        # Credentials must not be in payload
        assert "AIza" not in str(data)
        assert "password" not in str(data).lower()


def test_rag_handles_malformed_llm_citation_output():
    """Verify that when LLM returns invalid citations, CitationValidator catches them safely."""
    mock_vs = MagicMock()
    mock_vs.search.return_value = [{
        "chunk_id": "HTN_01",
        "document_name": "Cardio.pdf",
        "document_id": "DOC_HTN_01",
        "page_number": 1,
        "score": 0.88,
        "text": "Hypertension guidelines emphasize dietary sodium restriction."
    }]
    service = RAGService(vector_store=mock_vs)
    
    mock_gemini = MagicMock()
    mock_gemini.generate_answer.return_value = {
        "answer": "Hypertension is high blood pressure [Source 99]. Treatment requires surgery [Source 100].",
        "model": "mock-gemini-3.5-flash",
        "disclaimer": "Medical disclaimer.",
        "generation_time_ms": 110.0,
        "status": "success"
    }
    
    res = service.generate_rag_answer(
        question="What is hypertension?",
        gemini_service=mock_gemini
    )
    # Invalid citations [Source 99] must be stripped or flagged
    assert "[Source 99]" not in res["answer"]


def test_vector_store_corrupted_metadata_safely_handled(tmp_path):
    """Verify that if a vector store directory has corrupted metadata, it handles gracefully."""
    bad_meta = tmp_path / "metadata.json"
    bad_meta.write_text("{invalid json", encoding="utf-8")
    
    # Vector store instantiation should not crash the entire process
    store = VectorStoreService(storage_dir=tmp_path)
    assert store.count() == 0 or len(store.metadata) == 0


def test_vector_store_read_only_invariant_preserved_after_failures():
    """Verify that after running failure tests, production FAISS index has exactly 744 vectors."""
    import faiss
    idx = faiss.read_index("data/vector_store/index.faiss")
    meta = json.load(open("data/vector_store/metadata.json", "r", encoding="utf-8"))
    assert idx.ntotal == 744
    assert len(meta["records"]) == 744
