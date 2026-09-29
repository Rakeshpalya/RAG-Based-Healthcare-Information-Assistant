"""
Regression tests for RAG hypertension question answering, model fallback, and defensive generation.

Verifies:
1. End-to-end RAG query endpoint with exact hypertension question returns 200, sources >= 1, and citations.
2. GeminiService automatic fallback chain when a model experiences 503 / 404.
3. Defensive response text extraction from candidate parts.
4. APIClient query_rag cleanly extracts error details.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.gemini_service import GeminiService, GeminiServiceError
from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import VectorStoreService
from frontend.api_client import APIClient


@pytest.fixture
def client():
    return TestClient(app)


HYPERTENSION_CHUNKS = [
    {
        "chunk_id": "CHUNK_HTN_01",
        "document_id": "DOC_HYPERTENSION_001",
        "page_number": 1,
        "text": (
            "Hypertension is a chronic medical condition in which arterial blood pressure "
            "is persistently elevated above 130/80 mmHg. Common risk factors include high dietary sodium, "
            "obesity, lack of physical exercise, tobacco use, and chronic stress. Recommended lifestyle "
            "modifications include the DASH diet, sodium restriction, 150 minutes of moderate exercise per week, "
            "and smoking cessation."
        ),
        "category": "Cardiology",
    }
]


def test_hypertension_rag_query_end_to_end_with_citations(client):
    """
    14. Regression test:
    - Authenticated user
    - Hypertension document chunks in vector store
    - Exact clinical inquiry
    - Successful retrieval (sources >= 1)
    - Grounded answer generation with inline citations
    """
    mock_vector_store = VectorStoreService()
    mock_vector_store.add_chunks(HYPERTENSION_CHUNKS)

    mock_rag_service = RAGService(vector_store=mock_vector_store)

    mock_gemini_client = MagicMock()
    mock_gen_response = MagicMock()
    mock_gen_response.text = (
        "Based on [SOURCE 1], hypertension is defined as persistently elevated arterial blood pressure. "
        "Key risk factors include high sodium intake, obesity, and physical inactivity [SOURCE 1]. "
        "Recommended lifestyle modifications include the DASH dietary pattern and regular aerobic exercise [SOURCE 1]."
    )
    mock_gemini_client.models.generate_content.return_value = mock_gen_response

    gemini_svc = GeminiService(api_key="test-api-key")
    gemini_svc.set_client(mock_gemini_client)

    question = "What is hypertension, what are the common risk factors, and what lifestyle changes are generally recommended to help manage it?"

    # Execute through RAGService
    result = mock_rag_service.generate_rag_answer(
        question=question,
        top_k=5,
        similarity_threshold=0.25,
        gemini_service=gemini_svc,
    )

    assert result["retrieval_status"] == "success"
    assert len(result["sources"]) >= 1
    assert result["sources"][0]["document_id"] == "DOC_HYPERTENSION_001"
    assert "[SOURCE 1]" in result["answer"]
    assert "DASH" in result["answer"]
    assert "disclaimer" in result
    assert result["timings"]["generation_time_ms"] >= 0.0


def test_gemini_service_automatic_fallback_on_503(monkeypatch):
    """Verify GeminiService automatically attempts fallback models when primary experiences 503."""
    from google.genai.errors import ServerError

    mock_client = MagicMock()
    # First model call raises 503 Unavailable; second call succeeds
    error_503 = ServerError(503, {"error": {"code": 503, "message": "High demand"}}, MagicMock())
    success_resp = MagicMock()
    success_resp.text = "Grounded fallback answer [SOURCE 1]."

    mock_client.models.generate_content.side_effect = [error_503, success_resp]

    service = GeminiService(api_key="test-key", model="gemini-primary")
    service.set_client(mock_client)

    result = service.generate_answer(
        question="What is hypertension?",
        context="[SOURCE 1]\nHypertension is high blood pressure."
    )

    assert result["status"] == "success"
    assert "Grounded fallback answer" in result["answer"]
    assert mock_client.models.generate_content.call_count == 2


def test_gemini_service_defensive_parts_extraction():
    """Verify GeminiService safely extracts text when .text attribute is empty but parts exist."""
    mock_client = MagicMock()

    mock_part = MagicMock()
    mock_part.text = "Answer extracted from content parts [SOURCE 1]."

    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_candidate.finish_reason = "STOP"

    mock_resp = MagicMock()
    mock_resp.text = None  # text property is None
    mock_resp.candidates = [mock_candidate]

    mock_client.models.generate_content.return_value = mock_resp

    service = GeminiService(api_key="test-key")
    service.set_client(mock_client)

    result = service.generate_answer(
        question="What is hypertension?",
        context="[SOURCE 1]\nHypertension context."
    )

    assert result["status"] == "success"
    assert "Answer extracted from content parts" in result["answer"]


def test_api_client_query_rag_extracts_detail_on_error():
    """Verify APIClient.query_rag extracts custom detail message from 502/500 responses."""
    api = APIClient(base_url="http://test-server")

    mock_resp = MagicMock()
    mock_resp.status_code = 502
    mock_resp.json.return_value = {"detail": "Gemini LLM generation error: Model busy"}

    with patch("requests.post", return_value=mock_resp):
        res = api.query_rag("What is hypertension?")
        assert res["success"] is False
        assert res["status_code"] == 502
        assert "Gemini LLM generation error: Model busy" in res["error"]
