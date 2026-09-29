import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.main import app
from backend.api.agent_router import execute_agent_workflow
from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiService


client = TestClient(app)


def test_agent_router_successful_query():
    """22. Test successful query dispatch through POST /agent/query."""
    mock_rag = MagicMock(spec=RAGService)
    mock_rag.query.return_value = {
        "question": "What is hypertension?",
        "retrieval_status": "success",
        "context": "[SOURCE 1]\nDocument: DOC_CARDIO_001\nChunk ID: CHUNK_0\n\nHypertension is blood pressure above 130/80 mmHg.",
        "retrieved_chunks": [{"chunk_id": "CHUNK_0", "similarity_score": 0.85}],
        "sources": [{"source_index": 1, "chunk_id": "CHUNK_0", "document_id": "DOC_CARDIO_001"}],
        "timings": {"total_retrieval_time_ms": 15.0}
    }

    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.generate_answer.return_value = {
        "answer": "According to medical evidence [Source 1], hypertension is defined as blood pressure above 130/80 mmHg.",
        "model": "gemini-2.5-flash",
        "timings": {"generation_time_ms": 110.0}
    }

    resp = execute_agent_workflow(
        question="What is hypertension?",
        explanation_level="simple",
        top_k=5,
        rag_service=mock_rag,
        gemini_service=mock_gemini
    )

    assert resp.intent in ("RESEARCH_QUESTION", "GENERAL_HEALTH_INFORMATION")
    assert resp.agent == "research_agent"
    assert resp.safety["allowed"] is True
    assert "blood pressure above 130/80" in resp.answer
    assert len(resp.sources) == 1
    assert resp.citations["is_valid"] is True
    assert "total_time_ms" in resp.timings
    print("[PASS] test_agent_router_successful_query passed.")


def test_agent_router_safety_blocked_query():
    """23. Test that dangerous / emergency inquiries are blocked at the router layer with zero RAG/LLM latency."""
    # Emergency symptom inquiry
    emergency_resp = execute_agent_workflow(
        question="I have crushing chest pain and cannot breathe, what should I do?"
    )
    assert emergency_resp.intent == "SAFETY_SENSITIVE"
    assert emergency_resp.agent == "safety_agent"
    assert emergency_resp.safety["allowed"] is False
    assert emergency_resp.safety["category"] == "EMERGENCY_SYMPTOM"
    assert "EMERGENCY ADVISORY" in emergency_resp.answer
    assert emergency_resp.sources == []

    # Diagnostic inquiry
    diag_resp = execute_agent_workflow(
        question="Do I have diabetes based on feeling thirsty?"
    )
    assert diag_resp.intent == "SAFETY_SENSITIVE"
    assert diag_resp.agent == "safety_agent"
    assert diag_resp.safety["allowed"] is False
    assert diag_resp.safety["category"] == "DIAGNOSIS_REQUEST"
    assert "cannot provide personal medical diagnoses" in diag_resp.answer
    print("[PASS] test_agent_router_safety_blocked_query passed.")


def test_api_endpoint_post_agent_query():
    """24. Test HTTP POST /agent/query endpoint integration via TestClient."""
    # 1. Test emergency inquiry blocked via HTTP
    response = client.post(
        "/agent/query",
        json={"question": "I am having severe chest pain right now"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "SAFETY_SENSITIVE"
    assert data["safety"]["allowed"] is False
    assert "EMERGENCY ADVISORY" in data["answer"]

    # 2. Test empty query handled cleanly
    response_empty = client.post(
        "/agent/query",
        json={"question": "   "}
    )
    assert response_empty.status_code == 200
    data_empty = response_empty.json()
    assert data_empty["safety"]["allowed"] is False
    assert data_empty["safety"]["category"] == "EMPTY_QUERY"
    print("[PASS] test_api_endpoint_post_agent_query passed.")


if __name__ == "__main__":
    print("Running Agent Router Unit Tests...")
    test_agent_router_successful_query()
    test_agent_router_safety_blocked_query()
    test_api_endpoint_post_agent_query()
    print("\n[SUCCESS] All 3 Agent Router unit tests passed successfully!")
