"""
Integration Tests for POST /api/medical-chat Endpoint.

Verifies:
- HTTP 200 on valid medical questions
- Response schema: answer, session_id, status, agent
- Session ID generation and preservation
- HTTP 400 on empty / whitespace input
- Re-export compatibility in backend/routers/medical_chat_router.py
- Zero impact on existing RAG routes (/rag/query, /rag/retrieve)
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_medical_chat_endpoint_success():
    """Verifies successful medical chat request and response schema."""
    mock_response = {
        "answer": "Asthma is a chronic respiratory condition causing airway inflammation.",
        "session_id": "api-session-123",
        "status": "success",
        "agent": "HealthAI Medical Assistant",
    }

    with patch("backend.api.medical_chat_router.generate_medical_chat_response", return_value=mock_response):
        response = client.post(
            "/api/medical-chat",
            json={
                "message": "What is asthma?",
                "session_id": "api-session-123"
            }
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["session_id"] == "api-session-123"
    assert "Asthma" in data["answer"]
    assert data["agent"] == "HealthAI Medical Assistant"


def test_medical_chat_endpoint_auto_session_id():
    """Verifies that an omitted session_id is automatically generated."""
    mock_response = {
        "answer": "Type 1 diabetes is an autoimmune condition.",
        "session_id": "auto-generated-session-uuid",
        "status": "success",
        "agent": "HealthAI Medical Assistant",
    }

    with patch("backend.api.medical_chat_router.generate_medical_chat_response", return_value=mock_response):
        response = client.post(
            "/api/medical-chat",
            json={"message": "What is Type 1 diabetes?"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "auto-generated-session-uuid"
    assert data["status"] == "success"


def test_medical_chat_endpoint_empty_message():
    """Verifies that empty/whitespace message returns HTTP 400 Bad Request."""
    response = client.post(
        "/api/medical-chat",
        json={"message": "   "}
    )
    assert response.status_code == 400
    assert "cannot be empty" in response.json().get("detail", "").lower()


def test_medical_chat_router_reexport_compatibility():
    """Verifies that backend/routers/medical_chat_router.py re-exports router cleanly."""
    from backend.routers.medical_chat_router import router
    assert router is not None
    assert any(route.path == "/api/medical-chat" for route in router.routes)


def test_existing_rag_routes_unaffected():
    """Verifies that existing RAG routes (/rag/query, /rag/retrieve) remain registered and accessible."""
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    openapi_paths = list(app.openapi()["paths"].keys())
    assert "/rag/query" in openapi_paths
    assert "/rag/retrieve" in openapi_paths
    assert "/api/medical-chat" in openapi_paths


def test_api_client_send_medical_chat_success():
    """Verifies APIClient.send_medical_chat invokes requests.post correctly without session attribute error."""
    from frontend.api_client import APIClient
    client_instance = APIClient(base_url="http://127.0.0.1:8000")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "answer": "Hypertension is persistently elevated blood pressure.",
        "session_id": "sess-456",
        "status": "success",
        "agent": "HealthAI Medical Assistant",
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = client_instance.send_medical_chat("What is hypertension?", session_id="sess-456")

        assert res["success"] is True
        assert res["data"]["answer"] == "Hypertension is persistently elevated blood pressure."
        assert res["data"]["session_id"] == "sess-456"
        assert res["data"]["agent"] == "HealthAI Medical Assistant"
        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        assert call_args[0] == "http://127.0.0.1:8000/api/medical-chat"
        assert call_kwargs["json"] == {"message": "What is hypertension?", "session_id": "sess-456"}


def test_api_client_send_medical_chat_error_handling():
    """Verifies APIClient.send_medical_chat handles connection and timeout errors gracefully."""
    import requests
    from frontend.api_client import APIClient
    client_instance = APIClient(base_url="http://127.0.0.1:8000")

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
        res = client_instance.send_medical_chat("What is asthma?")
        assert res["success"] is False
        assert "unavailable" in res["error"].lower()

    with patch("requests.post", side_effect=requests.exceptions.Timeout):
        res = client_instance.send_medical_chat("What is asthma?")
        assert res["success"] is False
        assert "timed out" in res["error"].lower()

