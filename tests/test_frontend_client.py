import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import requests
from frontend.api_client import APIClient
from frontend.utils.helpers import (
    format_similarity,
    clean_filename,
    format_bytes,
    format_timestamp,
    truncate_text,
)


def test_helpers():
    """Verify frontend formatting helpers function reliably and safely."""
    # 1. format_similarity must NEVER say 'medical confidence' or 'certainty'
    score_str = format_similarity(0.8523)
    assert "relevance" in score_str.lower()
    assert "confidence" not in score_str.lower()
    assert "certainty" not in score_str.lower()
    assert format_similarity(None) == "N/A"

    # 2. clean_filename
    assert clean_filename("C:\\data\\reports\\cardio.pdf") == "cardio.pdf"
    assert clean_filename("/var/docs/summary.pdf") == "summary.pdf"
    assert clean_filename(None) == "Unnamed Document"

    # 3. format_bytes
    assert format_bytes(500) == "500 B"
    assert format_bytes(2048) == "2.0 KB"
    assert format_bytes(1048576 * 2) == "2.00 MB"
    assert format_bytes(None) == "Unknown size"

    # 4. truncate_text
    short_text = "This is a short message."
    assert truncate_text(short_text, 50) == short_text
    long_text = "Word " * 50
    truncated = truncate_text(long_text, 30)
    assert len(truncated) <= 35
    assert truncated.endswith("...")

    # 5. format_timestamp
    assert "Unknown" in format_timestamp(None)
    formatted = format_timestamp("2026-09-13T20:30:00Z")
    assert "Sep 13, 2026" in formatted

    print("[PASS] test_helpers passed.")


def test_health_check_handling():
    """Verify health check handling for connected, disconnected, and timeout states."""
    client = APIClient(base_url="http://test-backend:8000")

    # 1. Success
    with patch("requests.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {"status": "healthy", "service": "AI-Healthcare-Agent"}
        mock_get.return_value = mock_res

        res = client.health_check()
        assert res["connected"] is True
        assert res["status"] == "healthy"
        assert res["error"] is None

    # 2. Connection Error
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        res = client.health_check()
        assert res["connected"] is False
        assert res["status"] == "disconnected"
        assert "Ensure FastAPI is running" in res["error"]

    # 3. Timeout Error
    with patch("requests.get", side_effect=requests.exceptions.Timeout("Read timeout")):
        res = client.health_check()
        assert res["connected"] is False
        assert res["status"] == "timeout"
        assert "timed out" in res["error"]

    print("[PASS] test_health_check_handling passed.")


def test_upload_document_handling():
    """Verify PDF upload handling for successful parsing, HTTP errors, and timeouts."""
    client = APIClient(base_url="http://test-backend:8000")

    # 1. Successful upload
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "filename": "clinical_trial.pdf",
            "number_of_pages": 3,
            "extracted_text_length": 1500,
            "extracted_text": "Sample clinical trial findings..."
        }
        mock_post.return_value = mock_res

        res = client.upload_document(b"%PDF-1.4 test", "clinical_trial.pdf")
        assert res["success"] is True
        assert res["data"]["number_of_pages"] == 3

    # 2. Rejection / 400 error
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 400
        mock_res.json.return_value = {"detail": "Invalid file type. Only PDF files are supported."}
        mock_post.return_value = mock_res

        res = client.upload_document(b"fake txt", "notes.txt")
        assert res["success"] is False
        assert "Only PDF files are supported" in res["error"]

    # 3. Timeout
    with patch("requests.post", side_effect=requests.exceptions.Timeout()):
        res = client.upload_document(b"%PDF", "large.pdf")
        assert res["success"] is False
        assert "timed out" in res["error"]

    print("[PASS] test_upload_document_handling passed.")


def test_list_and_get_documents():
    """Verify document list retrieval and detailed metadata fetching."""
    client = APIClient(base_url="http://test-backend:8000")

    # 1. List documents
    with patch("requests.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = [
            {"id": 1, "filename": "doc1.pdf", "num_pages": 4, "num_chunks": 12, "status": "processed"},
            {"id": 2, "filename": "doc2.pdf", "num_pages": 2, "num_chunks": 6, "status": "processed"}
        ]
        mock_get.return_value = mock_res

        docs = client.list_documents()
        assert len(docs) == 2
        assert docs[0]["filename"] == "doc1.pdf"

    # 2. Get document by ID
    with patch("requests.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {"id": 1, "filename": "doc1.pdf", "num_pages": 4}
        mock_get.return_value = mock_res

        doc = client.get_document(1)
        assert doc is not None
        assert doc["id"] == 1

    # 3. Get document 404
    with patch("requests.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 404
        mock_get.return_value = mock_res

        doc = client.get_document(999)
        assert doc is None

    print("[PASS] test_list_and_get_documents passed.")


def test_send_agent_query_safe_and_blocked():
    """Verify agent query workflow for both safe informational and blocked responses."""
    client = APIClient(base_url="http://test-backend:8000")

    # 1. Safe Informational Response
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "question": "What is hypertension?",
            "intent": "MEDICAL_EXPLANATION",
            "agent": "explanation_agent",
            "safety": {"allowed": True, "category": "SAFE_INFORMATIONAL"},
            "answer": "Hypertension is high blood pressure [Source 1].",
            "sources": [
                {
                    "source_index": 1,
                    "chunk_id": "MED_CHUNK_0",
                    "document_id": "cardio.pdf",
                    "page_number": 1,
                    "similarity_score": 0.88,
                    "preview_text": "Hypertension is defined as blood pressure above 130/80 mmHg."
                }
            ],
            "citations": {"is_valid": True, "valid_citations": [1]},
            "timings": {"safety_check_time_ms": 0.12, "total_time_ms": 18.5},
            "disclaimer": "This tool provides informational support."
        }
        mock_post.return_value = mock_res

        res = client.send_agent_query("What is hypertension?", explanation_level="simple")
        assert res["success"] is True
        data = res["data"]
        assert data["intent"] == "MEDICAL_EXPLANATION"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["similarity_score"] == 0.88

    # 2. Blocked Emergency Response
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "question": "I have sudden severe chest pain and left arm numbness",
            "intent": "SAFETY_SENSITIVE",
            "agent": "safety_agent",
            "safety": {
                "allowed": False,
                "category": "EMERGENCY_SYMPTOMS",
                "requires_emergency_guidance": True
            },
            "answer": "EMERGENCY WARNING: Please call 911 or go to the nearest emergency room immediately.",
            "sources": [],
            "citations": {"is_valid": True},
            "timings": {"safety_check_time_ms": 0.15, "total_time_ms": 0.22},
            "disclaimer": "Emergency disclaimer."
        }
        mock_post.return_value = mock_res

        res = client.send_agent_query("I have severe chest pain")
        assert res["success"] is True
        data = res["data"]
        assert data["safety"]["allowed"] is False
        assert data["safety"]["requires_emergency_guidance"] is True
        assert "EMERGENCY" in data["answer"]

    print("[PASS] test_send_agent_query_safe_and_blocked passed.")


def test_conversations_and_messages():
    """Verify conversation creation, listing, and message persistence via APIClient."""
    client = APIClient(base_url="http://test-backend:8000")

    # 1. Create conversation
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 201
        mock_res.json.return_value = {
            "id": 42,
            "title": "Diabetes Consultation",
            "created_at": "2026-09-13T21:00:00Z",
            "updated_at": "2026-09-13T21:00:00Z"
        }
        mock_post.return_value = mock_res

        conv_res = client.create_conversation("Diabetes Consultation")
        assert conv_res["success"] is True
        assert conv_res["data"]["id"] == 42

    # 2. List conversations
    with patch("requests.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = [{"id": 42, "title": "Diabetes Consultation"}]
        mock_get.return_value = mock_res

        convs = client.list_conversations()
        assert len(convs) == 1
        assert convs[0]["id"] == 42

    # 3. Append message
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 201
        mock_res.json.return_value = {
            "id": 101,
            "conversation_id": 42,
            "sender": "user",
            "text": "What is HbA1c?",
            "created_at": "2026-09-13T21:01:00Z"
        }
        mock_post.return_value = mock_res

        msg_res = client.append_conversation_message(42, "user", "What is HbA1c?")
        assert msg_res["success"] is True
        assert msg_res["data"]["id"] == 101

    # 4. Get conversation messages
    with patch("requests.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = [
            {"id": 101, "sender": "user", "text": "What is HbA1c?"},
            {"id": 102, "sender": "assistant", "text": "HbA1c measures average blood sugar."}
        ]
        mock_get.return_value = mock_res

        messages = client.get_conversation_messages(42)
        assert len(messages) == 2
        assert messages[0]["sender"] == "user"
        assert messages[1]["sender"] == "assistant"

    print("[PASS] test_conversations_and_messages passed.")


if __name__ == "__main__":
    test_helpers()
    test_health_check_handling()
    test_upload_document_handling()
    test_list_and_get_documents()
    test_send_agent_query_safe_and_blocked()
    test_conversations_and_messages()
    print("[SUCCESS] All 6 frontend client & UI component unit tests passed successfully!")
