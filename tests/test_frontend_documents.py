"""
Unit and integration tests for Streamlit Frontend Document Ingestion Integration (Phase 13, Step 2).

Verifies all 14 required specifications:
1. PDF upload request is constructed correctly.
2. Bearer token is attached.
3. Multipart file is sent correctly.
4. Successful upload response is parsed.
5. 400 error handled.
6. 401 error handled.
7. 403 error handled.
8. 413 error handled.
9. 500 error handled.
10. Connection failure handled.
11. JWT is never returned/displayed by API client.
12. user_id is not required from frontend.
13. Document list uses authenticated API client.
14. Successful upload triggers document refresh behavior.
"""

import io
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import requests

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from frontend.api_client import APIClient


@pytest.fixture
def client():
    """Returns an APIClient instance pointed to a test backend."""
    return APIClient(base_url="http://127.0.0.1:8000")


# ==============================================================================
# 1. PDF upload request is constructed correctly
# ==============================================================================
def test_pdf_upload_request_constructed_correctly(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "document_id": 101,
            "filename": "clinical_trial.pdf",
            "status": "completed",
            "pages": 4,
            "num_chunks": 12,
            "user_id": 5,
        }
        mock_post.return_value = mock_res

        res = client.upload_document(b"%PDF-1.4 test content", "clinical_trial.pdf")
        assert res["success"] is True

        # Verify endpoint URL and HTTP method
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://127.0.0.1:8000/documents/upload"
        assert "files" in kwargs
        assert "file" in kwargs["files"]


# ==============================================================================
# 2. Bearer token is attached
# ==============================================================================
def test_bearer_token_attached(client):
    client.set_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test_token")

    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {"document_id": 1, "filename": "doc.pdf"}
        mock_post.return_value = mock_res

        client.upload_document(b"%PDF-1.4...", "doc.pdf")
        _, kwargs = mock_post.call_args
        headers = kwargs.get("headers", {})
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test_token"


# ==============================================================================
# 3. Multipart file is sent correctly
# ==============================================================================
def test_multipart_file_sent_correctly(client):
    class MockStreamlitFile:
        def __init__(self, name: str, data: bytes):
            self.name = name
            self._data = data

        def getvalue(self) -> bytes:
            return self._data

    st_file = MockStreamlitFile("cardiology_guideline.pdf", b"%PDF-1.4 raw binary data")

    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {"document_id": 2, "filename": "cardiology_guideline.pdf"}
        mock_post.return_value = mock_res

        res = client.upload_document(st_file)
        assert res["success"] is True

        _, kwargs = mock_post.call_args
        files_sent = kwargs["files"]
        filename_sent, bytes_sent, mime_sent = files_sent["file"]
        assert filename_sent == "cardiology_guideline.pdf"
        assert bytes_sent == b"%PDF-1.4 raw binary data"
        assert mime_sent == "application/pdf"


# ==============================================================================
# 4. Successful upload response is parsed
# ==============================================================================
def test_successful_upload_response_parsed(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "document_id": 55,
            "filename": "oncology_protocol.pdf",
            "status": "completed",
            "pages": 8,
            "number_of_pages": 8,
            "num_chunks": 24,
            "extracted_text_length": 5400,
            "extracted_text": "Oncology clinical trial protocol summary.",
            "user_id": 12,
        }
        mock_post.return_value = mock_res

        res = client.upload_document(b"%PDF-1.4", "oncology_protocol.pdf")
        assert res["success"] is True
        assert res["error"] is None
        data = res["data"]
        assert data["document_id"] == 55
        assert data["filename"] == "oncology_protocol.pdf"
        assert data["status"] == "completed"
        assert data["pages"] == 8
        assert data["num_chunks"] == 24


# ==============================================================================
# 5. 400 error handled
# ==============================================================================
def test_400_error_handled(client):
    # Case A: Backend returns specific 400 detail
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 400
        mock_res.json.return_value = {"detail": "No extractable text found in PDF 'scanned.pdf'."}
        mock_post.return_value = mock_res

        res = client.upload_document(b"%PDF-1.4 empty", "scanned.pdf")
        assert res["success"] is False
        assert res["status_code"] == 400
        assert "No extractable text found" in res["error"]

    # Case B: Generic 400 returns friendly readable text message
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 400
        mock_res.json.return_value = {}
        mock_post.return_value = mock_res

        res = client.upload_document(b"%PDF-1.4 empty", "scanned.pdf")
        assert res["success"] is False
        assert res["status_code"] == 400
        assert res["error"] == "Unable to process this PDF. Please check that it contains readable text."


# ==============================================================================
# 6. 401 error handled
# ==============================================================================
def test_401_error_handled(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 401
        mock_res.json.return_value = {"detail": "Could not validate credentials."}
        mock_post.return_value = mock_res

        res = client.upload_document(b"%PDF-1.4", "doc.pdf")
        assert res["success"] is False
        assert res["status_code"] == 401
        assert res["error"] == "Your session has expired. Please sign in again."


# ==============================================================================
# 7. 403 error handled
# ==============================================================================
def test_403_error_handled(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 403
        mock_res.json.return_value = {"detail": "Access denied: Cannot assign document to another user."}
        mock_post.return_value = mock_res

        res = client.upload_document(b"%PDF-1.4", "doc.pdf")
        assert res["success"] is False
        assert res["status_code"] == 403
        assert res["error"] == "You do not have permission to upload this document."


# ==============================================================================
# 8. 413 error handled
# ==============================================================================
def test_413_error_handled(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 413
        mock_res.json.return_value = {"detail": "Payload Too Large"}
        mock_post.return_value = mock_res

        res = client.upload_document(b"%PDF-1.4 large", "large.pdf")
        assert res["success"] is False
        assert res["status_code"] == 413
        assert res["error"] == "The PDF is too large. Maximum size is 10 MB."


# ==============================================================================
# 9. 500 error handled
# ==============================================================================
def test_500_error_handled(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 500
        mock_res.json.return_value = {"detail": "Internal processing failure."}
        mock_post.return_value = mock_res

        res = client.upload_document(b"%PDF-1.4", "doc.pdf")
        assert res["success"] is False
        assert res["status_code"] == 500
        assert res["error"] == "Document processing failed. Please try again."


# ==============================================================================
# 10. Connection failure handled
# ==============================================================================
def test_connection_failure_handled(client):
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        res = client.upload_document(b"%PDF-1.4", "doc.pdf")
        assert res["success"] is False
        assert res["status_code"] is None
        assert res["error"] == "Backend is unavailable. Please make sure the FastAPI server is running."


# ==============================================================================
# 11. JWT is never returned/displayed by API client
# ==============================================================================
def test_jwt_never_returned_or_displayed(client):
    client.set_token("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.sensitive_jwt_token")

    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "document_id": 88,
            "filename": "confidential_health_record.pdf",
            "status": "completed",
            "pages": 2,
            "num_chunks": 6,
            "user_id": 9,
        }
        mock_post.return_value = mock_res

        res = client.upload_document(b"%PDF-1.4", "confidential_health_record.pdf")
        assert res["success"] is True

        # Ensure no token or sensitive authorization headers are returned in result
        serialized_str = str(res)
        assert "sensitive_jwt_token" not in serialized_str
        assert "eyJ" not in serialized_str
        assert "Authorization" not in res


# ==============================================================================
# 12. user_id is not required from frontend
# ==============================================================================
def test_user_id_not_required_from_frontend(client):
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {"document_id": 99, "filename": "doc.pdf"}
        mock_post.return_value = mock_res

        # Call upload without specifying any user_id
        res = client.upload_document(b"%PDF-1.4", "doc.pdf")
        assert res["success"] is True

        _, kwargs = mock_post.call_args
        # Neither files nor data should include a client-injected user_id
        assert "user_id" not in kwargs.get("data", {})


# ==============================================================================
# 13. Document list uses authenticated API client
# ==============================================================================
def test_document_list_uses_authenticated_api_client(client):
    client.set_token("token_user_42")

    with patch("requests.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = [
            {"id": 1, "filename": "lab_results.pdf", "num_pages": 3, "num_chunks": 9, "status": "completed"}
        ]
        mock_get.return_value = mock_res

        docs = client.list_documents(limit=50)
        assert len(docs) == 1
        assert docs[0]["filename"] == "lab_results.pdf"

        # Verify Bearer token attached and user_id omitted
        args, kwargs = mock_get.call_args
        assert args[0] == "http://127.0.0.1:8000/documents"
        assert kwargs["headers"]["Authorization"] == "Bearer token_user_42"
        assert "user_id" not in kwargs["params"]


# ==============================================================================
# 14. Successful upload triggers document refresh behavior
# ==============================================================================
def test_successful_upload_triggers_document_refresh(client):
    # Simulate an upload followed by document list retrieval
    with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
        upload_resp = MagicMock()
        upload_resp.status_code = 200
        upload_resp.json.return_value = {
            "document_id": 105,
            "filename": "new_radiology_report.pdf",
            "status": "completed",
            "pages": 3,
            "num_chunks": 10,
            "user_id": 15,
        }
        mock_post.return_value = upload_resp

        # Ingestion call
        upload_res = client.upload_document(b"%PDF-1.4", "new_radiology_report.pdf")
        assert upload_res["success"] is True
        new_doc_id = upload_res["data"]["document_id"]
        assert new_doc_id == 105

        # Refreshed list call
        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.json.return_value = [
            {"id": 105, "filename": "new_radiology_report.pdf", "num_pages": 3, "num_chunks": 10, "status": "completed"},
            {"id": 100, "filename": "older_report.pdf", "num_pages": 1, "num_chunks": 3, "status": "completed"},
        ]
        mock_get.return_value = list_resp

        refreshed_docs = client.list_documents()
        assert any(d["id"] == new_doc_id for d in refreshed_docs)
