"""
Unit and integration tests for frontend authentication flows (Phase 12, Step 7).

Verifies:
1. Login success with token storage
2. Login failure (invalid credentials)
3. Registration success (with and without email confirmation)
4. Registration failure (duplicate user / weak password)
5. Session state and token lifecycle
6. /auth/me profile retrieval
7. Logout and session destruction
8. Authorization Bearer header formatting and attachment
9. HTTP 401 Unauthorized handling (token invalidation)
10. HTTP 403 Forbidden handling
11. Unauthenticated requests isolation
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import requests
import pytest
from frontend.api_client import APIClient


@pytest.fixture
def client():
    """Creates a fresh APIClient for each test."""
    return APIClient(base_url="http://test-backend:8000")


def test_login_success(client):
    """Test successful user login stores access token and returns user details."""
    fake_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.dummy_test_token"
    fake_user = {"id": "user-uuid-1234", "email": "doctor@hospital.org", "role": "authenticated"}

    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "user": fake_user,
            "session": {
                "access_token": fake_token,
                "token_type": "bearer",
                "expires_in": 3600,
            }
        }
        mock_post.return_value = mock_res

        result = client.login("doctor@hospital.org", "Secr3tP@ssword!")

        assert result["success"] is True
        assert result["token"] == fake_token
        assert result["user"]["email"] == "doctor@hospital.org"
        assert client.token == fake_token

        # Verify Authorization header is correctly constructed
        headers = client.get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {fake_token}"

        # Verify API called with correct endpoint and payload
        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        call_json = mock_post.call_args[1]["json"]
        assert call_url == "http://test-backend:8000/auth/login"
        assert call_json["email"] == "doctor@hospital.org"


def test_login_failure_invalid_credentials(client):
    """Test login failure with invalid credentials returns clear error and does not set token."""
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 401
        mock_res.json.return_value = {"detail": "Invalid login credentials"}
        mock_post.return_value = mock_res

        result = client.login("doctor@hospital.org", "WrongPassword")

        assert result["success"] is False
        assert result["token"] is None
        assert result["status_code"] == 401
        assert "Invalid login credentials" in result["error"]
        assert client.token is None
        assert "Authorization" not in client.get_headers()


def test_login_connection_and_timeout_errors(client):
    """Test login handling when backend is unreachable or times out."""
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
        res = client.login("test@hospital.org", "pass123")
        assert res["success"] is False
        assert "Verify FastAPI is running" in res["error"]

    with patch("requests.post", side_effect=requests.exceptions.Timeout):
        res = client.login("test@hospital.org", "pass123")
        assert res["success"] is False
        assert "timed out" in res["error"]


def test_registration_success_direct_session(client):
    """Test successful user registration without email confirmation requirement."""
    fake_token = "signup_token_abc"
    fake_user = {"id": "new-user-5678", "email": "newdoc@hospital.org"}

    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 201
        mock_res.json.return_value = {
            "user": fake_user,
            "session": {"access_token": fake_token}
        }
        mock_post.return_value = mock_res

        result = client.signup("newdoc@hospital.org", "ValidPass123!")

        assert result["success"] is True
        assert result["needs_email_confirmation"] is False
        assert result["data"]["user"]["email"] == "newdoc@hospital.org"


def test_registration_success_needs_email_confirmation(client):
    """Test successful user registration when email confirmation is enabled (session is None)."""
    fake_user = {"id": "confirm-user-999", "email": "confirm@hospital.org"}

    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 201
        mock_res.json.return_value = {
            "user": fake_user,
            "session": None
        }
        mock_post.return_value = mock_res

        result = client.signup("confirm@hospital.org", "ValidPass123!")

        assert result["success"] is True
        assert result["needs_email_confirmation"] is True
        assert result["data"]["user"]["id"] == "confirm-user-999"


def test_registration_failure(client):
    """Test registration failure returns clean error description."""
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 400
        mock_res.json.return_value = {"detail": "User already registered"}
        mock_post.return_value = mock_res

        result = client.signup("existing@hospital.org", "ValidPass123!")

        assert result["success"] is False
        assert "User already registered" in result["error"]


def test_token_lifecycle_and_headers(client):
    """Verify setting, clearing, and injecting the token into request headers."""
    assert client.token is None
    assert client.get_headers() == {}

    client.set_token("  my_token_value  ")
    assert client.token == "my_token_value"
    assert client.get_headers() == {"Authorization": "Bearer my_token_value"}

    # Include custom additional headers
    headers = client.get_headers({"X-Custom-Header": "HealthAI"})
    assert headers["Authorization"] == "Bearer my_token_value"
    assert headers["X-Custom-Header"] == "HealthAI"

    client.set_token(None)
    assert client.token is None
    assert client.get_headers() == {}


def test_get_current_user_success(client):
    """Test /auth/me returns the active authenticated user profile."""
    client.set_token("active_token")

    with patch("requests.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "user": {
                "id": "uuid-me-001",
                "email": "researcher@clinic.org",
                "created_at": "2026-09-14T00:00:00Z",
            }
        }
        mock_get.return_value = mock_res

        res = client.get_current_user()

        assert res["success"] is True
        assert res["user"]["id"] == "uuid-me-001"
        assert res["user"]["email"] == "researcher@clinic.org"
        mock_get.assert_called_once_with(
            "http://test-backend:8000/auth/me",
            headers={"Authorization": "Bearer active_token"},
            timeout=client.DEFAULT_TIMEOUT_FAST,
        )


def test_get_current_user_unauthorized_clears_token(client):
    """Test /auth/me returning 401 automatically clears the client token."""
    client.set_token("expired_token")

    with patch("requests.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 401
        mock_res.json.return_value = {"detail": "Invalid or expired JWT token"}
        mock_get.return_value = mock_res

        res = client.get_current_user()

        assert res["success"] is False
        assert res["status_code"] == 401
        assert client.token is None
        assert "Authorization" not in client.get_headers()


def test_logout_terminates_session_and_clears_token(client):
    """Test logout invokes POST /auth/logout and clears active token from memory."""
    client.set_token("token_to_clear")

    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_post.return_value = mock_res

        res = client.logout()

        assert res["success"] is True
        assert client.token is None
        assert "Authorization" not in client.get_headers()
        mock_post.assert_called_once()


def test_bearer_header_attached_to_protected_endpoints(client):
    """Verify Bearer token is attached across all protected frontend API operations."""
    client.set_token("valid_patient_bearer_token")

    with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
        # Mock responses
        mock_post_res = MagicMock()
        mock_post_res.status_code = 201
        mock_post_res.json.return_value = {"id": 1, "status": "ok"}
        mock_post.return_value = mock_post_res

        mock_get_res = MagicMock()
        mock_get_res.status_code = 200
        mock_get_res.json.return_value = [{"id": 1, "filename": "doc.pdf"}]
        mock_get.return_value = mock_get_res

        # 1. register_document_metadata
        client.register_document_metadata(
            filename="clinical_trial.pdf",
            file_path="/data/clinical_trial.pdf",
            file_size_bytes=1024,
            num_pages=2,
        )
        assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer valid_patient_bearer_token"

        # 2. list_documents
        client.list_documents()
        assert mock_get.call_args[1]["headers"]["Authorization"] == "Bearer valid_patient_bearer_token"

        # 3. create_conversation
        client.create_conversation("Oncology Consult")
        assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer valid_patient_bearer_token"

        # 4. append_conversation_message
        client.append_conversation_message(conversation_id=1, sender="user", text="What is the dosage?")
        assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer valid_patient_bearer_token"


def test_401_and_403_handling_on_protected_endpoints(client):
    """Verify HTTP 401 Unauthorized and 403 Forbidden are handled cleanly without unhandled exceptions."""
    client.set_token("invalid_or_forbidden_token")

    # 1. Test 401 on register_document_metadata
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 401
        mock_res.json.return_value = {"detail": "Invalid or expired JWT token"}
        mock_post.return_value = mock_res

        res = client.register_document_metadata("test.pdf", "/test.pdf")
        assert res["success"] is False
        assert res.get("status_code") == 401
        assert "unauthenticated" in res["error"].lower() or "session expired" in res["error"].lower()

    # 2. Test 403 on create_conversation
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 403
        mock_res.json.return_value = {"detail": "Access forbidden"}
        mock_post.return_value = mock_res

        res = client.create_conversation("Forbidden Consult")
        assert res["success"] is False
        assert res.get("status_code") == 403
        assert "access denied" in res["error"].lower() or "forbidden" in res["error"].lower()
