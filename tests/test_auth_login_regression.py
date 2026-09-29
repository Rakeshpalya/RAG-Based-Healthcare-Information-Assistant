"""
Regression tests for authentication login, URL normalization, and session handling.

Validates:
1. normalize_supabase_url strips /rest/v1 and trailing slashes.
2. APIClient.login() -> backend /auth/login -> successful session when credentials valid.
3. APIClient.login() -> backend /auth/login -> 401 Unauthorized when credentials invalid.
4. FastAPI auth router handles Supabase Auth errors cleanly with safe diagnostic logging.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import normalize_supabase_url, settings
from frontend.api_client import APIClient


@pytest.fixture
def client():
    return TestClient(app)


def test_normalize_supabase_url_strips_rest_v1():
    """Verify normalize_supabase_url strips /rest/v1, /rest/v1/, and trailing slashes."""
    assert normalize_supabase_url("https://example.supabase.co/rest/v1/") == "https://example.supabase.co"
    assert normalize_supabase_url("https://example.supabase.co/rest/v1") == "https://example.supabase.co"
    assert normalize_supabase_url("https://example.supabase.co/") == "https://example.supabase.co"
    assert normalize_supabase_url("https://example.supabase.co") == "https://example.supabase.co"
    assert normalize_supabase_url(None) is None
    assert normalize_supabase_url("") is None


def test_auth_login_endpoint_success(client):
    """Verify POST /auth/login returns 200 with user and session token when Supabase succeeds."""
    mock_user = {
        "id": "e58eca37-5771-4302-bb11-5086fce72952",
        "email": "ram11234@gmail.com",
        "created_at": "2026-09-14T11:00:27Z",
        "user_metadata": {"email_verified": True},
    }
    mock_session = {
        "access_token": "mock-valid-supabase-jwt-token",
        "token_type": "bearer",
        "expires_in": 3600,
        "expires_at": 1789440000,
    }

    with patch("backend.api.auth_router.auth_service.login") as mock_login:
        mock_login.return_value = {
            "user": mock_user,
            "session": mock_session,
        }

        resp = client.post(
            "/auth/login",
            json={"email": "ram11234@gmail.com", "password": "valid_password"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["message"] == "Login successful"
        assert data["user"]["email"] == "ram11234@gmail.com"
        assert data["session"]["access_token"] == "mock-valid-supabase-jwt-token"


def test_auth_login_endpoint_invalid_credentials_returns_401(client):
    """Verify POST /auth/login returns 401 Unauthorized when invalid credentials provided."""
    with patch("backend.api.auth_router.auth_service.login") as mock_login:
        from supabase_auth.errors import AuthApiError
        mock_login.side_effect = AuthApiError("Invalid login credentials", 400, "invalid_credentials")

        resp = client.post(
            "/auth/login",
            json={"email": "ram11234@gmail.com", "password": "wrong_password"},
        )

        assert resp.status_code == 401
        data = resp.json()
        assert data["detail"] == "Invalid email or password"


def test_api_client_login_success_and_token_storage():
    """Verify APIClient.login() stores bearer token on success."""
    api = APIClient(base_url="http://test-server")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": "Login successful",
        "user": {"id": "user-123", "email": "ram11234@gmail.com"},
        "session": {"access_token": "valid-token-xyz"},
    }

    with patch("requests.post", return_value=mock_resp):
        res = api.login("ram11234@gmail.com", "valid_password")

        assert res["success"] is True
        assert res["token"] == "valid-token-xyz"
        assert api.token == "valid-token-xyz"
        assert api.get_headers()["Authorization"] == "Bearer valid-token-xyz"


def test_api_client_login_failure():
    """Verify APIClient.login() handles 401 failure gracefully and clears token."""
    api = APIClient(base_url="http://test-server")

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"detail": "Invalid email or password"}
    mock_resp.text = '{"detail": "Invalid email or password"}'

    with patch("requests.post", return_value=mock_resp):
        res = api.login("ram11234@gmail.com", "wrong_password")

        assert res["success"] is False
        assert res["token"] is None
        assert res["error"] == "Invalid email or password"
        assert api.token is None
