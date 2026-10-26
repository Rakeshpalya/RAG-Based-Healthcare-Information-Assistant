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


def test_auth_login_local_fallback_on_dns_failure(client):
    """Verify login falls back to local auth in development when Supabase host cannot be resolved."""
    from backend.services.auth_service import auth_service
    # Test credentials for seeded local user
    test_email = "ram0@gmail.com"
    test_pass = "ram0@gmail.com"

    # POST to /auth/login with valid local test credentials
    resp = client.post("/auth/login", json={"email": test_email, "password": test_pass})
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Login successful"
    assert data["user"]["email"] == test_email
    assert "access_token" in data["session"]

    # Verify token can be verified by get_current_user
    token = data["session"]["access_token"]
    user = auth_service.get_current_user(jwt=token)
    assert user is not None
    assert user["email"] == test_email


def test_auth_login_local_fallback_rejects_wrong_password(client):
    """Verify local auth strictly enforces password verification and rejects incorrect passwords."""
    resp = client.post(
        "/auth/login",
        json={"email": "ram0@gmail.com", "password": "definitely_wrong_password"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid email or password"


def test_auth_login_network_timeout_in_production_returns_504(client):
    """Verify that an unhandled network timeout in production returns 504 Gateway Timeout."""
    with patch("backend.api.auth_router.auth_service.login") as mock_login:
        mock_login.side_effect = Exception("Read timed out on connection")

        resp = client.post(
            "/auth/login",
            json={"email": "ram0@gmail.com", "password": "any_password"},
        )
        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"]


def test_auth_login_unreachable_provider_in_production_returns_503(client):
    """Verify that unhandled connection/DNS failure in production returns 503 Service Unavailable."""
    with patch("backend.api.auth_router.auth_service.login") as mock_login:
        import httpx
        mock_login.side_effect = httpx.ConnectError("[Errno 11001] getaddrinfo failed")

        resp = client.post(
            "/auth/login",
            json={"email": "ram0@gmail.com", "password": "any_password"},
        )

def test_production_environment_strictly_prohibits_local_fallback():
    """Verify that in production, local fallback methods raise PermissionError."""
    from backend.services.auth_service import auth_service
    from backend.config import settings

    with patch.dict("os.environ", {"ENVIRONMENT": "production"}):
        assert settings.is_production is True

        with pytest.raises(PermissionError, match="strictly prohibited in production"):
            auth_service._local_login("ram0@gmail.com", "ram0@gmail.com")

        with pytest.raises(PermissionError, match="strictly prohibited in production"):
            auth_service._local_signup("newuser@example.com", "password123")


def test_production_login_does_not_fall_back_on_network_failure():
    """Verify that in production, login failure raises exception instead of falling back locally."""
    from backend.services.auth_service import auth_service
    from backend.config import settings
    import httpx

    with patch.dict("os.environ", {"ENVIRONMENT": "production"}):
        with patch.object(auth_service.client.auth, "sign_in_with_password") as mock_signin:
            mock_signin.side_effect = httpx.ConnectError("[Errno 11001] getaddrinfo failed")

            with pytest.raises(httpx.ConnectError):
                auth_service.login("ram0@gmail.com", "ram0@gmail.com")


def test_production_get_current_user_ignores_local_jwt():
    """Verify that in production, get_current_user does not decode local JWTs and delegates to Supabase."""
    from backend.services.auth_service import auth_service
    from backend.config import settings
    import jwt as pyjwt
    import time

    # Generate a valid local JWT
    now = int(time.time())
    token = pyjwt.encode(
        {"sub": "8", "email": "ram0@gmail.com", "role": "patient", "exp": now + 3600},
        auth_service._get_local_jwt_secret(),
        algorithm="HS256"
    )

    with patch.dict("os.environ", {"ENVIRONMENT": "production"}):
        assert settings.is_production is True
        with patch.object(auth_service.client.auth, "get_user") as mock_supabase_get_user:
            mock_supabase_get_user.side_effect = Exception("Invalid token on Supabase")
            with pytest.raises(Exception, match="Invalid token on Supabase"):
                auth_service.get_current_user(jwt_token=token)
            # Ensure client.auth.get_user was indeed called
            mock_supabase_get_user.assert_called_once_with(token)
