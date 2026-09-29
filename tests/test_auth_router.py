"""
Unit tests for FastAPI Authentication API Router (/auth).

Tests:
1. Signup success (POST /auth/signup)
2. Signup validation failure (POST /auth/signup)
3. Signup authentication failure (POST /auth/signup)
4. Login success (POST /auth/login)
5. Login authentication failure (POST /auth/login)
6. Logout success (POST /auth/logout)
7. Current-user success (GET /auth/me)
8. Current-user unauthenticated/error case (GET /auth/me)
9. Router registration confirmation in OpenAPI schema
10. Existing routes (/health, /) remain working

All external auth service calls are strictly mocked with zero real network requests.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.main import app

client = TestClient(app)


def test_signup_success():
    """Verify POST /auth/signup creates account and returns 201 with structured user and session."""
    mock_user = {
        "id": "usr_mock_001",
        "email": "newpatient@example.com",
        "created_at": "2026-09-14T00:00:00Z",
    }
    mock_session = {
        "access_token": "mock-access-token-123",
        "token_type": "bearer",
        "expires_in": 3600,
        "expires_at": 1726272000,
    }

    with patch("backend.api.auth_router.auth_service.signup") as mock_signup:
        mock_signup.return_value = {
            "user": mock_user,
            "session": mock_session,
        }

        payload = {
            "email": "newpatient@example.com",
            "password": "SecurePassword123!",
        }
        response = client.post("/auth/signup", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["message"] == "User registered successfully"
        assert data["user"]["email"] == "newpatient@example.com"
        assert data["user"]["id"] == "usr_mock_001"
        assert data["session"]["access_token"] == "mock-access-token-123"
        assert data["session"]["token_type"] == "bearer"
        mock_signup.assert_called_once_with(
            "newpatient@example.com",
            "SecurePassword123!",
        )
    print("[PASS] test_signup_success passed.")


def test_signup_validation_failure():
    """Verify POST /auth/signup rejects invalid emails or short passwords with 422."""
    # 1. Invalid email format
    invalid_email_payload = {
        "email": "not-a-valid-email",
        "password": "ValidPassword123!",
    }
    response = client.post("/auth/signup", json=invalid_email_payload)
    assert response.status_code == 422

    # 2. Short password (< 6 characters)
    short_pw_payload = {
        "email": "patient@example.com",
        "password": "123",
    }
    response = client.post("/auth/signup", json=short_pw_payload)
    assert response.status_code == 422

    # 3. Missing fields
    missing_payload = {"email": "patient@example.com"}
    response = client.post("/auth/signup", json=missing_payload)
    assert response.status_code == 422
    print("[PASS] test_signup_validation_failure passed.")


def test_signup_authentication_failure():
    """Verify POST /auth/signup handles service-level registration failures with 400 Bad Request."""
    with patch("backend.api.auth_router.auth_service.signup") as mock_signup:
        mock_signup.side_effect = Exception("User already registered with this email")

        payload = {
            "email": "existing@example.com",
            "password": "SecurePassword123!",
        }
        response = client.post("/auth/signup", json=payload)

        assert response.status_code == 400
        assert "User already registered" in response.json()["detail"]
    print("[PASS] test_signup_authentication_failure passed.")


def test_login_success():
    """Verify POST /auth/login returns 200 OK with session and user metadata."""
    mock_user = {
        "id": "usr_mock_002",
        "email": "clinician@example.com",
        "created_at": "2026-09-14T00:00:00Z",
    }
    mock_session = {
        "access_token": "mock-clinician-token-456",
        "token_type": "bearer",
        "expires_in": 3600,
    }

    with patch("backend.api.auth_router.auth_service.login") as mock_login:
        mock_login.return_value = {
            "user": mock_user,
            "session": mock_session,
        }

        payload = {
            "email": "clinician@example.com",
            "password": "StrongPassword789!",
        }
        response = client.post("/auth/login", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Login successful"
        assert data["user"]["email"] == "clinician@example.com"
        assert data["session"]["access_token"] == "mock-clinician-token-456"
        mock_login.assert_called_once_with(
            "clinician@example.com",
            "StrongPassword789!",
        )
    print("[PASS] test_login_success passed.")


def test_login_authentication_failure():
    """Verify POST /auth/login handles invalid credentials with 401 Unauthorized."""
    with patch("backend.api.auth_router.auth_service.login") as mock_login:
        mock_login.side_effect = Exception("Invalid login credentials")

        payload = {
            "email": "clinician@example.com",
            "password": "WrongPassword!",
        }
        response = client.post("/auth/login", json=payload)

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid email or password"

    # Also test when login returns None for user or session
    with patch("backend.api.auth_router.auth_service.login") as mock_login:
        mock_login.return_value = {"user": None, "session": None}
        response = client.post("/auth/login", json=payload)
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid email or password"
    print("[PASS] test_login_authentication_failure passed.")


def test_logout_success():
    """Verify POST /auth/logout returns 200 OK and success message."""
    with patch("backend.api.auth_router.auth_service.logout") as mock_logout:
        mock_logout.return_value = {"status": "success"}

        response = client.post("/auth/logout")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["message"] == "User signed out successfully"
        mock_logout.assert_called_once()
    print("[PASS] test_logout_success passed.")


def test_current_user_success():
    """Verify GET /auth/me returns 200 OK with authenticated user profile."""
    mock_user = {
        "id": "usr_mock_003",
        "email": "activeuser@example.com",
        "created_at": "2026-09-14T00:00:00Z",
    }

    with patch("backend.api.auth_router.auth_service.get_current_user") as mock_get_user:
        mock_get_user.return_value = mock_user

        response = client.get("/auth/me")

        assert response.status_code == 200
        data = response.json()
        assert data["user"]["id"] == "usr_mock_003"
        assert data["user"]["email"] == "activeuser@example.com"
        mock_get_user.assert_called_once()
    print("[PASS] test_current_user_success passed.")


def test_current_user_unauthenticated():
    """Verify GET /auth/me returns 401 Unauthorized when unauthenticated."""
    with patch("backend.api.auth_router.auth_service.get_current_user") as mock_get_user:
        mock_get_user.return_value = None

        response = client.get("/auth/me")

        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    # Also test when get_current_user raises an exception
    with patch("backend.api.auth_router.auth_service.get_current_user") as mock_get_user:
        mock_get_user.side_effect = Exception("Session expired")

        response = client.get("/auth/me")
        assert response.status_code == 401
        assert response.json()["detail"] == "Could not validate credentials"
    print("[PASS] test_current_user_unauthenticated passed.")


def test_auth_router_registered():
    """Verify authentication routes are registered in the FastAPI OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema.get("paths", {})

    assert "/auth/signup" in paths
    assert "post" in paths["/auth/signup"]

    assert "/auth/login" in paths
    assert "post" in paths["/auth/login"]

    assert "/auth/logout" in paths
    assert "post" in paths["/auth/logout"]

    assert "/auth/me" in paths
    assert "get" in paths["/auth/me"]
    print("[PASS] test_auth_router_registered passed.")


def test_existing_routes_still_work():
    """Verify existing Phase 1–11 endpoints (/health and /) continue to function."""
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "healthy"

    root_res = client.get("/")
    assert root_res.status_code == 200
    assert "Welcome" in root_res.json()["message"]
    print("[PASS] test_existing_routes_still_work passed.")


if __name__ == "__main__":
    print("Running FastAPI auth router tests...")
    test_signup_success()
    test_signup_validation_failure()
    test_signup_authentication_failure()
    test_login_success()
    test_login_authentication_failure()
    test_logout_success()
    test_current_user_success()
    test_current_user_unauthenticated()
    test_auth_router_registered()
    test_existing_routes_still_work()
    print("All FastAPI auth router tests passed successfully!")
