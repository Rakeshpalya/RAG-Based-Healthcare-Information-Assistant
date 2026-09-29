"""
Unit tests for FastAPI Authentication Dependencies (backend/api/auth_dependencies.py).

Verifies:
1. Valid Bearer token -> authenticated user
2. Missing Authorization header -> 401 Unauthorized
3. Wrong authentication scheme (e.g., Basic, Token) -> 401 Unauthorized
4. Empty token (Bearer with only whitespace) -> 401 Unauthorized
5. Invalid token (cryptographic failure) -> 401 Unauthorized
6. Expired/failed token -> 401 Unauthorized
7. Supabase authentication failure (returns None) -> 401 Unauthorized
8. Correct authenticated user payload is returned
9. Bearer token is strictly forwarded to Supabase verification
10. Sensitive token is never leaked in error responses or headers

All external Supabase calls are mocked with zero real network requests.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any
import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.api.auth_dependencies import get_current_user

# Mock application isolating the get_current_user dependency
mock_app = FastAPI()


@mock_app.get("/protected-test")
def protected_test_route(user: Any = Depends(get_current_user)):
    return {"status": "success", "user": user}


client = TestClient(mock_app)


def test_valid_bearer_token_returns_authenticated_user():
    """1. Verify valid Bearer token validates and returns authenticated user with 200 OK."""
    mock_user = {
        "id": "usr_valid_001",
        "email": "doctor@hospital.org",
        "role": "clinician",
    }
    sample_jwt = "header.payload.signature"

    with patch("backend.api.auth_dependencies.auth_service.get_current_user") as mock_auth:
        mock_auth.return_value = mock_user

        response = client.get(
            "/protected-test",
            headers={"Authorization": f"Bearer {sample_jwt}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["user"]["id"] == "usr_valid_001"
        assert data["user"]["email"] == "doctor@hospital.org"
        mock_auth.assert_called_once_with(jwt=sample_jwt)
    print("[PASS] test_valid_bearer_token_returns_authenticated_user passed.")


def test_missing_authorization_header_returns_401():
    """2. Verify missing Authorization header returns 401 with WWW-Authenticate header."""
    response = client.get("/protected-test")
    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"
    assert response.json()["detail"] == "Missing Authorization header"
    print("[PASS] test_missing_authorization_header_returns_401 passed.")


def test_wrong_authentication_scheme_returns_401():
    """3. Verify non-Bearer schemes (e.g. Basic, Token, Digest) return 401."""
    # Basic scheme
    res_basic = client.get(
        "/protected-test",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    assert res_basic.status_code == 401
    assert res_basic.headers.get("WWW-Authenticate") == "Bearer"
    assert "Invalid authentication scheme" in res_basic.json()["detail"]

    # Token scheme
    res_token = client.get(
        "/protected-test",
        headers={"Authorization": "Token some-api-key"},
    )
    assert res_token.status_code == 401
    assert "Invalid authentication scheme" in res_token.json()["detail"]
    print("[PASS] test_wrong_authentication_scheme_returns_401 passed.")


def test_empty_token_returns_401():
    """4. Verify Bearer header with empty or whitespace-only token returns 401."""
    # Empty token after Bearer
    res_empty = client.get(
        "/protected-test",
        headers={"Authorization": "Bearer "},
    )
    assert res_empty.status_code == 401
    assert res_empty.headers.get("WWW-Authenticate") == "Bearer"
    assert res_empty.json()["detail"] == "Empty bearer token provided"

    # Whitespace token
    res_ws = client.get(
        "/protected-test",
        headers={"Authorization": "Bearer     "},
    )
    assert res_ws.status_code == 401
    assert res_ws.json()["detail"] == "Empty bearer token provided"
    print("[PASS] test_empty_token_returns_401 passed.")


def test_invalid_token_returns_401():
    """5. Verify invalid token causing cryptographic verification failure returns 401."""
    with patch("backend.api.auth_dependencies.auth_service.get_current_user") as mock_auth:
        mock_auth.side_effect = Exception("JWT cryptographic operation failed")

        response = client.get(
            "/protected-test",
            headers={"Authorization": "Bearer invalid.signature.token"},
        )

        assert response.status_code == 401
        assert response.headers.get("WWW-Authenticate") == "Bearer"
        assert response.json()["detail"] == "Invalid or expired token"
    print("[PASS] test_invalid_token_returns_401 passed.")


def test_expired_or_failed_token_returns_401():
    """6. Verify expired token error from Supabase returns 401."""
    with patch("backend.api.auth_dependencies.auth_service.get_current_user") as mock_auth:
        mock_auth.side_effect = Exception("Token has expired")

        response = client.get(
            "/protected-test",
            headers={"Authorization": "Bearer expired.access.token"},
        )

        assert response.status_code == 401
        assert response.headers.get("WWW-Authenticate") == "Bearer"
        assert response.json()["detail"] == "Invalid or expired token"
    print("[PASS] test_expired_or_failed_token_returns_401 passed.")


def test_supabase_authentication_failure_returns_401():
    """7. Verify Supabase get_current_user returning None returns 401."""
    with patch("backend.api.auth_dependencies.auth_service.get_current_user") as mock_auth:
        mock_auth.return_value = None

        response = client.get(
            "/protected-test",
            headers={"Authorization": "Bearer unverified.token"},
        )

        assert response.status_code == 401
        assert response.headers.get("WWW-Authenticate") == "Bearer"
        assert response.json()["detail"] == "Could not validate credentials"
    print("[PASS] test_supabase_authentication_failure_returns_401 passed.")


def test_correct_authenticated_user_returned():
    """8. Verify the exact user object returned by Supabase Auth is received."""
    mock_user = {
        "id": "usr_specific_999",
        "email": "sarah.connor@resistance.org",
        "app_metadata": {"provider": "email"},
        "user_metadata": {"first_name": "Sarah"},
    }

    with patch("backend.api.auth_dependencies.auth_service.get_current_user") as mock_auth:
        mock_auth.return_value = mock_user

        response = client.get(
            "/protected-test",
            headers={"Authorization": "Bearer verified.token.payload"},
        )

        assert response.status_code == 200
        assert response.json()["user"] == mock_user
    print("[PASS] test_correct_authenticated_user_returned passed.")


def test_token_passed_to_supabase_verification():
    """9. Verify the raw bearer token is passed unmodified to Supabase get_current_user."""
    test_token = "secret.token.string.to.verify"

    with patch("backend.api.auth_dependencies.auth_service.get_current_user") as mock_auth:
        mock_auth.return_value = {"id": "usr_100"}

        response = client.get(
            "/protected-test",
            headers={"Authorization": f"Bearer {test_token}"},
        )

        assert response.status_code == 200
        mock_auth.assert_called_once_with(jwt=test_token)
    print("[PASS] test_token_passed_to_supabase_verification passed.")


def test_sensitive_token_not_in_error_responses():
    """10. Verify sensitive bearer token strings are NEVER leaked in error responses."""
    sensitive_token = "super_secret_jwt_token_never_leak_this_12345"

    with patch("backend.api.auth_dependencies.auth_service.get_current_user") as mock_auth:
        mock_auth.side_effect = Exception("Invalid signature")

        response = client.get(
            "/protected-test",
            headers={"Authorization": f"Bearer {sensitive_token}"},
        )

        assert response.status_code == 401
        # Token must never appear in response body or headers
        assert sensitive_token not in response.text
        assert sensitive_token not in str(response.headers)
    print("[PASS] test_sensitive_token_not_in_error_responses passed.")


if __name__ == "__main__":
    print("Running FastAPI auth dependencies unit tests...")
    test_valid_bearer_token_returns_authenticated_user()
    test_missing_authorization_header_returns_401()
    test_wrong_authentication_scheme_returns_401()
    test_empty_token_returns_401()
    test_invalid_token_returns_401()
    test_expired_or_failed_token_returns_401()
    test_supabase_authentication_failure_returns_401()
    test_correct_authenticated_user_returned()
    test_token_passed_to_supabase_verification()
    test_sensitive_token_not_in_error_responses()
    print("All FastAPI auth dependencies unit tests passed successfully!")
