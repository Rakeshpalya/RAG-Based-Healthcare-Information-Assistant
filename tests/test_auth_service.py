"""
Unit tests for Supabase AuthService.

Verifies:
1. Missing SUPABASE_URL error handling
2. Missing SUPABASE_PUBLISHABLE_KEY error handling
3. Supabase client initialization
4. signup calling client.auth.sign_up with expected payload
5. login calling client.auth.sign_in_with_password with expected payload
6. logout calling client.auth.sign_out
7. get_current_user calling client.auth.get_user

All network calls are strictly mocked with zero real API calls or exposed credentials.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.services.auth_service import AuthService


class DummyAuthResponse:
    """Mock structure returned by Supabase sign_up and sign_in_with_password."""

    def __init__(self, user=None, session=None):
        self.user = user
        self.session = session


class DummyUserResponse:
    """Mock structure returned by Supabase get_user."""

    def __init__(self, user=None):
        self.user = user


def test_missing_supabase_url_raises_value_error():
    """Verify ValueError is raised if SUPABASE_URL is missing or empty."""
    with patch("backend.services.auth_service.settings.SUPABASE_URL", None):
        with pytest.raises(ValueError) as exc_info:
            AuthService(supabase_url=None, supabase_key="dummy_publishable_key")
        assert "SUPABASE_URL" in str(exc_info.value)

    # Empty string url
    with pytest.raises(ValueError) as exc_info:
        AuthService(supabase_url="   ", supabase_key="dummy_publishable_key")
    assert "SUPABASE_URL" in str(exc_info.value)
    print("[PASS] test_missing_supabase_url_raises_value_error passed.")


def test_missing_supabase_key_raises_value_error():
    """Verify ValueError is raised if SUPABASE_PUBLISHABLE_KEY is missing or empty."""
    with patch("backend.services.auth_service.settings.SUPABASE_PUBLISHABLE_KEY", None):
        with pytest.raises(ValueError) as exc_info:
            AuthService(supabase_url="https://dummy.supabase.co", supabase_key=None)
        assert "SUPABASE_PUBLISHABLE_KEY" in str(exc_info.value)

    # Empty string key
    with pytest.raises(ValueError) as exc_info:
        AuthService(supabase_url="https://dummy.supabase.co", supabase_key="   ")
    assert "SUPABASE_PUBLISHABLE_KEY" in str(exc_info.value)
    print("[PASS] test_missing_supabase_key_raises_value_error passed.")


def test_client_initialization():
    """Verify Supabase client initializes correctly with valid URL and publishable key."""
    with patch("backend.services.auth_service.create_client") as mock_create_client:
        mock_client_instance = MagicMock()
        mock_create_client.return_value = mock_client_instance

        service = AuthService(
            supabase_url="https://test.supabase.co",
            supabase_key="test-anon-key",
        )

        mock_create_client.assert_called_once_with(
            "https://test.supabase.co",
            "test-anon-key",
        )
        assert service.client == mock_client_instance
        assert service.supabase_url == "https://test.supabase.co"
        assert service.supabase_key == "test-anon-key"
    print("[PASS] test_client_initialization passed.")


def test_signup_calls_supabase_sign_up():
    """Verify signup forwards email/password to client.auth.sign_up and returns user/session."""
    mock_client = MagicMock()
    mock_user = {"id": "usr_001", "email": "patient@example.com"}
    mock_session = {"access_token": "mock-token", "token_type": "bearer"}
    mock_client.auth.sign_up.return_value = DummyAuthResponse(
        user=mock_user,
        session=mock_session,
    )

    service = AuthService(client=mock_client)
    result = service.signup("patient@example.com", "SecurePassword123!")

    mock_client.auth.sign_up.assert_called_once_with(
        {
            "email": "patient@example.com",
            "password": "SecurePassword123!",
        }
    )
    assert result["user"] == mock_user
    assert result["session"] == mock_session
    print("[PASS] test_signup_calls_supabase_sign_up passed.")


def test_login_calls_supabase_sign_in_with_password():
    """Verify login forwards credentials to client.auth.sign_in_with_password and returns user/session."""
    mock_client = MagicMock()
    mock_user = {"id": "usr_001", "email": "patient@example.com"}
    mock_session = {"access_token": "mock-token", "token_type": "bearer"}
    mock_client.auth.sign_in_with_password.return_value = DummyAuthResponse(
        user=mock_user,
        session=mock_session,
    )

    service = AuthService(client=mock_client)
    result = service.login("patient@example.com", "SecurePassword123!")

    mock_client.auth.sign_in_with_password.assert_called_once_with(
        {
            "email": "patient@example.com",
            "password": "SecurePassword123!",
        }
    )
    assert result["user"] == mock_user
    assert result["session"] == mock_session
    print("[PASS] test_login_calls_supabase_sign_in_with_password passed.")


def test_logout_calls_supabase_sign_out():
    """Verify logout calls client.auth.sign_out and returns success status."""
    mock_client = MagicMock()
    mock_client.auth.sign_out.return_value = None

    service = AuthService(client=mock_client)
    result = service.logout()

    mock_client.auth.sign_out.assert_called_once()
    assert result["status"] == "success"
    print("[PASS] test_logout_calls_supabase_sign_out passed.")


def test_get_current_user_no_jwt():
    """Verify get_current_user calls client.auth.get_user() without arguments when no JWT is provided."""
    mock_client = MagicMock()
    mock_user = {"id": "usr_001", "email": "patient@example.com"}
    mock_client.auth.get_user.return_value = DummyUserResponse(user=mock_user)

    service = AuthService(client=mock_client)
    user = service.get_current_user()

    mock_client.auth.get_user.assert_called_once_with()
    assert user == mock_user
    print("[PASS] test_get_current_user_no_jwt passed.")


def test_get_current_user_with_jwt():
    """Verify get_current_user passes jwt to client.auth.get_user(jwt) when provided."""
    mock_client = MagicMock()
    mock_user = {"id": "usr_002", "email": "doctor@example.com"}
    mock_client.auth.get_user.return_value = DummyUserResponse(user=mock_user)

    service = AuthService(client=mock_client)
    user = service.get_current_user("mock-bearer-token")

    mock_client.auth.get_user.assert_called_once_with("mock-bearer-token")
    assert user == mock_user
    print("[PASS] test_get_current_user_with_jwt passed.")


def test_get_current_user_returns_none_when_unauthenticated():
    """Verify get_current_user returns None if Supabase get_user returns None."""
    mock_client = MagicMock()
    mock_client.auth.get_user.return_value = None

    service = AuthService(client=mock_client)
    user = service.get_current_user()

    mock_client.auth.get_user.assert_called_once_with()
    assert user is None
    print("[PASS] test_get_current_user_returns_none_when_unauthenticated passed.")


def test_auth_payload_dict_compatibility():
    """Verify _extract_auth_payload and get_current_user gracefully handle dictionary responses."""
    mock_client = MagicMock()
    mock_dict_response = {
        "user": {"id": "usr_003", "email": "researcher@example.com"},
        "session": {"access_token": "mock-token-3"},
    }
    mock_client.auth.sign_up.return_value = mock_dict_response
    mock_client.auth.get_user.return_value = {"user": {"id": "usr_003"}}

    service = AuthService(client=mock_client)
    signup_result = service.signup("researcher@example.com", "Pass123!")
    assert signup_result["user"] == {"id": "usr_003", "email": "researcher@example.com"}
    assert signup_result["session"] == {"access_token": "mock-token-3"}

    current_user = service.get_current_user()
    assert current_user == {"id": "usr_003"}
    print("[PASS] test_auth_payload_dict_compatibility passed.")


if __name__ == "__main__":
    print("Running Supabase AuthService unit tests...")
    test_missing_supabase_url_raises_value_error()
    test_missing_supabase_key_raises_value_error()
    test_client_initialization()
    test_signup_calls_supabase_sign_up()
    test_login_calls_supabase_sign_in_with_password()
    test_logout_calls_supabase_sign_out()
    test_get_current_user_no_jwt()
    test_get_current_user_with_jwt()
    test_get_current_user_returns_none_when_unauthenticated()
    test_auth_payload_dict_compatibility()
    print("All Supabase AuthService unit tests passed successfully!")
