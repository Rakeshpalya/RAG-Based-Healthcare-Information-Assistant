"""
Supabase Authentication Service for AI-Healthcare-Agent.

Provides centralized authentication operations (user registration, login,
logout, and current user retrieval) backed by Supabase Auth (GoTrue).
"""

from typing import Optional, Dict, Any
from supabase import create_client, Client

from backend.config import settings


class AuthService:
    """
    Service wrapper for Supabase authentication operations.

    Responsibilities:
    1. Supabase client initialization using publishable anon key.
    2. User signup with email and password.
    3. User login with email and password.
    4. User logout / session termination.
    5. Fetching current authenticated user profile.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        client: Optional[Client] = None,
    ) -> None:
        """
        Initialize AuthService.

        Args:
            supabase_url: Optional Supabase URL override. Defaults to settings.SUPABASE_URL.
            supabase_key: Optional Supabase publishable key override. Defaults to settings.SUPABASE_PUBLISHABLE_KEY.
            client: Optional pre-configured Supabase Client (useful for dependency injection and testing).

        Raises:
            ValueError: If SUPABASE_URL or SUPABASE_PUBLISHABLE_KEY is missing or empty.
        """
        if client is not None:
            self.client: Client = client
            self.supabase_url: str = str(supabase_url) if supabase_url else ""
            self.supabase_key: str = str(supabase_key) if supabase_key else ""
            return

        url = supabase_url if supabase_url is not None else settings.SUPABASE_URL
        key = supabase_key if supabase_key is not None else settings.SUPABASE_PUBLISHABLE_KEY

        if not url or not str(url).strip():
            raise ValueError(
                "SUPABASE_URL is not configured. "
                "Please configure SUPABASE_URL in your environment or .env file."
            )

        if not key or not str(key).strip():
            raise ValueError(
                "SUPABASE_PUBLISHABLE_KEY is not configured. "
                "Please configure SUPABASE_PUBLISHABLE_KEY in your environment or .env file."
            )

        raw_url = str(url).strip()
        clean_url = raw_url.split("/rest/v1")[0].rstrip("/")
        self.supabase_url: str = clean_url
        self.supabase_key: str = str(key).strip()
        self.client: Client = create_client(self.supabase_url, self.supabase_key)

    @staticmethod
    def _extract_auth_payload(response: Any) -> Dict[str, Any]:
        """
        Extract user and session objects safely from an AuthResponse or dict.

        Args:
            response: AuthResponse object or response dict.

        Returns:
            Dict containing 'user' and 'session'.
        """
        if isinstance(response, dict):
            return {
                "user": response.get("user"),
                "session": response.get("session"),
            }
        return {
            "user": getattr(response, "user", None),
            "session": getattr(response, "session", None),
        }

    def signup(self, email: str, password: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Registers a new user with email and password via Supabase Auth.
        Automatically retries on transient network/connection timeouts.

        Args:
            email: User's email address.
            password: User's chosen password.
            max_retries: Maximum retry attempts for transient timeouts.

        Returns:
            Dict containing 'user' and 'session'.
        """
        clean_email = email.strip().lower()
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.auth.sign_up(
                    {
                        "email": clean_email,
                        "password": password,
                    }
                )
                return self._extract_auth_payload(response)
            except Exception as exc:
                last_error = exc
                err_type = type(exc).__name__
                err_str = str(exc).lower()
                if err_type == "AuthApiError" or "already registered" in err_str:
                    raise exc
                is_timeout = any(t in err_type.lower() or t in err_str for t in ["timeout", "connect", "handshake", "ssl", "remotedisconnected"])
                if is_timeout and attempt < max_retries - 1:
                    import time
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise exc
        if last_error:
            raise last_error

    def login(self, email: str, password: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Authenticates an existing user with email and password via Supabase Auth.
        Automatically retries on transient network/connection/SSL handshake timeouts.

        Args:
            email: User's email address.
            password: User's password.
            max_retries: Maximum retry attempts for transient timeouts.

        Returns:
            Dict containing 'user' and 'session'.
        """
        clean_email = email.strip().lower()
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.auth.sign_in_with_password(
                    {
                        "email": clean_email,
                        "password": password,
                    }
                )
                return self._extract_auth_payload(response)
            except Exception as exc:
                last_error = exc
                err_type = type(exc).__name__
                err_str = str(exc).lower()
                # If explicit credential rejection, do not retry
                if err_type == "AuthApiError" or "invalid login credentials" in err_str:
                    raise exc
                is_timeout = any(t in err_type.lower() or t in err_str for t in ["timeout", "connect", "handshake", "ssl", "remotedisconnected"])
                if is_timeout and attempt < max_retries - 1:
                    import time
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise exc
        if last_error:
            raise last_error

    def logout(self) -> Dict[str, Any]:
        """
        Logs out the current user by terminating the active session.

        Returns:
            Dict indicating successful logout status.
        """
        self.client.auth.sign_out()
        return {
            "status": "success",
            "message": "User signed out successfully",
        }

    def get_current_user(self, jwt: Optional[str] = None) -> Optional[Any]:
        """
        Retrieves the currently authenticated user.

        Args:
            jwt: Optional JWT token string for bearer verification.
                 If omitted, checks the client's current session state.

        Returns:
            User object or dict, or None if not authenticated.
        """
        if jwt is not None:
            response = self.client.auth.get_user(jwt)
        else:
            response = self.client.auth.get_user()

        if response is None:
            return None

        if isinstance(response, dict):
            return response.get("user", response)

        return getattr(response, "user", response)


# Global default instance configured from settings
auth_service = AuthService()
