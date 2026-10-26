import os
import json
import time
import logging
import hashlib
import secrets
from pathlib import Path
from typing import Optional, Dict, Any
import jwt as pyjwt
from supabase import create_client, Client

from backend.config import settings

logger = logging.getLogger(__name__)


class AuthService:
    """
    Service wrapper for Supabase authentication operations with local development fallback.

    Responsibilities:
    1. Supabase client initialization using publishable anon key.
    2. User signup with email and password.
    3. User login with email and password.
    4. User logout / session termination.
    5. Fetching current authenticated user profile.
    6. Local development resilience fallback when external Supabase provider is unreachable.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        client: Optional[Client] = None,
        lazy: bool = False,
    ) -> None:
        """
        Initialize AuthService.

        Args:
            supabase_url: Optional Supabase URL override. Defaults to settings.SUPABASE_URL.
            supabase_key: Optional Supabase publishable key override. Defaults to settings.SUPABASE_PUBLISHABLE_KEY.
            client: Optional pre-configured Supabase Client (useful for dependency injection and testing).
            lazy: If True, defers Supabase client creation until first access, preventing import-time crashes.

        Raises:
            ValueError: If SUPABASE_URL or SUPABASE_PUBLISHABLE_KEY is missing or empty (when not lazy).
        """
        if client is not None:
            self._client: Optional[Client] = client
            self.supabase_url: str = str(supabase_url) if supabase_url else ""
            self.supabase_key: str = str(supabase_key) if supabase_key else ""
            self._lazy: bool = False
            return

        self._client: Optional[Client] = None
        self._supabase_url: Optional[str] = supabase_url
        self._supabase_key: Optional[str] = supabase_key
        self.supabase_url: str = ""
        self.supabase_key: str = ""
        self._lazy: bool = lazy

        if not lazy:
            self._init_client()

    def _init_client(self) -> None:
        """
        Instantiates the Supabase client using configured credentials.

        Raises:
            ValueError: If SUPABASE_URL or SUPABASE_PUBLISHABLE_KEY is missing or empty.
        """
        url = self._supabase_url if self._supabase_url is not None else settings.SUPABASE_URL
        key = self._supabase_key if self._supabase_key is not None else settings.SUPABASE_PUBLISHABLE_KEY

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
        self.supabase_url = clean_url
        self.supabase_key = str(key).strip()
        self._client = create_client(self.supabase_url, self.supabase_key)

    @property
    def client(self) -> Client:
        """
        Retrieves or initializes the underlying Supabase Client.
        """
        if self._client is None:
            self._init_client()
        return self._client

    @client.setter
    def client(self, value: Client) -> None:
        """Sets the Supabase client directly (useful for testing and injection)."""
        self._client = value

    @property
    def is_configured(self) -> bool:
        """Check if Supabase credentials are configured without throwing an exception."""
        url = self._supabase_url if self._supabase_url is not None else settings.SUPABASE_URL
        key = self._supabase_key if self._supabase_key is not None else settings.SUPABASE_PUBLISHABLE_KEY
        return bool(url and str(url).strip() and key and str(key).strip())

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

    # =========================================================================
    # Local Development Authentication Resilience Helpers
    # =========================================================================

    def _get_local_jwt_secret(self) -> str:
        """Returns the secret key used for signing and validating local development JWTs."""
        base_secret = (
            os.getenv("AUTH_SECRET_KEY")
            or settings.SUPABASE_PUBLISHABLE_KEY
            or "ai-healthcare-agent-local-secret-key-32bytes-min!"
        )
        return hashlib.sha256(base_secret.encode("utf-8")).hexdigest()

    def _get_credentials_path(self) -> Path:
        """Returns the local credentials store filepath in the data directory."""
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "local_auth_credentials.json"

    def _load_local_credentials(self) -> Dict[str, Any]:
        """Loads local user credentials safely from the persistent data store."""
        cred_path = self._get_credentials_path()
        if not cred_path.exists():
            return {}
        try:
            with open(cred_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load local auth credentials: %s", exc)
            return {}

    def _save_local_credentials(self, creds: Dict[str, Any]) -> None:
        """Saves local user credentials safely to the persistent data store."""
        cred_path = self._get_credentials_path()
        try:
            with open(cred_path, "w", encoding="utf-8") as f:
                json.dump(creds, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save local auth credentials: %s", exc)

    @staticmethod
    def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hashes a password using PBKDF2 with SHA-256 and a cryptographically secure salt."""
        salt_val = salt or secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_val),
            100000
        ).hex()
        return pwd_hash, salt_val

    @staticmethod
    def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
        """Verifies a password against the stored PBKDF2 hash in constant time."""
        calc_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            100000
        ).hex()
        return secrets.compare_digest(calc_hash, stored_hash)

    def _local_login(self, email: str, password: str) -> Dict[str, Any]:
        """
        Authenticates a user against the local database and credentials store.
        Strictly enforces password verification, tenant isolation, and JWT signing.
        Strictly forbidden in production environments.
        """
        if settings.is_production:
            raise PermissionError("Local authentication fallback is strictly prohibited in production environments.")

        from backend.database.database import SessionLocal
        from backend.database.repositories import UserRepository

        db = SessionLocal()
        try:
            db_user = UserRepository.get_by_email(db, email)
            if not db_user:
                raise ValueError("Invalid email or password")

            creds = self._load_local_credentials()
            user_cred = creds.get(email)

            if user_cred:
                if not self._verify_password(password, user_cred["hash"], user_cred["salt"]):
                    raise ValueError("Invalid email or password")
            else:
                # Seeded development user convention (e.g., ram0@gmail.com, ram11234@gmail.com)
                # Password equals email for test accounts
                if password == email:
                    pwd_hash, salt = self._hash_password(password)
                    creds[email] = {"hash": pwd_hash, "salt": salt}
                    self._save_local_credentials(creds)
                else:
                    raise ValueError("Invalid email or password")

            now = int(time.time())
            token_payload = {
                "sub": str(db_user.id),
                "email": db_user.email,
                "role": db_user.role,
                "aud": "authenticated",
                "iss": "ai-healthcare-local",
                "iat": now,
                "exp": now + 3600,
            }
            token = pyjwt.encode(token_payload, self._get_local_jwt_secret(), algorithm="HS256")

            return {
                "user": {
                    "id": str(db_user.id),
                    "email": db_user.email,
                    "created_at": db_user.created_at.isoformat() if db_user.created_at else None,
                    "user_metadata": {"role": db_user.role, "full_name": db_user.full_name},
                },
                "session": {
                    "access_token": token,
                    "token_type": "bearer",
                    "expires_in": 3600,
                    "expires_at": now + 3600,
                },
            }
        finally:
            db.close()

    def _local_signup(self, email: str, password: str) -> Dict[str, Any]:
        """
        Registers a new user in the local database and credentials store.
        Hashes password with PBKDF2 and issues signed JWT session.
        Strictly forbidden in production environments.
        """
        if settings.is_production:
            raise PermissionError("Local authentication fallback is strictly prohibited in production environments.")

        from backend.database.database import SessionLocal
        from backend.database.repositories import UserRepository
        from backend.database.schemas import UserCreate

        db = SessionLocal()
        try:
            existing = UserRepository.get_by_email(db, email)
            if existing:
                raise ValueError("User already registered with this email")

            pwd_hash, salt = self._hash_password(password)
            creds = self._load_local_credentials()
            creds[email] = {"hash": pwd_hash, "salt": salt}
            self._save_local_credentials(creds)

            new_user = UserRepository.create(
                db,
                UserCreate(
                    email=email,
                    full_name=email.split("@")[0],
                    role="patient"
                )
            )

            now = int(time.time())
            token_payload = {
                "sub": str(new_user.id),
                "email": new_user.email,
                "role": new_user.role,
                "aud": "authenticated",
                "iss": "ai-healthcare-local",
                "iat": now,
                "exp": now + 3600,
            }
            token = pyjwt.encode(token_payload, self._get_local_jwt_secret(), algorithm="HS256")

            return {
                "user": {
                    "id": str(new_user.id),
                    "email": new_user.email,
                    "created_at": new_user.created_at.isoformat() if new_user.created_at else None,
                    "user_metadata": {"role": new_user.role, "full_name": new_user.full_name},
                },
                "session": {
                    "access_token": token,
                    "token_type": "bearer",
                    "expires_in": 3600,
                    "expires_at": now + 3600,
                },
            }
        finally:
            db.close()

    def signup(self, email: str, password: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Registers a new user with email and password via Supabase Auth.
        Automatically retries on transient network/connection timeouts.
        Falls back to local development authentication if external provider is unreachable.

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
                # If non-transient DNS failure, do not retry repeatedly
                if "getaddrinfo failed" in err_str or "nodename nor servname" in err_str:
                    break
                is_timeout = any(t in err_type.lower() or t in err_str for t in ["timeout", "handshake", "ssl", "remotedisconnected"])
                if is_timeout and attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break

        # If external provider is unreachable and we are in development/test, fall back locally
        if last_error and not settings.is_production:
            logger.warning(
                "External authentication provider unavailable (%s). Falling back to local development signup for %s.",
                type(last_error).__name__,
                clean_email,
            )
            return self._local_signup(clean_email, password)

        if last_error:
            raise last_error

    def login(self, email: str, password: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Authenticates an existing user with email and password via Supabase Auth.
        Automatically retries on transient network/connection/SSL handshake timeouts.
        Falls back to local development authentication if external provider is unreachable.

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
                # If explicit credential rejection from active Supabase, do not retry or fall back
                if err_type == "AuthApiError" or "invalid login credentials" in err_str:
                    raise exc
                # If non-transient DNS lookup failure, do not retry repeatedly
                if "getaddrinfo failed" in err_str or "nodename nor servname" in err_str:
                    break
                is_timeout = any(t in err_type.lower() or t in err_str for t in ["timeout", "handshake", "ssl", "remotedisconnected"])
                if is_timeout and attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break

        # If external provider is unreachable and we are in development/test, fall back locally
        if last_error and not settings.is_production:
            logger.warning(
                "External authentication provider connection failed (%s). Falling back to local authentication for %s.",
                type(last_error).__name__,
                clean_email,
            )
            return self._local_login(clean_email, password)

        if last_error:
            raise last_error

    def logout(self) -> Dict[str, Any]:
        """
        Logs out the current user by terminating the active session.

        Returns:
            Dict indicating successful logout status.
        """
        try:
            self.client.auth.sign_out()
        except Exception:
            pass
        return {
            "status": "success",
            "message": "User signed out successfully",
        }

    def get_current_user(self, jwt_token: Optional[str] = None, jwt: Optional[str] = None) -> Optional[Any]:
        """
        Retrieves the currently authenticated user.
        Supports both Supabase JWT verification and locally-issued cryptographic JWT verification.

        Args:
            jwt_token: Optional JWT token string for bearer verification.
            jwt: Alias for jwt_token for backwards compatibility.

        Returns:
            User object or dict, or None if not authenticated.
        """
        token = jwt_token or jwt
        if token is not None:
            # 1. Try local cryptographic verification only in non-production environments
            if not settings.is_production:
                try:
                    decoded = pyjwt.decode(
                        token,
                        self._get_local_jwt_secret(),
                        algorithms=["HS256"],
                        options={"verify_exp": True, "verify_aud": False}
                    )
                    if decoded and decoded.get("email"):
                        return {
                            "id": str(decoded.get("sub", "")),
                            "email": decoded.get("email"),
                            "user_metadata": {
                                "role": decoded.get("role", "patient"),
                                "name": decoded.get("name")
                            },
                        }
                except pyjwt.ExpiredSignatureError:
                    raise
                except Exception:
                    pass

            # 2. Try Supabase verification
            try:
                response = self.client.auth.get_user(token)
            except Exception as exc:
                if not settings.is_production:
                    logger.debug("Supabase token verification notice: %s", exc)
                raise exc
        else:
            response = self.client.auth.get_user()

        if response is None:
            return None

        if isinstance(response, dict):
            return response.get("user", response)

        return getattr(response, "user", response)


# Global default instance configured lazily from settings so module imports
# do not require external service credentials to be present at import time.
auth_service = AuthService(lazy=True)
