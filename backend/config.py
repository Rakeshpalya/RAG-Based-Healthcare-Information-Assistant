import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv


# ============================================================
# Load environment variables from .env
# ============================================================

root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / ".env"

load_dotenv(dotenv_path=env_path)


def normalize_supabase_url(url: Optional[str]) -> Optional[str]:
    """
    Normalizes Supabase project URL by stripping accidental /rest/v1 or trailing slashes.
    Guarantees the URL represents the clean project root for GoTrue Auth.
    """
    if not url or not str(url).strip():
        return None
    raw = str(url).strip()
    return raw.split("/rest/v1")[0].rstrip("/")


# ============================================================
# Application Settings
# ============================================================

class Settings:
    """
    Centralized configuration management for AI-Healthcare-Agent (Phases 1–9).

    Provides strictly validated development, test, and production configurations,
    preventing secret leakage into logs or frontend code while clearly
    distinguishing missing configuration from downstream service failures.
    """

    # --------------------------------------------------------
    # Application & Environment Settings
    # --------------------------------------------------------

    APP_NAME: str = os.getenv(
        "APP_NAME",
        "AI Healthcare Research & Patient Assistance Agent"
    )

    ENVIRONMENT: str = os.getenv(
        "ENVIRONMENT",
        "development"
    )

    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    HOST: str = os.getenv(
        "HOST",
        "127.0.0.1"
    )

    PORT: int = int(
        os.getenv("PORT", "8000")
    )

    # --------------------------------------------------------
    # Observability & Safety Logging Settings (Phase 9)
    # --------------------------------------------------------

    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "json"
    )

    LOG_LEVEL: str = os.getenv(
        "LOG_LEVEL",
        "INFO"
    )

    # In production, user queries in logs are masked/redacted to protect PHI/PII
    SAFE_LOG_MODE: bool = os.getenv(
        "SAFE_LOG_MODE",
        "true" if os.getenv("ENVIRONMENT", "development").lower() == "production" else "false"
    ).lower() in ("true", "1", "yes")

    # --------------------------------------------------------
    # Gemini LLM Settings
    # --------------------------------------------------------

    GEMINI_API_KEY: Optional[str] = os.getenv(
        "GEMINI_API_KEY"
    )

    GEMINI_MODEL: str = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite"
    )

    GEMINI_TEMPERATURE: float = float(
        os.getenv("GEMINI_TEMPERATURE", "0.0")
    )

    GEMINI_MAX_OUTPUT_TOKENS: int = int(
        os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024")
    )

    GEMINI_TIMEOUT_SECONDS: float = float(
        os.getenv("GEMINI_TIMEOUT_SECONDS", "30.0")
    )

    # --------------------------------------------------------
    # LLM Response Cache Settings (Phase 3.3)
    # --------------------------------------------------------

    LLM_CACHE_ENABLED: bool = os.getenv(
        "LLM_CACHE_ENABLED", "true"
    ).lower() in ("true", "1", "yes")

    LLM_CACHE_TTL_SECONDS: int = int(
        os.getenv("LLM_CACHE_TTL_SECONDS", "3600")
    )

    LLM_CACHE_MAX_ENTRIES: int = int(
        os.getenv("LLM_CACHE_MAX_ENTRIES", "1000")
    )

    # --------------------------------------------------------
    # OpenAI Settings (for HealthAI Medical Chatbot)
    # --------------------------------------------------------

    OPENAI_API_KEY: Optional[str] = os.getenv(
        "OPENAI_API_KEY"
    )

    OPENAI_MODEL: str = os.getenv(
        "OPENAI_MODEL",
        "gpt-4o-mini"
    )

    # --------------------------------------------------------
    # RAG Retrieval Settings
    # --------------------------------------------------------

    RAG_TOP_K: int = int(
        os.getenv("RAG_TOP_K", "5")
    )

    RAG_SIMILARITY_THRESHOLD: float = float(
        os.getenv(
            "RAG_SIMILARITY_THRESHOLD",
            "0.25"
        )
    )

    RAG_LATENCY_P95_BUDGET_MS: float = float(
        os.getenv("RAG_LATENCY_P95_BUDGET_MS", "3000.0")
    )

    # --------------------------------------------------------
    # Evaluation & Observability Settings (Phase 5)
    # --------------------------------------------------------

    EVALUATION_DATA_DIR: str = os.getenv(
        "EVALUATION_DATA_DIR",
        str(root_dir / "tests" / "evaluation_data")
    )

    EVALUATION_REPORTS_DIR: str = os.getenv(
        "EVALUATION_REPORTS_DIR",
        str(root_dir / "evaluation_reports")
    )

    # --------------------------------------------------------
    # Database Settings & Connection Pooling (Phase 9)
    # --------------------------------------------------------

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/ai_healthcare"
    )

    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))
    DB_CONNECT_TIMEOUT: int = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))

    # --------------------------------------------------------
    # Supabase Settings
    # --------------------------------------------------------

    SUPABASE_URL: Optional[str] = normalize_supabase_url(
        os.getenv("SUPABASE_URL")
    )

    SUPABASE_PUBLISHABLE_KEY: Optional[str] = os.getenv(
        "SUPABASE_PUBLISHABLE_KEY"
    )

    # --------------------------------------------------------
    # Security & Rate Limiting Settings (Phase 8/9/3.4)
    # --------------------------------------------------------

    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    GENERAL_REQUESTS_PER_MINUTE: int = int(os.getenv("GENERAL_REQUESTS_PER_MINUTE", "30"))
    STREAMING_REQUESTS_PER_MINUTE: int = int(os.getenv("STREAMING_REQUESTS_PER_MINUTE", "10"))
    MAX_CONCURRENT_LLM_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_LLM_REQUESTS", "10"))
    REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60.0"))
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0"))

    # --------------------------------------------------------
    # Redis Shared Infrastructure Settings (Phase 3.5)
    # --------------------------------------------------------

    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "false").lower() in ("true", "1", "yes")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD", None)
    REDIS_SOCKET_TIMEOUT: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", "2.0"))
    REDIS_CONNECT_TIMEOUT: float = float(os.getenv("REDIS_CONNECT_TIMEOUT", "2.0"))
    REDIS_MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
    REDIS_RETRY_ON_TIMEOUT: bool = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() in ("true", "1", "yes")

    # ========================================================
    # Environment Detection Helpers
    # ========================================================

    @property
    def is_production(self) -> bool:
        """True if operating in production environment."""
        env = (os.getenv("ENVIRONMENT") or self.ENVIRONMENT).lower()
        return env == "production"

    @property
    def is_test(self) -> bool:
        """True if operating in test/testing environment."""
        env = (os.getenv("ENVIRONMENT") or self.ENVIRONMENT).lower()
        return env in ("test", "testing")

    @property
    def is_development(self) -> bool:
        """True if operating in local/development environment."""
        env = (os.getenv("ENVIRONMENT") or self.ENVIRONMENT).lower()
        return env in ("development", "dev", "local")

    # ========================================================
    # Reload Settings
    # ========================================================

    @classmethod
    def reload(cls) -> None:
        """
        Reload all environment variables from the .env file.
        """
        load_dotenv(
            dotenv_path=env_path,
            override=True
        )

        # Application
        cls.APP_NAME = os.getenv(
            "APP_NAME",
            "AI Healthcare Research & Patient Assistance Agent"
        )

        cls.ENVIRONMENT = os.getenv(
            "ENVIRONMENT",
            "development"
        )

        cls.DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

        cls.HOST = os.getenv(
            "HOST",
            "127.0.0.1"
        )

        cls.PORT = int(
            os.getenv("PORT", "8000")
        )

        cls.LOG_FORMAT = os.getenv(
            "LOG_FORMAT",
            "json"
        )

        cls.LOG_LEVEL = os.getenv(
            "LOG_LEVEL",
            "INFO"
        )

        cls.SAFE_LOG_MODE = os.getenv(
            "SAFE_LOG_MODE",
            "true" if os.getenv("ENVIRONMENT", "development").lower() == "production" else "false"
        ).lower() in ("true", "1", "yes")

        # Database
        cls.DATABASE_URL = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/ai_healthcare"
        )

        cls.DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
        cls.DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
        cls.DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        cls.DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))
        cls.DB_CONNECT_TIMEOUT = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))

        # Supabase
        cls.SUPABASE_URL = normalize_supabase_url(
            os.getenv("SUPABASE_URL")
        )

        cls.SUPABASE_PUBLISHABLE_KEY = os.getenv(
            "SUPABASE_PUBLISHABLE_KEY"
        )

        # Gemini
        cls.GEMINI_API_KEY = os.getenv(
            "GEMINI_API_KEY"
        )

        cls.GEMINI_MODEL = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.5-flash-lite"
        )

        cls.GEMINI_TEMPERATURE = float(
            os.getenv("GEMINI_TEMPERATURE", "0.0")
        )

        cls.GEMINI_MAX_OUTPUT_TOKENS = int(
            os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024")
        )

        cls.GEMINI_TIMEOUT_SECONDS = float(
            os.getenv("GEMINI_TIMEOUT_SECONDS", "30.0")
        )

        # LLM Cache
        cls.LLM_CACHE_ENABLED = os.getenv(
            "LLM_CACHE_ENABLED", "true"
        ).lower() in ("true", "1", "yes")
        cls.LLM_CACHE_TTL_SECONDS = int(
            os.getenv("LLM_CACHE_TTL_SECONDS", "3600")
        )
        cls.LLM_CACHE_MAX_ENTRIES = int(
            os.getenv("LLM_CACHE_MAX_ENTRIES", "1000")
        )

        # OpenAI (for HealthAI Medical Chatbot)
        cls.OPENAI_API_KEY = os.getenv(
            "OPENAI_API_KEY"
        )

        cls.OPENAI_MODEL = os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini"
        )

        # RAG
        cls.RAG_TOP_K = int(
            os.getenv("RAG_TOP_K", "5")
        )

        cls.RAG_SIMILARITY_THRESHOLD = float(
            os.getenv(
                "RAG_SIMILARITY_THRESHOLD",
                "0.25"
            )
        )

        # Evaluation & Observability (Phase 5)
        cls.EVALUATION_DATA_DIR = os.getenv(
            "EVALUATION_DATA_DIR",
            str(root_dir / "tests" / "evaluation_data")
        )

        cls.EVALUATION_REPORTS_DIR = os.getenv(
            "EVALUATION_REPORTS_DIR",
            str(root_dir / "evaluation_reports")
        )

        cls.RAG_LATENCY_P95_BUDGET_MS = float(
            os.getenv("RAG_LATENCY_P95_BUDGET_MS", "3000.0")
        )

        # Security & Rate Limiting (Phase 3.4)
        cls.RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")
        cls.RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        cls.GENERAL_REQUESTS_PER_MINUTE = int(os.getenv("GENERAL_REQUESTS_PER_MINUTE", "30"))
        cls.STREAMING_REQUESTS_PER_MINUTE = int(os.getenv("STREAMING_REQUESTS_PER_MINUTE", "10"))
        cls.MAX_CONCURRENT_LLM_REQUESTS = int(os.getenv("MAX_CONCURRENT_LLM_REQUESTS", "10"))
        cls.REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60.0"))
        cls.LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0"))

        # Redis Shared Infrastructure (Phase 3.5)
        cls.REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() in ("true", "1", "yes")
        cls.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        cls.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        cls.REDIS_DB = int(os.getenv("REDIS_DB", "0"))
        cls.REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
        cls.REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "2.0"))
        cls.REDIS_CONNECT_TIMEOUT = float(os.getenv("REDIS_CONNECT_TIMEOUT", "2.0"))
        cls.REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
        cls.REDIS_RETRY_ON_TIMEOUT = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() in ("true", "1", "yes")

    # ========================================================
    # Gemini API Key Validation
    # ========================================================

    @classmethod
    def validate_gemini_api_key(cls) -> str:
        """
        Validate that GEMINI_API_KEY is configured.

        Returns:
            Valid Gemini API key.

        Raises:
            ValueError:
                If the API key is missing or empty.
        """
        api_key = os.environ.get("GEMINI_API_KEY", cls.GEMINI_API_KEY or "")

        if not api_key or not api_key.strip():
            raise ValueError(
                "Gemini API key is not configured. "
                "Please set 'GEMINI_API_KEY' in your .env file "
                "or environment before requesting LLM answer generation."
            )

        return api_key.strip()

    # ========================================================
    # Masked Credentials (Safe for logs & UI)
    # ========================================================

    @classmethod
    def get_masked_api_key(cls) -> str:
        """
        Return a masked representation of the Gemini API key.
        This is safe for logs or status checks.
        """
        api_key = os.environ.get("GEMINI_API_KEY", cls.GEMINI_API_KEY or "")

        if not api_key or not api_key.strip():
            return "NOT_CONFIGURED"

        clean = api_key.strip()
        if len(clean) <= 8:
            return "****"

        return f"{clean[:4]}...{clean[-4:]}"

    @classmethod
    def get_masked_database_url(cls) -> str:
        """
        Return a sanitized representation of the database URL, redacting user/password.
        """
        db_url = os.environ.get("DATABASE_URL", cls.DATABASE_URL or "")
        if not db_url or not db_url.strip():
            return "NOT_CONFIGURED"
        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:[REDACTED_PASSWORD]@", db_url)

    # ========================================================
    # Safe Sanitized Config Representation
    # ========================================================

    @classmethod
    def get_sanitized_config_dict(cls) -> Dict[str, Any]:
        """
        Returns a dictionary of all active configurations with all credentials,
        API keys, and passwords safely masked.
        """
        env = os.getenv("ENVIRONMENT") or cls.ENVIRONMENT
        return {
            "app_name": cls.APP_NAME,
            "environment": env,
            "host": cls.HOST,
            "port": cls.PORT,
            "log_format": cls.LOG_FORMAT,
            "safe_log_mode": cls.SAFE_LOG_MODE,
            "gemini_api_key_configured": bool(os.environ.get("GEMINI_API_KEY", cls.GEMINI_API_KEY or "")),
            "gemini_masked_key": cls.get_masked_api_key(),
            "gemini_model": cls.GEMINI_MODEL,
            "gemini_temperature": cls.GEMINI_TEMPERATURE,
            "gemini_max_output_tokens": cls.GEMINI_MAX_OUTPUT_TOKENS,
            "gemini_timeout_seconds": cls.GEMINI_TIMEOUT_SECONDS,
            "rag_top_k": cls.RAG_TOP_K,
            "rag_similarity_threshold": cls.RAG_SIMILARITY_THRESHOLD,
            "rag_latency_p95_budget_ms": cls.RAG_LATENCY_P95_BUDGET_MS,
            "database_url_masked": cls.get_masked_database_url(),
            "llm_cache_enabled": cls.LLM_CACHE_ENABLED,
            "llm_cache_ttl_seconds": cls.LLM_CACHE_TTL_SECONDS,
            "llm_cache_max_entries": cls.LLM_CACHE_MAX_ENTRIES,
            "rate_limit_enabled": cls.RATE_LIMIT_ENABLED,
            "rate_limit_per_minute": cls.RATE_LIMIT_PER_MINUTE,
            "redis_enabled": cls.REDIS_ENABLED,
            "redis_host": cls.REDIS_HOST,
            "redis_port": cls.REDIS_PORT,
            "redis_db": cls.REDIS_DB,
            "redis_password_configured": bool(cls.REDIS_PASSWORD),
            "debug": cls.DEBUG,
        }

    @classmethod
    def reload_from_env(cls) -> None:
        """Reload configuration values dynamically from os.environ."""
        cls.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        cls.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        cls.GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
        cls.GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))
        cls.GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "30.0"))
        cls.LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
        cls.LLM_CACHE_TTL_SECONDS = int(os.getenv("LLM_CACHE_TTL_SECONDS", "3600"))
        cls.LLM_CACHE_MAX_ENTRIES = int(os.getenv("LLM_CACHE_MAX_ENTRIES", "1000"))
        cls.REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() in ("true", "1", "yes")
        cls.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        cls.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        cls.REDIS_DB = int(os.getenv("REDIS_DB", "0"))
        cls.REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

    # ========================================================
    # Production Configuration Validation (Phase 9)
    # ========================================================

    @classmethod
    def validate_production_config(cls) -> Dict[str, Any]:
        """
        Validates that required production configuration invariants are satisfied.
        Clearly distinguishes MISSING_CONFIGURATION from runtime SERVICE_FAILURE.
        Verifies that no backend secrets are present in frontend VITE_* variables.

        Returns:
            Dict containing validation status, errors, warnings, and masked config.
        """
        errors: List[str] = []
        warnings: List[str] = []
        env = (os.getenv("ENVIRONMENT") or cls.ENVIRONMENT).lower()

        # Check frontend VITE_* variables for backend secret leakage
        for k, v in os.environ.items():
            if k.startswith("VITE_") and v:
                v_str = str(v)
                if (os.getenv("GEMINI_API_KEY") and v_str == os.getenv("GEMINI_API_KEY")) or \
                   (os.getenv("OPENAI_API_KEY") and v_str == os.getenv("OPENAI_API_KEY")) or \
                   "AIza" in v_str or "sk-" in v_str:
                    errors.append(f"CRITICAL_SECURITY_LEAK: Secret detected in frontend variable '{k}'")

        # In production mode, require essential credentials and strict invariants
        if env == "production":
            debug_val = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
            if debug_val:
                errors.append("PRODUCTION_INVARIANT_VIOLATION: DEBUG mode must be disabled in production.")

            safe_log = os.getenv("SAFE_LOG_MODE", "true").lower() in ("true", "1", "yes")
            if not safe_log:
                warnings.append("PRODUCTION_WARNING: SAFE_LOG_MODE should be enabled in production to protect PHI/PII.")

            api_key = os.environ.get("GEMINI_API_KEY", cls.GEMINI_API_KEY or "")
            if not api_key or not api_key.strip():
                warnings.append("MISSING_CONFIGURATION: GEMINI_API_KEY is not set for production answer generation.")

            db_url = os.environ.get("DATABASE_URL", cls.DATABASE_URL or "")
            if not db_url or not db_url.strip():
                errors.append("MISSING_CONFIGURATION: DATABASE_URL is required in production.")

        return {
            "valid": len(errors) == 0,
            "environment": env,
            "errors": errors,
            "warnings": warnings,
            "masked_config": cls.get_sanitized_config_dict(),
        }


# ============================================================
# Global Settings Instance
# ============================================================

settings = Settings()