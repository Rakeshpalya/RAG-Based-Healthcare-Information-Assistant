"""
Unit tests for Phase 9 Production Configuration System.
Validates environment separation, credential masking, production validation,
and prevention of secret exposure in frontend VITE_* variables.
"""

import os
import pytest
from backend.config import Settings, settings


def test_environment_profile_helpers(monkeypatch):
    """Validates is_production, is_test, and is_development helper properties."""
    s = Settings()

    monkeypatch.setenv("ENVIRONMENT", "production")
    assert s.is_production is True
    assert s.is_test is False
    assert s.is_development is False

    monkeypatch.setenv("ENVIRONMENT", "test")
    assert s.is_production is False
    assert s.is_test is True
    assert s.is_development is False

    monkeypatch.setenv("ENVIRONMENT", "development")
    assert s.is_production is False
    assert s.is_test is False
    assert s.is_development is True


def test_credential_masking(monkeypatch):
    """Ensures API keys and database URLs are masked in logs and status objects."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyFakeSecret1234567890abcdef")
    monkeypatch.setenv("DATABASE_URL", "postgresql://myuser:supersecretpass@db.internal:5432/healthcare_db")

    masked_key = Settings.get_masked_api_key()
    assert "supersecretpass" not in masked_key
    assert masked_key.startswith("AIza")
    assert masked_key.endswith("cdef")
    assert "..." in masked_key

    masked_db = Settings.get_masked_database_url()
    assert "supersecretpass" not in masked_db
    assert "[REDACTED_PASSWORD]" in masked_db
    assert "db.internal:5432/healthcare_db" in masked_db


def test_sanitized_config_dict(monkeypatch):
    """Ensures get_sanitized_config_dict never exposes unmasked credentials."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSySecretTestingKey99999")
    monkeypatch.setenv("DATABASE_URL", "postgresql://prod_admin:topsecret123@prod-cluster:5432/prod_db")

    config_dict = Settings.get_sanitized_config_dict()

    assert "topsecret123" not in str(config_dict)
    assert "AIzaSySecretTestingKey99999" not in str(config_dict)
    assert config_dict["gemini_api_key_configured"] is True
    assert "[REDACTED_PASSWORD]" in config_dict["database_url_masked"]


def test_prevent_secret_leak_in_vite_vars(monkeypatch):
    """Flags critical error if any VITE_* variable contains a backend secret."""
    secret_key = "AIzaSyDangerousSecretKey123456"
    monkeypatch.setenv("GEMINI_API_KEY", secret_key)
    monkeypatch.setenv("VITE_ACCIDENTAL_KEY", secret_key)

    validation = Settings.validate_production_config()
    assert validation["valid"] is False
    assert any("CRITICAL_SECURITY_LEAK" in err for err in validation["errors"])


def test_distinguish_missing_config_from_service_failure(monkeypatch):
    """Verifies that missing configuration is clearly flagged with MISSING_CONFIGURATION."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("ENVIRONMENT", "production")

    validation = Settings.validate_production_config()
    # In production without key, it must emit a clear MISSING_CONFIGURATION warning
    assert any("MISSING_CONFIGURATION" in w for w in validation["warnings"])
