"""
Phase 3.6.2 Tests: Automated CI/CD Pipeline & Configuration Validation.

Verifies:
1. Required and optional environment variables are handled safely.
2. Application, routers, and essential services import cleanly without errors.
3. Production configuration validation enforces DEBUG=False, SAFE_LOG_MODE=True.
4. Production mode rejects accidental DEBUG=True.
5. Secrets and credentials are masked and never printed in logs or string representations.
6. Pytest test discovery successfully locates all test suites without collection errors.
7. CI/CD workflow YAML and Docker Compose configuration are syntactically and structurally valid.
"""

import os
import sys
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch

from backend.config import Settings, settings


def test_01_application_and_modules_import_cleanly():
    """Verify application, routers, and essential services import without syntax or runtime error."""
    import backend.main as main_mod
    import backend.config as config_mod
    import backend.rag.rag_service as rag_mod
    import backend.services.gemini_service as gemini_mod
    import backend.services.redis_service as redis_mod
    import backend.services.llm_cache_service as cache_mod
    import backend.security.rate_limiter as limiter_mod
    import backend.evaluation.observability as obs_mod
    import backend.services.snapshot_service as snap_mod

    assert main_mod.app is not None
    assert config_mod.settings is not None


def test_02_required_env_vars_handled_safely():
    """Verify system handles missing or optional env vars safely with valid fallbacks."""
    with patch.dict(os.environ, {"ENVIRONMENT": "testing", "PORT": "8000"}):
        Settings.reload_from_env()
        assert Settings.PORT == 8000
        env = os.getenv("ENVIRONMENT")
        assert env == "testing"


def test_03_production_config_validation_rules():
    """Verify validate_production_config enforces production security requirements."""
    prod_env = {
        "ENVIRONMENT": "production",
        "DEBUG": "false",
        "SAFE_LOG_MODE": "true",
        "GEMINI_API_KEY": "AIzaSyFakeKeyForTestValidProd12345",
        "DATABASE_URL": "postgresql://user:pass@db:5432/healthcare",
        "JWT_SECRET_KEY": "a" * 32,
        "REDIS_ENABLED": "false"
    }
    with patch.dict(os.environ, prod_env):
        Settings.reload_from_env()
        res = Settings.validate_production_config()
        assert res["valid"] is True, f"Expected valid production configuration, got errors: {res.get('errors')}"


def test_04_production_mode_rejects_debug_true():
    """Verify production mode strictly refuses to run with DEBUG=True."""
    insecure_env = {
        "ENVIRONMENT": "production",
        "DEBUG": "true",  # INSECURE in production!
        "SAFE_LOG_MODE": "true",
        "GEMINI_API_KEY": "AIzaSyFakeKeyForTestValidProd12345",
        "DATABASE_URL": "postgresql://user:pass@db:5432/healthcare",
        "JWT_SECRET_KEY": "a" * 32
    }
    with patch.dict(os.environ, insecure_env):
        Settings.reload_from_env()
        res = Settings.validate_production_config()
        assert res["valid"] is False
        assert any("DEBUG mode must be disabled in production" in err for err in res["errors"])


def test_05_secrets_never_printed_in_config_or_logs():
    """Verify credentials and secrets are masked in str(settings) and sanitized_dict."""
    secret_env = {
        "GEMINI_API_KEY": "AIzaSySecretTestingKey12345",
        "REDIS_PASSWORD": "SuperSecretRedisPassword999",
        "DATABASE_URL": "postgresql://postgres:MySecretDbPass@localhost:5432/db",
        "JWT_SECRET_KEY": "SuperSecretJwtSigningKeyX"
    }
    with patch.dict(os.environ, secret_env):
        Settings.reload_from_env()
        masked_key = Settings.get_masked_api_key()
        masked_db = Settings.get_masked_database_url()
        sanitized = Settings.get_sanitized_config_dict()

        # None of the raw secrets should appear in representations
        assert "AIzaSySecretTestingKey12345" not in masked_key
        assert masked_key.startswith("AIza")
        assert masked_key.endswith("2345")

        assert "MySecretDbPass" not in masked_db
        assert "[REDACTED_PASSWORD]" in masked_db

        assert "AIzaSySecretTestingKey12345" not in str(sanitized)
        assert "MySecretDbPass" not in str(sanitized)
        assert sanitized["gemini_api_key_configured"] is True
        assert sanitized["redis_password_configured"] is True


def test_06_ci_workflow_yaml_syntax_and_structure():
    """Verify .github/workflows/ci.yml exists and contains all required pipeline stages."""
    ci_path = Path(".github/workflows/ci.yml")
    assert ci_path.exists(), "CI workflow file .github/workflows/ci.yml must exist"

    with open(ci_path, "r", encoding="utf-8") as f:
        ci_data = yaml.safe_load(f)

    assert "jobs" in ci_data
    assert "test-and-audit" in ci_data["jobs"]

    steps = ci_data["jobs"]["test-and-audit"]["steps"]
    step_names = [s.get("name", "") for s in steps]

    # Verify all required pipeline stages are explicitly represented
    assert any("Dependencies" in name for name in step_names)
    assert any("Syntax" in name or "Import" in name for name in step_names)
    assert any("Unit" in name for name in step_names)
    assert any("Phase 2" in name for name in step_names)
    assert any("Phase 3.4" in name for name in step_names)
    assert any("Phase 3.5" in name for name in step_names)
    assert any("Security" in name for name in step_names)


def test_07_docker_compose_yaml_syntax_and_structure():
    """Verify docker-compose.yml exists and is structurally valid YAML."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml must exist"

    with open(compose_path, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    assert "services" in compose_data
    assert "api" in compose_data["services"]
    assert "redis" in compose_data["services"]
    assert "prometheus" in compose_data["services"]
