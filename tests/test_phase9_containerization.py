"""
Unit tests for Phase 9 Production Containerization.
Validates Dockerfile, docker-compose.yml, and .dockerignore for security,
healthchecks, multi-stage builds, non-root execution, and read-only vector store mounts.
"""

import os
import yaml
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_dockerfile_security_and_structure():
    """Validates Dockerfile multi-stage build, non-root user, and healthcheck."""
    dockerfile_path = os.path.join(REPO_ROOT, "Dockerfile")
    assert os.path.exists(dockerfile_path), "Dockerfile must exist"

    with open(dockerfile_path, "r", encoding="utf-8") as fh:
        content = fh.read()

    # Multi-stage build
    assert "AS builder" in content
    assert "AS runtime" in content

    # Non-root user execution
    assert "USER appuser" in content or "USER 10001" in content

    # Container healthcheck
    assert "HEALTHCHECK" in content
    assert "/health" in content


def test_docker_compose_configuration():
    """Validates docker-compose.yml services, healthchecks, and vector store read-only mount."""
    compose_path = os.path.join(REPO_ROOT, "docker-compose.yml")
    assert os.path.exists(compose_path), "docker-compose.yml must exist"

    with open(compose_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    assert "services" in data
    services = data["services"]

    # Postgres service
    assert "postgres" in services
    assert "healthcheck" in services["postgres"]

    # API service
    assert "api" in services
    api = services["api"]
    assert "healthcheck" in api
    assert "depends_on" in api

    # Read-only vector store invariant
    volumes = api.get("volumes", [])
    vector_mounts = [v for v in volumes if "vector_store" in v]
    assert len(vector_mounts) > 0, "Vector store volume must be mounted"
    assert any(":ro" in v for v in vector_mounts), "Vector store must be mounted strictly read-only (:ro)"


def test_dockerignore_excludes_secrets_and_caches():
    """Ensures .dockerignore excludes virtualenvs, test suites, and .env files."""
    dockerignore_path = os.path.join(REPO_ROOT, ".dockerignore")
    assert os.path.exists(dockerignore_path), ".dockerignore must exist"

    with open(dockerignore_path, "r", encoding="utf-8") as fh:
        content = fh.read()

    assert ".env" in content
    assert "venv/" in content
    assert "tests/" in content
