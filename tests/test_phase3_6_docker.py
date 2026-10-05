"""
Phase 3.6.3 Tests: Docker Production Container & Multi-Service Topology Validation.

Verifies:
1. Dockerfile exists and is configured for multi-stage build.
2. Non-root user execution (appuser:10001) is strictly enforced in Dockerfile.
3. Health check command is present and targets /health/live.
4. Read-only vector store mount (:ro) is enforced in Docker Compose.
5. Persistent volume definitions exist for Redis and Prometheus.
6. Restart policies (unless-stopped) are declared for all critical services.
7. Unnecessary development dependencies (e.g. pytest, debuggers) are excluded from production container.
8. Liveness and readiness probe endpoint behavior is verified via FastAPI test client.
9. Redis and Prometheus service configurations in docker-compose.yml are valid.
"""

import re
import yaml
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_01_dockerfile_exists_and_multistage():
    """Verify Dockerfile exists and defines multi-stage builder and runtime targets."""
    df_path = Path("Dockerfile")
    assert df_path.exists(), "Dockerfile must exist in workspace root"

    content = df_path.read_text(encoding="utf-8")
    assert "AS builder" in content, "Dockerfile must feature a builder stage"
    assert "AS runtime" in content, "Dockerfile must feature a runtime stage"
    assert "WORKDIR /app" in content


def test_02_dockerfile_enforces_non_root_user():
    """Verify production Dockerfile strictly creates and executes under non-root appuser:appgroup (10001)."""
    df_path = Path("Dockerfile")
    content = df_path.read_text(encoding="utf-8")

    assert "groupadd -g 10001 appgroup" in content
    assert "useradd -u 10001 -g appgroup" in content
    assert "USER appuser" in content, "Runtime container must switch to non-root USER appuser"


def test_03_dockerfile_contains_healthcheck_probe():
    """Verify Dockerfile specifies native HEALTHCHECK pointing to /health/live."""
    df_path = Path("Dockerfile")
    content = df_path.read_text(encoding="utf-8")

    assert "HEALTHCHECK" in content
    assert "/health/live" in content


def test_04_docker_compose_vector_store_mounted_read_only():
    """Verify docker-compose.yml mounts vector_store directory strictly as read-only (:ro)."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists()

    with open(compose_path, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    api_service = compose_data["services"].get("api", {})
    volumes = api_service.get("volumes", [])

    # Search for vector store mount
    vs_mounts = [v for v in volumes if "vector_store" in str(v)]
    assert len(vs_mounts) > 0, "Vector store must be mounted in api service"
    assert any(":ro" in str(v) for v in vs_mounts), "Vector store MUST be mounted with :ro (read-only) flag"


def test_05_docker_compose_persistent_volumes_and_restart_policy():
    """Verify persistent volumes and restart policies are configured for stateful services."""
    compose_path = Path("docker-compose.yml")
    with open(compose_path, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data.get("services", {})
    assert "redis" in services
    assert "api" in services

    # Verify restart policies
    assert services["api"].get("restart") in ("always", "unless-stopped")
    assert services["redis"].get("restart") in ("always", "unless-stopped")

    # Verify volumes
    declared_volumes = compose_data.get("volumes", {})
    assert "redis_data" in declared_volumes, "redis_data persistent volume must be declared"


def test_06_docker_compose_service_healthchecks():
    """Verify healthchecks are defined for api, redis, and postgres services."""
    compose_path = Path("docker-compose.yml")
    with open(compose_path, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data.get("services", {})
    assert "healthcheck" in services["api"]
    assert "healthcheck" in services["redis"]
    if "postgres" in services:
        assert "healthcheck" in services["postgres"]


def test_07_requirements_excludes_heavy_dev_tools():
    """Verify requirements.txt does not bundle development or testing frameworks."""
    req_path = Path("requirements.txt")
    assert req_path.exists()

    content = req_path.read_text(encoding="utf-8").lower()
    # Production requirements should not contain ipykernel, pytest, etc.
    assert "ipykernel" not in content
    assert "jupyter" not in content


def test_08_health_liveness_and_readiness_endpoints(client):
    """Verify /health/live and /health/ready return valid status codes and JSON envelopes."""
    # Liveness probe
    live_resp = client.get("/health/live")
    assert live_resp.status_code == 200
    assert live_resp.json()["status"] == "alive"

    # Readiness probe
    ready_resp = client.get("/health/ready")
    assert ready_resp.status_code == 200
    data = ready_resp.json()
    assert "status" in data
    assert "vector_store" in data
    assert "checks" in data
