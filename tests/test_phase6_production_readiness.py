"""
Unit and integration tests for Phase 6 Production Containerization & Deployment Readiness.

Verifies:
1. Consolidated /health endpoint reports all subsystem statuses.
2. /health/liveness returns instant HTTP 200 alive state.
3. /health/readiness validates 744 vector count and core subsystem health non-destructively.
4. Production startup validation executes successfully on baseline environment.
5. Production startup validation detects and rejects invariant breaches safely.
6. Vector store remains strictly read-only and unmutated across all checks.
"""

import json
import pytest
import faiss
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.main import app
from backend.startup_validation import (
    validate_production_startup,
    StartupValidationError,
    EXPECTED_VECTOR_COUNT,
    EXPECTED_METADATA_COUNT,
    EXPECTED_EMBEDDING_DIM,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_health_consolidated_endpoint(client):
    """Verify GET /health returns operational readiness with all subsystem fields."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "healthy"
    assert data["service"] == "AI-Healthcare-Agent"
    assert data["vector_store"] == "ready"
    assert data["embedding_service"] == "ready"
    assert data["safety_engine"] == "ready"
    assert "database" in data
    assert "llm_config" in data


def test_health_liveness_probe(client):
    """Verify GET /health/liveness returns immediate 200 alive status."""
    res = client.get("/health/liveness")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "alive"
    assert data["service"] == "AI-Healthcare-Agent"


def test_health_readiness_probe_success(client):
    """Verify GET /health/readiness returns 200 and confirms exact 744 vector count."""
    res = client.get("/health/readiness")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "ready"
    assert "checks" in data
    assert data["checks"]["vector_store"]["status"] == "ready"
    assert data["checks"]["vector_store"]["vector_count"] == 744
    assert data["checks"]["embedding_service"]["status"] == "ready"
    assert data["checks"]["embedding_service"]["dimension"] == 384
    assert data["checks"]["safety_engine"]["status"] == "ready"


def test_health_readiness_probe_fails_on_unready_vector_store(client):
    """Verify GET /health/readiness returns 503 if vector store count is unready."""
    mock_vs = MagicMock()
    mock_vs.count.return_value = 0  # uninitialized

    with patch("backend.services.vector_store_service.get_vector_store_service", return_value=mock_vs):
        res = client.get("/health/readiness")
        assert res.status_code == 503
        data = res.json()
        assert data["status"] == "not_ready"
        assert data["checks"]["vector_store"]["status"] == "degraded"


def test_startup_validation_success():
    """Verify production startup validation passes on pristine repository state."""
    result = validate_production_startup()
    assert result["status"] == "PASS"
    assert result["passed"] is True
    assert result["vector_count"] == EXPECTED_VECTOR_COUNT
    assert result["metadata_count"] == EXPECTED_METADATA_COUNT
    assert result["embedding_dimension"] == EXPECTED_EMBEDDING_DIM
    assert len(result["errors"]) == 0


def test_startup_validation_detects_vector_count_mismatch(tmp_path):
    """Verify startup validation detects corrupted vector store count and raises StartupValidationError."""
    # Create fake corrupted vector store in tmp_path
    corrupt_dir = tmp_path / "data" / "vector_store"
    corrupt_dir.mkdir(parents=True)

    import faiss
    corrupt_idx = faiss.IndexFlatIP(384)
    # Add 10 dummy vectors instead of 744
    import numpy as np
    dummy_vecs = np.zeros((10, 384), dtype=np.float32)
    corrupt_idx.add(dummy_vecs)
    faiss.write_index(corrupt_idx, str(corrupt_dir / "index.faiss"))

    with open(corrupt_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"count": 10, "records": [{"id": i} for i in range(10)]}, f)

    with pytest.raises(StartupValidationError) as exc_info:
        validate_production_startup(root_dir=tmp_path)

    err_msg = str(exc_info.value)
    assert "FAISS invariant breach" in err_msg or "Metadata invariant breach" in err_msg


def test_startup_validation_vector_store_read_only():
    """Verify startup validation causes zero mutations to FAISS index or metadata."""
    faiss_path = Path("data/vector_store/index.faiss")
    meta_path = Path("data/vector_store/metadata.json")

    idx_before = faiss.read_index(str(faiss_path)).ntotal
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_before = json.load(f)["count"]

    assert idx_before == 744
    assert meta_before == 744

    # Run startup validation multiple times
    for _ in range(3):
        res = validate_production_startup()
        assert res["passed"] is True

    idx_after = faiss.read_index(str(faiss_path)).ntotal
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_after = json.load(f)["count"]

    assert idx_after == idx_before == 744
    assert meta_after == meta_before == 744
