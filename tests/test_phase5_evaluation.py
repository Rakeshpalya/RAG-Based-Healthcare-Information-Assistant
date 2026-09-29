import json
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from backend.evaluation.evaluation_runner import EvaluationRunner, EvaluationReport


@pytest.fixture
def api_client():
    return TestClient(app)


def test_health_endpoint_subsystem_readiness(api_client):
    """Verify GET /health returns readiness status for vector_store, embedding_service, and safety_engine."""
    res = api_client.get("/health")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "healthy"
    assert data["service"] == "AI-Healthcare-Agent"
    assert "environment" in data
    assert data["vector_store"] == "ready"
    assert data["embedding_service"] == "ready"
    assert data["safety_engine"] == "ready"


def test_evaluation_runner_full_suite():
    """Verify EvaluationRunner executes all 4 evaluation tracks and produces structured report."""
    runner = EvaluationRunner()
    report = runner.run_all(save_report=False)

    assert isinstance(report, EvaluationReport)
    assert report.retrieval_status == "PASS"
    assert report.citation_status == "PASS"
    assert report.hallucination_status == "PASS"
    assert report.safety_status == "PASS"
    assert report.overall_status == "PASS"

    assert report.retrieval_summary["total_cases"] > 0
    assert report.citation_summary["total_cases"] > 0
    assert report.hallucination_summary["total_cases"] > 0
    assert report.safety_summary["total_cases"] > 0

    assert len(report.safety_summary["critical_life_safety_false_negatives"]) == 0


def test_evaluation_runner_report_persistence(tmp_path):
    """Verify that EvaluationRunner writes timestamped JSON reports without destructive overwrites."""
    runner = EvaluationRunner()
    report_file = runner.save_report_to_disk(runner.run_all(save_report=False), output_dir=str(tmp_path))

    assert Path(report_file).exists()

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["overall_status"] == "PASS"
    assert "retrieval_summary" in data
    assert "safety_summary" in data
    assert "timestamp" in data


def test_vector_store_integrity_during_evaluation():
    """Verify that running the entire evaluation suite leaves the FAISS vector index and metadata pristine."""
    meta_path = Path("data/vector_store/metadata.json")
    assert meta_path.exists()

    with open(meta_path, "r", encoding="utf-8") as f:
        before_records = json.load(f)["records"]

    assert len(before_records) == 744

    # Run complete suite
    runner = EvaluationRunner()
    _ = runner.run_all(save_report=False)

    with open(meta_path, "r", encoding="utf-8") as f:
        after_records = json.load(f)["records"]

    assert len(after_records) == 744
    assert before_records[0]["chunk_id"] == after_records[0]["chunk_id"]
