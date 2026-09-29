import pytest
from scripts.load_test import run_benchmark_endpoint
from fastapi.testclient import TestClient
from backend.main import app
from backend.security import rate_limiter

client = TestClient(app)


def test_health_endpoint_load_concurrency():
    """Verify that GET /health handles concurrent traffic with 0% error rate."""
    # Warm up dependencies
    client.get("/health")
    orig_rpm = rate_limiter.requests_per_minute
    rate_limiter.requests_per_minute = 10000
    try:
        res = run_benchmark_endpoint(
            client=client,
            endpoint="/health",
            method="GET",
            concurrency=5,
            total_requests=20
        )
        assert res["failed_requests"] == 0
        assert res["error_rate_percent"] == 0.0
        assert res["status_code_breakdown"].get(200, 0) == 20
        assert res["p95_latency_ms"] < 250.0  # P95 health check must be snappy once warm
    finally:
        rate_limiter.requests_per_minute = orig_rpm
        rate_limiter.reset()


def test_rag_retrieve_load_concurrency():
    """Verify that POST /rag/retrieve handles concurrent retrieval within SLA budget."""
    orig_rpm = rate_limiter.requests_per_minute
    rate_limiter.requests_per_minute = 10000
    try:
        res = run_benchmark_endpoint(
            client=client,
            endpoint="/rag/retrieve",
            method="POST",
            payload={"question": "What are hypertension symptoms?"},
            concurrency=5,
            total_requests=10
        )
        assert res["failed_requests"] == 0
        assert res["error_rate_percent"] == 0.0
        assert res["status_code_breakdown"].get(200, 0) == 10
        assert res["p95_latency_ms"] < 1500.0  # Well within 1500ms latency budget
    finally:
        rate_limiter.requests_per_minute = orig_rpm
        rate_limiter.reset()
