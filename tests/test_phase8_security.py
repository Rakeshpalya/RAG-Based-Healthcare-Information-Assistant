import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.security import rate_limiter

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def test_oversized_payload_rejection():
    """Verify that request with oversized content-length is rejected with HTTP 413."""
    headers = {"Content-Length": str(15 * 1024 * 1024)}  # 15 MB
    res = client.post("/rag/retrieve", json={"question": "What is hypertension?"}, headers=headers)
    assert res.status_code == 413
    assert "exceeds maximum allowed limit" in res.json()["detail"]


def test_security_headers_present():
    """Verify security headers (X-Content-Type-Options, X-Frame-Options, etc.) on responses."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert res.headers.get("X-XSS-Protection") == "1; mode=block"
    assert res.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


def test_malformed_json_rejection():
    """Verify that malformed JSON payloads return 422 Unprocessable Entity."""
    res = client.post(
        "/rag/retrieve",
        content="{'invalid': json, missing_quote}",
        headers={"Content-Type": "application/json"}
    )
    assert res.status_code == 422


def test_rag_query_empty_question_rejected():
    """Verify that empty question string violates Pydantic min_length=1."""
    res = client.post("/rag/query", json={"question": ""})
    assert res.status_code == 422


def test_rag_query_extremely_long_question_rejected():
    """Verify that question exceeding max_length (2000 chars) is rejected."""
    long_question = "What is hypertension? " * 150  # ~3300 characters
    res = client.post("/rag/query", json={"question": long_question})
    assert res.status_code == 422


def test_rag_query_invalid_top_k_rejected():
    """Verify that invalid top_k (< 1 or > 50) is rejected."""
    res_low = client.post("/rag/query", json={"question": "What is hypertension?", "top_k": 0})
    assert res_low.status_code == 422

    res_high = client.post("/rag/query", json={"question": "What is hypertension?", "top_k": 100})
    assert res_high.status_code == 422


def test_rag_query_invalid_similarity_threshold_rejected():
    """Verify that invalid similarity_threshold (< -1.0 or > 1.0) is rejected."""
    res = client.post("/rag/query", json={"question": "What is hypertension?", "similarity_threshold": 2.5})
    assert res.status_code == 422


def test_unexpected_fields_forbidden():
    """Verify that extra unexpected fields in request body are rejected with 422 (extra='forbid')."""
    res = client.post("/rag/query", json={
        "question": "What is hypertension?",
        "malicious_extra_field": "injected_value"
    })
    assert res.status_code == 422


def test_rate_limiter_throttling():
    """Verify that exceeding rate limit triggers HTTP 429 with Retry-After header."""
    # Temporarily set tight limit for testing
    orig_rpm = rate_limiter.requests_per_minute
    try:
        rate_limiter.requests_per_minute = 3
        # Send 3 allowed requests
        for _ in range(3):
            res = client.post("/rag/retrieve", json={"question": "What is hypertension?"})
            assert res.status_code == 200

        # 4th request must be throttled
        res_blocked = client.post("/rag/retrieve", json={"question": "What is hypertension?"})
        assert res_blocked.status_code == 429
        assert "Retry-After" in res_blocked.headers
        assert "Rate limit exceeded" in res_blocked.json()["detail"]
    finally:
        rate_limiter.requests_per_minute = orig_rpm


def test_auth_token_format_enforcement():
    """Verify that malformed Authorization header is rejected."""
    # Non-bearer scheme
    res = client.get("/conversations", headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert res.status_code == 401
    assert "Bearer token required" in res.json()["detail"]

    # Empty token
    res = client.get("/conversations", headers={"Authorization": "Bearer "})
    assert res.status_code == 401
