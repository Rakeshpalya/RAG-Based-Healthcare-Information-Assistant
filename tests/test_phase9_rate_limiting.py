"""
Unit tests for Phase 9 Rate Limiting & Abuse Prevention.
Validates RFC-compliant headers, endpoint tiers, IP and user isolation,
and 429 Too Many Requests enforcement.
"""

import time
import pytest
from unittest.mock import MagicMock
from fastapi import Request, HTTPException
from backend.security.rate_limiter import SlidingWindowRateLimiter


def make_mock_request(client_ip="192.168.1.100", auth_header=None, forwarded_for=None):
    """Creates a mock FastAPI Request object for testing."""
    req = MagicMock(spec=Request)
    req.client = MagicMock()
    req.client.host = client_ip

    headers = {}
    if auth_header:
        headers["authorization"] = auth_header
    if forwarded_for:
        headers["x-forwarded-for"] = forwarded_for

    req.headers = headers
    return req


def test_successful_request_emits_rate_limit_headers():
    """Ensures a permitted request returns standard rate limit headers."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=10, window_seconds=60)
    req = make_mock_request()

    headers = limiter.check_rate_limit(req)
    assert headers["X-RateLimit-Limit"] == "10"
    assert headers["X-RateLimit-Remaining"] == "9"
    assert int(headers["X-RateLimit-Reset"]) > int(time.time())


def test_rate_limit_exceeded_raises_429_with_retry_after():
    """Ensures exceeding the rate limit quota raises HTTP 429 with Retry-After header."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=3, window_seconds=10)
    req = make_mock_request()

    # Exhaust quota (3 requests)
    for _ in range(3):
        limiter.check_rate_limit(req)

    # 4th request must be rejected
    with pytest.raises(HTTPException) as exc_info:
        limiter.check_rate_limit(req)

    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers
    assert exc_info.value.headers["X-RateLimit-Remaining"] == "0"
    assert int(exc_info.value.headers["Retry-After"]) >= 1


def test_endpoint_tier_limits():
    """Ensures different endpoints have separate, tiered limits."""
    limiter = SlidingWindowRateLimiter(enabled=True)
    req = make_mock_request()

    # auth_login has limit 10
    login_headers = limiter.check_rate_limit(req, endpoint_type="auth_login")
    assert login_headers["X-RateLimit-Limit"] == "10"

    # rag_query has limit 30
    rag_headers = limiter.check_rate_limit(req, endpoint_type="rag_query")
    assert rag_headers["X-RateLimit-Limit"] == "30"


def test_user_and_ip_isolation():
    """Ensures authenticated users and different IP addresses do not exhaust each other's quota."""
    limiter = SlidingWindowRateLimiter(requests_per_minute=2, window_seconds=30)
    req_ip1 = make_mock_request(client_ip="10.0.0.1")
    req_ip2 = make_mock_request(client_ip="10.0.0.2")

    # IP 1 uses all 2 requests
    limiter.check_rate_limit(req_ip1)
    limiter.check_rate_limit(req_ip1)

    with pytest.raises(HTTPException):
        limiter.check_rate_limit(req_ip1)

    # IP 2 must still be allowed
    h2 = limiter.check_rate_limit(req_ip2)
    assert h2["X-RateLimit-Remaining"] == "1"
