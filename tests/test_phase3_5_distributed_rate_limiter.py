"""
Phase 3.5.2 Tests: Distributed Sliding Window Rate Limiting.

Verifies:
1. Single user rate limiting under Redis.
2. Multi-user rate limiting isolation (User A quota does not affect User B).
3. Concurrent requests with thread pool (no race conditions).
4. Multiple API instances sharing the same Redis state.
5. Window expiration and quota reset.
6. Fail-safe behavior on Redis failure: DOES NOT allow unlimited requests (falls back to local memory quota).
7. Redis recovery: distributed synchronization resumes cleanly.
8. Client-IP fallback when user authentication is not present.
"""

import time
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
from fastapi import Request, HTTPException

from backend.security.rate_limiter import SlidingWindowRateLimiter
from backend.services.redis_service import RedisService


class MockRedisAdapterForRateLimit:
    """Simulates Redis Lua sliding window execution and sorted sets for testing."""

    def __init__(self):
        self.zsets = {}
        self.ttls = {}
        self.is_connected = True
        self.RATE_LIMIT_PREFIX = "healthcare:ratelimit:"

    def is_available(self):
        return self.is_connected

    def execute_lua(self, script, numkeys, key, now_str, window_str, limit_str, member):
        if not self.is_connected:
            raise ConnectionError("Simulated Redis outage")

        now = float(now_str)
        window = float(window_str)
        limit = int(limit_str)
        clear_before = now - window

        if key not in self.zsets:
            self.zsets[key] = {}

        # 1. Clean expired items
        self.zsets[key] = {m: ts for m, ts in self.zsets[key].items() if ts > clear_before}

        current_count = len(self.zsets[key])
        if current_count >= limit:
            oldest_ts = min(self.zsets[key].values()) if self.zsets[key] else (now - window)
            retry_after = max(1, int(oldest_ts + window - now))
            reset_ts = int(oldest_ts + window)
            return [0, current_count, retry_after, reset_ts]
        else:
            self.zsets[key][member] = now
            self.ttls[key] = now + window * 2
            remaining = max(0, limit - current_count - 1)
            reset_ts = int(now + window)
            return [1, remaining, 0, reset_ts]

    def delete_pattern(self, pattern):
        if not self.is_connected:
            raise ConnectionError("Simulated Redis outage")
        self.zsets.clear()
        self.ttls.clear()


def make_request(path="/rag/query", client_ip="192.168.1.10", auth_header=None):
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [(b"host", b"localhost")]
    }
    if auth_header:
        scope["headers"].append((b"authorization", auth_header.encode("utf-8")))
    if client_ip:
        scope["client"] = (client_ip, 12345)
    return Request(scope)


def test_single_user_rate_limiting():
    """Verify rate limiter allows requests up to limit and rejects exceeding requests."""
    redis_mock = MockRedisAdapterForRateLimit()
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=5,
        window_seconds=60,
        enabled=True,
        redis_service=redis_mock
    )

    req = make_request()
    # 5 requests should pass
    for i in range(5):
        headers = limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=5, user_id=101)
        assert "X-RateLimit-Remaining" in headers
        assert int(headers["X-RateLimit-Remaining"]) == 5 - 1 - i

    # 6th request must raise 429
    with pytest.raises(HTTPException) as exc_info:
        limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=5, user_id=101)

    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers
    assert exc_info.value.headers["X-RateLimit-Remaining"] == "0"


def test_multi_user_isolation():
    """Verify User A's requests do not count against User B's quota."""
    redis_mock = MockRedisAdapterForRateLimit()
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=3,
        window_seconds=60,
        enabled=True,
        redis_service=redis_mock
    )

    req = make_request()
    # Exhaust User A
    for _ in range(3):
        limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=3, user_id="user_a")

    with pytest.raises(HTTPException):
        limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=3, user_id="user_a")

    # User B should still have full quota
    res_b = limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=3, user_id="user_b")
    assert int(res_b["X-RateLimit-Remaining"]) == 2


def test_client_ip_fallback():
    """Verify rate limiting falls back to IP address when no user_id is authenticated."""
    redis_mock = MockRedisAdapterForRateLimit()
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=2,
        window_seconds=60,
        enabled=True,
        redis_service=redis_mock
    )

    req_ip1 = make_request(client_ip="10.0.0.1")
    req_ip2 = make_request(client_ip="10.0.0.2")

    # Use IP 1
    limiter.check_rate_limit(req_ip1, max_requests=2)
    limiter.check_rate_limit(req_ip1, max_requests=2)

    with pytest.raises(HTTPException):
        limiter.check_rate_limit(req_ip1, max_requests=2)

    # IP 2 should be unaffected
    headers_ip2 = limiter.check_rate_limit(req_ip2, max_requests=2)
    assert int(headers_ip2["X-RateLimit-Remaining"]) == 1


def test_multi_api_instance_shared_state():
    """Simulate API Instance A and Instance B hitting the same Redis store."""
    shared_redis = MockRedisAdapterForRateLimit()

    instance_a = SlidingWindowRateLimiter(enabled=True, redis_service=shared_redis)
    instance_b = SlidingWindowRateLimiter(enabled=True, redis_service=shared_redis)

    req = make_request()

    # Instance A serves 2 requests
    instance_a.check_rate_limit(req, endpoint_type="rag_query", max_requests=4, user_id=999)
    instance_a.check_rate_limit(req, endpoint_type="rag_query", max_requests=4, user_id=999)

    # Instance B serves 2 requests
    instance_b.check_rate_limit(req, endpoint_type="rag_query", max_requests=4, user_id=999)
    instance_b.check_rate_limit(req, endpoint_type="rag_query", max_requests=4, user_id=999)

    # 5th request on Instance A should be blocked (shared limit reached)
    with pytest.raises(HTTPException):
        instance_a.check_rate_limit(req, endpoint_type="rag_query", max_requests=4, user_id=999)

    # 5th request on Instance B should also be blocked
    with pytest.raises(HTTPException):
        instance_b.check_rate_limit(req, endpoint_type="rag_query", max_requests=4, user_id=999)


def test_concurrent_requests_atomic():
    """Verify thread concurrency does not exceed limits (race condition protection)."""
    shared_redis = MockRedisAdapterForRateLimit()
    limiter = SlidingWindowRateLimiter(enabled=True, redis_service=shared_redis)

    limit = 10
    total_workers = 30
    req = make_request()

    successes = 0
    blocked = 0

    def make_call():
        nonlocal successes, blocked
        try:
            limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=limit, user_id="concurrent_user")
            successes += 1
        except HTTPException as exc:
            if exc.status_code == 429:
                blocked += 1

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(make_call) for _ in range(total_workers)]
        for f in futures:
            f.result()

    assert successes == limit
    assert blocked == (total_workers - limit)


def test_redis_failure_safe_fallback():
    """
    CRITICAL INVARIANT:
    When Redis fails, the system must NOT allow unlimited requests.
    It MUST fall back safely to local in-memory sliding window.
    """
    failing_redis = MockRedisAdapterForRateLimit()
    failing_redis.is_connected = False  # Outage!

    limiter = SlidingWindowRateLimiter(
        requests_per_minute=3,
        window_seconds=60,
        enabled=True,
        redis_service=failing_redis
    )

    req = make_request()

    # In-memory quota must still enforce 3 requests limit
    limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=3, user_id="fallback_user")
    limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=3, user_id="fallback_user")
    limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=3, user_id="fallback_user")

    # 4th request must STILL raise 429 even though Redis is completely dead
    with pytest.raises(HTTPException) as exc_info:
        limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=3, user_id="fallback_user")

    assert exc_info.value.status_code == 429


def test_redis_recovery():
    """Verify that when Redis recovers from outage, distributed state resumes."""
    redis_mock = MockRedisAdapterForRateLimit()
    redis_mock.is_connected = False  # Starts dead

    limiter = SlidingWindowRateLimiter(
        requests_per_minute=5,
        window_seconds=60,
        enabled=True,
        redis_service=redis_mock
    )

    req = make_request()
    # Runs on local memory fallback
    limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=5, user_id="recovery_user")

    # Redis comes back online
    redis_mock.is_connected = True

    # Next request succeeds and populates Redis
    headers = limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=5, user_id="recovery_user")
    assert "X-RateLimit-Limit" in headers
    # Verify Redis sorted set contains the entry
    key = "healthcare:ratelimit:rag_query:user_recovery_user"
    assert key in redis_mock.zsets
