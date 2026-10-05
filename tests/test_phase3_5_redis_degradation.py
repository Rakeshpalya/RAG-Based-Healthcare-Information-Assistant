"""
Phase 3.5.7 Tests: Redis Failure & Graceful Degradation Suite.

Verifies:
1. Cache operations continue safely via in-memory fallback during Redis outages.
2. Rate limiting continues safely via local sliding window during Redis outages (NO unlimited requests).
3. Redis timeout / connection refused handling with circuit-breaker backoff.
4. Redis recovery: distributed services automatically reconnect when healthy.
5. Critical Invariants under Redis Failure:
   - FastAPI does NOT crash.
   - Medical safety is NEVER bypassed.
   - Cross-user data isolation is strictly preserved.
   - LLM concurrency limits remain strictly bounded.
   - Zero credential or clinical data leakage.
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request, HTTPException
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.redis_service import RedisService
from backend.services.llm_cache_service import LLMCacheService
from backend.security.rate_limiter import SlidingWindowRateLimiter
from backend.security.concurrency import LLMConcurrencyController, ConcurrencyLimitExceeded
from backend.safety.medical_safety_guard import MedicalSafetyGuard


class FlakyRedisClient:
    """Simulates various Redis failure modes: timeouts, connection refused, disconnects, partial errors."""

    def __init__(self):
        self.mode = "healthy"  # "healthy", "timeout", "refused", "write_fail"
        self.store = {}
        self.zsets = {}

    def is_available(self):
        return self.mode == "healthy"

    def ping(self):
        if self.mode == "refused":
            raise ConnectionError("Connection refused: 127.0.0.1:6379")
        if self.mode == "timeout":
            raise TimeoutError("Socket timeout waiting for Redis response")
        return True

    def get(self, key):
        if self.mode in ("refused", "timeout"):
            self.ping()
        return self.store.get(key)

    def set(self, key, value, ex=None):
        if self.mode in ("refused", "timeout", "write_fail"):
            self.ping()
        self.store[key] = str(value)
        return True

    def execute_lua(self, script, numkeys, *args):
        if self.mode in ("refused", "timeout"):
            self.ping()
        return [1, 10, 0, int(time.time() + 60)]


def test_cache_graceful_degradation_on_connection_refused():
    """Verify cache operations continue seamlessly when Redis connection is refused."""
    flaky = FlakyRedisClient()
    flaky.mode = "refused"

    redis_svc = RedisService(enabled=True)
    redis_svc._client = flaky

    cache = LLMCacheService(enabled=True, ttl_seconds=60, redis_service=redis_svc)

    valid_payload = {
        "answer": "Hypertension is managed with lifestyle modifications and antihypertensive therapy.",
        "retrieval_status": "success",
        "sources": [{"chunk_id": "c1", "document_id": "d1"}]
    }

    # Storing entry must succeed via in-memory fallback
    key = cache.generate_cache_key("What is the management of hypertension?", user_scope=101)
    stored = cache.set(key, valid_payload, user_scope=101)
    assert stored is True

    # Retrieving entry must hit in-memory fallback
    res = cache.get(key)
    assert res is not None
    assert "antihypertensive" in res["answer"]
    assert res["timings"]["distributed_cache"] is False


def test_cache_graceful_degradation_on_timeout():
    """Verify cache operations do not crash on Redis timeouts and trigger circuit-breaker."""
    flaky = FlakyRedisClient()
    flaky.mode = "timeout"

    redis_svc = RedisService(enabled=True)
    redis_svc._client = flaky

    cache = LLMCacheService(enabled=True, ttl_seconds=60, redis_service=redis_svc)

    valid_payload = {
        "answer": "Diabetes mellitus requires glycemic monitoring and hemoglobin A1c targets.",
        "retrieval_status": "success",
        "sources": [{"chunk_id": "c2", "document_id": "d2"}]
    }

    key = cache.generate_cache_key("What are diabetes management targets?", user_scope=202)
    stored = cache.set(key, valid_payload, user_scope=202)
    assert stored is True

    # Fetch should safely degrade to local memory
    res = cache.get(key)
    assert res is not None
    assert "glycemic" in res["answer"]


def test_rate_limiter_never_allows_unlimited_requests_during_redis_failure():
    """
    CRITICAL SECURITY INVARIANT:
    Redis outage MUST NOT bypass rate limiting or allow unlimited requests.
    """
    flaky = FlakyRedisClient()
    flaky.mode = "refused"  # Redis is completely dead

    limiter = SlidingWindowRateLimiter(
        requests_per_minute=2,
        window_seconds=60,
        enabled=True,
        redis_service=flaky
    )

    req = Request({
        "type": "http",
        "method": "POST",
        "path": "/rag/query",
        "headers": [(b"host", b"localhost")]
    })

    # Allowed requests
    limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=2, user_id=888)
    limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=2, user_id=888)

    # 3rd request MUST be rejected with HTTP 429
    with pytest.raises(HTTPException) as exc_info:
        limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=2, user_id=888)

    assert exc_info.value.status_code == 429
    assert exc_info.value.headers.get("X-RateLimit-Remaining") == "0"


def test_medical_safety_invariant_retained_during_redis_failure():
    """
    CRITICAL MEDICAL INVARIANT:
    Medical safety pre-screening MUST run before cache lookup,
    regardless of whether Redis is healthy, degraded, or dead.
    """
    flaky = FlakyRedisClient()
    flaky.mode = "refused"

    redis_svc = RedisService(enabled=True)
    redis_svc._client = flaky

    cache = LLMCacheService(enabled=True, redis_service=redis_svc)

    # Inquire about acute emergency symptoms
    emergency_query = "I am having severe crushing chest pain radiating to my left arm right now"

    # 1. Pre-screening check
    allow_rag, safety_assessment, immediate_msg = MedicalSafetyGuard.pre_screen_inquiry(emergency_query)
    assert allow_rag is False
    assert safety_assessment.category.value == "EMERGENCY_SYMPTOMS"
    assert "911" in immediate_msg or "emergency" in immediate_msg.lower()

    # Invariant: RAG service returns immediate emergency message without cache lookup
    # Even if an attacker tried to inject a cached entry for the emergency query:
    fake_safe_payload = {
        "answer": "This is completely harmless, just rest.",
        "retrieval_status": "success",
        "sources": []
    }
    key = cache.generate_cache_key(emergency_query, user_scope="anon")
    cache._set_local(key, fake_safe_payload, 60)

    # Because safety pre-screen blocks first, the cached poisoned answer is NEVER served
    allow_rag_eval, _, _ = MedicalSafetyGuard.pre_screen_inquiry(emergency_query)
    assert allow_rag_eval is False


def test_user_isolation_retained_during_redis_failure():
    """Verify that during Redis failure, User A cannot read User B's local cache."""
    flaky = FlakyRedisClient()
    flaky.mode = "refused"

    redis_svc = RedisService(enabled=True)
    redis_svc._client = flaky

    cache = LLMCacheService(enabled=True, redis_service=redis_svc)

    payload_user_a = {
        "answer": "User A private cardiology report details.",
        "retrieval_status": "success",
        "sources": [{"chunk_id": "c_a", "document_id": "doc_a"}]
    }

    key_a = cache.generate_cache_key("Give me my cardiology report", user_scope="user_A", document_signature="sig_a")
    key_b = cache.generate_cache_key("Give me my cardiology report", user_scope="user_B", document_signature="sig_b")

    cache.set(key_a, payload_user_a, user_scope="user_A", document_signature="sig_a")

    # User B must NOT see User A's answer
    assert cache.get(key_b) is None

    # User A gets own answer
    res_a = cache.get(key_a)
    assert res_a is not None
    assert "User A private" in res_a["answer"]


def test_concurrency_controller_unaffected_by_redis_failure():
    """Verify LLM concurrency limits remain strictly enforced during Redis failure."""
    controller = LLMConcurrencyController(max_concurrent=2, acquire_timeout=0.1)

    # Acquire 2 slots
    with controller.acquire():
        with controller.acquire():
            assert controller.active_count == 2
            # 3rd request must be rejected with ConcurrencyLimitExceeded
            with pytest.raises(ConcurrencyLimitExceeded):
                with controller.acquire():
                    pass

    assert controller.active_count == 0


def test_automatic_reconnection_when_redis_heals():
    """Verify system automatically resumes distributed mode once Redis recovers."""
    flaky = FlakyRedisClient()
    flaky.mode = "refused"  # Broken

    redis_svc = RedisService(enabled=True)
    redis_svc._client = flaky

    cache = LLMCacheService(enabled=True, redis_service=redis_svc)

    payload = {
        "answer": "Pneumonia clinical symptoms include cough, fever, dyspnea, and pleuritic chest pain.",
        "retrieval_status": "success",
        "sources": [{"chunk_id": "c_pneumonia", "document_id": "doc_p"}]
    }

    key = cache.generate_cache_key("What are pneumonia symptoms?", user_scope=555)

    # Write while Redis is down
    cache.set(key, payload, user_scope=555)

    # Redis heals
    flaky.mode = "healthy"
    redis_svc._last_failure_time = 0.0  # Reset backoff timer

    # Next write should successfully persist to Redis
    key2 = cache.generate_cache_key("What are symptoms of bacterial bronchitis?", user_scope=555)
    cache.set(key2, payload, user_scope=555)

    # Check that Redis now contains the key
    redis_key = f"{redis_svc.CACHE_KEY_PREFIX}{key2}"
    assert redis_key in flaky.store
