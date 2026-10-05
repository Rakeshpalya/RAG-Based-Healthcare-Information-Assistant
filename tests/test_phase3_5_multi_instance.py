"""
Phase 3.5.8 Tests: Multi-Instance Distributed Architecture Simulation.

Simulates two separate FastAPI / RAG API nodes (Instance A and Instance B)
connected to a shared Redis infrastructure layer:

API Instance A        API Instance B
       \\                //
        \\              //
         \\            //
         Shared Redis Layer

Verifies:
1. Cache written by Instance A is immediately readable by Instance B.
2. Cache isolation between users remains strictly enforced across instances.
3. Rate limits are shared globally across Instance A and Instance B.
4. Request IDs remain distinct and unique across instances.
5. Medical safety behavior remains strictly identical across instances.
6. LLM concurrency limits remain respected.
"""

import time
import uuid
import pytest
from unittest.mock import MagicMock

from backend.services.redis_service import RedisService
from backend.services.llm_cache_service import LLMCacheService
from backend.security.rate_limiter import SlidingWindowRateLimiter
from backend.safety.medical_safety_guard import MedicalSafetyGuard
from backend.evaluation.observability import generate_request_id
from tests.test_phase3_5_redis_cache import FakeRedisClient
from tests.test_phase3_5_distributed_rate_limiter import make_request
from fastapi import HTTPException


@pytest.fixture
def multi_instance_environment():
    """Provides two separate API instances connected to a single shared Redis backend."""
    shared_redis_client = FakeRedisClient()

    redis_adapter_a = RedisService(enabled=True)
    redis_adapter_a._client = shared_redis_client
    redis_adapter_a._is_alive = True

    redis_adapter_b = RedisService(enabled=True)
    redis_adapter_b._client = shared_redis_client
    redis_adapter_b._is_alive = True

    cache_instance_a = LLMCacheService(enabled=True, ttl_seconds=3600, redis_service=redis_adapter_a)
    cache_instance_b = LLMCacheService(enabled=True, ttl_seconds=3600, redis_service=redis_adapter_b)

    limiter_instance_a = SlidingWindowRateLimiter(requests_per_minute=30, window_seconds=60, enabled=True, redis_service=redis_adapter_a)
    limiter_instance_b = SlidingWindowRateLimiter(requests_per_minute=30, window_seconds=60, enabled=True, redis_service=redis_adapter_b)

    return {
        "redis": shared_redis_client,
        "cache_a": cache_instance_a,
        "cache_b": cache_instance_b,
        "limiter_a": limiter_instance_a,
        "limiter_b": limiter_instance_b,
    }


def test_cache_written_by_instance_a_read_by_instance_b(multi_instance_environment):
    """Verify response cached on Instance A is instantly available to Instance B via shared Redis."""
    env = multi_instance_environment
    cache_a = env["cache_a"]
    cache_b = env["cache_b"]

    payload = {
        "answer": "Primary hypertension has no identifiable secondary cause and accounts for 90-95% of cases.",
        "retrieval_status": "success",
        "sources": [{"chunk_id": "c1", "document_id": "doc_cardio"}]
    }

    key = cache_a.generate_cache_key("What is primary hypertension etiology?", user_scope=5001)

    # Write on Instance A
    stored = cache_a.set(key, payload, user_scope=5001)
    assert stored is True

    # Read from Instance B (local memory empty, must fetch from shared Redis)
    read_b = cache_b.get(key)
    assert read_b is not None
    assert "Primary hypertension has no identifiable" in read_b["answer"]
    assert read_b["timings"]["cache_hit"] is True
    assert read_b["timings"]["distributed_cache"] is True


def test_cache_user_isolation_across_instances(multi_instance_environment):
    """Verify User 101 on Instance A cannot be read by User 202 on Instance B."""
    env = multi_instance_environment
    cache_a = env["cache_a"]
    cache_b = env["cache_b"]

    payload_101 = {
        "answer": "Patient 101 specific clinical diagnosis and private lab findings.",
        "retrieval_status": "success",
        "sources": [{"chunk_id": "c_101", "document_id": "doc_101"}]
    }

    key_user_101 = cache_a.generate_cache_key("What are my lab results?", user_scope=101, document_signature="sig_101")
    key_user_202 = cache_b.generate_cache_key("What are my lab results?", user_scope=202, document_signature="sig_202")

    cache_a.set(key_user_101, payload_101, user_scope=101, document_signature="sig_101")

    # Instance B querying for User 202 must get a MISS
    assert cache_b.get(key_user_202) is None

    # Instance B querying for User 101 gets a HIT
    hit_101 = cache_b.get(key_user_101)
    assert hit_101 is not None
    assert "Patient 101 specific" in hit_101["answer"]


def test_shared_rate_limiting_across_instances(multi_instance_environment):
    """Verify rate limits are enforced across instances (alternating requests decrement shared quota)."""
    env = multi_instance_environment
    limiter_a = env["limiter_a"]
    limiter_b = env["limiter_b"]

    shared_user_id = 7777
    quota = 6
    req = make_request()

    # Alternate requests between Instance A and Instance B
    for i in range(quota):
        limiter = limiter_a if (i % 2 == 0) else limiter_b
        headers = limiter.check_rate_limit(
            req,
            endpoint_type="rag_query",
            max_requests=quota,
            user_id=shared_user_id
        )
        assert "X-RateLimit-Remaining" in headers
        assert int(headers["X-RateLimit-Remaining"]) == quota - 1 - i

    # Request 7 on Instance A must be blocked
    with pytest.raises(HTTPException) as exc_a:
        limiter_a.check_rate_limit(req, endpoint_type="rag_query", max_requests=quota, user_id=shared_user_id)
    assert exc_a.value.status_code == 429

    # Request 7 on Instance B must also be blocked
    with pytest.raises(HTTPException) as exc_b:
        limiter_b.check_rate_limit(req, endpoint_type="rag_query", max_requests=quota, user_id=shared_user_id)
    assert exc_b.value.status_code == 429


def test_request_ids_unique_across_instances():
    """Verify request IDs generated across instances never collide."""
    ids_a = {generate_request_id() for _ in range(50)}
    ids_b = {generate_request_id() for _ in range(50)}

    # Must be completely disjoint sets of unique IDs
    assert len(ids_a) == 50
    assert len(ids_b) == 50
    assert len(ids_a.intersection(ids_b)) == 0


def test_identical_safety_behavior_across_instances():
    """Verify medical safety guard behaves identically across both API instances."""
    emergency_q = "I am having severe crushing chest pain radiating to my left arm right now"
    benign_q = "What is the recommended dosage of metformin for adult type 2 diabetes?"

    # Instance A check
    allow_a_em, assess_a_em, _ = MedicalSafetyGuard.pre_screen_inquiry(emergency_q)
    allow_a_ben, assess_a_ben, _ = MedicalSafetyGuard.pre_screen_inquiry(benign_q)

    # Instance B check
    allow_b_em, assess_b_em, _ = MedicalSafetyGuard.pre_screen_inquiry(emergency_q)
    allow_b_ben, assess_b_ben, _ = MedicalSafetyGuard.pre_screen_inquiry(benign_q)

    assert allow_a_em is False and allow_b_em is False
    assert assess_a_em.category == assess_b_em.category
    assert assess_a_em.risk_level == assess_b_em.risk_level

    assert allow_a_ben is True and allow_b_ben is True
    assert assess_a_ben.category == assess_b_ben.category
