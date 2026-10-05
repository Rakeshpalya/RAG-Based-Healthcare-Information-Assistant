"""
Phase 3.6.6 Tests: Disaster Recovery & Degradation Scenarios.

Simulates and verifies:
1. Redis unavailable:
   - Layered cache falls back transparently to L1 in-memory LRU.
   - Distributed rate limiter falls back to local thread-safe history; rate limits strictly enforced.
   - MedicalSafetyGuard executes and intercepts dangerous queries with zero bypass.
2. Redis data lost / flushed:
   - Cache transparently regenerates on subsequent queries; no crashes.
3. Cache subsystem completely unavailable / disabled:
   - System continues servicing queries safely via vector retrieval + LLM synthesis.
   - Never fabricates or hallucinates responses.
4. Vector store failure / unavailable / empty:
   - Returns safe failure envelope ('no_relevant_context').
   - Never hallucinates medical evidence.
5. Gemini API unavailable (503 / 500 error):
   - Returns safe fallback envelope ('service_error' / 'service_unavailable').
   - Contains mandatory medical disclaimer; does not fabricate advice.
   - Unsuccessful responses are never cached.
6. Gemini timeout (request timeout exceeded):
   - Returns safe timeout response; catches error cleanly without traceback leakage.
7. Prometheus metrics scrape failure / exposition failure:
   - RAG pipeline execution continues unaffected.
8. Application restart recovery:
   - In-memory state regenerates cleanly from persistent storage on restart.
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from fastapi import HTTPException

from backend.rag.rag_service import RAGService
from backend.services.redis_service import RedisService
from backend.services.llm_cache_service import LLMCacheService
from backend.security.rate_limiter import SlidingWindowRateLimiter
from backend.safety.medical_safety_guard import MedicalSafetyGuard
from backend.services.gemini_service import GeminiService, GeminiServiceError
from backend.services.vector_store_service import VectorStoreService, get_vector_store_service
from tests.test_phase3_5_redis_cache import FakeRedisClient
from tests.test_phase3_5_e2e import make_request


@pytest.fixture
def dr_env():
    """Sets up an isolated disaster recovery environment."""
    fake_redis = FakeRedisClient()
    redis_svc = RedisService(enabled=True)
    redis_svc._client = fake_redis
    redis_svc._is_alive = True

    cache_svc = LLMCacheService(enabled=True, ttl_seconds=3600, redis_service=redis_svc)
    limiter_svc = SlidingWindowRateLimiter(requests_per_minute=5, window_seconds=60, enabled=True, redis_service=redis_svc)

    vs = get_vector_store_service()

    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.model = "gemini-3.5-flash-lite"
    mock_gemini.temperature = 0.0
    mock_gemini.max_output_tokens = 1024
    mock_gemini.generate_answer.return_value = {
        "answer": "Clinical trial demonstrates metformin efficacy in type 2 diabetes [Source 1].",
        "model": "gemini-3.5-flash-lite",
        "input_tokens": 100,
        "output_tokens": 30,
        "generation_time_ms": 120.0,
        "gemini_calls_count": 1,
        "disclaimer": "This information is educational only."
    }

    rag = RAGService(vector_store=vs)

    return {
        "redis_client": fake_redis,
        "redis_svc": redis_svc,
        "cache": cache_svc,
        "rate_limiter": limiter_svc,
        "gemini": mock_gemini,
        "rag": rag,
        "vector_store": vs
    }


def test_01_redis_unavailable_degradation(dr_env):
    """Scenario 1: Redis goes down -> local cache & local rate limit fallback, safety remains active."""
    fake_redis = dr_env["redis_client"]
    fake_redis.is_connected = False  # Hard outage

    rag = dr_env["rag"]
    cache = dr_env["cache"]
    limiter = dr_env["rate_limiter"]
    gemini = dr_env["gemini"]

    # 1. Safety remains active before any cache or LLM
    emergency_q = "I have acute crushing chest pain radiating to my left arm!"
    res_safety = rag.generate_rag_answer(question=emergency_q, user_id=1, gemini_service=gemini)
    assert res_safety["retrieval_status"] == "safety_intercepted"
    assert "EMERGENCY" in res_safety["answer"]

    # 2. Local rate limiter fallback strictly enforces limit during Redis outage
    req = make_request()
    for _ in range(5):
        limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=5, user_id=99)

    with pytest.raises(HTTPException) as exc_info:
        limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=5, user_id=99)
    assert exc_info.value.status_code == 429

    # 3. Local in-memory cache continues serving
    q = "What is the efficacy of metformin in type 2 diabetes?"
    with patch("backend.services.llm_cache_service.get_llm_cache_service", return_value=cache):
        res1 = rag.generate_rag_answer(question=q, user_id=1, gemini_service=gemini)
        assert res1["retrieval_status"] in ("success", "grounded_boundary")
        # Second call hits L1 local memory
        res2 = rag.generate_rag_answer(question=q, user_id=1, gemini_service=gemini)
        assert res2["timings"]["cache_hit"] is True


def test_02_redis_data_lost_recovers_gracefully(dr_env):
    """Scenario 2: Redis loses data (flush/restart) -> cache transparently repopulates."""
    rag = dr_env["rag"]
    cache = dr_env["cache"]
    gemini = dr_env["gemini"]
    q = "What is the efficacy of metformin in type 2 diabetes?"

    with patch("backend.services.llm_cache_service.get_llm_cache_service", return_value=cache):
        # Initial query caches
        rag.generate_rag_answer(question=q, user_id=1, gemini_service=gemini)

        # Simulate Redis data loss
        dr_env["redis_client"].store.clear()
        dr_env["redis_client"].zsets.clear()
        cache._cache.clear()

        # Next query executes cleanly without crash and re-caches
        res = rag.generate_rag_answer(question=q, user_id=1, gemini_service=gemini)
        assert res["retrieval_status"] in ("success", "grounded_boundary")
        assert "metformin" in res["answer"].lower()


def test_03_cache_completely_disabled_no_fabrication(dr_env):
    """Scenario 3: Cache is disabled -> pipeline continues normally without hallucination."""
    cache = dr_env["cache"]
    cache.enabled = False

    rag = dr_env["rag"]
    gemini = dr_env["gemini"]
    q = "What is the efficacy of metformin in type 2 diabetes?"

    with patch("backend.services.llm_cache_service.get_llm_cache_service", return_value=cache):
        res = rag.generate_rag_answer(question=q, user_id=1, gemini_service=gemini)
        assert res["retrieval_status"] in ("success", "grounded_boundary")
        assert res["timings"]["cache_hit"] is False
        assert "metformin" in res["answer"].lower()


def test_04_vector_store_unavailable_returns_safe_envelope():
    """Scenario 4: Vector store empty or corrupted -> returns safe failure, no fabricated answers."""
    empty_vs = VectorStoreService(dimension=384, storage_dir=Path("tmp/non_existent_dir"))
    rag = RAGService(vector_store=empty_vs)

    res = rag.generate_rag_answer(
        question="What is the treatment for hypertension?",
        user_id=1
    )
    assert res["retrieval_status"] == "no_relevant_context"
    assert res["timings"]["llm_called"] is False
    assert len(res["retrieved_chunks"]) == 0
    # Must not contain fabricated treatment instructions
    assert "take 500mg" not in res["answer"].lower()


def test_05_gemini_unavailable_safe_fallback(dr_env):
    """Scenario 5: Gemini 503 / network failure -> returns safe failure envelope, not cached."""
    rag = dr_env["rag"]
    cache = dr_env["cache"]

    failing_gemini = MagicMock(spec=GeminiService)
    failing_gemini.generate_answer.side_effect = GeminiServiceError("Gemini API 503 Service Unavailable")

    q = "What is the efficacy of metformin in type 2 diabetes?"
    with patch("backend.services.llm_cache_service.get_llm_cache_service", return_value=cache):
        res = rag.generate_rag_answer(
            question=q,
            user_id=1,
            gemini_service=failing_gemini
        )
        assert res["retrieval_status"] in ("service_error", "service_unavailable", "error")
        assert "temporarily unavailable" in res["answer"].lower()
        # Invariant: failed generation must NEVER be cached
        key = cache.generate_cache_key(q, user_scope=1)
        assert cache.get(key) is None


def test_06_gemini_timeout_handling(dr_env):
    """Scenario 6: Gemini timeout -> caught cleanly, safe message, no raw credentials."""
    rag = dr_env["rag"]
    cache = dr_env["cache"]

    timeout_gemini = MagicMock(spec=GeminiService)
    timeout_gemini.generate_answer.side_effect = GeminiServiceError("API call timed out after 30.0s")

    q = "What is the efficacy of metformin in type 2 diabetes?"
    with patch("backend.services.llm_cache_service.get_llm_cache_service", return_value=cache):
        res = rag.generate_rag_answer(
            question=q,
            user_id=1,
            gemini_service=timeout_gemini
        )
        assert res["retrieval_status"] in ("service_error", "service_unavailable", "error")
        assert "AIza" not in res["answer"]
        assert "secret" not in res["answer"].lower()


def test_07_prometheus_scrape_failure_does_not_affect_rag(dr_env):
    """Scenario 7: Prometheus scrape error or metrics exception does not crash RAG query pipeline."""
    rag = dr_env["rag"]
    gemini = dr_env["gemini"]
    q = "What is the efficacy of metformin in type 2 diabetes?"

    with patch("backend.evaluation.observability.ProductionMetricsCollector.record_rag_event", side_effect=RuntimeError("Prometheus collector failure")):
        # Query must still succeed even if metrics collector encountered an internal error
        res = rag.generate_rag_answer(question=q, user_id=1, gemini_service=gemini)
        assert res["retrieval_status"] in ("success", "grounded_boundary")
        assert "metformin" in res["answer"].lower()


def test_08_application_restart_state_recovery():
    """Scenario 8: On application restart, persistent vector store loads cleanly from disk."""
    prod_store_dir = Path("data/vector_store")
    reloaded_vs = VectorStoreService(storage_dir=prod_store_dir, dimension=384)
    assert reloaded_vs.load() is True
    assert reloaded_vs.count() == 744
    assert len(reloaded_vs.metadata_store) == 744
