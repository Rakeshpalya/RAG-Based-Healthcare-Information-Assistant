"""
Phase 3.5.10 Integration Test: End-to-End Distributed Production Pipeline.

Tests the complete authoritative flow:
User
  ↓
Authentication
  ↓
Rate limiter (Distributed Redis)
  ↓
Request ID
  ↓
Medical Safety Pre-Screen
  ↓
RAG Retrieval (FAISS 744 vectors)
  ↓
Sufficiency Gate
  ↓
Redis Cache Lookup (Layered L1/L2)
  ↓
LLM Concurrency Controller
  ↓
Gemini Answer Generation
  ↓
Citation Validation
  ↓
Grounding Validation
  ↓
Hallucination Guard
  ↓
Medical Safety Post-Screen
  ↓
Redis Cache Storage (Validated only)
  ↓
Final Safe Response

Also tests the 14 mandatory conditions:
1. Cache HIT
2. Cache MISS
3. Different users (User isolation)
4. Different documents (Document scoping)
5. Document update (Cache invalidation)
6. Redis unavailable (Graceful degradation)
7. LLM unavailable (Resilience)
8. Vector store unavailable / empty
9. Rate-limit exceeded (HTTP 429)
10. Streaming (SSE events with request_id)
11. Concurrent requests
12. Metrics (Prometheus exposition)
13. Snapshot (Atomic verified backup)
14. Restore (Two-phase verified restore)
"""

import time
import copy
import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from fastapi import Request, HTTPException
from fastapi.testclient import TestClient


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
from backend.main import app
from backend.rag.rag_service import RAGService
from backend.services.redis_service import RedisService
from backend.services.llm_cache_service import LLMCacheService
from backend.security.rate_limiter import SlidingWindowRateLimiter
from backend.security.concurrency import LLMConcurrencyController
from backend.safety.medical_safety_guard import MedicalSafetyGuard
from backend.evaluation.observability import get_metrics_collector
from backend.services.snapshot_service import VectorStoreSnapshotService
from backend.services.vector_store_service import VectorStoreService, get_vector_store_service
from tests.test_phase3_5_redis_cache import FakeRedisClient


@pytest.fixture
def e2e_env():
    """Sets up a complete isolated distributed test harness with mock Redis and vector store."""
    fake_redis = FakeRedisClient()

    redis_svc = RedisService(enabled=True)
    redis_svc._client = fake_redis
    redis_svc._is_alive = True

    cache_svc = LLMCacheService(enabled=True, ttl_seconds=3600, redis_service=redis_svc)
    rate_limiter_svc = SlidingWindowRateLimiter(requests_per_minute=30, window_seconds=60, enabled=True, redis_service=redis_svc)

    # Use active vector store
    vs = get_vector_store_service()

    mock_gemini = MagicMock()
    mock_gemini.model = "gemini-3.5-flash-lite"
    mock_gemini.temperature = 0.0
    mock_gemini.max_output_tokens = 1024
    mock_gemini.generate_answer.return_value = {
        "answer": "Clinical trial demonstrates metformin efficacy in type 2 diabetes [Source 1].",
        "model": "gemini-3.5-flash-lite",
        "input_tokens": 120,
        "output_tokens": 45,
        "generation_time_ms": 150.0,
        "gemini_calls_count": 1,
        "disclaimer": "This information is educational only."
    }

    rag = RAGService(vector_store=vs)

    return {
        "redis_client": fake_redis,
        "redis_svc": redis_svc,
        "cache": cache_svc,
        "rate_limiter": rate_limiter_svc,
        "vector_store": vs,
        "gemini": mock_gemini,
        "rag": rag,
        "client": TestClient(app)
    }


def test_01_and_02_cache_miss_then_hit(e2e_env):
    """Test 1 & 2: Cache MISS on first query, and Cache HIT on second query."""
    rag = e2e_env["rag"]
    gemini = e2e_env["gemini"]
    cache = e2e_env["cache"]

    q = "What is the efficacy of metformin in type 2 diabetes?"
    user_id = 1

    # First call: Cache MISS
    with patch("backend.services.llm_cache_service.get_llm_cache_service", return_value=cache):
        res1 = rag.generate_rag_answer(question=q, user_id=user_id, gemini_service=gemini)
        assert res1["retrieval_status"] in ("success", "grounded_boundary")
        assert res1["timings"]["cache_hit"] is False
        assert gemini.generate_answer.call_count == 1

        # Second call: Cache HIT
        res2 = rag.generate_rag_answer(question=q, user_id=user_id, gemini_service=gemini)
        assert res2["timings"]["cache_hit"] is True
        assert res2["timings"]["llm_called"] is False
        # Gemini should NOT have been called a second time
        assert gemini.generate_answer.call_count == 1
        assert "metformin" in res2["answer"].lower()


def test_03_different_users_cache_isolation(e2e_env):
    """Test 3: Different users have strictly isolated caches."""
    rag = e2e_env["rag"]
    gemini = e2e_env["gemini"]
    cache = e2e_env["cache"]

    q = "What is the efficacy of metformin in type 2 diabetes?"

    with patch("backend.services.llm_cache_service.get_llm_cache_service", return_value=cache):
        # User 1 queries and caches
        res_u1 = rag.generate_rag_answer(question=q, user_id=1, gemini_service=gemini)
        assert res_u1["timings"]["cache_hit"] is False
        assert res_u1["retrieval_status"] in ("success", "grounded_boundary")

        # User 2 queries identical question -> MUST be cache MISS and must not see User 1's cache
        res_u2 = rag.generate_rag_answer(question=q, user_id=2, gemini_service=gemini)
        assert res_u2.get("timings", {}).get("cache_hit", False) is False
        # Ensure User 2 does not get User 1's cached content
        assert "Metformin improves" not in res_u2.get("answer", "")


def test_04_different_documents_scoping(e2e_env):
    """Test 4: Different document signatures do not cross-hit in cache."""
    cache = e2e_env["cache"]

    payload = {
        "answer": "Doc A answer on asthma.",
        "retrieval_status": "success",
        "sources": [{"chunk_id": "c1", "document_id": "doc_A"}]
    }

    key_doc_a = cache.generate_cache_key("asthma symptoms", user_scope=300, document_signature="doc_A_hash")
    key_doc_b = cache.generate_cache_key("asthma symptoms", user_scope=300, document_signature="doc_B_hash")

    cache.set(key_doc_a, payload, user_scope=300, document_signature="doc_A_hash")

    assert cache.get(key_doc_b) is None
    assert cache.get(key_doc_a) is not None


def test_05_document_update_invalidates_cache(e2e_env):
    """Test 5: Uploading or updating a document invalidates cached responses."""
    cache = e2e_env["cache"]

    payload = {
        "answer": "Lab results before document update.",
        "retrieval_status": "success",
        "sources": [{"chunk_id": "c1", "document_id": "doc_lab_123"}]
    }

    key = cache.generate_cache_key("What are my lab results?", user_scope=400, document_signature="doc_lab_123")
    cache.set(key, payload, user_scope=400, document_signature="doc_lab_123")
    assert cache.get(key) is not None

    # Simulate document update invalidation
    invalidated = cache.delete_by_document("doc_lab_123")
    assert invalidated >= 1
    assert cache.get(key) is None


def test_06_redis_unavailable_graceful_degradation(e2e_env):
    """Test 6: Application continues safely when Redis is dead."""
    fake_redis = e2e_env["redis_client"]
    fake_redis.is_connected = False  # Outage

    rag = e2e_env["rag"]
    gemini = e2e_env["gemini"]
    cache = e2e_env["cache"]

    q = "What is the efficacy of metformin in type 2 diabetes?"
    user_id = 1

    with patch("backend.services.llm_cache_service.get_llm_cache_service", return_value=cache):
        # Query should succeed via local in-memory fallback
        res = rag.generate_rag_answer(question=q, user_id=user_id, gemini_service=gemini)
        assert res["retrieval_status"] in ("success", "grounded_boundary")
        assert "timings" in res


def test_07_llm_unavailable_resilience(e2e_env):
    """Test 7: When Gemini fails, RAG pipeline returns safe error status without crashing."""
    rag = e2e_env["rag"]
    cache = e2e_env["cache"]

    from backend.services.gemini_service import GeminiServiceError
    failing_gemini = MagicMock()
    failing_gemini.generate_answer.side_effect = GeminiServiceError("Google Gemini API 503 Unavailable")

    q = "What is the efficacy of metformin in type 2 diabetes?"
    user_id = 1

    with patch("backend.services.llm_cache_service.get_llm_cache_service", return_value=cache):
        res = rag.generate_rag_answer(
            question=q,
            user_id=user_id,
            gemini_service=failing_gemini
        )
        assert res["retrieval_status"] in ("service_error", "service_unavailable", "error")
        # Invariant: error response must NEVER be cached
        key = cache.generate_cache_key(q, user_scope=user_id)
        assert cache.get(key) is None


def test_08_vector_store_empty_insufficient_context(e2e_env):
    """Test 8: When vector store finds no relevant evidence, returns insufficient context."""
    empty_vs = VectorStoreService(dimension=384, storage_dir=Path("tmp/empty_store"))
    rag_empty = RAGService(vector_store=empty_vs)

    res = rag_empty.generate_rag_answer(
        question="Unanswerable obscure medical condition xyz999",
        user_id=700,
        gemini_service=e2e_env["gemini"]
    )
    assert res["retrieval_status"] == "no_relevant_context"
    assert res["timings"]["llm_called"] is False


def test_09_rate_limit_exceeded(e2e_env):
    """Test 9: Rate limiter blocks requests beyond limit with HTTP 429."""
    limiter = e2e_env["rate_limiter"]
    req = make_request()

    for _ in range(5):
        limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=5, user_id=800)

    with pytest.raises(HTTPException) as exc_info:
        limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=5, user_id=800)

    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers


def test_10_streaming_rag(e2e_env):
    """Test 10: Streaming yields status and token events with request correlation."""
    rag = e2e_env["rag"]
    cache = e2e_env["cache"]
    gemini = e2e_env["gemini"]

    with patch("backend.services.llm_cache_service.get_llm_cache_service", return_value=cache):
        stream_gen = rag.generate_rag_stream(
            question="What is chronic kidney disease?",
            user_id=900,
            gemini_service=gemini
        )
        events = list(stream_gen)
        assert len(events) > 0
        event_types = [e[0] for e in events]
        assert "status" in event_types
        assert "token" in event_types or "complete" in event_types


def test_11_concurrent_llm_limits(e2e_env):
    """Test 11: LLMConcurrencyController bounds active slots and releases reliably."""
    controller = LLMConcurrencyController(max_concurrent=3, acquire_timeout=0.1)

    with controller.acquire():
        with controller.acquire():
            with controller.acquire():
                assert controller.active_count == 3
                # 4th slot times out
                with pytest.raises(Exception):
                    with controller.acquire():
                        pass

    assert controller.active_count == 0


def test_12_prometheus_metrics_exposition(e2e_env):
    """Test 12: GET /metrics exposes valid Prometheus text format without PHI or secrets."""
    client = e2e_env["client"]
    resp = client.get("/metrics", headers={"Accept": "text/plain"})
    assert resp.status_code == 200
    text = resp.text

    assert "rag_requests_total" in text
    assert "rag_cache_hits_total" in text
    assert "user_id" not in text
    assert "AIza" not in text


def test_13_and_14_snapshot_and_restore(tmp_path):
    """Test 13 & 14: Automated atomic snapshot and two-phase restore."""
    prod_store_dir = Path("data/vector_store")
    assert prod_store_dir.exists()

    snaps_dir = tmp_path / "e2e_snapshots"
    restore_dir = tmp_path / "e2e_restore"

    service = VectorStoreSnapshotService(
        vector_store_dir=prod_store_dir,
        snapshots_dir=snaps_dir
    )

    # 13. Snapshot Creation
    manifest = service.create_snapshot(
        snapshot_id="e2e_snap_test",
        validate_production_invariants=True
    )
    assert manifest["vector_count"] == 744
    assert manifest["metadata_count"] == 744
    assert manifest["dimension"] == 384

    # 14. Restore to isolated target
    snap_path = snaps_dir / "e2e_snap_test"
    restore_res = service.restore_snapshot(
        snapshot_dir=snap_path,
        target_dir=restore_dir,
        activate_in_memory=False
    )
    assert restore_res["restored"] is True
    assert restore_res["vector_count"] == 744
    assert restore_res["metadata_count"] == 744
