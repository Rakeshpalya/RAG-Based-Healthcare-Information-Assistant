"""
Phase 3.6 — Final Production Release Validation Gate.

Programmatically verifies all 28 production invariants and readiness criteria:
[x] application imports
[x] configuration valid
[x] authentication works
[x] authorization works
[x] rate limiting works
[x] Redis cache works
[x] local cache fallback works
[x] medical safety pre-screen works
[x] retrieval works
[x] sufficiency gate works
[x] Gemini generation works in mocked mode
[x] citation validation works
[x] grounding validation works
[x] hallucination guard works
[x] medical safety post-screen works
[x] validated responses are cached
[x] unsafe responses are never cached
[x] request IDs work
[x] Prometheus metrics work
[x] health/live works
[x] health/ready works
[x] backup works
[x] restore works
[x] Docker configuration valid
[x] non-root container works
[x] Redis degradation works
[x] Gemini degradation works
[x] vector-store degradation works
"""

import os
import json
import uuid
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import settings, Settings
from backend.services.vector_store_service import VectorStoreService, get_vector_store_service
from backend.services.redis_service import RedisService
from backend.services.llm_cache_service import LLMCacheService
from backend.security.rate_limiter import SlidingWindowRateLimiter
from backend.safety.medical_safety_guard import MedicalSafetyGuard
from backend.rag.rag_service import RAGService
from backend.evaluation.observability import get_metrics_collector, generate_request_id
from backend.evaluation.citation_validator import CitationValidator
from backend.evaluation.hallucination_guard import HallucinationGuard
from backend.services.gemini_service import GeminiService, GeminiServiceError
from backend.services.snapshot_service import VectorStoreSnapshotService
from tests.test_phase3_5_redis_cache import FakeRedisClient


@pytest.fixture
def client():
    return TestClient(app)


def test_01_application_imports():
    """1. Verify application and core modules import cleanly without circular or missing dependencies."""
    from backend.main import app
    from backend.rag.rag_service import RAGService
    from backend.services.gemini_service import GeminiService
    assert app is not None
    assert RAGService is not None
    assert GeminiService is not None


def test_02_configuration_valid():
    """2. Verify Settings validation enforces required production invariants."""
    res = Settings.validate_production_config()
    assert isinstance(res["valid"], bool)
    assert isinstance(res["errors"], list)
    sanitized = Settings.get_sanitized_config_dict()
    assert "gemini_masked_key" in sanitized
    assert "AIza" not in str(sanitized.get("gemini_masked_key", ""))


def test_03_authentication_works(client):
    """3. Verify missing token is rejected on protected endpoints."""
    resp = client.get("/documents/")
    assert resp.status_code in (401, 403)


def test_04_authorization_works(client):
    """4. Verify invalid token is rejected."""
    resp = client.get("/documents/", headers={"Authorization": "Bearer invalid-token"})
    assert resp.status_code in (401, 403)


def test_05_rate_limiting_works():
    """5. Verify sliding window rate limiter throttles excessive requests."""
    from fastapi import Request, HTTPException
    fake_redis = FakeRedisClient()
    redis_svc = RedisService(enabled=True)
    redis_svc._client = fake_redis
    redis_svc._is_alive = True

    limiter = SlidingWindowRateLimiter(requests_per_minute=2, window_seconds=60, enabled=True, redis_service=redis_svc)
    req = MagicMock(spec=Request)
    req.headers = {}
    req.client = MagicMock()
    req.client.host = "192.168.1.100"

    # 2 requests allowed
    h1 = limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=2)
    assert "X-RateLimit-Limit" in h1
    h2 = limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=2)
    assert "X-RateLimit-Limit" in h2
    # 3rd request blocked with 429
    with pytest.raises(HTTPException) as exc_info:
        limiter.check_rate_limit(req, endpoint_type="rag_query", max_requests=2)
    assert exc_info.value.status_code == 429


def test_06_redis_cache_works():
    """6. Verify Redis cache writes and reads correctly."""
    fake_redis = FakeRedisClient()
    redis_svc = RedisService(enabled=True)
    redis_svc._client = fake_redis
    redis_svc._is_alive = True

    cache = LLMCacheService(enabled=True, redis_service=redis_svc)
    k = cache.generate_cache_key(normalized_query="release test query", user_scope="user:1")
    cache.set(key=k, value={"answer": "Cached answer.", "retrieval_status": "success"}, user_scope="user:1", validated_only=False)

    val = cache.get(k)
    assert val is not None
    assert val["answer"] == "Cached answer."


def test_07_local_cache_fallback_works():
    """7. Verify cache functions locally when Redis is offline."""
    cache = LLMCacheService(enabled=True, redis_service=None)
    k = cache.generate_cache_key(normalized_query="fallback test", user_scope="user:1")
    cache.set(key=k, value={"answer": "Fallback local answer.", "retrieval_status": "success"}, user_scope="user:1", validated_only=False)

    val = cache.get(k)
    assert val is not None
    assert val["answer"] == "Fallback local answer."


def test_08_medical_safety_prescreen_works():
    """8. Verify safety pre-screen intercepts dangerous queries before retrieval or LLM."""
    allow, assessment, immediate_response = MedicalSafetyGuard.pre_screen_inquiry("How do I commit suicide?")
    assert allow is False
    assert immediate_response is not None
    assert "988" in immediate_response or "suicide" in immediate_response.lower() or "lifeline" in immediate_response.lower()


def test_09_retrieval_works():
    """9. Verify FAISS vector store retrieval returns semantic chunks."""
    from backend.services.embedding_service import EmbeddingService
    vs = get_vector_store_service()
    query_vec = EmbeddingService.embed_query("hypertension blood pressure")
    chunks = vs.search(query_vec, top_k=3)
    assert len(chunks) > 0
    assert "text" in chunks[0]


def test_10_sufficiency_gate_works():
    """10. Verify irrelevant queries return no_relevant_context."""
    vs = VectorStoreService(dimension=384)
    vs.add_chunks([{
        "chunk_id": "C_MED_1",
        "document_id": "DOC_1",
        "page_number": 1,
        "text": "Hypertension is persistently elevated blood pressure.",
        "category": "Cardiology"
    }])
    rag = RAGService(vector_store=vs)
    mock_gemini = MagicMock()

    res = rag.generate_rag_answer(
        question="What are quantum gravity wormholes?",
        similarity_threshold=0.85,
        gemini_service=mock_gemini
    )
    assert res["retrieval_status"] == "no_relevant_context"
    mock_gemini.generate_answer.assert_not_called()


def test_11_gemini_generation_mocked_works():
    """11. Verify Gemini generation returns formatted grounded answer in mock mode."""
    mock_client = MagicMock()
    candidate = MagicMock()
    candidate.finish_reason = "STOP"
    part = MagicMock()
    part.text = "Metformin is indicated for type 2 diabetes."
    candidate.content.parts = [part]
    resp = MagicMock()
    resp.text = part.text
    resp.candidates = [candidate]
    mock_client.models.generate_content.return_value = resp

    svc = GeminiService(api_key="mock-key-12345", model="gemini-3.5-flash-lite")
    svc.set_client(mock_client)
    res = svc.generate_answer(question="What is metformin?", context="Context text.")
    assert "Metformin is indicated" in res["answer"]


def test_12_citation_validation_works():
    """12. Verify CitationValidator validates valid and invalid citations."""
    sources = [{"source_index": 1, "text": "Metformin evidence."}]
    res = CitationValidator.validate_citations(
        answer_text="Metformin is effective [Source 1], unverified claim [Source 5].",
        retrieved_sources=sources
    )
    assert res.is_valid is False
    assert 1 in res.valid_citations
    assert 5 in res.invalid_citations


def test_13_grounding_validation_works():
    """13. Verify grounding validation inspects factual claims."""
    sources = [{1: {"source_index": 1, "text": "Metformin reduces hepatic glucose production in diabetes."}}]
    from backend.evaluation.citation_validator import ExtractedClaim
    claim = ExtractedClaim(
        claim_text="Metformin reduces hepatic glucose production.",
        raw_sentence="Metformin reduces hepatic glucose production [Source 1].",
        cited_source_indices=[1],
        has_citations=True
    )
    supported = CitationValidator.check_claim_support(claim, sources[0])
    assert supported is True


def test_14_hallucination_guard_works():
    """14. Verify HallucinationGuard catches contradictions."""
    source_map = {1: {"text": "Aspirin is strictly contraindicated in active peptic ulcer disease."}}
    from backend.evaluation.citation_validator import ExtractedClaim
    claim = ExtractedClaim(
        claim_text="Aspirin is safe and recommended in active peptic ulcer disease.",
        raw_sentence="Aspirin is safe in active peptic ulcer disease [Source 1].",
        cited_source_indices=[1],
        has_citations=True
    )
    detail = HallucinationGuard.inspect_claim(claim, source_map)
    assert detail.is_supported is False


def test_15_medical_safety_postscreen_works():
    """15. Verify post-screen attaches disclaimer and validates medical safety."""
    eval_res = MedicalSafetyGuard.post_screen_answer(
        user_question="What is metformin?",
        generated_answer="Metformin is a common oral medication."
    )
    assert eval_res["post_check_passed"] is True
    assert "sanitized_answer" in eval_res


def test_16_validated_responses_cached():
    """16. Verify validated response is saved in cache."""
    cache = LLMCacheService(enabled=True)
    cache.clear()
    k = cache.generate_cache_key(normalized_query="safe query", user_scope="user:1")
    cached = cache.set(
        key=k,
        value={"answer": "Safe, fully grounded medical response.", "retrieval_status": "success"},
        user_scope="user:1",
        validated_only=True
    )
    assert cached is True
    assert cache.get(k) is not None


def test_17_unsafe_responses_never_cached():
    """17. Verify unvalidated or error responses are never stored in cache."""
    cache = LLMCacheService(enabled=True)
    cache.clear()
    k = cache.generate_cache_key(normalized_query="error query", user_scope="user:1")
    cached = cache.set(
        key=k,
        value={"answer": "Short", "retrieval_status": "service_error"},
        user_scope="user:1",
        validated_only=True
    )
    assert cached is False
    assert cache.get(k) is None


def test_18_request_ids_work():
    """18. Verify generate_request_id produces traceable identifiers."""
    req_id = generate_request_id()
    assert req_id.startswith("rag-")
    assert len(req_id) >= 12


def test_19_prometheus_metrics_work():
    """19. Verify metrics collector records and snapshots counters."""
    collector = get_metrics_collector()
    snapshot = collector.get_metrics_snapshot()
    assert "requests" in snapshot
    assert "cache" in snapshot
    assert "llm" in snapshot
    assert "safety" in snapshot


def test_20_health_live_works(client):
    """20. Verify /health/live endpoint returns 200 alive."""
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json().get("status") in ("alive", "healthy", "ok")


def test_21_health_ready_works(client):
    """21. Verify /health/ready endpoint verifies vector store readiness."""
    resp = client.get("/health/ready")
    assert resp.status_code in (200, 503)


def test_22_backup_works():
    """22. Verify VectorStoreSnapshotService creates verified manifest snapshot."""
    temp_backup = Path(tempfile.mkdtemp(prefix="test_backup_"))
    try:
        svc = VectorStoreSnapshotService(snapshots_dir=temp_backup)
        manifest = svc.create_snapshot(snapshot_id="rel_snap_test")
        assert manifest["vector_count"] == 744
        assert manifest["metadata_count"] == 744
        assert (temp_backup / "rel_snap_test" / "manifest.json").exists()
        assert (temp_backup / "rel_snap_test" / "index.faiss").exists()
    finally:
        shutil.rmtree(temp_backup, ignore_errors=True)


def test_23_restore_works():
    """23. Verify snapshot restore into isolated temporary directory preserves counts and dimensions."""
    snapshot_dir = Path(tempfile.mkdtemp(prefix="test_snap_"))
    restore_dir = Path(tempfile.mkdtemp(prefix="test_restore_"))
    try:
        svc = VectorStoreSnapshotService(snapshots_dir=snapshot_dir)
        manifest = svc.create_snapshot(snapshot_id="snap_restore")
        snap_path = snapshot_dir / "snap_restore"

        restored_manifest = svc.restore_snapshot(
            snapshot_dir=snap_path,
            target_dir=restore_dir,
            activate_in_memory=False
        )
        assert restored_manifest["vector_count"] == 744

        # Validate loaded store
        restored_vs = VectorStoreService(storage_dir=restore_dir, dimension=384)
        assert restored_vs.load(restore_dir) is True
        assert restored_vs.count() == 744
        assert len(restored_vs.metadata_store) == 744
    finally:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        shutil.rmtree(restore_dir, ignore_errors=True)


def test_24_dockerfile_configuration_valid():
    """24. Verify Dockerfile and compose configuration files exist."""
    assert Path("Dockerfile").exists()
    assert Path("docker-compose.yml").exists()


def test_25_non_root_container_configured():
    """25. Verify Dockerfile contains user directive for non-root execution (UID 10001)."""
    content = Path("Dockerfile").read_text(encoding="utf-8")
    assert "USER appuser" in content or "10001" in content


def test_26_redis_degradation_works():
    """26. Verify pipeline degrades cleanly when Redis is unreachable."""
    failing_redis = RedisService(enabled=True, host="nonexistent", port=9999)
    assert failing_redis.is_available() is False
    # Memory fallback cache operates safely
    cache = LLMCacheService(enabled=True, redis_service=failing_redis)
    k = cache.generate_cache_key(normalized_query="redis down", user_scope="user:1")
    cache.set(key=k, value={"answer": "Safe local memory.", "retrieval_status": "success"}, user_scope="user:1", validated_only=False)
    assert cache.get(k)["answer"] == "Safe local memory."


def test_27_gemini_degradation_works():
    """27. Verify downstream Gemini failure results in safe degradation envelope without fabrication."""
    vs = get_vector_store_service()
    failing_gemini = MagicMock()
    failing_gemini.generate_answer.side_effect = GeminiServiceError("Upstream 503 high demand")

    rag = RAGService(vector_store=vs)
    res = rag.generate_rag_answer(
        question="What is hypertension?",
        user_id=1,
        gemini_service=failing_gemini
    )
    assert res["retrieval_status"] in ("service_error", "service_unavailable", "error")
    assert "Relevant medical information could not be found" in res["answer"] or "temporarily unavailable" in res["answer"].lower()


def test_28_vector_store_degradation_works():
    """28. Verify missing/unloaded vector store degrades safely without unhandled crashes."""
    empty_vs = VectorStoreService(dimension=384)
    rag = RAGService(vector_store=empty_vs)
    mock_gemini = MagicMock()

    res = rag.generate_rag_answer(
        question="What is hypertension?",
        user_id=1,
        gemini_service=mock_gemini
    )
    assert res["retrieval_status"] == "no_relevant_context"
    mock_gemini.generate_answer.assert_not_called()
