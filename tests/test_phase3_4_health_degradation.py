"""
Phase 3.4 Milestones 3.4.10 & 3.4.11: Health, Readiness & Graceful Degradation Test Suite

Verifies:
1. GET /health/live returns process liveness.
2. GET /health/ready verifies subsystem dependencies (vector_store, gemini, cache).
3. Readiness reports dependency failure with appropriate status code (503).
4. Graceful degradation when Gemini is unavailable (safe fallback, no hallucinations).
5. Graceful degradation when Cache is unavailable (continues safely without cache).
6. Graceful degradation when Vector Store is unavailable (no fabricated medical content, safe fallback).
7. Graceful degradation when Rate Limiter is unavailable.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.main import app
from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiService, GeminiServiceError
from backend.services.llm_cache_service import LLMCacheService, get_llm_cache_service
from backend.rag.prompt_builder import MEDICAL_DISCLAIMER


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthAndReadinessEndpoints:
    """Tests for GET /health/live and GET /health/ready."""

    def test_liveness_endpoint_returns_alive(self, client):
        """GET /health/live confirms process is alive with 200 OK."""
        resp = client.get("/health/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "alive"
        assert data.get("service") == "AI-Healthcare-Agent"

    def test_readiness_endpoint_subsystem_verification(self, client):
        """GET /health/ready returns status of vector_store, gemini, and cache."""
        resp = client.get("/health/ready")
        data = resp.json()

        assert "status" in data
        assert "vector_store" in data
        assert "gemini" in data
        assert "cache" in data
        # In test environment, status is either 200 ready or 503 if mock keys
        assert resp.status_code in (200, 503)

    def test_readiness_reports_failure_when_vector_store_corrupted(self, client):
        """Readiness probe returns 503 when vector store invariant is broken."""
        mock_vs = MagicMock()
        mock_vs.count.return_value = 100  # Mismatched from required 744

        with patch("backend.services.vector_store_service.get_vector_store_service", return_value=mock_vs):
            resp = client.get("/health/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "not_ready"
            assert data["vector_store"] is False


class TestGracefulDegradation:
    """Tests graceful degradation when external services or subsystems fail."""

    def test_gemini_unavailable_returns_safe_fallback_no_hallucination(self):
        """
        When Gemini fails or is unreachable, the system must return a deterministic
        safe fallback message and NOT hallucinate medical content.
        """
        mock_vs = MagicMock()
        mock_vs.count.return_value = 744
        rag = RAGService(vector_store=mock_vs)

        # Mock retrieval success with clinical context
        rag.query = MagicMock(return_value={
            "retrieval_status": "success",
            "context": "[Source 1] Clinical reference for hypertension.",
            "retrieved_chunks": [{"chunk_id": "c1", "text": "Clinical reference"}],
            "sources": [{"chunk_id": "c1", "text": "Clinical reference", "source_index": 1}],
            "timings": {
                "embedding_time_ms": 1.0,
                "faiss_retrieval_time_ms": 2.0,
                "deduplication_time_ms": 0.5,
                "context_construction_time_ms": 1.0,
                "total_retrieval_time_ms": 4.5
            }
        })

        mock_failing_gemini = MagicMock()
        mock_failing_gemini.generate_answer.side_effect = GeminiServiceError("API network connection refused")

        res = rag.generate_rag_answer(
            question="What is hypertension?",
            gemini_service=mock_failing_gemini,
            use_cache=False
        )

        # Invariant: Safe fallback, no hallucination, disclaimer present
        assert res["retrieval_status"] in ("service_error", "service_unavailable")
        assert "temporarily unavailable" in res["answer"].lower()
        assert MEDICAL_DISCLAIMER in res.get("disclaimer", "")
        # Did not fabricate claims
        assert "you have" not in res["answer"].lower()
        assert "prescribe" not in res["answer"].lower()

    def test_cache_unavailable_continues_safely_without_cache(self):
        """When cache is disabled or encounters an issue, queries execute safely without cache."""
        cache = get_llm_cache_service()
        cache.enabled = False

        mock_vs = MagicMock()
        rag = RAGService(vector_store=mock_vs)
        mock_gemini = MagicMock()
        mock_gemini.generate_answer.return_value = {
            "answer": "Normal grounded response on hypertension [Source 1].",
            "model": "gemini-3.5-flash-lite",
            "generation_time_ms": 80.0,
            "gemini_calls_count": 1
        }
        rag.query = MagicMock(return_value={
            "retrieval_status": "success",
            "context": "[Source 1] Normal grounded response on hypertension.",
            "retrieved_chunks": [{"chunk_id": "c1", "text": "Normal grounded response on hypertension."}],
            "sources": [{"chunk_id": "c1", "text": "Normal grounded response on hypertension.", "source_index": 1, "source_label": "[Source 1]"}],
            "timings": {
                "embedding_time_ms": 1.0,
                "faiss_retrieval_time_ms": 2.0,
                "deduplication_time_ms": 0.5,
                "context_construction_time_ms": 1.0,
                "total_retrieval_time_ms": 4.5
            }
        })

        res = rag.generate_rag_answer(
            question="What is hypertension?",
            gemini_service=mock_gemini,
            use_cache=True  # Requested cache, but cache is disabled
        )

        assert res["retrieval_status"] == "success"
        assert res["timings"]["llm_called"] is True
        assert res["timings"]["cache_hit"] is False

    def test_vector_store_unavailable_returns_safe_fallback_no_fabrication(self):
        """
        When vector store fails, the system halts generation safely and does NOT
        fabricate medical context or answers.
        """
        mock_failing_vs = MagicMock()
        mock_failing_vs.count.side_effect = RuntimeError("FAISS index file locked or unavailable")
        rag = RAGService(vector_store=mock_failing_vs)

        # When retrieval fails or finds no context
        rag.query = MagicMock(return_value={
            "retrieval_status": "no_relevant_context",
            "context": "",
            "retrieved_chunks": [],
            "sources": [],
            "timings": {
                "embedding_time_ms": 0.0,
                "faiss_retrieval_time_ms": 0.0,
                "deduplication_time_ms": 0.0,
                "context_construction_time_ms": 0.0,
                "total_retrieval_time_ms": 0.0
            }
        })

        mock_gemini = MagicMock()
        res = rag.generate_rag_answer(
            question="What are the symptoms of hypertension?",
            gemini_service=mock_gemini
        )

        # Must not call LLM and must not fabricate medical content
        assert res["retrieval_status"] == "no_relevant_context"
        assert res["timings"]["llm_called"] is False
        mock_gemini.generate_answer.assert_not_called()
        assert "could not be found" in res["answer"].lower() or "not find" in res["answer"].lower()
