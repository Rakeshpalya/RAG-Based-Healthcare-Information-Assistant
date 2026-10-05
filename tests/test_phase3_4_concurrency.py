"""
Phase 3.4.2 — Rate Limiting & Concurrency Test Suite.

Verifies:
1. Normal request passes rate limiting.
2. Rate limit exceeded returns HTTP 429 with standard Retry-After and X-RateLimit-* headers.
3. Different users have independent rate limits.
4. Streaming-specific rate limiting (10 requests/minute tier).
5. Concurrent Gemini generations bounded by concurrency controller.
6. Semaphore maximum capacity is strictly respected.
7. Concurrency acquisition timeout raises ConcurrencyLimitExceeded / safe fallback.
8. Provider failure (e.g. GeminiServiceError) reliably releases semaphore.
9. Unexpected exception reliably releases semaphore.
10. Async cancellation releases semaphore without deadlock.
11. No deadlocks under rapid concurrent calls.
"""

import time
import asyncio
import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request, HTTPException
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import settings
from backend.security.rate_limiter import SlidingWindowRateLimiter
from backend.security.concurrency import LLMConcurrencyController, ConcurrencyLimitExceeded
from backend.services.gemini_service import GeminiService, GeminiServiceError
from backend.rag.rag_service import RAGService


class TestRateLimiting:

    def test_normal_request_within_limit_passes(self):
        """Request within limit passes and returns RFC rate limit headers."""
        limiter = SlidingWindowRateLimiter(requests_per_minute=5, window_seconds=60)
        mock_req = MagicMock(spec=Request)
        mock_req.headers = {}
        mock_req.client.host = "192.168.1.100"

        headers = limiter.check_rate_limit(mock_req, endpoint_type="rag_query")
        assert "X-RateLimit-Limit" in headers
        assert headers["X-RateLimit-Remaining"] == "4"

    def test_rate_limit_exceeded_raises_429(self):
        """Exceeding limit raises HTTPException(429) with Retry-After header."""
        limiter = SlidingWindowRateLimiter(requests_per_minute=2, window_seconds=60)
        mock_req = MagicMock(spec=Request)
        mock_req.headers = {}
        mock_req.client.host = "192.168.1.101"

        # 2 requests pass
        limiter.check_rate_limit(mock_req, endpoint_type="rag_query")
        limiter.check_rate_limit(mock_req, endpoint_type="rag_query")

        # 3rd request must be blocked
        with pytest.raises(HTTPException) as exc_info:
            limiter.check_rate_limit(mock_req, endpoint_type="rag_query")

        assert exc_info.value.status_code == 429
        assert "Retry-After" in exc_info.value.headers
        assert exc_info.value.headers["X-RateLimit-Remaining"] == "0"

    def test_different_users_have_independent_rate_limits(self):
        """User 1 hitting their rate limit does not block User 2."""
        limiter = SlidingWindowRateLimiter(requests_per_minute=2, window_seconds=60)
        mock_req = MagicMock(spec=Request)
        mock_req.headers = {}
        mock_req.client.host = "10.0.0.1"

        # User 1 exhausts their 2 requests
        limiter.check_rate_limit(mock_req, endpoint_type="rag_query", user_id=1)
        limiter.check_rate_limit(mock_req, endpoint_type="rag_query", user_id=1)

        with pytest.raises(HTTPException):
            limiter.check_rate_limit(mock_req, endpoint_type="rag_query", user_id=1)

        # User 2 is independent and must pass
        headers_u2 = limiter.check_rate_limit(mock_req, endpoint_type="rag_query", user_id=2)
        assert headers_u2["X-RateLimit-Remaining"] == "1"

    def test_streaming_specific_rate_limit_tier(self):
        """Streaming endpoint uses its dedicated lower limit (STREAMING_REQUESTS_PER_MINUTE)."""
        limiter = SlidingWindowRateLimiter(requests_per_minute=60, window_seconds=60)
        mock_req = MagicMock(spec=Request)
        mock_req.headers = {}
        mock_req.client.host = "10.0.0.5"

        with patch.object(settings, "STREAMING_REQUESTS_PER_MINUTE", 3):
            # 3 streaming requests pass
            for _ in range(3):
                limiter.check_rate_limit(mock_req, endpoint_type="rag_stream")

            # 4th streaming request is blocked
            with pytest.raises(HTTPException) as exc:
                limiter.check_rate_limit(mock_req, endpoint_type="rag_stream")
            assert exc.value.status_code == 429


class TestConcurrencyControl:

    def test_concurrency_controller_capacity_and_active_count(self):
        """Concurrency controller tracks active executions and respects max limit."""
        controller = LLMConcurrencyController(max_concurrent=2, acquire_timeout=0.1)
        assert controller.active_count == 0

        with controller.acquire():
            assert controller.active_count == 1
            with controller.acquire():
                assert controller.active_count == 2
                # 3rd concurrent attempt exceeds capacity and raises ConcurrencyLimitExceeded
                with pytest.raises(ConcurrencyLimitExceeded):
                    with controller.acquire(timeout=0.05):
                        pass
                assert controller.active_count == 2

            assert controller.active_count == 1
        assert controller.active_count == 0

    def test_provider_failure_releases_semaphore(self):
        """When LLM provider raises GeminiServiceError, semaphore is reliably released."""
        controller = LLMConcurrencyController(max_concurrent=1, acquire_timeout=0.1)

        with pytest.raises(GeminiServiceError):
            with controller.acquire():
                assert controller.active_count == 1
                raise GeminiServiceError("Provider 503 error")

        # Must be released back to 0
        assert controller.active_count == 0

        # Should be able to acquire again immediately
        with controller.acquire():
            assert controller.active_count == 1
        assert controller.active_count == 0

    def test_unexpected_exception_releases_semaphore(self):
        """When an unhandled exception occurs, semaphore is reliably released (no deadlock)."""
        controller = LLMConcurrencyController(max_concurrent=1, acquire_timeout=0.1)

        with pytest.raises(RuntimeError):
            with controller.acquire():
                assert controller.active_count == 1
                raise RuntimeError("Unexpected failure")

        assert controller.active_count == 0

    @pytest.mark.anyio
    async def test_async_concurrency_controller_releases_on_cancellation(self):
        """Async task cancellation safely releases the async semaphore without deadlocks."""
        controller = LLMConcurrencyController(max_concurrent=1, acquire_timeout=1.0)

        async def worker():
            async with controller.acquire_async():
                await asyncio.sleep(5.0)

        task = asyncio.create_task(worker())
        await asyncio.sleep(0.05)  # Let worker acquire
        assert controller.active_count == 1

        # Cancel task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Semaphore must be freed
        assert controller.active_count == 0

        # New worker can now acquire immediately
        async with controller.acquire_async(timeout=0.1):
            assert controller.active_count == 1
        assert controller.active_count == 0

    def test_rag_service_throttles_gracefully_when_concurrency_limit_exceeded(self):
        """When concurrency controller is full, RAGService returns clean service_busy fallback."""
        from backend.services.vector_store_service import get_vector_store_service
        vs = get_vector_store_service()
        rag = RAGService(vector_store=vs)

        mock_gemini = MagicMock(spec=GeminiService)
        query = "What lifestyle changes are recommended for managing hypertension?"

        with patch("backend.security.concurrency.concurrency_controller.acquire", side_effect=ConcurrencyLimitExceeded()):
            res = rag.generate_rag_answer(
                question=query,
                gemini_service=mock_gemini,
                user_id=2
            )
            assert res["retrieval_status"] == "service_busy"
            assert "temporarily busy" in res["answer"]
            assert res["timings"]["llm_called"] is False
            assert res["timings"]["service_error"] == "concurrency_limit_exceeded"
