"""
Phase 3.6 — Production Deployment & Reliability: LLM Resilience Tests.

Verifies the reliability, timeout bounds, and retry classification of GeminiService:
1. Successful generation (prompt + context)
2. Bounded timeout handling
3. Connection failure with secret redaction
4. Transient 5xx (503 Service Unavailable) backoff and recovery
5. Rate limit (429 Resource Exhausted) handling
6. Malformed provider response (empty / missing candidates)
7. Cancellation handling (immediate abort, no retries)
8. Bounded retry exhaustion (never retries indefinitely)
9. Safety blocks are never retried as provider failures
10. Failed / unsafe LLM responses are never cached in RAGService
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from concurrent.futures import CancelledError

from backend.services.gemini_service import GeminiService, GeminiServiceError
from backend.services.llm_cache_service import LLMCacheService
from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import VectorStoreService


class MockResponse:
    def __init__(self, text="Metformin is first-line pharmacotherapy.", finish_reason="STOP"):
        self.text = text
        candidate = MagicMock()
        candidate.finish_reason = finish_reason
        part = MagicMock()
        part.text = text
        candidate.content.parts = [part]
        self.candidates = [candidate]
        self.usage_metadata = MagicMock()
        self.usage_metadata.prompt_token_count = 150
        self.usage_metadata.candidates_token_count = 50


@pytest.fixture(autouse=True)
def fast_retry_timings(monkeypatch):
    """Speed up retry backoffs during tests so tests execute in milliseconds."""
    monkeypatch.setattr(GeminiService, "INITIAL_RETRY_DELAY_SEC", 0.001)
    monkeypatch.setattr(GeminiService, "MAX_RETRY_DELAY_SEC", 0.01)


def test_01_successful_generation():
    """Verify standard successful generation completes cleanly."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MockResponse("Grounded answer.")

    svc = GeminiService(api_key="mock-key-12345", model="gemini-3.5-flash-lite")
    svc.set_client(mock_client)

    result = svc.generate_answer(question="What is hypertension?", context="Hypertension is high blood pressure.")
    assert "Grounded answer" in result["answer"]
    assert result["model"] == "gemini-3.5-flash-lite"
    assert result["gemini_calls_count"] == 1


def test_02_timeout_handling():
    """Verify request timeout raises GeminiServiceError with sanitized message and does not hang."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = TimeoutError("Deadline exceeded: 504 Gateway Timeout")

    svc = GeminiService(api_key="mock-key-12345", model="gemini-3.5-flash-lite")
    svc.set_client(mock_client)

    with pytest.raises(GeminiServiceError) as exc_info:
        svc.generate_answer(question="What is diabetes?", context="Diabetes context.")

    assert "timed out" in str(exc_info.value).lower() or "timeout" in str(exc_info.value).lower()
    assert "mock-key-12345" not in str(exc_info.value)


def test_03_connection_failure_bounded_and_sanitized():
    """Verify connection failures retry within bounds and never leak secrets."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = ConnectionResetError("Connection refused by endpoint AIzaSySecretKey999")

    svc = GeminiService(api_key="AIzaSySecretKey999", model="gemini-3.5-flash-lite")
    svc.set_client(mock_client)

    with pytest.raises(GeminiServiceError) as exc_info:
        svc.generate_answer(question="What is asthma?", context="Asthma context.")

    msg = str(exc_info.value)
    assert "AIzaSySecretKey999" not in msg
    assert "[REDACTED_API_KEY]" in msg or "failed across all models" in msg


def test_04_transient_5xx_backoff_and_recovery():
    """Verify transient 503 error retries and succeeds on next attempt without failing."""
    mock_client = MagicMock()
    # First attempt fails with 503, second attempt succeeds
    mock_client.models.generate_content.side_effect = [
        Exception("503 Service Unavailable: High demand temporarily"),
        MockResponse("Recovered answer on retry.")
    ]

    svc = GeminiService(api_key="mock-key-12345", model="gemini-3.5-flash-lite")
    svc.set_client(mock_client)

    result = svc.generate_answer(question="What is asthma?", context="Asthma context.")
    assert "Recovered answer on retry" in result["answer"]
    assert result["gemini_calls_count"] == 2


def test_05_rate_limit_429_exhausted():
    """Verify 429 Resource Exhausted / Rate limit converts to high-demand user-friendly error."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 Resource exhausted: Rate limit exceeded")

    svc = GeminiService(api_key="mock-key-12345", model="gemini-3.5-flash-lite")
    svc.set_client(mock_client)

    with pytest.raises(GeminiServiceError) as exc_info:
        svc.generate_answer(question="What is asthma?", context="Asthma context.")

    assert "temporarily unavailable due to high demand" in str(exc_info.value).lower()


def test_06_malformed_provider_response():
    """Verify malformed response (empty or None text) fails gracefully without crashing."""
    mock_client = MagicMock()
    # Returns object with empty text and no candidates
    empty_resp = MagicMock()
    empty_resp.text = ""
    empty_resp.candidates = []
    mock_client.models.generate_content.return_value = empty_resp

    svc = GeminiService(api_key="mock-key-12345", model="gemini-3.5-flash-lite")
    svc.set_client(mock_client)

    with pytest.raises(GeminiServiceError) as exc_info:
        svc.generate_answer(question="What is asthma?", context="Asthma context.")

    assert "empty response" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()


def test_07_cancellation_handling():
    """Verify cancelled requests abort immediately without looping through retries or fallback models."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = CancelledError("Request was cancelled by client")

    svc = GeminiService(api_key="mock-key-12345", model="gemini-3.5-flash-lite")
    svc.set_client(mock_client)

    with pytest.raises(CancelledError):
        svc.generate_answer(question="What is asthma?", context="Asthma context.")

    # Must abort on the very first attempt (1 call, zero retries)
    assert mock_client.models.generate_content.call_count == 1


def test_08_retry_exhaustion_is_bounded():
    """Verify retries never loop indefinitely and stop after candidate models are exhausted."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("Persistent unrecoverable internal error")

    svc = GeminiService(api_key="mock-key-12345", model="gemini-3.5-flash-lite")
    svc.set_client(mock_client)

    with pytest.raises(GeminiServiceError):
        svc.generate_answer(question="What is asthma?", context="Asthma context.")

    # Number of calls must be strictly bounded: len(candidate_models) * (MAX_RETRIES + 1)
    # Non-transient errors break out of retry loop immediately per candidate model
    total_calls = mock_client.models.generate_content.call_count
    assert total_calls > 0
    assert total_calls <= len(svc.FALLBACK_MODELS) + 1


def test_09_safety_blocks_never_retried_as_provider_failures():
    """Verify safety blocks raise immediately without retrying as if they were server errors."""
    mock_client = MagicMock()
    # Response with finish_reason="SAFETY" and empty text
    safety_resp = MagicMock()
    safety_resp.text = ""
    candidate = MagicMock()
    candidate.finish_reason = "SAFETY"
    safety_resp.candidates = [candidate]
    mock_client.models.generate_content.return_value = safety_resp

    svc = GeminiService(api_key="mock-key-12345", model="gemini-3.5-flash-lite")
    svc.set_client(mock_client)

    with pytest.raises(GeminiServiceError) as exc_info:
        svc.generate_answer(question="Dangerous prompt", context="Context")

    assert "safety filters" in str(exc_info.value).lower()
    # Must NOT retry: call count should be exactly 1
    assert mock_client.models.generate_content.call_count == 1


def test_10_unsuccessful_llm_responses_never_cached():
    """Verify that when LLM fails or errors out, RAGService never writes error responses to cache."""
    cache = LLMCacheService(enabled=True)
    cache.clear()

    # Pre-populate dummy vector store with a valid chunk
    vs = VectorStoreService(dimension=384)
    vs.add_chunks([{
        "chunk_id": "CHUNK_RES_001",
        "document_id": "DOC_RES_001",
        "page_number": 1,
        "text": "Metformin hydrochloride is an oral biguanide antihyperglycemic medication.",
        "category": "Pharmacology"
    }])

    failing_gemini = MagicMock()
    failing_gemini.generate_answer.side_effect = GeminiServiceError("LLM downstream service failure")

    with patch("backend.services.llm_cache_service.get_llm_cache_service", return_value=cache):
        rag = RAGService(vector_store=vs)
        res = rag.generate_rag_answer(
            question="What is metformin hydrochloride?",
            user_id=1,
            gemini_service=failing_gemini
        )

    assert res["retrieval_status"] in ("service_error", "service_unavailable", "error")
    # Verify cache is completely empty - error was NOT stored
    assert len(cache._cache) == 0
    assert cache._hits == 0
