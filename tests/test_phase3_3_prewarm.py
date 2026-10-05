"""
Phase 3.3 Gemini Client Pre-Warming & Singleton Management Tests.

Verifies:
1. Client initialized once and reused across requests.
2. Pre-warming does NOT invoke an external generation call.
3. Missing or invalid API key is handled safely without crashing.
4. Credentials and API keys are redacted from logs, errors, and status outputs.
5. FastAPI lifespan context manager warms the client at startup safely.
6. Observability: warm status and duration are captured.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.services.gemini_service import GeminiService, GeminiServiceError
from backend.main import lifespan, app


@pytest.fixture(autouse=True)
def reset_gemini_client():
    """Ensure clean client state before and after each test."""
    original_client = GeminiService._client
    original_warm = GeminiService._client_warm
    GeminiService.reset_client()
    yield
    GeminiService._client = original_client
    GeminiService._client_warm = original_warm


class TestGeminiPreWarmAndSingleton:
    """Tests verifying singleton reuse, pre-warming, and credential safety."""

    def test_singleton_client_reuse_across_instances(self):
        """Test: Repeated requests across different service instances reuse the exact same client."""
        mock_client = MagicMock()
        GeminiService.set_client(mock_client)

        svc1 = GeminiService()
        svc2 = GeminiService()

        c1 = svc1.get_client()
        c2 = svc2.get_client()

        assert c1 is c2
        assert c1 is mock_client

    def test_prewarm_does_not_invoke_llm_generation(self):
        """Test: Pre-warming initializes SDK client but does NOT make generation calls."""
        mock_genai_client = MagicMock()

        with patch("google.genai.Client", return_value=mock_genai_client):
            status = GeminiService.prewarm_client(api_key="AIzaSyDummyTestKeyForPrewarm12345678901")

            assert status["warm"] is True
            assert status["status"] == "success"
            assert status["warmup_duration_ms"] >= 0.0
            assert GeminiService.is_client_warm() is True

            # Verify NO generation was invoked
            assert mock_genai_client.models.generate_content.call_count == 0
            assert mock_genai_client.models.generate_content_stream.call_count == 0

    def test_prewarm_already_warm_idempotency(self):
        """Test: Repeated prewarm calls are idempotent and return already_warm."""
        mock_client = MagicMock()
        GeminiService.set_client(mock_client)
        GeminiService._client_warm = True

        status = GeminiService.prewarm_client(api_key="dummy_key")
        assert status["status"] == "already_warm"
        assert status["warm"] is True

    def test_missing_api_key_handled_safely(self):
        """Test: Missing API key fails safely without crashing startup."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            with patch("backend.config.settings.validate_gemini_api_key", side_effect=ValueError("GEMINI_API_KEY is not configured")):
                status = GeminiService.prewarm_client(api_key=None)
                assert status["warm"] is False
                assert status["status"] == "unconfigured"
                assert "not configured" in status.get("error", "")

    def test_secret_redaction_in_prewarm_and_errors(self):
        """Test: Raw API key is NEVER exposed in pre-warm outputs or error messages."""
        real_fake_key = "AIzaSyTestSecretKeyMustNeverBeLogged1234"
        with patch("google.genai.Client", side_effect=Exception(f"Failed connection with {real_fake_key}")):
            status = GeminiService.prewarm_client(api_key=real_fake_key)
            assert status["warm"] is False
            assert status["status"] == "failed"
            error_str = status.get("error", "")
            # Ensure raw key was redacted
            assert real_fake_key not in error_str
            assert "[REDACTED_API_KEY]" in error_str

    def test_fastapi_lifespan_prewarms_client(self):
        """Test: FastAPI application startup runs lifespan pre-warming without error."""
        with patch.object(GeminiService, "prewarm_client", return_value={"status": "success", "warm": True, "warmup_duration_ms": 1.23}) as mock_prewarm:
            with TestClient(app) as client:
                res = client.get("/health")
                assert res.status_code == 200
                data = res.json()
                assert "llm_cache" in data
                assert "gemini_client_warm" in data
            assert mock_prewarm.call_count >= 1
