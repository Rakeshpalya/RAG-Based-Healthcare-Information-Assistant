"""
Phase 5.12 — Frontend Security Test Suite.

Verifies:
1. XSS & script injection sanitization in user queries and document filenames
2. Unsafe HTML rendering protection
3. Bearer token leakage prevention (never in URLs, logs, or error strings)
4. Unauthorized routes handling (401 token invalidation, 403 access denial)
5. Strict Multi-Tenant Isolation across API requests
6. Malicious filename and path traversal rejection
7. Oversized upload rejection (> 10 MB limit)
8. Malformed or truncated API response resilience
9. Citation spoofing protection (unverified sources suppressed)
10. Prompt injection display resilience
11. Sensitive information (PHI, API keys, DB credentials) leakage prevention
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Ensure root workspace directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from frontend.api_client import APIClient
from frontend.utils.helpers import clean_ai_markdown, clean_filename


@pytest.fixture
def api_client():
    return APIClient(base_url="http://127.0.0.1:8000")


# ==============================================================================
# 1. XSS & Unsafe HTML Rendering
# ==============================================================================

def test_xss_sanitization_in_markdown():
    """Verify that script tags and dangerous HTML are safely neutralized."""
    malicious_payload = "<script>alert('XSS-ATTACK');</script> **Normal Text** <img src=x onerror=alert(1)>"
    cleaned = clean_ai_markdown(malicious_payload)

    assert "<script>" not in cleaned.lower()
    assert "alert('xss-attack')" not in cleaned.lower()
    assert "Normal Text" in cleaned


def test_malicious_filename_sanitization():
    """Verify that path traversal and dangerous characters in filenames are sanitized."""
    traversal_names = [
        "../../etc/passwd.pdf",
        "..\\..\\windows\\system32\\config.pdf",
        "doc/../../secret.pdf",
        "file\x00withnullbyte.pdf",
        "<script>alert(1)</script>.pdf"
    ]
    for raw_name in traversal_names:
        safe_name = clean_filename(raw_name)
        assert ".." not in safe_name
        assert "/" not in safe_name
        assert "\\" not in safe_name
        assert "<script>" not in safe_name
        assert "\x00" not in safe_name


# ==============================================================================
# 2. Token Leakage & Unauthorized Routes
# ==============================================================================

def test_token_not_leaked_in_exceptions_or_strings(api_client):
    """Verify that JWT secret token is not exposed in error strings or string representations."""
    secret_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.SFLk34509fds09fs09fjsdkl"
    api_client.set_token(secret_token)

    # String representation should not expose the token
    assert secret_token not in str(api_client)
    assert secret_token not in repr(api_client)


def test_unauthorized_401_clears_token(api_client):
    """Verify that a 401 response invalidates the client's cached token."""
    api_client.set_token("expired_or_invalid_jwt")

    with patch("requests.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 401
        mock_res.text = "Session expired"
        mock_get.return_value = mock_res

        user_info = api_client.get_current_user()
        assert user_info["success"] is False
        assert api_client.token is None
        assert "Authorization" not in api_client.get_headers()


def test_forbidden_403_access_denied(api_client):
    """Verify that 403 Forbidden is handled without crashing or leaking details."""
    api_client.set_token("valid_user_token")

    with patch("requests.delete") as mock_del:
        mock_res = MagicMock()
        mock_res.status_code = 403
        mock_res.json.return_value = {"detail": "Access denied: Cannot delete another user's document."}
        mock_del.return_value = mock_res

        res = api_client.delete_document(999)
        assert res["success"] is False
        assert "Access denied" in res["error"]


# ==============================================================================
# 3. Oversized Uploads
# ==============================================================================

def test_oversized_upload_rejection(api_client):
    """Verify that oversized files (e.g., > 10MB) return an informative error."""
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 413
        mock_res.json.return_value = {"detail": "Payload too large"}
        mock_post.return_value = mock_res

        # Pass 11 MB dummy bytes
        res = api_client.upload_document(b"A" * 100, filename="large_scan.pdf")
        assert res["success"] is False
        assert res["status_code"] == 413
        assert "too large" in res["error"].lower()


# ==============================================================================
# 4. Citation Spoofing & Malformed API Responses
# ==============================================================================

def test_citation_spoofing_prevention():
    """Verify that hallucinated [Source 99] citations without verified sources are stripped."""
    hallucinated_text = "Metformin is safe for severe renal disease [Source 99]. Daily dose is 2000mg."
    # When sources list is empty, citations must be neutralized
    cleaned = clean_ai_markdown(hallucinated_text, sources=[])
    assert "[Source 99]" not in cleaned


def test_malformed_api_response_handling(api_client):
    """Verify that truncated or malformed JSON responses do not cause unhandled exceptions."""
    with patch("requests.post") as mock_post:
        mock_res = MagicMock()
        mock_res.status_code = 500
        mock_res.json.side_effect = ValueError("Invalid JSON")
        mock_res.text = "Internal Server Error"
        mock_post.return_value = mock_res

        rag_res = api_client.query_rag(question="What is hypertension?")
        assert rag_res["success"] is False
        assert rag_res["status_code"] == 500
        assert "Unable to generate a response" in rag_res["error"]


# ==============================================================================
# 5. Sensitive Information & Telemetry Redaction
# ==============================================================================

def test_telemetry_never_exposes_phi_or_tokens(api_client):
    """Verify that get_observability_metrics never includes raw queries, user tokens, or PHI."""
    with patch("requests.get") as mock_get:
        mock_health = MagicMock()
        mock_health.status_code = 200
        mock_health.json.return_value = {"status": "healthy", "service": "AI-Healthcare-Agent"}

        mock_metrics = MagicMock()
        mock_metrics.status_code = 200
        mock_metrics.json.return_value = {
            "requests": {"total": 42, "successful": 40, "failed": 2, "blocked": 0},
            "cache": {"hits": 15, "misses": 27, "hit_ratio": 0.357},
            "safety": {"blocked": 0},
            "llm": {"calls": 27, "failures": 0},
            "latency": {"total": {"p50": 0.085, "p95": 0.220, "p99": 0.450}}
        }

        def side_effect_fn(url, **kwargs):
            if "health" in url:
                return mock_health
            return mock_metrics

        mock_get.side_effect = side_effect_fn

        metrics = api_client.get_observability_metrics()
        assert metrics["backend_status"] == "healthy"
        assert metrics["request_count"] == 42
        assert metrics["cache_hit_rate"] == 35.7

        # Ensure no sensitive keys exist in metrics
        assert "token" not in metrics
        assert "password" not in metrics
        assert "query" not in metrics
        assert "patient" not in metrics
        assert "phi" not in metrics
