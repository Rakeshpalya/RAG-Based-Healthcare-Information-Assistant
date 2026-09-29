"""
Unit tests for frontend backend connectivity helper and environment resolution.
Tests check_backend_health(), resolve_backend_url(), and error isolation.
"""

import os
from unittest.mock import patch, MagicMock
import requests
import pytest

from frontend.api_client import APIClient, resolve_backend_url


def test_resolve_backend_url_priority():
    """Verify URL resolution follows strict priority order."""
    # 1. Explicit argument overrides everything
    with patch.dict(os.environ, {"API_BASE_URL": "http://env-api:8000", "BACKEND_URL": "http://env-backend:8000"}):
        assert resolve_backend_url("http://explicit:9000") == "http://explicit:9000"

    # 2. API_BASE_URL takes precedence over BACKEND_URL
    with patch.dict(os.environ, {"API_BASE_URL": "http://api-base:8000", "BACKEND_URL": "http://backend-url:8000"}):
        assert resolve_backend_url() == "http://api-base:8000"

    # 3. BACKEND_URL takes precedence over FASTAPI_URL
    with patch.dict(os.environ, {"BACKEND_URL": "http://backend-url:8000", "FASTAPI_URL": "http://fastapi-url:8000"}, clear=True):
        assert resolve_backend_url() == "http://backend-url:8000"

    # 4. FASTAPI_URL used when others are absent
    with patch.dict(os.environ, {"FASTAPI_URL": "http://fastapi-url:8000"}, clear=True):
        assert resolve_backend_url() == "http://fastapi-url:8000"

    # 5. HOST and PORT used when URLs absent
    with patch.dict(os.environ, {"HOST": "0.0.0.0", "PORT": "8080"}, clear=True):
        assert resolve_backend_url() == "http://0.0.0.0:8080"


def test_resolve_backend_url_normalizes_localhost():
    """Verify localhost is normalized to 127.0.0.1 for Windows IPv4 stability."""
    assert resolve_backend_url("http://localhost:8000") == "http://127.0.0.1:8000"
    assert resolve_backend_url("http://localhost:8000/") == "http://127.0.0.1:8000"
    assert resolve_backend_url("https://localhost:8443") == "https://127.0.0.1:8443"
    assert resolve_backend_url("http://localhost") == "http://127.0.0.1"


def test_check_backend_health_success():
    """Verify check_backend_health returns connected=True on HTTP 200."""
    client = APIClient(base_url="http://127.0.0.1:8000")
    with patch("requests.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.json.return_value = {
            "status": "healthy",
            "service": "AI-Healthcare-Agent",
            "environment": "development",
        }
        mock_get.return_value = mock_res

        res = client.check_backend_health(timeout=1.5)
        assert res["connected"] is True
        assert res["status"] == "healthy"
        assert res["error"] is None
        assert res["service"] == "AI-Healthcare-Agent"


def test_check_backend_health_connection_error():
    """Verify check_backend_health handles ConnectionError gracefully."""
    client = APIClient(base_url="http://127.0.0.1:8000")
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        res = client.check_backend_health(timeout=1.5)
        assert res["connected"] is False
        assert res["status"] == "disconnected"
        assert "Ensure FastAPI is running" in res["error"]


def test_check_backend_health_timeout():
    """Verify check_backend_health handles Timeout gracefully."""
    client = APIClient(base_url="http://127.0.0.1:8000")
    with patch("requests.get", side_effect=requests.exceptions.Timeout("Read timeout")):
        res = client.check_backend_health(timeout=2.0)
        assert res["connected"] is False
        assert res["status"] == "timeout"
        assert "timed out after 2 seconds" in res["error"]


def test_check_backend_health_never_leaks_secrets():
    """Verify check_backend_health response contains no credentials or database details."""
    client = APIClient(base_url="http://127.0.0.1:8000")
    with patch("requests.get") as mock_get:
        mock_res = MagicMock()
        mock_res.status_code = 500
        mock_res.text = "Internal Server Error with postgresql://user:secret@db.supabase.co"
        mock_get.return_value = mock_res

        res = client.check_backend_health(timeout=1.5)
        assert res["connected"] is False
        assert "secret" not in str(res).lower()
        assert "postgresql" not in str(res).lower()
