"""
Unit tests for Phase 9 Structured Observability and Logging.
Validates HTTP request logging, PHI query masking, correlation tracking,
and credential sanitization.
"""

import os
import json
import pytest
from backend.evaluation.observability import (
    generate_request_id,
    mask_clinical_query,
    sanitize_log_dict,
    HTTPRequestLogEvent,
    RAGStructuredLogEvent,
    StructuredRAGLogger
)


def test_clinical_query_masking_in_safe_mode():
    """Ensures clinical queries are masked in safe mode but preserved in dev mode."""
    query = "Patient has severe acute chest pain radiating to left arm and shortness of breath"

    # Dev mode: safe_mode=False
    dev_logged = mask_clinical_query(query, safe_mode=False)
    assert dev_logged == query

    # Production/Safe mode: safe_mode=True
    safe_logged = mask_clinical_query(query, safe_mode=True)
    assert "MASKED_PHI" in safe_logged
    assert "Patient has" in safe_logged
    assert "shortness of breath" not in safe_logged
    assert "len=" in safe_logged
    assert "hash=" in safe_logged


def test_http_request_log_event():
    """Validates HTTP request log event generation, serialization, and sanitization."""
    req_id = generate_request_id()
    http_event = HTTPRequestLogEvent(
        request_id=req_id,
        endpoint="/rag/query",
        method="POST",
        status_code=200,
        latency_ms=142.5,
        retrieval_status="context_grounded",
        safety_status="PASSED",
        error_category=None,
        client_ip="127.0.0.1"
    )

    d = http_event.to_dict()
    assert d["request_id"] == req_id
    assert d["endpoint"] == "/rag/query"
    assert d["method"] == "POST"
    assert d["status_code"] == 200
    assert d["latency_ms"] == 142.5
    assert d["safety_status"] == "PASSED"

    json_str = http_event.to_json()
    parsed = json.loads(json_str)
    assert parsed["request_id"] == req_id
    assert parsed["status_code"] == 200


def test_extended_sanitization_pii():
    """Ensures emails, phone numbers, and bearer tokens are sanitized."""
    raw_dict = {
        "user_email": "doctor.smith@hospital.org",
        "emergency_contact": "Call 1-800-555-0199 immediately",
        "auth_header": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0",
        "safe_field": "Standard clinical diagnosis"
    }

    sanitized = sanitize_log_dict(raw_dict)
    assert sanitized["user_email"] == "[REDACTED_EMAIL]"
    assert "[REDACTED_PHONE]" in sanitized["emergency_contact"]
    assert "[REDACTED]" in sanitized["auth_header"]
    assert sanitized["safe_field"] == "Standard clinical diagnosis"


def test_structured_http_logger_emission():
    """Validates emission of HTTP log events via StructuredRAGLogger."""
    req_id = generate_request_id()
    event = HTTPRequestLogEvent(
        request_id=req_id,
        endpoint="/health",
        method="GET",
        status_code=200,
        latency_ms=4.8
    )

    emitted_dict = StructuredRAGLogger.emit_http_log(event, use_json=True)
    assert emitted_dict["request_id"] == req_id
    assert emitted_dict["endpoint"] == "/health"
    assert emitted_dict["status_code"] == 200
