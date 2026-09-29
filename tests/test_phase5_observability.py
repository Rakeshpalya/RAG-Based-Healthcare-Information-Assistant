import json
import pytest
from backend.evaluation.observability import (
    generate_request_id,
    sanitize_value,
    sanitize_log_dict,
    RAGStructuredLogEvent,
    StructuredRAGLogger
)
from backend.evaluation.latency_evaluator import LatencyEvaluator, LatencySummary


def test_generate_request_id():
    """Verify unique traceable request ID generation."""
    id1 = generate_request_id()
    id2 = generate_request_id()
    assert id1.startswith("rag-")
    assert id2.startswith("rag-")
    assert id1 != id2
    assert len(id1) >= 10


def test_credential_sanitization():
    """Verify sensitive credentials, keys, tokens, and SSNs are thoroughly redacted."""
    raw_payload = {
        "api_key": "AIzaSyD-abc12345678901234567890123456",
        "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-IDc-TEST",
        "user_query": "Patient SSN is 000-12-3456 with severe headache",
        "normal_text": "Symptoms of hypertension",
        "nested": {
            "password": "SuperSecretPassword123!",
            "gemini_api_key": "secret_gemini_key",
            "openai_api_key": "sk-123456789012345678901234"
        }
    }

    sanitized = sanitize_log_dict(raw_payload)

    assert sanitized["api_key"] == "[REDACTED_CREDENTIAL]"
    assert sanitized["authorization"] == "[REDACTED_CREDENTIAL]"
    assert "[REDACTED_SSN]" in sanitized["user_query"]
    assert "000-12-3456" not in sanitized["user_query"]
    assert sanitized["normal_text"] == "Symptoms of hypertension"
    assert sanitized["nested"]["password"] == "[REDACTED_CREDENTIAL]"
    assert sanitized["nested"]["gemini_api_key"] == "[REDACTED_CREDENTIAL]"
    assert sanitized["nested"]["openai_api_key"] == "[REDACTED_CREDENTIAL]"


def test_structured_log_event_serialization():
    """Verify RAGStructuredLogEvent serialization to dict and JSON."""
    event = RAGStructuredLogEvent(
        request_id="rag-test12345",
        query="What is hypertension?",
        safety_classification="benign",
        risk_level="none",
        retrieval_status="success",
        embedding_latency_ms=12.5,
        vector_search_latency_ms=4.2,
        context_construction_latency_ms=1.1,
        llm_latency_ms=450.0,
        total_latency_ms=467.8,
        number_of_retrieved_chunks=3,
        similarity_scores=[0.88, 0.76, 0.65],
        citation_validation_passed=True,
        citation_coverage=1.0,
        hallucination_detected=False,
        hallucination_types=[],
        final_status="SUCCESS"
    )

    d = event.to_dict()
    assert d["request_id"] == "rag-test12345"
    assert d["total_latency_ms"] == 467.8
    assert d["citation_validation_passed"] is True

    json_str = event.to_json()
    parsed = json.loads(json_str)
    assert parsed["query"] == "What is hypertension?"
    assert parsed["final_status"] == "SUCCESS"


def test_structured_logger_emission():
    """Verify StructuredRAGLogger emits structured logs safely."""
    event = RAGStructuredLogEvent(
        request_id="rag-emit001",
        query="Safe medical query",
        safety_classification="benign",
        risk_level="none",
        retrieval_status="success",
        embedding_latency_ms=10.0,
        vector_search_latency_ms=5.0,
        context_construction_latency_ms=1.0,
        llm_latency_ms=200.0,
        total_latency_ms=216.0,
        number_of_retrieved_chunks=2,
        similarity_scores=[0.85, 0.72],
        citation_validation_passed=True,
        citation_coverage=1.0,
        hallucination_detected=False,
        hallucination_types=[],
        final_status="SUCCESS"
    )

    emitted = StructuredRAGLogger.emit_rag_log(event, use_json=True)
    assert emitted["request_id"] == "rag-emit001"


def test_latency_evaluator_percentiles():
    """Verify LatencyEvaluator computes Mean, Median, P95, and P99 correctly."""
    # 100 sample latencies from 1 to 100 ms
    samples = [float(i) for i in range(1, 101)]

    summary = LatencyEvaluator.compute_summary("test_stage", samples)

    assert isinstance(summary, LatencySummary)
    assert summary.count == 100
    assert summary.min_ms == 1.0
    assert summary.max_ms == 100.0
    # Mean of 1..100 = 50.5
    assert summary.mean_ms == 50.5
    # Median = 50.5
    assert summary.median_ms == 50.5
    # P95 should be ~95.05
    assert 94.0 <= summary.p95_ms <= 96.0
    # P99 should be ~99.01
    assert 98.0 <= summary.p99_ms <= 100.0


def test_latency_evaluator_empty_samples():
    """Verify LatencyEvaluator handles empty samples gracefully."""
    summary = LatencyEvaluator.compute_summary("empty_stage", [])
    assert summary.count == 0
    assert summary.mean_ms == 0.0
    assert summary.p95_ms == 0.0
