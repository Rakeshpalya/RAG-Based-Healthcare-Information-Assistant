"""
Phase 3.5.3 Tests: Prometheus Observability Metrics & Exposition.

Verifies:
1. /metrics returns valid Prometheus exposition (version 0.0.4) on standard scrape requests.
2. Counters increase as RAG events and HTTP requests occur.
3. Histograms work correctly with buckets, sum, and count.
4. Strict low-cardinality enforcement: NEVER exposes user_id, request_id, queries, or document_ids as labels.
5. Zero secret leakage: no API keys or passwords in /metrics.
6. Zero PHI/clinical leakage in Prometheus metric lines.
7. Backward compatibility: /metrics continues returning structured JSON for standard API consumers.
"""

import re
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.evaluation.observability import (
    get_metrics_collector,
    RAGStructuredLogEvent,
    HTTPRequestLogEvent,
    StructuredRAGLogger
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_metrics():
    collector = get_metrics_collector()
    collector.reset()
    yield
    collector.reset()


def test_metrics_prometheus_exposition_format(client):
    """Verify /metrics returns valid RFC-compliant Prometheus exposition."""
    # Test via Accept header (Prometheus scraper standard)
    resp = client.get("/metrics", headers={"Accept": "text/plain"})
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    text = resp.text

    # Must contain essential metric families with HELP and TYPE
    expected_families = [
        "rag_requests_total",
        "rag_cache_hits_total",
        "rag_cache_misses_total",
        "rag_cache_errors_total",
        "rag_cache_hit_ratio",
        "rag_llm_calls_total",
        "rag_llm_successes_total",
        "rag_llm_failures_total",
        "rag_llm_retries_total",
        "rag_llm_active_concurrency",
        "rag_safety_blocked_total",
        "rag_safety_emergency_blocks_total",
        "rag_safety_prompt_injection_blocks_total",
        "rag_validation_failures_total",
        "rag_retrieval_total",
        "rag_insufficient_context_total",
        "rag_request_duration_seconds",
        "rag_retrieval_latency_seconds",
        "rag_cache_latency_seconds",
        "rag_llm_latency_seconds",
        "rag_ttft_seconds",
        "rag_ttfe_seconds",
    ]

    for family in expected_families:
        assert f"# HELP {family}" in text, f"Missing HELP for {family}"
        assert f"# TYPE {family}" in text, f"Missing TYPE for {family}"


def test_metrics_counters_increment(client):
    """Verify counters increase as events are recorded."""
    collector = get_metrics_collector()

    # Emit a successful grounded RAG event
    e_success = RAGStructuredLogEvent(
        request_id="rag-test-111",
        query="What causes hypertension?",
        safety_classification="GENERAL_MEDICAL_INQUIRY",
        risk_level="LOW",
        retrieval_status="success",
        embedding_latency_ms=10.0,
        vector_search_latency_ms=25.0,
        context_construction_latency_ms=5.0,
        llm_latency_ms=300.0,
        total_latency_ms=340.0,
        number_of_retrieved_chunks=3,
        similarity_scores=[0.85, 0.75],
        citation_validation_passed=True,
        citation_coverage=1.0,
        hallucination_detected=False,
        hallucination_types=[],
        final_status="SUCCESS",
        user_id=1234,
        llm_called=True,
        cache_hit=False,
        cache_miss=True
    )
    StructuredRAGLogger.emit_rag_log(e_success)

    # Emit an emergency safety blocked event
    e_emergency = RAGStructuredLogEvent(
        request_id="rag-test-222",
        query="I have crushing chest pain radiating to my arm",
        safety_classification="EMERGENCY_SYMPTOMS",
        risk_level="CRITICAL",
        retrieval_status="safety_intercepted",
        embedding_latency_ms=0.0,
        vector_search_latency_ms=0.0,
        context_construction_latency_ms=0.0,
        llm_latency_ms=0.0,
        total_latency_ms=5.0,
        number_of_retrieved_chunks=0,
        similarity_scores=[],
        citation_validation_passed=True,
        citation_coverage=1.0,
        hallucination_detected=False,
        hallucination_types=[],
        final_status="SAFETY_INTERCEPTED",
        user_id=5678,
        llm_called=False
    )
    StructuredRAGLogger.emit_rag_log(e_emergency)

    resp = client.get("/metrics?format=prometheus")
    assert resp.status_code == 200
    text = resp.text

    assert 'rag_requests_total{status="successful"} 1' in text
    assert 'rag_requests_total{status="blocked"} 1' in text
    assert "rag_llm_calls_total 1" in text
    assert "rag_safety_blocked_total 1" in text
    assert "rag_safety_emergency_blocks_total 1" in text
    assert "rag_cache_misses_total 1" in text
    assert "rag_retrieval_total 1" in text


def test_metrics_histograms(client):
    """Verify Prometheus histograms correctly calculate buckets, count, and sum."""
    collector = get_metrics_collector()

    e = RAGStructuredLogEvent(
        request_id="rag-hist-1",
        query="What is type 2 diabetes?",
        safety_classification="GENERAL_MEDICAL_INQUIRY",
        risk_level="LOW",
        retrieval_status="success",
        embedding_latency_ms=15.0,
        vector_search_latency_ms=45.0,
        context_construction_latency_ms=5.0,
        llm_latency_ms=500.0,
        total_latency_ms=565.0,
        number_of_retrieved_chunks=2,
        similarity_scores=[0.88],
        citation_validation_passed=True,
        citation_coverage=1.0,
        hallucination_detected=False,
        hallucination_types=[],
        final_status="SUCCESS",
        llm_called=True,
        time_to_first_token_ms=120.0
    )
    StructuredRAGLogger.emit_rag_log(e)

    resp = client.get("/metrics/prometheus")
    assert resp.status_code == 200
    text = resp.text

    # Request duration histogram: 565 ms = 0.565 s -> should be in le="1.0" bucket and +Inf
    assert 'rag_request_duration_seconds_count 1' in text
    assert 'rag_request_duration_seconds_bucket{le="+Inf"} 1' in text
    assert 'rag_request_duration_seconds_bucket{le="1.0"} 1' in text

    # TTFT histogram: 120 ms = 0.12 s -> in le="0.25" bucket
    assert 'rag_ttft_seconds_count 1' in text
    assert 'rag_ttft_seconds_bucket{le="0.25"} 1' in text


def test_low_cardinality_no_user_ids_or_requests():
    """
    CRITICAL PROMETHEUS REQUIREMENT:
    Labels must NOT contain high-cardinality values:
    user_id, request_id, query text, or document_id.
    """
    collector = get_metrics_collector()

    private_user_id = 998877
    private_req_id = "rag-secret-token-xyz"
    private_query = "Sensitive patient clinical condition 404"
    private_doc = "confidential_health_record_123.pdf"

    e = RAGStructuredLogEvent(
        request_id=private_req_id,
        query=private_query,
        safety_classification="GENERAL_MEDICAL_INQUIRY",
        risk_level="LOW",
        retrieval_status="success",
        embedding_latency_ms=10.0,
        vector_search_latency_ms=20.0,
        context_construction_latency_ms=5.0,
        llm_latency_ms=200.0,
        total_latency_ms=235.0,
        number_of_retrieved_chunks=1,
        similarity_scores=[0.9],
        citation_validation_passed=True,
        citation_coverage=1.0,
        hallucination_detected=False,
        hallucination_types=[],
        final_status="SUCCESS",
        user_id=private_user_id,
        llm_called=True
    )
    StructuredRAGLogger.emit_rag_log(e)

    exposition = collector.get_prometheus_exposition()

    # None of these private high-cardinality fields may appear in Prometheus text
    assert str(private_user_id) not in exposition
    assert private_req_id not in exposition
    assert private_query not in exposition
    assert private_doc not in exposition
    assert "user_id" not in exposition
    assert "request_id" not in exposition


def test_no_secrets_in_metrics_exposition(client):
    """Verify no API keys or credentials appear in the metrics output."""
    resp = client.get("/metrics/prometheus")
    assert resp.status_code == 200
    text = resp.text

    assert "AIza" not in text
    assert "sk-" not in text
    assert "password" not in text.lower()
    assert "secret" not in text.lower()


def test_backward_compatibility_json_snapshot(client):
    """Verify GET /metrics without scrape headers returns backward-compatible JSON snapshot."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    data = resp.json()

    # Required Phase 3.4 keys
    assert "requests" in data
    assert "cache" in data
    assert "llm" in data
    assert "safety" in data
    assert "latency" in data
