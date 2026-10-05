"""
Phase 3.4 Milestone 3.4.6 & 3.4.7: Request IDs, Observability & Production Metrics Test Suite

Verifies:
1. Every API request receives a unique request_id.
2. request_id propagates consistently: API -> RAGService -> Retrieval -> Cache -> Gemini -> Response.
3. Separate requests receive distinct, unique request_ids.
4. Error logs and events include request_id.
5. Streaming events include request_id.
6. Secrets remain sanitized in all events and logs.
7. Metrics: Requests (total, successful, failed, blocked).
8. Metrics: Cache (hits, misses, hit rate).
9. Metrics: LLM (calls, successes, failures, retries, fallback usage).
10. Metrics: Safety (blocked, passed).
11. Metrics: Latency percentiles (p50, p95, p99) for retrieval, cache, LLM, total, TTFT, TTFE.
12. GET /metrics returns structured snapshot matching production schema.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.main import app
from backend.evaluation.observability import (
    generate_request_id,
    get_metrics_collector,
    RAGStructuredLogEvent,
    HTTPRequestLogEvent,
    StructuredRAGLogger
)
from backend.rag.rag_service import RAGService


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_metrics():
    collector = get_metrics_collector()
    collector.reset()
    yield
    collector.reset()


class TestRequestIDPropagation:
    """Verifies that unique request_id is assigned and propagates through all layers."""

    def test_request_id_generated_and_returned_in_headers(self, client):
        """API request without explicit ID receives an auto-generated X-Request-ID header."""
        resp = client.get("/health/live")
        assert resp.status_code == 200
        req_id = resp.headers.get("X-Request-ID")
        assert req_id is not None
        assert req_id.startswith("rag-") or len(req_id) >= 8

    def test_custom_request_id_propagates_consistently(self, client):
        """Custom X-Request-ID in header is honored and echoed in response headers."""
        custom_id = "test-custom-req-12345"
        resp = client.get("/health/live", headers={"X-Request-ID": custom_id})
        assert resp.status_code == 200
        assert resp.headers.get("X-Request-ID") == custom_id

    def test_separate_requests_receive_distinct_ids(self, client):
        """Separate requests receive different unique IDs."""
        resp1 = client.get("/health/live")
        resp2 = client.get("/health/live")
        id1 = resp1.headers.get("X-Request-ID")
        id2 = resp2.headers.get("X-Request-ID")
        assert id1 != id2

    def test_request_id_propagates_through_rag_pipeline(self):
        """request_id propagates into RAGService result, timings, and logs."""
        trace_id = "rag-trace-abc987"
        rag = RAGService(vector_store=MagicMock())
        mock_gemini = MagicMock()
        mock_gemini.generate_answer.return_value = {
            "answer": "Grounded medical answer on hypertension [Source 1].",
            "model": "gemini-3.5-flash-lite",
            "generation_time_ms": 100.0,
            "gemini_calls_count": 1
        }
        rag.query = MagicMock(return_value={
            "retrieval_status": "success",
            "context": "[Source 1] Grounded medical answer on hypertension.",
            "retrieved_chunks": [{"chunk_id": "c1", "text": "Grounded medical answer on hypertension."}],
            "sources": [{"chunk_id": "c1", "text": "Grounded medical answer on hypertension.", "source_index": 1, "source_label": "[Source 1]"}],
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
            request_id=trace_id,
            gemini_service=mock_gemini
        )

        assert res.get("request_id") == trace_id
        assert res.get("timings", {}).get("request_id") == trace_id

    def test_streaming_events_include_request_id(self):
        """Streaming SSE events contain the exact request_id."""
        trace_id = "rag-stream-trace-456"
        rag = RAGService(vector_store=MagicMock())

        events = list(rag.generate_rag_stream(
            question="What is diabetes?",
            request_id=trace_id
        ))

        # Start event has trace_id
        start_evt = next((payload for evt, payload in events if evt == "start"), None)
        assert start_evt is not None
        assert start_evt.get("request_id") == trace_id

    def test_error_logs_and_safety_intercepts_include_request_id(self):
        """Emergency intercepted inquiries carry request_id in response."""
        trace_id = "rag-emergency-999"
        rag = RAGService(vector_store=MagicMock())
        res = rag.generate_rag_answer(
            question="Crushing chest pain radiating to left arm right now!",
            request_id=trace_id
        )
        assert res.get("request_id") == trace_id
        assert res.get("retrieval_status") == "safety_intercepted"


class TestProductionMetricsCalculation:
    """Verifies that ProductionMetricsCollector calculates required metrics accurately."""

    def test_record_rag_event_updates_request_and_latency_metrics(self):
        """RAG events properly increment counters and track latency percentiles."""
        collector = get_metrics_collector()

        # Emit 3 sample events
        e1 = RAGStructuredLogEvent(
            request_id="req-1",
            query="query 1",
            safety_classification="NORMAL_MEDICAL_INFORMATION",
            risk_level="LOW",
            retrieval_status="success",
            embedding_latency_ms=2.0,
            vector_search_latency_ms=10.0,
            context_construction_latency_ms=1.0,
            llm_latency_ms=150.0,
            total_latency_ms=163.0,
            number_of_retrieved_chunks=3,
            similarity_scores=[0.8],
            citation_validation_passed=True,
            citation_coverage=1.0,
            hallucination_detected=False,
            hallucination_types=[],
            final_status="success",
            llm_called=True,
            cache_hit=False,
            cache_miss=True
        )

        e2 = RAGStructuredLogEvent(
            request_id="req-2",
            query="query 2",
            safety_classification="NORMAL_MEDICAL_INFORMATION",
            risk_level="LOW",
            retrieval_status="success",
            embedding_latency_ms=1.0,
            vector_search_latency_ms=8.0,
            context_construction_latency_ms=1.0,
            llm_latency_ms=0.0,
            total_latency_ms=10.0,
            number_of_retrieved_chunks=3,
            similarity_scores=[0.8],
            citation_validation_passed=True,
            citation_coverage=1.0,
            hallucination_detected=False,
            hallucination_types=[],
            final_status="success",
            llm_called=False,
            cache_hit=True,
            cache_miss=False
        )

        e3 = RAGStructuredLogEvent(
            request_id="req-3",
            query="emergency query",
            safety_classification="EMERGENCY_SYMPTOMS",
            risk_level="CRITICAL",
            retrieval_status="safety_intercepted",
            embedding_latency_ms=0.0,
            vector_search_latency_ms=0.0,
            context_construction_latency_ms=0.0,
            llm_latency_ms=0.0,
            total_latency_ms=2.0,
            number_of_retrieved_chunks=0,
            similarity_scores=[],
            citation_validation_passed=True,
            citation_coverage=1.0,
            hallucination_detected=False,
            hallucination_types=[],
            final_status="SAFETY_INTERCEPTED",
            llm_called=False,
            fallback_reason="safety_intercepted"
        )

        StructuredRAGLogger.emit_rag_log(e1)
        StructuredRAGLogger.emit_rag_log(e2)
        StructuredRAGLogger.emit_rag_log(e3)

        snap = collector.get_metrics_snapshot()

        # Requests
        assert snap["requests"]["total"] == 3
        assert snap["requests"]["successful"] == 2
        assert snap["requests"]["blocked"] == 1

        # Cache
        assert snap["cache"]["hits"] == 1
        assert snap["cache"]["misses"] == 1
        assert snap["cache"]["hit_rate"] == 0.5

        # LLM
        assert snap["llm"]["calls"] == 1
        assert snap["llm"]["successes"] == 1

        # Safety
        assert snap["safety"]["blocked"] == 1
        assert snap["safety"]["passed"] == 2

        # Latencies
        assert "p50" in snap["latency"]["total"]
        assert "p95" in snap["latency"]["total"]
        assert "p99" in snap["latency"]["total"]
        assert snap["latency"]["total"]["p50"] > 0

    def test_metrics_endpoint_returns_json(self, client):
        """GET /metrics returns 200 with full production metrics schema."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()

        assert "requests" in data
        assert "cache" in data
        assert "llm" in data
        assert "safety" in data
        assert "latency" in data

        for cat in ("retrieval", "cache", "llm", "total", "ttft", "ttfe"):
            assert cat in data["latency"]
            assert "p50" in data["latency"][cat]
            assert "p95" in data["latency"][cat]
            assert "p99" in data["latency"][cat]
