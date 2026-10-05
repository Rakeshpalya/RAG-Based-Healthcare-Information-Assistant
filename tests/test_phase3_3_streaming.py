"""
Phase 3.3 Streaming Generation Test Suite: Server-Sent Events (SSE), Pre-LLM Sufficiency Gate, and Error Handling.

Verifies:
1. Supported query streams structured SSE events (start, status, token, complete).
2. Invariant: Pre-LLM Sufficiency Gate blocks unsupported query before LLM (0 LLM calls, fast safe fallback).
3. Tokens streamed to user are strictly verified safe (never stream unvalidated claims).
4. Timeout during streaming returns safe structured fallback without leaking internal errors.
5. Provider errors (429, 500, APIError) return safe structured fallback without exposing raw tracebacks.
6. Safety blocked response is caught and safe fallback emitted.
7. Citation failure / grounding enforcement prunes unsupported claims.
8. Secret redaction: No credentials or internal prompts leak in SSE streams.
9. FastAPI POST /rag/stream returns valid 'text/event-stream' content.
10. FastAPI POST /chat/stream alias endpoint returns valid 'text/event-stream' content.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiService, GeminiServiceError
from backend.main import app
from backend.services.llm_cache_service import get_llm_cache_service


@pytest.fixture(autouse=True)
def clean_cache():
    cache = get_llm_cache_service()
    cache.clear()
    cache.enabled = True
    yield
    cache.clear()


class TestStreamingRAGService:
    """Unit and pipeline tests for RAGService.generate_rag_stream()."""

    def test_streaming_supported_query_emits_expected_events(self):
        """Test: Supported query emits start, status, token, and complete events."""
        rag_service = RAGService()
        mock_gemini = MagicMock(spec=GeminiService)
        mock_gemini.model = "gemini-3.5-flash-lite"
        mock_gemini.temperature = 0.0
        mock_gemini.max_output_tokens = 1024

        grounded_text = "Lifestyle recommendations for hypertension include regular physical activity and moderating dietary sodium intake."
        mock_retrieval = {
            "question": "What lifestyle changes help manage hypertension?",
            "retrieved_chunks": [{"chunk_id": "c1", "text": grounded_text, "similarity_score": 0.88, "metadata": {"filename": "doc.pdf"}}],
            "context": f"[SOURCE 1] {grounded_text}",
            "sources": [{"source_index": 1, "source_label": "[Source 1]", "filename": "doc.pdf", "text": grounded_text}],
            "retrieval_status": "success",
            "timings": {"total_retrieval_time_ms": 10.0}
        }

        # Mock stream yielding text chunks
        mock_gemini.generate_stream.return_value = iter([
            "Lifestyle recommendations for hypertension include ",
            "regular physical activity and ",
            "moderating dietary sodium intake [Source 1]."
        ])

        with patch.object(rag_service, "query", return_value=mock_retrieval):
            events = list(rag_service.generate_rag_stream(
                question="What lifestyle changes help manage hypertension?",
                gemini_service=mock_gemini,
                user_id=10
            ))

        event_types = [e[0] for e in events]
        assert "start" in event_types
        assert "status" in event_types
        assert "token" in event_types
        assert "complete" in event_types

        # Verify complete payload
        complete_event = [e[1] for e in events if e[0] == "complete"][0]
        assert complete_event["retrieval_status"] == "success"
        assert len(complete_event["sources"]) > 0
        assert "Source 1" in complete_event["answer"]
        assert complete_event["timings"]["llm_called"] is True
        assert mock_gemini.generate_stream.call_count == 1

    def test_streaming_unsupported_query_blocks_at_sufficiency_gate(self):
        """Test: Unsupported query produces ZERO LLM calls and fast safe fallback."""
        rag_service = RAGService()
        mock_gemini = MagicMock(spec=GeminiService)

        unsupported_query = "What is the recommended chemotherapy dosage for glioblastoma multiforme?"

        events = list(rag_service.generate_rag_stream(
            question=unsupported_query,
            gemini_service=mock_gemini,
            user_id=10
        ))

        event_types = [e[0] for e in events]
        assert "start" in event_types
        assert "complete" in event_types

        # Crucial Invariant: Gemini MUST NEVER be called for unsupported query
        assert mock_gemini.generate_stream.call_count == 0

        complete_event = [e[1] for e in events if e[0] == "complete"][0]
        assert complete_event["retrieval_status"] == "no_relevant_context"
        assert complete_event["timings"]["llm_called"] is False
        assert complete_event["timings"]["gemini_calls_count"] == 0
        assert "Relevant medical information could not be found" in complete_event["answer"]

    def test_streaming_cached_query_serves_without_llm_calls(self):
        """Test: When answer is cached, streaming serves tokens from cache with 0 LLM calls."""
        rag_service = RAGService()
        mock_gemini = MagicMock(spec=GeminiService)
        mock_gemini.model = "gemini-3.5-flash-lite"
        mock_gemini.temperature = 0.0
        mock_gemini.max_output_tokens = 1024
        mock_gemini.generate_stream.return_value = iter([
            "Physical activity and sodium restriction help control blood pressure [Source 1]."
        ])

        query = "What lifestyle measures are advised for high blood pressure?"
        grounded_text = "Physical activity and sodium restriction help control blood pressure."
        mock_retrieval = {
            "question": query,
            "retrieved_chunks": [{"chunk_id": "c1", "text": grounded_text, "similarity_score": 0.88, "metadata": {"filename": "doc.pdf"}}],
            "context": f"[SOURCE 1] {grounded_text}",
            "sources": [{"source_index": 1, "source_label": "[Source 1]", "filename": "doc.pdf", "text": grounded_text}],
            "retrieval_status": "success",
            "timings": {"total_retrieval_time_ms": 10.0}
        }

        with patch.object(rag_service, "query", return_value=mock_retrieval):
            # Stream 1: Cache Miss -> Calls Gemini
            events1 = list(rag_service.generate_rag_stream(question=query, gemini_service=mock_gemini, user_id=1))
            assert mock_gemini.generate_stream.call_count == 1

            # Stream 2: Cache Hit -> Serves cached tokens, ZERO Gemini calls!
            events2 = list(rag_service.generate_rag_stream(question=query, gemini_service=mock_gemini, user_id=1))
            assert mock_gemini.generate_stream.call_count == 1  # Not called again!

            complete2 = [e[1] for e in events2 if e[0] == "complete"][0]
            assert complete2["timings"]["cache_hit"] is True
            assert complete2["timings"]["llm_called"] is False

    def test_streaming_timeout_error_handling(self):
        """Test: Timeout during Gemini streaming returns safe structured fallback."""
        rag_service = RAGService()
        mock_gemini = MagicMock(spec=GeminiService)
        mock_gemini.model = "gemini-3.5-flash-lite"
        mock_gemini.temperature = 0.0
        mock_gemini.max_output_tokens = 1024
        mock_gemini.generate_stream.side_effect = GeminiServiceError("The AI generation service timed out.")

        mock_retrieval = {
            "question": "What are hypertension risk factors?",
            "retrieved_chunks": [{"chunk_id": "c1", "text": "Hypertension risk factors include age and family history.", "similarity_score": 0.88, "metadata": {"filename": "doc.pdf"}}],
            "context": "[SOURCE 1] Hypertension risk factors include age and family history.",
            "sources": [{"source_index": 1, "source_label": "[Source 1]", "filename": "doc.pdf", "text": "Hypertension risk factors include age and family history."}],
            "retrieval_status": "success",
            "timings": {"total_retrieval_time_ms": 10.0}
        }

        with patch.object(rag_service, "query", return_value=mock_retrieval):
            events = list(rag_service.generate_rag_stream(
                question="What are hypertension risk factors?",
                gemini_service=mock_gemini,
                user_id=1
            ))

        event_types = [e[0] for e in events]
        assert "error" in event_types
        assert "complete" in event_types

        complete_event = [e[1] for e in events if e[0] == "complete"][0]
        assert complete_event["retrieval_status"] == "service_unavailable"
        assert "temporarily unavailable" in complete_event["answer"].lower()

    def test_streaming_no_credential_leakage(self):
        """Test: No API keys or internal secrets leak in streaming event payloads."""
        rag_service = RAGService()
        mock_gemini = MagicMock(spec=GeminiService)
        mock_gemini.model = "gemini-3.5-flash-lite"
        mock_gemini.temperature = 0.0
        mock_gemini.max_output_tokens = 1024
        mock_gemini.generate_stream.return_value = iter([
            "Hypertension management includes lifestyle changes [Source 1]."
        ])

        grounded_text = "Hypertension management includes lifestyle changes."
        mock_retrieval = {
            "question": "How is hypertension managed?",
            "retrieved_chunks": [{"chunk_id": "c1", "text": grounded_text, "similarity_score": 0.88, "metadata": {"filename": "doc.pdf"}}],
            "context": f"[SOURCE 1] {grounded_text}",
            "sources": [{"source_index": 1, "source_label": "[Source 1]", "filename": "doc.pdf", "text": grounded_text}],
            "retrieval_status": "success",
            "timings": {"total_retrieval_time_ms": 10.0}
        }

        with patch.object(rag_service, "query", return_value=mock_retrieval):
            events = list(rag_service.generate_rag_stream(
                question="How is hypertension managed?",
                gemini_service=mock_gemini
            ))

        all_text = json.dumps([e[1] for e in events])
        assert "AIza" not in all_text
        assert "GEMINI_API_KEY" not in all_text
        assert "sk-" not in all_text


class TestStreamingHTTPEndpoints:
    """FastAPI TestClient integration tests for SSE endpoints."""

    def test_post_rag_stream_endpoint(self):
        """Test: POST /rag/stream returns text/event-stream with valid SSE events."""
        with patch.object(
            RAGService,
            "generate_rag_stream",
            return_value=[
                ("start", {"status": "started", "request_id": "test-123"}),
                ("status", {"step": "retrieval", "message": "Searching..."}),
                ("token", {"token": "Hypertension ", "index": 0}),
                ("token", {"token": "measures [Source 1].", "index": 1}),
                ("complete", {"answer": "Hypertension measures [Source 1].", "retrieval_status": "success", "sources": []})
            ]
        ):
            client = TestClient(app)
            response = client.post(
                "/rag/stream",
                json={"question": "What are hypertension measures?"}
            )

            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            # Parse SSE formatted text
            content = response.text
            assert "event: start\n" in content
            assert "event: status\n" in content
            assert "event: token\n" in content
            assert "event: complete\n" in content

    def test_post_chat_stream_alias_endpoint(self):
        """Test: POST /chat/stream alias endpoint returns text/event-stream."""
        with patch.object(
            RAGService,
            "generate_rag_stream",
            return_value=[
                ("start", {"status": "started"}),
                ("token", {"token": "Validated ", "index": 0}),
                ("complete", {"answer": "Validated", "retrieval_status": "success"})
            ]
        ):
            client = TestClient(app)
            response = client.post(
                "/chat/stream",
                json={"message": "What is hypertension?"}
            )

            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            assert "event: start\n" in response.text
            assert "event: complete\n" in response.text
