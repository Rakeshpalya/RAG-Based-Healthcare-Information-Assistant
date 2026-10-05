"""
Phase 3.3 Cache Test Suite: Comprehensive Safety, Isolation, TTL, Eviction, and Invariant Verification.

Verifies:
1. Cache HIT on identical query, user, and document scope.
2. Cache MISS on different user (strict user isolation).
3. Cache MISS on different document / signature.
4. Cache MISS on changed document version.
5. Cache MISS on different model or generation configuration.
6. Cache MISS on different prompt version.
7. Cache MISS on TTL expiration.
8. Cache disabled behavior (bypasses cache when disabled).
9. Invariant: Unvalidated, empty, or error responses are NEVER cached.
10. Cached responses preserve valid citations, sources, and disclaimers.
11. Cached responses CANNOT bypass MedicalSafetyGuard.
12. Cache isolation prevents cross-user medical response leakage.
13. LRU eviction respects max_entries bound.
14. Cache invalidation via delete_by_user and clear.
15. End-to-end RAGService integration: Hit avoids LLM call (llm_called=False, gemini_calls_count=0).
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from backend.services.llm_cache_service import (
    LLMCacheService,
    get_llm_cache_service,
    PROMPT_VERSION
)
from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiService


@pytest.fixture(autouse=True)
def clean_cache():
    """Ensure a clean, isolated cache for every test."""
    cache = get_llm_cache_service()
    cache.clear()
    cache.enabled = True
    yield
    cache.clear()
    cache.enabled = True


class TestLLMCacheSafetyAndIsolation:
    """Unit tests for LLMCacheService security, isolation, and bounds."""

    def test_cache_hit_identical_query_and_scope(self):
        """Test 1: Same query + same document + same user -> cache HIT."""
        cache = LLMCacheService(enabled=True, ttl_seconds=60, max_entries=100)
        key = cache.generate_cache_key(
            normalized_query="What are hypertension symptoms?",
            user_scope=101,
            document_signature="doc_sig_abc",
            model="gemini-3.5-flash-lite"
        )
        sample_response = {
            "question": "What are hypertension symptoms?",
            "answer": "Common symptoms include headache and dizziness [Source 1].",
            "retrieval_status": "success",
            "sources": [{"source_index": 1, "source_label": "[Source 1]"}]
        }
        stored = cache.set(key, sample_response, user_scope=101, document_signature="doc_sig_abc")
        assert stored is True

        cached = cache.get(key)
        assert cached is not None
        assert cached["answer"] == sample_response["answer"]
        assert cached["timings"]["cache_hit"] is True
        assert cached["timings"]["llm_called"] is False

    def test_cache_miss_different_user_isolation(self):
        """Test 2: Same query + different user -> cache MISS (user isolation)."""
        cache = LLMCacheService(enabled=True, ttl_seconds=60, max_entries=100)
        q = "What are the lifestyle measures for hypertension?"
        sig = "doc_sig_common"

        key_user1 = cache.generate_cache_key(normalized_query=q, user_scope=1, document_signature=sig)
        key_user2 = cache.generate_cache_key(normalized_query=q, user_scope=2, document_signature=sig)

        assert key_user1 != key_user2

        cache.set(key_user1, {
            "question": q,
            "answer": "Dietary sodium reduction is recommended [Source 1].",
            "retrieval_status": "success"
        }, user_scope=1, document_signature=sig)

        # User 1 hits cache
        assert cache.get(key_user1) is not None
        # User 2 misses cache
        assert cache.get(key_user2) is None

    def test_cache_miss_different_document_scope(self):
        """Test 3: Same query + different document -> cache MISS."""
        cache = LLMCacheService(enabled=True, ttl_seconds=60, max_entries=100)
        q = "What are the recommended treatments?"
        key_doc_a = cache.generate_cache_key(normalized_query=q, user_scope=10, document_signature="doc_a_hash")
        key_doc_b = cache.generate_cache_key(normalized_query=q, user_scope=10, document_signature="doc_b_hash")

        assert key_doc_a != key_doc_b

        cache.set(key_doc_a, {
            "question": q,
            "answer": "Treatment A recommendations [Source 1].",
            "retrieval_status": "success"
        }, user_scope=10, document_signature="doc_a_hash")

        assert cache.get(key_doc_a) is not None
        assert cache.get(key_doc_b) is None

    def test_cache_miss_changed_document_version(self):
        """Test 4: Same query + changed document version -> cache MISS."""
        cache = LLMCacheService(enabled=True, ttl_seconds=60, max_entries=100)
        q = "What are the clinical criteria?"
        key_v1 = cache.generate_cache_key(normalized_query=q, user_scope=5, document_signature="version_1_hash")
        key_v2 = cache.generate_cache_key(normalized_query=q, user_scope=5, document_signature="version_2_hash")

        cache.set(key_v1, {
            "question": q,
            "answer": "Version 1 guidelines [Source 1].",
            "retrieval_status": "success"
        }, user_scope=5, document_signature="version_1_hash")

        assert cache.get(key_v1) is not None
        assert cache.get(key_v2) is None

    def test_cache_miss_changed_model(self):
        """Test 5: Same query + changed model -> cache MISS."""
        cache = LLMCacheService(enabled=True, ttl_seconds=60, max_entries=100)
        q = "Summarize the findings."
        key_flash = cache.generate_cache_key(normalized_query=q, model="gemini-3.5-flash-lite")
        key_pro = cache.generate_cache_key(normalized_query=q, model="gemini-2.5-flash")

        assert key_flash != key_pro
        cache.set(key_flash, {"answer": "Flash summary [Source 1].", "retrieval_status": "success"})
        assert cache.get(key_flash) is not None
        assert cache.get(key_pro) is None

    def test_cache_miss_changed_prompt_version(self):
        """Test 6: Same query + changed prompt version -> cache MISS."""
        cache = LLMCacheService(enabled=True, ttl_seconds=60, max_entries=100)
        q = "Explain the mechanism."
        key_pv1 = cache.generate_cache_key(normalized_query=q, prompt_version="v3.2.0")
        key_pv2 = cache.generate_cache_key(normalized_query=q, prompt_version="v3.3.0")

        assert key_pv1 != key_pv2
        cache.set(key_pv1, {"answer": "Mechanism explanation [Source 1].", "retrieval_status": "success"})
        assert cache.get(key_pv1) is not None
        assert cache.get(key_pv2) is None

    def test_cache_expiration_ttl(self):
        """Test 7: Expired cache -> cache MISS."""
        cache = LLMCacheService(enabled=True, ttl_seconds=1, max_entries=100)
        key = cache.generate_cache_key(normalized_query="Expiring question")
        cache.set(key, {
            "answer": "This will expire quickly [Source 1].",
            "retrieval_status": "success"
        }, ttl_seconds=1)

        # Immediate lookup succeeds
        assert cache.get(key) is not None

        # Wait for expiration
        time.sleep(1.05)
        assert cache.get(key) is None
        stats = cache.stats()
        assert stats["expirations"] >= 1

    def test_cache_disabled_no_usage(self):
        """Test 8: Cache disabled -> no cache usage."""
        cache = LLMCacheService(enabled=False, ttl_seconds=60, max_entries=100)
        key = cache.generate_cache_key(normalized_query="Disabled cache test")
        stored = cache.set(key, {
            "answer": "Will not be stored [Source 1].",
            "retrieval_status": "success"
        })
        assert stored is False
        assert cache.get(key) is None
        assert cache.stats()["total_entries"] == 0

    def test_invalid_unvalidated_generation_never_cached(self):
        """Test 9: Invalid, ungrounded, or empty generation is NEVER cached."""
        cache = LLMCacheService(enabled=True, ttl_seconds=60, max_entries=100)
        key1 = cache.generate_cache_key(normalized_query="q1")
        key2 = cache.generate_cache_key(normalized_query="q2")
        key3 = cache.generate_cache_key(normalized_query="q3")

        # Refusal/fallback status: must not be cached
        res_no_ctx = {"answer": "No relevant info found.", "retrieval_status": "no_relevant_context"}
        assert cache.set(key1, res_no_ctx, validated_only=True) is False

        # Error status: must not be cached
        res_err = {"answer": "An error occurred.", "retrieval_status": "service_unavailable"}
        assert cache.set(key2, res_err, validated_only=True) is False

        # Empty/short answer: must not be cached
        res_empty = {"answer": "", "retrieval_status": "success"}
        assert cache.set(key3, res_empty, validated_only=True) is False

        assert cache.get(key1) is None
        assert cache.get(key2) is None
        assert cache.get(key3) is None

    def test_lru_eviction_bounded_capacity(self):
        """Test 13: Bounded capacity with LRU eviction."""
        cache = LLMCacheService(enabled=True, ttl_seconds=60, max_entries=3)
        k1 = cache.generate_cache_key(normalized_query="k1")
        k2 = cache.generate_cache_key(normalized_query="k2")
        k3 = cache.generate_cache_key(normalized_query="k3")
        k4 = cache.generate_cache_key(normalized_query="k4")

        cache.set(k1, {"answer": "Answer 1 [Source 1].", "retrieval_status": "success"})
        cache.set(k2, {"answer": "Answer 2 [Source 1].", "retrieval_status": "success"})
        cache.set(k3, {"answer": "Answer 3 [Source 1].", "retrieval_status": "success"})
        assert len(cache._cache) == 3

        # Access k1 to make it most recently used; k2 is now LRU
        _ = cache.get(k1)

        # Insert k4, triggering eviction of k2
        cache.set(k4, {"answer": "Answer 4 [Source 1].", "retrieval_status": "success"})
        assert len(cache._cache) == 3
        assert cache.get(k2) is None
        assert cache.get(k1) is not None
        assert cache.get(k3) is not None
        assert cache.get(k4) is not None
        assert cache.stats()["evictions"] >= 1

    def test_cache_invalidation_by_user(self):
        """Test 14: Cache invalidation by user scope on document update."""
        cache = LLMCacheService(enabled=True, ttl_seconds=60, max_entries=100)
        k_u1_a = cache.generate_cache_key(normalized_query="u1_a", user_scope=100)
        k_u1_b = cache.generate_cache_key(normalized_query="u1_b", user_scope=100)
        k_u2 = cache.generate_cache_key(normalized_query="u2", user_scope=200)

        cache.set(k_u1_a, {"answer": "u1 answer a [Source 1].", "retrieval_status": "success"}, user_scope=100)
        cache.set(k_u1_b, {"answer": "u1 answer b [Source 1].", "retrieval_status": "success"}, user_scope=100)
        cache.set(k_u2, {"answer": "u2 answer [Source 1].", "retrieval_status": "success"}, user_scope=200)

        # Invalidate User 100 entries (e.g. user uploaded a new document)
        deleted = cache.delete_by_user(100)
        assert deleted == 2
        assert cache.get(k_u1_a) is None
        assert cache.get(k_u1_b) is None
        assert cache.get(k_u2) is not None


class TestRAGServiceCacheIntegration:
    """Integration tests verifying RAGService caching behavior and safety guards."""

    def test_rag_service_cache_hit_avoids_llm_call(self):
        """Test 10 & 15: Cache hit returns valid citations, sources, and makes 0 LLM calls."""
        from backend.services.vector_store_service import get_vector_store_service
        vs = get_vector_store_service()
        rag_service = RAGService(vector_store=vs)
        mock_gemini = MagicMock(spec=GeminiService)
        mock_gemini.model = "gemini-3.5-flash-lite"
        mock_gemini.temperature = 0.0
        mock_gemini.max_output_tokens = 1024
        mock_gemini.generate_answer.return_value = {
            "answer": "Lifestyle modifications for hypertension include regular physical activity and sodium reduction [Source 1].",
            "model": "gemini-3.5-flash-lite",
            "disclaimer": "MEDICAL DISCLAIMER: For educational purposes only.",
            "generation_time_ms": 150.0,
            "api_request_time_ms": 140.0,
            "request_start_time": "2026-10-02T12:00:00Z",
            "gemini_calls_count": 1,
            "input_tokens": 120,
            "output_tokens": 40,
            "status": "success"
        }

        query = "What lifestyle changes are recommended for managing hypertension?"
        grounded_text = "Lifestyle modifications for hypertension include regular physical activity and sodium reduction."
        mock_retrieval = {
            "question": query,
            "retrieved_chunks": [
                {
                    "chunk_id": "chunk_0",
                    "text": grounded_text,
                    "similarity_score": 0.88,
                    "metadata": {"filename": "hypertension.pdf"}
                }
            ],
            "context": f"[SOURCE 1] {grounded_text}",
            "sources": [
                {
                    "source_index": 1,
                    "source_label": "[Source 1]",
                    "filename": "hypertension.pdf",
                    "text": grounded_text
                }
            ],
            "retrieval_status": "success",
            "timings": {"total_retrieval_time_ms": 10.0}
        }

        with patch.object(rag_service, "query", return_value=mock_retrieval):
            # Turn 1: Cache Miss -> Calls Gemini
            res1 = rag_service.generate_rag_answer(
                question=query,
                gemini_service=mock_gemini,
                user_id=None
            )
            assert res1["retrieval_status"] == "success"
            assert len(res1["sources"]) > 0
            assert mock_gemini.generate_answer.call_count == 1
            assert res1["timings"]["cache_hit"] is False
            assert res1["timings"]["llm_called"] is True

            # Turn 2: Cache Hit -> ZERO Gemini calls!
            res2 = rag_service.generate_rag_answer(
                question=query,
                gemini_service=mock_gemini,
                user_id=None
            )
            assert res2["retrieval_status"] == "success"
            assert res2["answer"] == res1["answer"]
            assert len(res2["sources"]) == len(res1["sources"])
            assert mock_gemini.generate_answer.call_count == 1  # Still 1, NOT incremented!
            assert res2["timings"]["cache_hit"] is True
            assert res2["timings"]["llm_called"] is False
            assert res2["timings"]["gemini_calls_count"] == 0

    def test_cached_response_cannot_bypass_medical_safety_guard(self):
        """Test 11: Emergency/harmful inquiries CANNOT bypass MedicalSafetyGuard via cache."""
        rag_service = RAGService()
        emergency_query = "I am having severe crushing chest pain, should I take aspirin?"

        # Emergency query must be intercepted immediately, never returning a cached response
        res = rag_service.generate_rag_answer(
            question=emergency_query,
            user_id=1
        )
        assert res["retrieval_status"] == "safety_intercepted"
        assert "emergency" in res["answer"].lower() or "911" in res["answer"]
        assert res["timings"]["llm_called"] is False

    def test_cache_does_not_leak_cross_user_responses(self):
        """Test 12: User B cannot retrieve User A's cached response."""
        rag_service = RAGService()
        mock_gemini = MagicMock(spec=GeminiService)
        mock_gemini.model = "gemini-3.5-flash-lite"
        mock_gemini.temperature = 0.0
        mock_gemini.max_output_tokens = 1024
        mock_gemini.generate_answer.side_effect = [
            {
                "answer": "User A clinical findings [Source 1].",
                "model": "gemini-3.5-flash-lite",
                "disclaimer": "DISCLAIMER",
                "generation_time_ms": 100.0,
                "status": "success",
                "gemini_calls_count": 1
            },
            {
                "answer": "User B clinical findings [Source 1].",
                "model": "gemini-3.5-flash-lite",
                "disclaimer": "DISCLAIMER",
                "generation_time_ms": 100.0,
                "status": "success",
                "gemini_calls_count": 1
            }
        ]

        q = "What is the recommended treatment plan?"
        mock_retrieval = {
            "question": q,
            "retrieved_chunks": [{"chunk_id": "c1", "text": "Treatment plan info", "metadata": {"filename": "plan.pdf"}}],
            "context": "[SOURCE 1] Treatment plan info",
            "sources": [{"source_index": 1, "source_label": "[Source 1]", "filename": "plan.pdf"}],
            "retrieval_status": "success",
            "timings": {"total_retrieval_time_ms": 10.0}
        }

        with patch.object(rag_service, "query", return_value=mock_retrieval):
            # User 1 queries
            res_u1 = rag_service.generate_rag_answer(question=q, gemini_service=mock_gemini, user_id=1)
            assert "User A" in res_u1["answer"]

            # User 2 queries same question: MUST NOT return User 1's cached response!
            res_u2 = rag_service.generate_rag_answer(question=q, gemini_service=mock_gemini, user_id=2)
            assert "User B" in res_u2["answer"]
            assert mock_gemini.generate_answer.call_count == 2
