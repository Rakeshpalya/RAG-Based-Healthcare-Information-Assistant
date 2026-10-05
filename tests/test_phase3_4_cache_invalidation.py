"""
Phase 3.4 Milestone 3.4.3: Cache Invalidation Test Suite

Verifies strict cache invalidation invariants:
1. Document added -> Cache invalidation (signature changes / delete_by_user)
2. Document deleted -> Cache invalidation (signature changes / delete_by_document)
3. Document replaced -> Cache invalidation (signature changes)
4. Document content changed -> Cache invalidation (content hash changes)
5. Metadata changed -> Cache invalidation (metadata hash changes)
6. User changed -> Cache key changes (user isolation)
7. Model changed -> Cache key changes
8. Temperature changed -> Cache key changes
9. max_output_tokens changed -> Cache key changes
10. Prompt version changed -> Cache key changes
11. Stale answers cannot survive document changes (end-to-end simulation)
12. Unvalidated LLM output is NEVER cached (safety, errors, ungrounded)
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from backend.services.llm_cache_service import (
    LLMCacheService,
    get_llm_cache_service,
    PROMPT_VERSION,
)
from backend.rag.rag_service import RAGService


@pytest.fixture(autouse=True)
def clean_cache():
    """Ensure clean cache before and after each test."""
    cache = get_llm_cache_service()
    cache.clear()
    cache.enabled = True
    yield
    cache.clear()
    cache.enabled = True


class TestCacheInvalidationFactors:
    """Tests all 10 cache invalidation triggers specified in Phase 3.4.3."""

    def test_invalidation_when_document_added(self):
        """1. Document added changes the user/document signature and causes cache miss."""
        mock_vs = MagicMock()
        mock_vs.metadata_store = [
            {"chunk_id": "c1", "text": "Original hypertension guidelines.", "user_id": 1, "document_id": "doc1"}
        ]
        rag = RAGService(vector_store=mock_vs)

        sig1 = rag.compute_document_signature(user_id=1, document_id="doc1")

        # Document chunk added
        mock_vs.metadata_store.append(
            {"chunk_id": "c2", "text": "New section on medication guidelines.", "user_id": 1, "document_id": "doc1"}
        )
        sig2 = rag.compute_document_signature(user_id=1, document_id="doc1")

        assert sig1 != sig2, "Document signature must change when a document chunk is added"

        cache = get_llm_cache_service()
        key1 = cache.generate_cache_key(
            normalized_query="hypertension treatment",
            user_scope=1,
            document_signature=sig1
        )
        key2 = cache.generate_cache_key(
            normalized_query="hypertension treatment",
            user_scope=1,
            document_signature=sig2
        )
        assert key1 != key2, "Cache key must change when document is added"

    def test_invalidation_when_document_deleted(self):
        """2. Document deleted removes cached entries via delete_by_document or signature change."""
        cache = get_llm_cache_service()
        sig = "doc_sig_doc99"
        key = cache.generate_cache_key(
            normalized_query="what is asthma",
            user_scope=5,
            document_signature=sig
        )
        cache.set(
            key,
            {
                "question": "what is asthma",
                "answer": "Asthma is a chronic respiratory condition [Source 1].",
                "retrieval_status": "success",
                "sources": [{"document_id": "doc99", "source_index": 1}]
            },
            user_scope=5,
            document_signature=sig
        )

        assert cache.get(key) is not None

        # Delete by document ID
        deleted = cache.delete_by_document("doc99")
        assert deleted >= 1
        assert cache.get(key) is None, "Cache entry must be purged after document is deleted"

    def test_invalidation_when_document_replaced(self):
        """3. Document replaced with different chunks changes document signature."""
        mock_vs = MagicMock()
        mock_vs.metadata_store = [
            {"chunk_id": "chunk_old_1", "text": "Old clinical protocol.", "user_id": 2, "document_id": "docA"}
        ]
        rag = RAGService(vector_store=mock_vs)

        sig_old = rag.compute_document_signature(user_id=2, document_id="docA")

        # Document replaced with new chunk
        mock_vs.metadata_store = [
            {"chunk_id": "chunk_new_1", "text": "Replaced updated clinical protocol.", "user_id": 2, "document_id": "docA"}
        ]
        sig_new = rag.compute_document_signature(user_id=2, document_id="docA")

        assert sig_old != sig_new, "Replacing a document must change its document signature"

    def test_invalidation_when_document_content_changed(self):
        """4. Document content changed (same chunk_id, modified text) changes signature."""
        mock_vs = MagicMock()
        mock_vs.metadata_store = [
            {"chunk_id": "c100", "text": "Target blood pressure is below 140/90.", "user_id": 3, "document_id": "doc_cardio"}
        ]
        rag = RAGService(vector_store=mock_vs)

        sig1 = rag.compute_document_signature(user_id=3, document_id="doc_cardio")

        # Update text content within the same chunk_id
        mock_vs.metadata_store[0]["text"] = "Target blood pressure is below 130/80 according to revised guidance."
        sig2 = rag.compute_document_signature(user_id=3, document_id="doc_cardio")

        assert sig1 != sig2, "Changing chunk text content MUST change the document signature"

    def test_invalidation_when_metadata_changed(self):
        """5. Metadata changed (version/updated_at/tag) changes document signature."""
        mock_vs = MagicMock()
        mock_vs.metadata_store = [
            {
                "chunk_id": "c200",
                "text": "Cardiology guideline.",
                "user_id": 4,
                "document_id": "doc_meta",
                "version": "1.0",
                "status": "draft"
            }
        ]
        rag = RAGService(vector_store=mock_vs)

        sig1 = rag.compute_document_signature(user_id=4, document_id="doc_meta")

        # Update metadata fields
        mock_vs.metadata_store[0]["version"] = "2.0"
        mock_vs.metadata_store[0]["status"] = "published"
        sig2 = rag.compute_document_signature(user_id=4, document_id="doc_meta")

        assert sig1 != sig2, "Modifying metadata fields MUST change the document signature"

    def test_invalidation_when_user_changed(self):
        """6. User changed results in different cache keys and complete isolation."""
        cache = get_llm_cache_service()
        key_user_a = cache.generate_cache_key(
            normalized_query="diabetes management",
            user_scope=100,
            document_signature="shared_sig"
        )
        key_user_b = cache.generate_cache_key(
            normalized_query="diabetes management",
            user_scope=200,
            document_signature="shared_sig"
        )
        assert key_user_a != key_user_b, "Different users MUST produce different cache keys"

    def test_invalidation_when_model_changed(self):
        """7. Model changed produces different cache key."""
        cache = get_llm_cache_service()
        key_flash = cache.generate_cache_key(
            normalized_query="asthma symptoms",
            user_scope=1,
            model="gemini-3.5-flash-lite"
        )
        key_pro = cache.generate_cache_key(
            normalized_query="asthma symptoms",
            user_scope=1,
            model="gemini-1.5-pro"
        )
        assert key_flash != key_pro, "Changing model MUST produce different cache key"

    def test_invalidation_when_temperature_changed(self):
        """8. Temperature changed produces different cache key."""
        cache = get_llm_cache_service()
        key_temp0 = cache.generate_cache_key(
            normalized_query="asthma symptoms",
            user_scope=1,
            temperature=0.0
        )
        key_temp7 = cache.generate_cache_key(
            normalized_query="asthma symptoms",
            user_scope=1,
            temperature=0.7
        )
        assert key_temp0 != key_temp7, "Changing temperature MUST produce different cache key"

    def test_invalidation_when_max_output_tokens_changed(self):
        """9. max_output_tokens changed produces different cache key."""
        cache = get_llm_cache_service()
        key_tok1024 = cache.generate_cache_key(
            normalized_query="asthma symptoms",
            user_scope=1,
            max_output_tokens=1024
        )
        key_tok2048 = cache.generate_cache_key(
            normalized_query="asthma symptoms",
            user_scope=1,
            max_output_tokens=2048
        )
        assert key_tok1024 != key_tok2048, "Changing max_output_tokens MUST produce different cache key"

    def test_invalidation_when_prompt_version_changed(self):
        """10. Prompt version changed produces different cache key."""
        cache = get_llm_cache_service()
        key_v1 = cache.generate_cache_key(
            normalized_query="asthma symptoms",
            user_scope=1,
            prompt_version="v3.3.0"
        )
        key_v2 = cache.generate_cache_key(
            normalized_query="asthma symptoms",
            user_scope=1,
            prompt_version="v3.4.0"
        )
        assert key_v1 != key_v2, "Bumping prompt version MUST produce different cache key"


class TestStaleAnswerSurvivalAndSafety:
    """Verifies that stale answers cannot survive document changes and unvalidated output is rejected."""

    def test_stale_answer_cannot_survive_document_changes(self):
        """
        Required pipeline behavior:
        Document V1 -> Generate answer -> Validated response cached
        Document V1 changes -> Document signature changes -> Previous cache MUST NOT be reused
        -> Retrieve new context -> Generate new answer -> Validate -> Cache new answer
        """
        mock_vs = MagicMock()
        mock_vs.count.return_value = 1
        mock_vs.metadata_store = [
            {"chunk_id": "c1", "text": "Version 1 guidelines: Take with water.", "user_id": 42, "document_id": "docV1"}
        ]

        rag = RAGService(vector_store=mock_vs)
        mock_gemini = MagicMock()
        mock_gemini.model = "gemini-3.5-flash-lite"
        mock_gemini.temperature = 0.0
        mock_gemini.max_output_tokens = 1024

        # Mock retrieval for V1
        rag.query = MagicMock(return_value={
            "retrieval_status": "success",
            "context": "[Source 1] Version 1 guidelines: Take with water.",
            "retrieved_chunks": [{"chunk_id": "c1", "text": "Version 1 guidelines: Take with water."}],
            "sources": [{"chunk_id": "c1", "text": "Version 1 guidelines: Take with water.", "document_name": "guidelines.pdf"}],
            "timings": {
                "embedding_time_ms": 1.0,
                "faiss_retrieval_time_ms": 2.0,
                "deduplication_time_ms": 0.5,
                "context_construction_time_ms": 1.0,
                "total_retrieval_time_ms": 4.5,
                "similarity_scores": [0.85],
                "retrieved_chunk_ids": ["c1"],
                "retrieved_document_names": ["guidelines.pdf"]
            }
        })

        mock_gemini.generate_answer.return_value = {
            "answer": "According to the initial guidelines, take the medication with water [Source 1].",
            "model": "gemini-3.5-flash-lite",
            "generation_time_ms": 150.0,
            "gemini_calls_count": 1
        }

        # Step 1: Run V1 query and verify cached
        res1 = rag.generate_rag_answer(
            question="How should the medication be taken?",
            user_id=42,
            document_id="docV1",
            gemini_service=mock_gemini,
            use_cache=True
        )
        assert res1["timings"]["llm_called"] is True
        assert "take the medication with water" in res1["answer"]

        # Step 2: Document V1 changes!
        mock_vs.metadata_store[0]["text"] = "Version 2 guidelines: Take with food, NOT with water alone."
        mock_gemini.generate_answer.return_value = {
            "answer": "According to updated guidelines, take with food, not with water alone [Source 1].",
            "model": "gemini-3.5-flash-lite",
            "generation_time_ms": 140.0,
            "gemini_calls_count": 1
        }
        rag.query = MagicMock(return_value={
            "retrieval_status": "success",
            "context": "[Source 1] Version 2 guidelines: Take with food, NOT with water alone.",
            "retrieved_chunks": [{"chunk_id": "c1", "text": "Version 2 guidelines: Take with food, NOT with water alone."}],
            "sources": [{"chunk_id": "c1", "text": "Version 2 guidelines: Take with food, NOT with water alone.", "document_name": "guidelines.pdf"}],
            "timings": {
                "embedding_time_ms": 1.0,
                "faiss_retrieval_time_ms": 2.0,
                "deduplication_time_ms": 0.5,
                "context_construction_time_ms": 1.0,
                "total_retrieval_time_ms": 4.5,
                "similarity_scores": [0.88],
                "retrieved_chunk_ids": ["c1"],
                "retrieved_document_names": ["guidelines.pdf"]
            }
        })

        # Step 3: Run query again - MUST NOT reuse old V1 answer!
        res2 = rag.generate_rag_answer(
            question="How should the medication be taken?",
            user_id=42,
            document_id="docV1",
            gemini_service=mock_gemini,
            use_cache=True
        )

        assert res2["timings"]["llm_called"] is True, "Must call LLM again because document signature changed"
        assert "take with food" in res2["answer"]
        assert "water alone" in res2["answer"]

    def test_unvalidated_llm_output_is_never_cached(self):
        """Unvalidated responses (errors, ungrounded, empty) are never stored in cache."""
        cache = get_llm_cache_service()
        key = "test_unvalidated_key"

        # 1. Error status
        stored_err = cache.set(key, {"answer": "Some error message.", "retrieval_status": "service_error"})
        assert stored_err is False
        assert cache.get(key) is None

        # 2. No relevant context
        stored_no_ctx = cache.set(key, {"answer": "No relevant context found.", "retrieval_status": "no_relevant_context"})
        assert stored_no_ctx is False
        assert cache.get(key) is None

        # 3. Empty or short answer
        stored_empty = cache.set(key, {"answer": "", "retrieval_status": "success"})
        assert stored_empty is False
        assert cache.get(key) is None

        # 4. Valid answer stores successfully
        stored_valid = cache.set(key, {
            "answer": "Valid grounded medical explanation [Source 1].",
            "retrieval_status": "success"
        })
        assert stored_valid is True
        assert cache.get(key) is not None
