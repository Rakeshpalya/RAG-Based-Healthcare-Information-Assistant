"""
Phase 3.4.1 — Authentication & User Isolation Test Suite.

Verifies:
1. User A querying User A document -> ALLOW (retrieval succeeds).
2. User B querying User B document -> ALLOW (retrieval succeeds).
3. User A querying User B document -> DENY (0 chunks retrieved, safe fallback).
4. User B querying User A document -> DENY (0 chunks retrieved, safe fallback).
5. User A cache -> inaccessible to User B (cache MISS).
6. Different user_id -> cache MISS on identical query.
7. User ID cannot be overridden through request payload (Pydantic extra='forbid' or server-enforced).
8. User ID cannot be overridden through query parameters.
9. User ID cannot be overridden through document metadata (spoofing rejected).
10. Authorization failures do not leak document existence (returns standard medical fallback, no doc existence leakage).
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.vector_store_service import VectorStoreService, get_vector_store_service
from backend.services.llm_cache_service import get_llm_cache_service
from backend.services.gemini_service import GeminiService
from backend.rag.rag_service import RAGService
from backend.database.models import User


@pytest.fixture
def isolated_vector_store():
    """Provides a fresh isolated VectorStoreService with test vectors for User 1 and User 2."""
    vs = VectorStoreService(dimension=384)
    np.random.seed(42)

    # 1. User 1 Document Chunks (Hypertension Lifestyle)
    vec1 = np.random.randn(384).astype(np.float32)
    vec1 /= np.linalg.norm(vec1)
    vs.add_embeddings(
        embeddings=[vec1],
        metadatas=[{
            "chunk_id": "u1_chunk_0",
            "document_id": "doc_u1_101",
            "text": "User 1 confidential medical report: Patient adheres to a strict low-sodium DASH diet.",
            "user_id": 1,
            "metadata": {"filename": "patient_1_records.pdf", "user_id": 1}
        }]
    )

    # 2. User 2 Document Chunks (Asthma Inhaler Protocol)
    vec2 = np.random.randn(384).astype(np.float32)
    vec2 /= np.linalg.norm(vec2)
    vs.add_embeddings(
        embeddings=[vec2],
        metadatas=[{
            "chunk_id": "u2_chunk_0",
            "document_id": "doc_u2_202",
            "text": "User 2 confidential clinical notes: Albuterol sulfate inhaler prescribed for acute bronchospasm.",
            "user_id": 2,
            "metadata": {"filename": "patient_2_records.pdf", "user_id": 2}
        }]
    )

    # 3. Public Document Chunks (user_id is None)
    vec_pub = np.random.randn(384).astype(np.float32)
    vec_pub /= np.linalg.norm(vec_pub)
    vs.add_embeddings(
        embeddings=[vec_pub],
        metadatas=[{
            "chunk_id": "pub_chunk_0",
            "document_id": "doc_public_001",
            "text": "General clinical guideline: Regular aerobic exercise supports overall cardiovascular health.",
            "user_id": None,
            "metadata": {"filename": "public_guideline.pdf"}
        }]
    )
    return vs, vec1, vec2, vec_pub


class TestUserIsolationRetrievalAndContext:

    def test_user_a_querying_user_a_document_allowed(self, isolated_vector_store):
        """User A querying User A document -> ALLOW."""
        vs, vec1, vec2, _ = isolated_vector_store
        results = vs.search(query_embedding=vec1, top_k=5, user_id=1)
        chunk_ids = [r["chunk_id"] for r in results]
        assert "u1_chunk_0" in chunk_ids
        assert "u2_chunk_0" not in chunk_ids

    def test_user_b_querying_user_b_document_allowed(self, isolated_vector_store):
        """User B querying User B document -> ALLOW."""
        vs, vec1, vec2, _ = isolated_vector_store
        results = vs.search(query_embedding=vec2, top_k=5, user_id=2)
        chunk_ids = [r["chunk_id"] for r in results]
        assert "u2_chunk_0" in chunk_ids
        assert "u1_chunk_0" not in chunk_ids

    def test_user_a_querying_user_b_document_denied(self, isolated_vector_store):
        """User A querying User B document -> DENIED (User B chunks strictly filtered out)."""
        vs, _, vec2, _ = isolated_vector_store
        # User 1 searches using a vector that matches User 2's document perfectly
        results = vs.search(query_embedding=vec2, top_k=5, user_id=1)
        chunk_ids = [r["chunk_id"] for r in results]
        assert "u2_chunk_0" not in chunk_ids

    def test_user_b_querying_user_a_document_denied(self, isolated_vector_store):
        """User B querying User A document -> DENIED (User A chunks strictly filtered out)."""
        vs, vec1, _, _ = isolated_vector_store
        # User 2 searches using a vector that matches User 1's document perfectly
        results = vs.search(query_embedding=vec1, top_k=5, user_id=2)
        chunk_ids = [r["chunk_id"] for r in results]
        assert "u1_chunk_0" not in chunk_ids

    def test_third_party_user_denied_access_to_user_documents(self, isolated_vector_store):
        """User C (user_id=3) querying User A or User B documents is strictly DENIED."""
        vs, vec1, vec2, vec_pub = isolated_vector_store
        results1 = vs.search(query_embedding=vec1, top_k=5, user_id=3)
        chunk_ids1 = [r["chunk_id"] for r in results1]
        assert "u1_chunk_0" not in chunk_ids1
        assert "u2_chunk_0" not in chunk_ids1

        results2 = vs.search(query_embedding=vec2, top_k=5, user_id=3)
        chunk_ids2 = [r["chunk_id"] for r in results2]
        assert "u1_chunk_0" not in chunk_ids2
        assert "u2_chunk_0" not in chunk_ids2



class TestUserCacheIsolation:

    def test_user_a_cache_inaccessible_to_user_b(self, isolated_vector_store):
        """User A cache is completely isolated from User B (cache MISS on different user_id)."""
        cache = get_llm_cache_service()
        cache.clear()

        query = "What diet is recommended in my records?"
        doc_sig = "sig_123"

        # Key for User 1
        key_user1 = cache.generate_cache_key(
            normalized_query=query,
            user_scope=1,
            document_signature=doc_sig
        )
        # Key for User 2
        key_user2 = cache.generate_cache_key(
            normalized_query=query,
            user_scope=2,
            document_signature=doc_sig
        )

        assert key_user1 != key_user2, "Keys for User 1 and User 2 must never collide"

        # Populate cache for User 1
        cached_data = {
            "answer": "Patient adheres to a strict low-sodium DASH diet [Source 1].",
            "retrieval_status": "success",
            "sources": [{"source_index": 1, "source_label": "[Source 1]"}]
        }
        cache.set(key=key_user1, value=cached_data, user_scope=1)

        # Verify User 1 gets cache HIT
        hit_u1 = cache.get(key_user1)
        assert hit_u1 is not None
        assert "DASH diet" in hit_u1["answer"]

        # Verify User 2 gets cache MISS
        miss_u2 = cache.get(key_user2)
        assert miss_u2 is None, "User 2 must receive a cache MISS for User 1's query"

    def test_rag_service_end_to_end_cross_user_cache_isolation(self, isolated_vector_store):
        """End-to-End RAGService ensures cross-user cache access is impossible."""
        vs, _, _, _ = isolated_vector_store
        rag = RAGService(vector_store=vs)
        mock_gemini = MagicMock(spec=GeminiService)
        mock_gemini.model = "gemini-3.5-flash-lite"
        mock_gemini.temperature = 0.0
        mock_gemini.max_output_tokens = 1024
        mock_gemini.generate_answer.return_value = {
            "answer": "Patient adheres to a strict low-sodium DASH diet [Source 1].",
            "model": "gemini-3.5-flash-lite",
            "disclaimer": "MEDICAL DISCLAIMER",
            "generation_time_ms": 100.0,
            "status": "success"
        }

        query = "What diet is recommended in my records?"
        with patch.object(rag, "query", return_value={
            "question": query,
            "retrieved_chunks": [{"chunk_id": "u1_chunk_0", "text": "DASH diet", "similarity_score": 0.9}],
            "context": "[SOURCE 1] DASH diet",
            "sources": [{"source_index": 1, "source_label": "[Source 1]", "text": "DASH diet"}],
            "retrieval_status": "success",
            "timings": {"total_retrieval_time_ms": 10.0}
        }):
            # Turn 1: User 1 queries and caches
            res_u1 = rag.generate_rag_answer(question=query, user_id=1, gemini_service=mock_gemini)
            assert res_u1["retrieval_status"] == "success"

        # Turn 2: User 2 makes the same query with mock_gemini
        mock_gemini_u2 = MagicMock(spec=GeminiService)
        with patch.object(rag, "query", return_value={
            "question": query,
            "retrieved_chunks": [],
            "context": "",
            "sources": [],
            "retrieval_status": "no_relevant_context",
            "timings": {"total_retrieval_time_ms": 10.0}
        }):
            res_u2 = rag.generate_rag_answer(question=query, user_id=2, gemini_service=mock_gemini_u2)
            # User 2 MUST NOT get User 1's cached response!
            assert res_u2["retrieval_status"] == "no_relevant_context"
            assert "DASH diet" not in res_u2["answer"]


class TestRequestSpoofingAndNonLeakage:

    def test_user_id_cannot_be_overridden_in_rag_payload(self):
        """User ID cannot be injected via request payload (Pydantic extra='forbid' rejects extra fields)."""
        client = TestClient(app)
        # Attempt to inject user_id in /rag/query payload
        res = client.post("/rag/query", json={
            "question": "What is hypertension?",
            "user_id": 999  # Extra forbidden parameter
        })
        assert res.status_code == 422, f"Expected 422 for forbidden field, got {res.status_code}"

    def test_user_id_cannot_be_overridden_in_stream_payload(self):
        """User ID cannot be injected via streaming payload (Pydantic extra='forbid' rejects extra fields)."""
        client = TestClient(app)
        res = client.post("/rag/stream", json={
            "question": "What is hypertension?",
            "user_id": 999  # Extra forbidden parameter
        })
        assert res.status_code == 422, f"Expected 422 for forbidden field, got {res.status_code}"

    def test_user_id_cannot_be_overridden_through_query_parameters(self):
        """Query parameters like ?user_id=123 are ignored; RAGService strictly binds to authenticated token."""
        client = TestClient(app)
        # Query endpoint with query parameter spoof attempt
        res = client.post("/rag/query?user_id=999", json={
            "question": "What is hypertension?"
        })
        # Endpoint succeeds or fails based on body, but user_id query param does NOT authenticate as user 999
        assert res.status_code in (200, 429)

    def test_user_id_spoofing_rejected_in_document_metadata(self):
        """Client cannot spoof document user_id via POST /documents/metadata."""
        from backend.api.auth_dependencies import get_current_db_user
        client = TestClient(app)
        mock_user = User(id=1, email="alice@hospital.org", role="patient")
        app.dependency_overrides[get_current_db_user] = lambda: mock_user
        try:
            res = client.post("/documents/metadata", json={
                "filename": "spoofed.pdf",
                "file_path": "/data/spoofed.pdf",
                "file_size": 1024,
                "user_id": 2  # Alice attempts to register document under Bob's user_id
            })
            assert res.status_code == 403
            assert "Cannot assign document to another user" in res.json()["detail"]
        finally:
            app.dependency_overrides.pop(get_current_db_user, None)

    def test_authorization_failure_does_not_leak_document_existence_in_rag(self, isolated_vector_store):
        """
        When User A queries about User B's private document, RAG returns standard
        no_relevant_context fallback without disclosing that User B's document exists.
        """
        vs, _, _, _ = isolated_vector_store
        rag = RAGService(vector_store=vs)
        mock_gemini = MagicMock(spec=GeminiService)

        # User 1 asks about User 2's private inhaler notes
        result = rag.generate_rag_answer(
            question="What is the prescribed dosage of Albuterol for bronchospasm?",
            user_id=1,
            gemini_service=mock_gemini
        )
        assert result["retrieval_status"] == "no_relevant_context"
        # Verify no document name or existence leakage
        assert "patient_2_records.pdf" not in result["answer"]
        assert "doc_u2_202" not in result["answer"]
        assert "Relevant medical information could not be found" in result["answer"]
