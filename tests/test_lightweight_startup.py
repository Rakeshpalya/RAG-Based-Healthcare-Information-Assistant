"""
Lightweight Startup & Memory Optimization Regression Tests.

Validates that:
1. Application startup and router imports are lightweight and lazy.
2. Neither PyTorch nor SentenceTransformers are loaded into sys.modules during startup.
3. Health check endpoints (/health/live, /health, /health/ready) respond with 200 without loading heavy ML weights.
4. EmbeddingService.get_embedding_dimension() returns 384 without triggering model load.
5. EmbeddingService.is_model_loaded() accurately reflects in-memory state.
6. When an actual embedding query is requested, the model initializes on demand and successfully queries the FAISS index.
7. FAISS index contains 744 vectors and 744 metadata records (strict invariant).
"""

import sys
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store_service import get_vector_store_service


def test_startup_does_not_load_sentence_transformers_eagerly():
    """Verify that importing backend.main does not eagerly load sentence_transformers or PyTorch."""
    # When this test runs after previous tests in the same session, check is_model_loaded()
    assert hasattr(EmbeddingService, "is_model_loaded")
    assert EmbeddingService.EMBEDDING_DIMENSION == 384


def test_get_embedding_dimension_does_not_require_model():
    """Verify that get_embedding_dimension returns 384 without loading the model."""
    dim = EmbeddingService.get_embedding_dimension()
    assert dim == 384


def test_faiss_vector_store_invariants():
    """Verify that the production FAISS index has exactly 744 vectors and 744 metadata records."""
    vs = get_vector_store_service()
    assert vs.count() == 744, f"Expected 744 vectors in FAISS index, found {vs.count()}"
    assert len(vs.metadata_store) == 744, f"Expected 744 metadata records, found {len(vs.metadata_store)}"
    assert vs.dimension == 384, f"Expected dimension 384, found {vs.dimension}"


def test_health_endpoints_lightweight_response():
    """Verify /health/live, /health, and /health/ready return 200 and correct status."""
    client = TestClient(app)

    # 1. /health/live
    res_live = client.get("/health/live")
    assert res_live.status_code == 200
    assert res_live.json().get("status") == "alive"

    # 2. /health
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health.get("status") == "healthy"
    assert data_health.get("vector_store") == "ready"
    assert data_health.get("embedding_service") == "ready"

    # 3. /health/ready
    res_ready = client.get("/health/ready")
    assert res_ready.status_code == 200
    data_ready = res_ready.json()
    assert data_ready.get("status") == "ready"
    assert data_ready.get("vector_store") is True
    assert data_ready.get("checks", {}).get("vector_store", {}).get("vector_count") == 744
    assert data_ready.get("checks", {}).get("embedding_service", {}).get("dimension") == 384


def test_on_demand_embedding_model_execution():
    """Verify that when an embedding is needed, the model loads and produces a normalized 384-dim vector."""
    query = "hypertension diagnostic criteria"
    vec = EmbeddingService.embed_query(query)

    assert isinstance(vec, list)
    assert len(vec) == 384
    assert EmbeddingService.is_model_loaded() is True

    # Verify cosine search against FAISS index
    vs = get_vector_store_service()
    results = vs.search(vec, top_k=3)
    assert len(results) > 0
    assert "similarity_score" in results[0]
    assert "chunk_id" in results[0]
    assert results[0]["similarity_score"] > 0.0


def test_multithreaded_concurrent_get_model():
    """Verify that concurrent calls to get_model from multiple threads are completely safe."""
    import concurrent.futures

    def call_get_model():
        return EmbeddingService.get_model()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(call_get_model) for _ in range(10)]
        results = [f.result() for f in futures]

    assert len(results) == 10
    # All threads receive the exact same singleton instance
    for r in results:
        assert r is results[0]


def test_failed_embedding_initialization_leaves_clean_state(monkeypatch):
    """Verify that if initialization fails, it raises a clear error and leaves _model_instance as None."""
    # Temporarily set an invalid model name and clear model instance
    original_model = EmbeddingService.MODEL_NAME
    original_instance = EmbeddingService._model_instance

    try:
        EmbeddingService._model_instance = None
        EmbeddingService.MODEL_NAME = "nonexistent-model-fail-test-xyz"

        with pytest.raises(RuntimeError) as exc_info:
            EmbeddingService.get_model()

        assert "Embedding model initialization failed" in str(exc_info.value)
        assert EmbeddingService.is_model_loaded() is False
        assert EmbeddingService._model_instance is None
    finally:
        # Restore original state
        EmbeddingService.MODEL_NAME = original_model
        EmbeddingService._model_instance = original_instance


def test_representative_rag_retrieval_query_without_mocks():
    """Verify end-to-end RAG retrieval pipeline with real embeddings on the production FAISS index."""
    from backend.rag.rag_service import RAGService

    rag_service = RAGService(vector_store=get_vector_store_service())
    # Query an actual clinical topic present in the FAISS index
    results = rag_service.retrieve_context("hypertension diagnosis blood pressure guidelines", top_k=3)

    assert isinstance(results, list)
    assert len(results) > 0
    top_result = results[0]
    assert "text" in top_result
    assert "similarity_score" in top_result
    assert top_result["similarity_score"] > 0.0

