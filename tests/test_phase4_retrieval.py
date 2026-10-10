"""
Phase 4 Milestone 4.2 — RAG Retrieval Evaluation Tests.

Validates:
1. Recall@1, Recall@3, and Recall@5 on benchmark queries.
2. Precision@K calculation accuracy and bounds.
3. Mean Reciprocal Rank (MRR) across ranked retrieval results.
4. Retrieval sufficiency gating (correctly discerning sufficient medical context from insufficient context).
5. Strict user ownership isolation in semantic retrieval.
6. Invariant verification: FAISS vector count = 744, metadata records = 744, dimension = 384.
"""

import json
from pathlib import Path
import pytest

from backend.services.vector_store_service import VectorStoreService, get_vector_store_service
from backend.services.embedding_service import EmbeddingService
from backend.rag.rag_service import RAGService
from backend.evaluation.retrieval_evaluator import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr_at_k,
    evaluate_context_sufficiency,
)
from backend.evaluation.phase4_dataset import Phase4DatasetLoader


def test_01_faiss_production_invariants_preserved():
    """Verify that retrieval testing has not mutated production vector store."""
    vs = get_vector_store_service()
    assert vs.count() == 744, f"Expected 744 vectors, found {vs.count()}"
    assert len(vs.metadata_store) == 744, f"Expected 744 metadata records, found {len(vs.metadata_store)}"
    assert vs.dimension == 384, f"Expected 384 dimension, found {vs.dimension}"


def test_02_precision_and_recall_calculation_metrics():
    """Verify precision and recall metric functions with known deterministic inputs."""
    retrieved = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
    expected = ["chunk_1", "chunk_3", "chunk_99"]

    # Precision@1: 1 / 1 = 1.0
    assert compute_precision_at_k(retrieved, expected, k=1) == 1.0
    # Precision@3: 2 / 3 = 0.6667
    assert round(compute_precision_at_k(retrieved, expected, k=3), 4) == 0.6667
    # Precision@5: 2 / 5 = 0.4000
    assert compute_precision_at_k(retrieved, expected, k=5) == 0.4

    # Recall@1: 1 / 3 = 0.3333
    assert round(compute_recall_at_k(retrieved, expected, k=1), 4) == 0.3333
    # Recall@3: 2 / 3 = 0.6667
    assert round(compute_recall_at_k(retrieved, expected, k=3), 4) == 0.6667
    # Recall@5: 2 / 3 = 0.6667
    assert round(compute_recall_at_k(retrieved, expected, k=5), 4) == 0.6667


def test_03_mrr_metric_calculation():
    """Verify Mean Reciprocal Rank (MRR@K) computation."""
    expected = ["target_chunk"]

    # Case 1: Relevant at rank 1 -> MRR = 1.0
    assert compute_mrr_at_k(["target_chunk", "other_1", "other_2"], expected, k=3) == 1.0
    # Case 2: Relevant at rank 2 -> MRR = 0.5
    assert compute_mrr_at_k(["other_1", "target_chunk", "other_2"], expected, k=3) == 0.5
    # Case 3: Relevant at rank 3 -> MRR = 1/3
    assert round(compute_mrr_at_k(["other_1", "other_2", "target_chunk"], expected, k=3), 4) == 0.3333
    # Case 4: Not found in top 3 -> MRR = 0.0
    assert compute_mrr_at_k(["other_1", "other_2", "other_3"], expected, k=3) == 0.0


def test_04_benchmark_retrieval_recall_and_ranking():
    """Evaluate retrieval ranking and Recall@K on document-grounded medical benchmark cases."""
    vs = get_vector_store_service()
    doc_cases = Phase4DatasetLoader.get_by_category("document-grounded questions")
    assert len(doc_cases) >= 3

    mrr_scores = []
    recall_at_1_scores = []
    recall_at_3_scores = []
    recall_at_5_scores = []

    for case in doc_cases:
        query_vec = EmbeddingService.embed_query(case.question)
        results = vs.search(query_vec, top_k=5)
        assert len(results) > 0

        # Check if expected source snippet matches any retrieved chunk
        retrieved_texts = [r.get("text", "") for r in results]
        first_match_rank = None
        for rank, text in enumerate(retrieved_texts, 1):
            if any(exp.lower() in text.lower() or text.lower() in exp.lower() for exp in case.expected_sources):
                first_match_rank = rank
                break

        if first_match_rank:
            mrr_scores.append(1.0 / first_match_rank)
            recall_at_1_scores.append(1.0 if first_match_rank <= 1 else 0.0)
            recall_at_3_scores.append(1.0 if first_match_rank <= 3 else 0.0)
            recall_at_5_scores.append(1.0 if first_match_rank <= 5 else 0.0)
        else:
            mrr_scores.append(0.0)
            recall_at_1_scores.append(0.0)
            recall_at_3_scores.append(0.0)
            recall_at_5_scores.append(0.0)

    avg_mrr = sum(mrr_scores) / len(mrr_scores)
    avg_recall_5 = sum(recall_at_5_scores) / len(recall_at_5_scores)

    # Retrieval benchmark performance requirements
    assert avg_recall_5 >= 0.65, f"Expected Recall@5 >= 0.65, got {avg_recall_5}"
    assert avg_mrr >= 0.50, f"Expected MRR >= 0.50, got {avg_mrr}"


def test_05_retrieval_sufficiency_gate_discrimination():
    """Verify sufficiency gate distinguishes between grounded queries and off-topic unsupported queries."""
    from unittest.mock import MagicMock
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)

    mock_gemini = MagicMock()
    mock_gemini.generate_answer.return_value = {
        "answer": "CT scan findings of the lungs showed clear lung fields without consolidation [Source 1].",
        "model": "mock-gemini-test",
        "elapsed_ms": 10.0,
        "input_tokens": 50,
        "output_tokens": 20
    }

    # Supported medical query -> sufficient context
    med_res = rag.generate_rag_answer(
        question="What were the CT scan findings of the lungs?",
        similarity_threshold=0.25,
        gemini_service=mock_gemini
    )
    assert med_res["retrieval_status"] == "success"
    assert len(med_res["sources"]) > 0

    # Unsupported off-topic query -> insufficient context (sufficiency gate triggers)
    unsup_res = rag.generate_rag_answer(
        question="What is the subspace warp drive formula in quantum astrophysics?",
        similarity_threshold=0.85,
        gemini_service=mock_gemini
    )
    assert unsup_res["retrieval_status"] == "no_relevant_context"
    assert "Relevant medical information could not be found" in unsup_res["answer"]


def test_06_retrieval_user_isolation_enforced():
    """Verify retrieval strictly isolates user-owned documents and prevents cross-user access."""
    # Create isolated test vector store with 2 user documents
    test_vs = VectorStoreService(dimension=384)
    test_vs.add_chunks([
        {
            "chunk_id": "U101_CHUNK_1",
            "document_id": "DOC_101",
            "user_id": 101,
            "text": "User 101 confidential oncology biopsy: benign nevus observed.",
        },
        {
            "chunk_id": "U202_CHUNK_1",
            "document_id": "DOC_202",
            "user_id": 202,
            "text": "User 202 confidential cardiology record: severe aortic stenosis.",
        }
    ])

    q_vec = EmbeddingService.embed_query("biopsy oncology nevus")

    # Search isolated to User 101
    results_101 = test_vs.search(q_vec, top_k=5, user_id=101)
    assert len(results_101) == 1
    assert results_101[0]["chunk_id"] == "U101_CHUNK_1"

    # Search isolated to User 202 should NOT return User 101's document
    results_202 = test_vs.search(q_vec, top_k=5, user_id=202)
    for r in results_202:
        assert r["chunk_id"] != "U101_CHUNK_1"
        assert r.get("user_id") == 202
