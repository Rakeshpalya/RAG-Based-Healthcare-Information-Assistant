import json
import pytest
from pathlib import Path

from backend.evaluation.retrieval_evaluator import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_hit_rate_at_k,
    compute_mrr_at_k,
    evaluate_context_sufficiency,
    evaluate_retrieval,
    StructuredRetrievalEvaluation,
    RetrievalEvaluator
)


def test_precision_at_k_calculations():
    """Verify Precision@K under perfect, partial, and zero retrieval conditions."""
    retrieved = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
    expected = {"chunk_1", "chunk_3", "chunk_99"}

    # At K=1: chunk_1 is in expected -> 1/1 = 1.0
    assert compute_precision_at_k(retrieved, expected, k=1) == 1.0

    # At K=2: chunk_1 match, chunk_2 miss -> 1/2 = 0.5
    assert compute_precision_at_k(retrieved, expected, k=2) == 0.5

    # At K=3: chunk_1 and chunk_3 match -> 2/3 ≈ 0.6667
    assert round(compute_precision_at_k(retrieved, expected, k=3), 4) == 0.6667

    # At K=5: chunk_1 and chunk_3 match -> 2/5 = 0.4
    assert compute_precision_at_k(retrieved, expected, k=5) == 0.4

    # Empty retrieved or empty expected
    assert compute_precision_at_k([], expected, k=5) == 0.0
    assert compute_precision_at_k(retrieved, set(), k=5) == 0.0
    assert compute_precision_at_k(retrieved, expected, k=0) == 0.0


def test_recall_at_k_calculations():
    """Verify Recall@K under perfect, partial, and empty conditions."""
    retrieved = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
    expected = {"chunk_1", "chunk_3"}

    # At K=1: 1 found out of 2 expected -> 0.5
    assert compute_recall_at_k(retrieved, expected, k=1) == 0.5

    # At K=3: 2 found out of 2 expected -> 1.0
    assert compute_recall_at_k(retrieved, expected, k=3) == 1.0

    # Empty expected: if nothing was expected, recall is 0.0 (no ground truth to recall)
    assert compute_recall_at_k(retrieved, set(), k=5) == 0.0

    # Empty retrieved with expected chunks -> 0.0
    assert compute_recall_at_k([], expected, k=5) == 0.0


def test_hit_rate_at_k():
    """Verify Hit Rate@K (binary presence of any relevant chunk in top K)."""
    retrieved = ["chunk_miss", "chunk_hit", "chunk_miss2"]
    expected = {"chunk_hit"}

    assert compute_hit_rate_at_k(retrieved, expected, k=1) == 0.0
    assert compute_hit_rate_at_k(retrieved, expected, k=2) == 1.0
    assert compute_hit_rate_at_k(retrieved, expected, k=5) == 1.0
    assert compute_hit_rate_at_k([], expected, k=5) == 0.0
    assert compute_hit_rate_at_k(retrieved, set(), k=5) == 0.0
    assert compute_hit_rate_at_k([], set(), k=5) == 1.0


def test_mrr_at_k():
    """Verify Mean Reciprocal Rank at various match ranks."""
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]

    # Match at rank 1 -> 1/1 = 1.0
    assert compute_mrr_at_k(retrieved, {"doc_a"}, k=4) == 1.0

    # Match at rank 2 -> 1/2 = 0.5
    assert compute_mrr_at_k(retrieved, {"doc_b"}, k=4) == 0.5

    # Match at rank 3 -> 1/3 ≈ 0.3333
    assert round(compute_mrr_at_k(retrieved, {"doc_c"}, k=4), 4) == 0.3333

    # Match beyond K or no match -> 0.0
    assert compute_mrr_at_k(retrieved, {"doc_c"}, k=2) == 0.0
    assert compute_mrr_at_k(retrieved, {"doc_z"}, k=4) == 0.0


def test_context_sufficiency_evaluation():
    """Verify context sufficiency evaluation with chunks, expected IDs, and empty states."""
    chunks_sufficient = [
        {"chunk_id": "c1", "similarity_score": 0.85, "text": "Hypertension treatment with thiazide"},
        {"chunk_id": "c2", "similarity_score": 0.75, "text": "Lifestyle modifications"}
    ]
    # Expected chunk c1 present with score >= 0.25 -> True
    assert evaluate_context_sufficiency(chunks_sufficient, ["c1", "c3"], min_similarity_threshold=0.25) is True

    # Expected chunk not present -> False
    assert evaluate_context_sufficiency(chunks_sufficient, ["c99"], min_similarity_threshold=0.25) is False

    # Low similarity score below threshold -> False
    chunks_low = [{"chunk_id": "c1", "similarity_score": 0.15}]
    assert evaluate_context_sufficiency(chunks_low, ["c1"], min_similarity_threshold=0.25) is False

    # Empty expected (out of domain) with no high-scoring chunks -> True
    assert evaluate_context_sufficiency(chunks_low, [], min_similarity_threshold=0.25) is True

    # Empty expected with high-scoring false positives -> False
    assert evaluate_context_sufficiency(chunks_sufficient, [], min_similarity_threshold=0.25) is False


def test_evaluate_retrieval_structured_output():
    """Verify evaluate_retrieval returns StructuredRetrievalEvaluation with correct metrics."""
    query = "What is the recommended first-line therapy for essential hypertension?"
    retrieved_chunks = [
        {"chunk_id": "chunk_hyp_01", "text": "First-line agents include ACE inhibitors, ARBs, CCBs, or thiazide diuretics.", "similarity_score": 0.85},
        {"chunk_id": "chunk_hyp_02", "text": "Blood pressure target is < 130/80 mmHg.", "similarity_score": 0.75},
        {"chunk_id": "chunk_unrelated", "text": "Unrelated paragraph about dermatology.", "similarity_score": 0.30},
    ]
    expected_chunk_ids = ["chunk_hyp_01", "chunk_hyp_02"]

    res = evaluate_retrieval(
        query=query,
        retrieved_chunks=retrieved_chunks,
        expected_chunk_ids=expected_chunk_ids,
        k=2
    )

    assert isinstance(res, StructuredRetrievalEvaluation)
    assert res.query == query
    assert res.precision_at_k == 1.0
    assert res.recall_at_k == 1.0
    assert res.hit_rate_at_k == 1.0
    assert res.mrr_at_k == 1.0
    assert res.passed is True
    assert len(res.retrieved_chunks) == 3
    assert len(res.relevant_chunks) == 2


def test_retrieval_evaluator_legacy_compatibility():
    """Verify legacy RetrievalEvaluator methods remain 100% backwards compatible."""
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5])
    test_cases = [
        {
            "query": "What causes hypertension?",
            "expected_chunk_ids": ["c1", "c2"],
            "retrieved_chunk_ids": ["c1", "c3", "c4"]
        }
    ]
    summary = evaluator.evaluate_all(test_cases)
    assert "mean_precision" in summary
    assert "mean_recall" in summary
    assert "mean_mrr" in summary
    assert summary["total_queries"] == 1


def test_vector_store_read_only_invariant():
    """Verify that retrieval evaluation never mutates the underlying vector store files."""
    faiss_path = Path("data/vector_store/index.faiss")
    meta_path = Path("data/vector_store/metadata.json")

    assert faiss_path.exists()
    assert meta_path.exists()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    initial_count = len(meta_data.get("records", []))
    assert initial_count == 744

    # Run retrieval evaluation
    evaluate_retrieval(
        query="hypertension treatment",
        retrieved_chunks=[{"chunk_id": "c1", "text": "test"}],
        expected_chunk_ids=["c1"],
        k=5
    )

    # Re-verify counts
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data_after = json.load(f)
    assert len(meta_data_after.get("records", [])) == 744
