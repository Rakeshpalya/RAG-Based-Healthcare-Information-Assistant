"""
Phase 2D: Retrieval Evaluation & Benchmarking Tests.

Verifies:
1. Evaluation dataset is valid (structure, required fields, query counts >= 30).
2. Every query has an expected relevance definition.
3. Metrics calculate correctly on controlled examples.
4. Precision@K works correctly.
5. Recall@K works correctly.
6. MRR works correctly.
7. Hit Rate works correctly.
8. Irrelevant evidence rate works correctly.
9. Empty retrieval results do not crash evaluation.
10. Fewer-than-K results do not crash evaluation.
11. Multi-document evaluation works.
12. Out-of-scope queries are evaluated separately.
13. Evaluation does not modify the vector store.
"""

import pytest
from unittest.mock import MagicMock
from tests.evaluation.retrieval_eval_dataset import EVALUATION_DATASET
from tests.evaluation.run_retrieval_evaluation import (
    calculate_query_metrics,
    matches_target,
    RetrievalEvaluator
)
from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import VectorStoreService


def test_1_evaluation_dataset_is_valid():
    """Test 1: Dataset contains at least 30 evaluation queries with required fields."""
    assert len(EVALUATION_DATASET) >= 30, f"Expected at least 30 queries, got {len(EVALUATION_DATASET)}"

    required_keys = {"id", "query", "expected_document_ids", "relevant_documents", "irrelevant_documents", "query_type", "difficulty"}
    for case in EVALUATION_DATASET:
        missing = required_keys - set(case.keys())
        assert not missing, f"Case {case.get('id')} missing keys: {missing}"
        assert case["query"].strip(), f"Case {case.get('id')} has empty query"
        assert case["query_type"] in {
            "single_document", "multi_document", "synonym", "multi_chunk",
            "out_of_scope", "ambiguous", "irrelevant", "exact_terminology", "natural_language"
        }, f"Case {case.get('id')} has invalid query_type {case['query_type']}"


def test_2_every_query_has_relevance_definition():
    """Test 2: Every query has an explicit relevance/boundary definition."""
    for case in EVALUATION_DATASET:
        q_type = case["query_type"]
        if q_type == "irrelevant":
            assert len(case["relevant_documents"]) == 0, f"Irrelevant query {case['id']} should have empty relevant docs"
            assert len(case["irrelevant_documents"]) > 0, f"Irrelevant query {case['id']} should specify irrelevant docs"
        elif q_type != "out_of_scope":
            assert len(case["relevant_documents"]) > 0, f"Query {case['id']} must define relevant documents"


def test_3_metrics_calculate_correctly_on_controlled_example():
    """
    Test 3: Controlled synthetic example.
    Retrieved: [DocA, DocB, DocC]
    Relevant: [DocA, DocD]
    Irrelevant: [DocB, DocC]
    """
    retrieved = [
        {"metadata": {"filename": "DocA.pdf"}, "similarity_score": 0.85},
        {"metadata": {"filename": "DocB.pdf"}, "similarity_score": 0.70},
        {"metadata": {"filename": "DocC.pdf"}, "similarity_score": 0.60},
    ]
    relevant = ["DocA.pdf", "DocD.pdf"]
    irrelevant = ["DocB.pdf", "DocC.pdf"]

    m = calculate_query_metrics(retrieved, relevant, irrelevant)

    # Precision@1: 1 / 1 = 1.0
    assert m["precision_1"] == 1.0
    # Precision@3: 1 / 3 = 0.3333
    assert m["precision_3"] == 0.3333
    # Precision@5: 1 / 5 = 0.2000
    assert m["precision_5"] == 0.2000

    # Recall@1: 1 / 2 = 0.50
    assert m["recall_1"] == 0.50
    # Recall@3: 1 / 2 = 0.50
    assert m["recall_3"] == 0.50

    # Hit Rate: 1.0 for all
    assert m["hit_rate_1"] == 1.0
    assert m["hit_rate_3"] == 1.0
    assert m["hit_rate_5"] == 1.0

    # MRR: DocA is at rank 1 -> 1 / 1 = 1.0
    assert m["mrr"] == 1.0

    # Evidence coverage: 1 of 2 relevant docs retrieved -> 0.5
    assert m["evidence_coverage"] == 0.5

    # Irrelevant rate: DocB and DocC are irrelevant (2 of 3) -> 2/3 = 0.6667
    assert m["irrelevant_rate"] == 0.6667


def test_4_precision_at_k():
    """Test 4: Precision@K decreases correctly as irrelevant items enter top-K."""
    # Top 1 relevant, next 4 irrelevant
    retrieved = [
        {"metadata": {"filename": "good.pdf"}},
        {"metadata": {"filename": "bad1.pdf"}},
        {"metadata": {"filename": "bad2.pdf"}},
        {"metadata": {"filename": "bad3.pdf"}},
        {"metadata": {"filename": "bad4.pdf"}},
    ]
    m = calculate_query_metrics(retrieved, ["good.pdf"], ["bad1.pdf"])
    assert m["precision_1"] == 1.0
    assert m["precision_3"] == round(1.0 / 3.0, 4)
    assert m["precision_5"] == 0.2


def test_5_recall_at_k():
    """Test 5: Recall@K increases as more relevant items are retrieved."""
    retrieved = [
        {"metadata": {"filename": "doc1.pdf"}},
        {"metadata": {"filename": "bad.pdf"}},
        {"metadata": {"filename": "doc2.pdf"}},
    ]
    relevant = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    m = calculate_query_metrics(retrieved, relevant, ["bad.pdf"])
    # Rank 1: 1 of 3
    assert m["recall_1"] == round(1.0 / 3.0, 4)
    # Rank 3: 2 of 3
    assert m["recall_3"] == round(2.0 / 3.0, 4)


def test_6_mrr_calculation():
    """Test 6: MRR handles relevant items at rank 1, 2, 3, or none."""
    # Relevant item at rank 2
    r_rank_2 = [
        {"metadata": {"filename": "bad.pdf"}},
        {"metadata": {"filename": "good.pdf"}},
    ]
    m2 = calculate_query_metrics(r_rank_2, ["good.pdf"], ["bad.pdf"])
    assert m2["mrr"] == 0.5

    # Relevant item at rank 3
    r_rank_3 = [
        {"metadata": {"filename": "bad1.pdf"}},
        {"metadata": {"filename": "bad2.pdf"}},
        {"metadata": {"filename": "good.pdf"}},
    ]
    m3 = calculate_query_metrics(r_rank_3, ["good.pdf"], ["bad1.pdf"])
    assert m3["mrr"] == round(1.0 / 3.0, 4)

    # No relevant item
    r_none = [{"metadata": {"filename": "bad1.pdf"}}]
    m_none = calculate_query_metrics(r_none, ["good.pdf"], ["bad1.pdf"])
    assert m_none["mrr"] == 0.0


def test_7_hit_rate_calculation():
    """Test 7: Hit Rate is 1 when at least one relevant document is found in top K, else 0."""
    retrieved = [{"metadata": {"filename": "bad.pdf"}}, {"metadata": {"filename": "good.pdf"}}]
    m = calculate_query_metrics(retrieved, ["good.pdf"], ["bad.pdf"])
    assert m["hit_rate_1"] == 0.0
    assert m["hit_rate_3"] == 1.0
    assert m["hit_rate_5"] == 1.0


def test_8_irrelevant_evidence_rate():
    """Test 8: Irrelevant evidence rate measures proportion of off-topic results."""
    # 0 irrelevant
    ret1 = [{"metadata": {"filename": "good1.pdf"}}, {"metadata": {"filename": "good2.pdf"}}]
    m1 = calculate_query_metrics(ret1, ["good1.pdf", "good2.pdf"], [])
    assert m1["irrelevant_rate"] == 0.0

    # 100% irrelevant
    ret2 = [{"metadata": {"filename": "bad1.pdf"}}, {"metadata": {"filename": "bad2.pdf"}}]
    m2 = calculate_query_metrics(ret2, ["good1.pdf"], ["bad1.pdf", "bad2.pdf"])
    assert m2["irrelevant_rate"] == 1.0


def test_9_empty_retrieval_does_not_crash():
    """Test 9: Guard against ZeroDivisionError when retrieved items is empty."""
    m = calculate_query_metrics([], ["doc1.pdf"], ["bad.pdf"])
    assert m["precision_1"] == 0.0
    assert m["precision_5"] == 0.0
    assert m["recall_1"] == 0.0
    assert m["hit_rate_1"] == 0.0
    assert m["mrr"] == 0.0
    assert m["irrelevant_rate"] == 0.0


def test_10_fewer_than_k_results_does_not_crash():
    """Test 10: Gracefully handles retrieving fewer than K items (e.g. only 1 item for K=5)."""
    retrieved = [{"metadata": {"filename": "doc1.pdf"}}]
    m = calculate_query_metrics(retrieved, ["doc1.pdf"], [])
    assert m["precision_1"] == 1.0
    assert m["precision_3"] == round(1.0 / 3.0, 4)
    assert m["precision_5"] == 0.2000
    assert m["hit_rate_5"] == 1.0


def test_11_multi_document_evaluation():
    """Test 11: Multi-document query measures coverage across multiple target PDFs."""
    retrieved = [
        {"metadata": {"filename": "cardio.pdf"}},
        {"metadata": {"filename": "diet.pdf"}},
        {"metadata": {"filename": "exercise.pdf"}},
    ]
    relevant = ["cardio.pdf", "diet.pdf", "exercise.pdf"]
    m = calculate_query_metrics(retrieved, relevant, [])
    assert m["evidence_coverage"] == 1.0
    assert m["recall_3"] == 1.0
    assert m["precision_3"] == 1.0
    assert m["irrelevant_rate"] == 0.0


def test_12_out_of_scope_queries_evaluated_separately():
    """
    Test 12: Out-of-scope query evaluation:
    - If 0 chunks retrieved or status is no_relevant_context -> perfect score (rejection success).
    - If irrelevant chunks retrieved -> penalized (irrelevant_rate = 1.0).
    """
    # Case 1: Successfully halted
    m_halted = calculate_query_metrics(
        retrieved_items=[],
        relevant_docs=[],
        irrelevant_docs=["any.pdf"],
        is_out_of_scope=True,
        retrieval_status="no_relevant_context"
    )
    assert m_halted["irrelevant_rate"] == 0.0
    assert m_halted["hit_rate_5"] == 1.0

    # Case 2: Hallucinated / false positive retrieval
    m_leaked = calculate_query_metrics(
        retrieved_items=[{"metadata": {"filename": "unrelated.pdf"}}],
        relevant_docs=[],
        irrelevant_docs=["unrelated.pdf"],
        is_out_of_scope=True,
        retrieval_status="success"
    )
    assert m_leaked["irrelevant_rate"] == 1.0
    assert m_leaked["hit_rate_5"] == 0.0


def test_13_evaluation_does_not_modify_vector_store():
    """Test 13: Running evaluation harness leaves vector store count strictly unchanged."""
    mock_vs = MagicMock(spec=VectorStoreService)
    mock_vs.count.return_value = 42
    mock_vs.search.return_value = [
        {"document_id": "doc1", "similarity_score": 0.80, "text": "Medical info", "metadata": {"filename": "doc1.pdf"}}
    ]

    rag = RAGService(vector_store=mock_vs)
    evaluator = RetrievalEvaluator(rag_service=rag)

    sample_dataset = [
        {
            "id": "T1",
            "query": "Test medical query",
            "relevant_documents": ["doc1.pdf"],
            "irrelevant_documents": [],
            "query_type": "single_document",
            "difficulty": "easy"
        }
    ]

    rep = evaluator.run_benchmark(dataset=sample_dataset)

    assert rep["vector_store_count_before"] == rep["vector_store_count_after"] == 42
    assert rep["total_queries"] == 1
    assert rep["total_passed"] == 1
