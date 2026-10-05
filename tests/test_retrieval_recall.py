"""
Phase 2E: Retrieval Recall Improvement Unit & Regression Tests.

Validates the 12 key recall requirements:
1. Hypertension synonym expansion
2. Diabetes/glycemic synonym expansion
3. Complications/sequelae synonym expansion
4. Multi-aspect query decomposition
5. Multi-query candidate merging
6. Duplicate candidate removal
7. Irrelevant medical document rejection
8. Ambiguous query safety
9. Preservation of Phase 2C precision
10. Top_k enforcement
11. Existing grounded-boundary behavior
12. User-document isolation
"""

import pytest
from typing import List, Dict, Any, Optional, Set, Union
from unittest.mock import MagicMock, patch

from backend.rag.query_expander import MedicalQueryExpander
from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import VectorStoreService


# ==============================================================================
# 1. Hypertension Synonym Expansion
# ==============================================================================
def test_1_hypertension_synonym_expansion():
    """Validates that high blood pressure queries expand and match hypertension evidence."""
    query = "What causes elevated arterial blood pressure?"
    terms = MedicalQueryExpander.get_expanded_terms(query)
    assert any("hypertension" in t.lower() for t in terms)

    expanded_q = MedicalQueryExpander.expand_query(query)
    assert "hypertension" in expanded_q.lower()

    # Verify _is_chunk_relevant_to_query matches chunk discussing hypertension
    chunk_text = "Primary hypertension is associated with genetics, sodium intake, and vascular stiffness."
    assert RAGService._is_chunk_relevant_to_query(query, chunk_text) is True


# ==============================================================================
# 2. Diabetes / Glycemic Synonym Expansion
# ==============================================================================
def test_2_diabetes_glycemic_synonym_expansion():
    """Validates that glycemic and blood sugar queries match diabetes and metformin evidence."""
    query = "What medications help regulate blood sugar in glycemic disorders?"
    terms = MedicalQueryExpander.get_expanded_terms(query)
    assert any(term in ["diabetes", "type 2 diabetes", "glucose", "insulin", "metformin"] for term in terms)

    chunk_text = "Metformin is the first-line pharmacotherapy for Type 2 Diabetes to control hyperglycemia."
    assert RAGService._is_chunk_relevant_to_query(query, chunk_text) is True


# ==============================================================================
# 3. Complications / Sequelae Synonym Expansion
# ==============================================================================
def test_3_complications_sequelae_synonym_expansion():
    """Validates that sequelae queries expand to complications and target organ damage."""
    query = "What are the common secondary sequelae of uncontrolled blood pressure?"
    terms = MedicalQueryExpander.get_expanded_terms(query)
    assert any(term in ["complications", "organ damage", "heart disease", "stroke"] for term in terms)

    chunk_text = "Chronic high blood pressure leads to secondary complications including stroke, nephropathy, and heart failure."
    assert RAGService._is_chunk_relevant_to_query(query, chunk_text) is True


# ==============================================================================
# 4. Multi-Aspect Query Decomposition
# ==============================================================================
def test_4_multi_aspect_query_decomposition():
    """Validates that compound queries are recognized and decomposed into focused sub-queries."""
    query = "Provide an overview of hypertension definition, risk factors, lifestyle measures, and complications."
    assert MedicalQueryExpander.is_multi_aspect_query(query) is True

    sub_queries = MedicalQueryExpander.decompose_multi_aspect_query(query)
    assert len(sub_queries) >= 3
    combined = " ".join(sub_queries).lower()
    assert "definition" in combined
    assert "risk factor" in combined or "factors" in combined
    assert "lifestyle" in combined
    assert "complications" in combined


# ==============================================================================
# 5. Multi-Query Candidate Merging
# ==============================================================================
def test_5_multi_query_candidate_merging():
    """Validates that candidate chunks from distinct aspects are merged during multi-aspect retrieval."""
    mock_vs = MagicMock(spec=VectorStoreService)

    # Sub-query 1 returns definition chunk; Sub-query 2 returns complications chunk
    chunk_def = {
        "document_id": "doc_def",
        "chunk_id": "c0",
        "similarity_score": 0.85,
        "text": "Hypertension is defined as persistent systolic BP >= 130 mmHg.",
        "metadata": {"filename": "def.pdf"}
    }
    chunk_comp = {
        "document_id": "doc_comp",
        "chunk_id": "c1",
        "similarity_score": 0.80,
        "text": "Complications of hypertension include stroke and myocardial infarction.",
        "metadata": {"filename": "comp.pdf"}
    }

    call_count = [0]
    def mock_search(vec, top_k=20, user_id=None):
        call_count[0] += 1
        if call_count[0] <= 1:
            return [chunk_def]
        return [chunk_comp]

    mock_vs.search.side_effect = mock_search

    rag = RAGService(vector_store=mock_vs)
    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        results = rag.retrieve_context(
            "hypertension definition, risk factors, and complications",
            top_k=5
        )

    doc_ids = [c["document_id"] for c in results]
    assert "doc_def" in doc_ids
    assert "doc_comp" in doc_ids


# ==============================================================================
# 6. Duplicate Candidate Removal
# ==============================================================================
def test_6_duplicate_candidate_removal():
    """Validates that multiple identical chunk copies across sub-queries are deduplicated."""
    chunk_copy1 = {
        "document_id": "doc1",
        "chunk_id": "c0",
        "similarity_score": 0.75,
        "text": "Regular aerobic exercise reduces systolic blood pressure by 5-8 mmHg.",
        "metadata": {"filename": "exercise.pdf"}
    }
    chunk_copy2 = {
        "document_id": "doc1",
        "chunk_id": "c0",
        "similarity_score": 0.82,  # higher score
        "text": "Regular aerobic exercise reduces systolic blood pressure by 5-8 mmHg.",
        "metadata": {"filename": "exercise.pdf"}
    }

    deduped = RAGService.deduplicate_chunks([chunk_copy1, chunk_copy2])
    assert len(deduped) == 1
    assert deduped[0]["similarity_score"] == 0.82


# ==============================================================================
# 7. Irrelevant Medical Document Rejection
# ==============================================================================
def test_7_irrelevant_medical_document_rejection():
    """Validates that out-of-scope medical questions are safely rejected by pre-LLM safety gate."""
    query = "What are the chemotherapy guidelines for pancreatic carcinoma in the general hypertension guideline?"
    irrelevant_chunks = [
        {
            "document_id": "doc_htn",
            "chunk_id": "c0",
            "similarity_score": 0.35,
            "text": "General clinical guideline for hypertension management and target BP.",
            "metadata": {"filename": "hypertension_summary.pdf"}
        }
    ]

    is_relevant, reason = RAGService.verify_relevance_and_sufficiency(
        question=query,
        retrieved_chunks=irrelevant_chunks,
        similarity_threshold=0.25
    )
    assert is_relevant is False
    assert "query_targets_condition" in reason or "no_core_query_terms" in reason


# ==============================================================================
# 8. Ambiguous Query Safety
# ==============================================================================
def test_8_ambiguous_query_safety():
    """Validates that ambiguous queries without sufficient grounded evidence halt generation safely."""
    query = "What treatment was prescribed for the patient?"
    unrelated_chunks = [
        {
            "document_id": "doc_diag",
            "chunk_id": "c0",
            "similarity_score": 0.30,
            "text": "Patient vitals and demographic data. Blood pressure 145/95 mmHg. No prescription written.",
            "metadata": {"filename": "triage.pdf"}
        }
    ]

    is_relevant, reason = RAGService.verify_relevance_and_sufficiency(
        question=query,
        retrieved_chunks=unrelated_chunks,
        similarity_threshold=0.25
    )
    assert is_relevant is False


# ==============================================================================
# 9. Preservation of Phase 2C Precision
# ==============================================================================
def test_9_preservation_of_phase_2c_precision():
    """Validates that Phase 2C dynamic relative pruning prunes weakly-related chunks when a strong match exists."""
    query = "What are the diagnostic blood pressure thresholds for hypertension?"
    strong_chunk = {
        "document_id": "doc_htn_thresh",
        "chunk_id": "c0",
        "similarity_score": 0.85,
        "text": "Hypertension is diagnosed when blood pressure is consistently 130/80 mmHg or higher.",
        "metadata": {"filename": "htn.pdf"}
    }
    weak_irrelevant_chunk = {
        "document_id": "doc_skin",
        "chunk_id": "c1",
        "similarity_score": 0.45,
        "text": "Eczema and dermatitis are inflammatory skin conditions presenting with erythema.",
        "metadata": {"filename": "derma.pdf"}
    }

    filtered = RAGService.filter_candidate_precision(
        query=query,
        chunks=[strong_chunk, weak_irrelevant_chunk],
        threshold=0.25
    )
    assert len(filtered) == 1
    assert filtered[0]["document_id"] == "doc_htn_thresh"


# ==============================================================================
# 10. Top_k Enforcement
# ==============================================================================
def test_10_top_k_enforcement():
    """Validates that retrieval output strictly adheres to top_k even after multi-query merging."""
    mock_vs = MagicMock(spec=VectorStoreService)

    # 10 distinct chunks returned across sub-queries
    many_chunks = [
        {
            "document_id": f"doc_{i}",
            "chunk_id": f"c_{i}",
            "similarity_score": 0.70 + (i * 0.01),
            "text": f"Hypertension clinical guideline fact number {i}.",
            "metadata": {"filename": f"doc_{i}.pdf"}
        }
        for i in range(10)
    ]
    mock_vs.search.return_value = many_chunks

    rag = RAGService(vector_store=mock_vs)
    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        retrieved = rag.retrieve_context("hypertension overview", top_k=3)

    assert len(retrieved) <= 3


# ==============================================================================
# 11. Existing Grounded-Boundary Behavior
# ==============================================================================
def test_11_existing_grounded_boundary_behavior():
    """Validates that explicit document boundary statements are preserved through precision filtering."""
    query = "What does the uploaded test document state about tuberculosis treatments?"
    boundary_chunk = {
        "document_id": "doc_boundary",
        "chunk_id": "c_boundary",
        "similarity_score": 0.42,
        "text": "Scope Boundary: This document does not contain information about tuberculosis treatments or regimens.",
        "metadata": {"filename": "HealthAI_RAG_Test_Document.pdf"}
    }

    filtered = RAGService.filter_candidate_precision(
        query=query,
        chunks=[boundary_chunk],
        threshold=0.25
    )
    assert len(filtered) == 1
    assert filtered[0]["document_id"] == "doc_boundary"


# ==============================================================================
# 12. User-Document Isolation
# ==============================================================================
def test_12_user_document_isolation():
    """Validates that multi-query retrieval passes user_id to all vector store searches."""
    mock_vs = MagicMock(spec=VectorStoreService)
    mock_vs.search.return_value = []

    rag = RAGService(vector_store=mock_vs)
    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        rag.retrieve_context(
            "hypertension definition, risk factors, and complications",
            top_k=5,
            user_id=42
        )

    # All calls to search must have user_id=42
    assert mock_vs.search.call_count >= 1
    for call_args in mock_vs.search.call_args_list:
        assert call_args.kwargs.get("user_id") == 42


# ==============================================================================
# Phase 2E.1: RecallEvaluator & Benchmark Unit and Integration Tests
# ==============================================================================


from backend.evaluation.recall_evaluator import (
    RecallEvaluator,
    BenchmarkQuery,
    QueryRecallResult,
    AggregateRecallMetrics,
    HYPERTENSION_LIFESTYLE_BENCHMARK,
    compute_concept_recall_at_k,
    compute_chunk_recall_at_k,
    verify_concept_in_text,
)


# ==============================================================================
# Phase 2E.1 - 1. Concept Verification & Text Matching Tests
# ==============================================================================

def test_verify_concept_in_text_exact_and_phrasal():
    """Verifies that clinical concept phrases match correctly in medical text."""
    sample_text = (
        "General approaches that may support healthy blood pressure include "
        "regular physical activity, maintaining a healthy weight when appropriate, "
        "choosing a balanced diet rich in vegetables, fruits, whole grains, "
        "moderating sodium intake, avoiding tobacco, limiting alcohol, and getting adequate sleep."
    )

    assert verify_concept_in_text("regular physical activity", sample_text) is True
    assert verify_concept_in_text("healthy weight", sample_text) is True
    assert verify_concept_in_text("balanced diet", sample_text) is True
    assert verify_concept_in_text("sodium", sample_text) is True
    assert verify_concept_in_text("tobacco", sample_text) is True
    assert verify_concept_in_text("alcohol", sample_text) is True
    assert verify_concept_in_text("sleep", sample_text) is True

    # Concept not present
    assert verify_concept_in_text("chemotherapy", sample_text) is False
    assert verify_concept_in_text("beta-blockers", sample_text) is False
    assert verify_concept_in_text("", sample_text) is False
    assert verify_concept_in_text("sodium", "") is False


def test_verify_concept_does_not_false_positive_on_generic_words():
    """Verifies that mentioning 'hypertension' does not trigger concepts like 'diet' or 'exercise'."""
    text_without_lifestyle = (
        "1. What Is Hypertension? Hypertension, commonly called high blood pressure, "
        "is a condition in which the force of blood against artery walls is persistently elevated."
    )
    assert verify_concept_in_text("regular physical activity", text_without_lifestyle) is False
    assert verify_concept_in_text("balanced diet", text_without_lifestyle) is False
    assert verify_concept_in_text("tobacco", text_without_lifestyle) is False


# ==============================================================================
# 2. Recall@K Calculation Tests
# ==============================================================================

def test_compute_concept_recall_exact_calculation():
    """Verifies exact recall calculation across different K values."""
    chunks = [
        {"chunk_id": "c1", "text": "Patient advised on regular physical activity and healthy weight."},
        {"chunk_id": "c2", "text": "Diet recommendations: balanced diet and moderate sodium."},
        {"chunk_id": "c3", "text": "Advised stopping tobacco and limiting alcohol with adequate sleep."},
    ]
    concepts = [
        "regular physical activity",
        "healthy weight",
        "balanced diet",
        "sodium",
        "tobacco",
        "alcohol",
        "sleep"
    ]

    # At K=1: c1 has 2/7 concepts
    r1 = compute_concept_recall_at_k(chunks, concepts, k=1)
    assert pytest.approx(r1, rel=1e-3) == 2.0 / 7.0

    # At K=2: c1 + c2 have 4/7 concepts
    r2 = compute_concept_recall_at_k(chunks, concepts, k=2)
    assert pytest.approx(r2, rel=1e-3) == 4.0 / 7.0

    # At K=3: c1 + c2 + c3 have 7/7 concepts
    r3 = compute_concept_recall_at_k(chunks, concepts, k=3)
    assert pytest.approx(r3, rel=1e-3) == 1.0


def test_compute_concept_recall_invalid_k_values():
    """Verifies that invalid, zero, or negative K values return 0.0 safely."""
    chunks = [{"chunk_id": "c1", "text": "regular physical activity and sleep"}]
    concepts = ["regular physical activity", "sleep"]

    assert compute_concept_recall_at_k(chunks, concepts, k=0) == 0.0
    assert compute_concept_recall_at_k(chunks, concepts, k=-1) == 0.0
    assert compute_concept_recall_at_k(chunks, concepts, k=-100) == 0.0
    assert compute_concept_recall_at_k(chunks, [], k=5) == 0.0


def test_compute_concept_recall_empty_retrieval_results():
    """Verifies that an empty chunk list returns 0.0 recall."""
    concepts = ["regular physical activity", "healthy weight"]
    assert compute_concept_recall_at_k([], concepts, k=1) == 0.0
    assert compute_concept_recall_at_k([], concepts, k=5) == 0.0
    assert compute_concept_recall_at_k([], concepts, k=10) == 0.0


def test_compute_concept_recall_duplicate_chunks():
    """Verifies that duplicate chunks do not artificially inflate or corrupt recall calculation."""
    dup_chunk = {"chunk_id": "c1", "text": "regular physical activity and healthy weight"}
    chunks_with_duplicates = [dup_chunk, dup_chunk, dup_chunk]
    concepts = ["regular physical activity", "healthy weight", "balanced diet"]

    # 2 out of 3 concepts found despite 3 duplicate chunks
    r1 = compute_concept_recall_at_k(chunks_with_duplicates, concepts, k=1)
    r3 = compute_concept_recall_at_k(chunks_with_duplicates, concepts, k=3)
    assert pytest.approx(r1, rel=1e-3) == 2.0 / 3.0
    assert pytest.approx(r3, rel=1e-3) == 2.0 / 3.0


def test_compute_concept_recall_missing_expected_concepts():
    """Verifies partial recall and missing concepts reporting."""
    chunks = [{"chunk_id": "c1", "text": "Sodium and tobacco were discussed."}]
    concepts = ["regular physical activity", "healthy weight", "sodium", "tobacco"]

    r = compute_concept_recall_at_k(chunks, concepts, k=1)
    assert pytest.approx(r, rel=1e-3) == 2.0 / 4.0  # 0.5


def test_compute_chunk_recall_at_k():
    """Verifies standard chunk-id based recall."""
    retrieved = ["c1", "c2", "c3", "c4"]
    expected = ["c1", "c5"]

    assert compute_chunk_recall_at_k(retrieved, expected, k=1) == 0.5  # c1 in top 1 of 2
    assert compute_chunk_recall_at_k(retrieved, expected, k=3) == 0.5  # c1 in top 3 of 2
    assert compute_chunk_recall_at_k([], expected, k=5) == 0.0
    assert compute_chunk_recall_at_k(retrieved, [], k=5) == 0.0
    assert compute_chunk_recall_at_k(retrieved, expected, k=0) == 0.0


# ==============================================================================
# 3. RecallEvaluator Unit Tests (Offline / Mocked RAG)
# ==============================================================================

class MockRAGService:
    """Mock RAGService for isolated offline tests."""
    def __init__(self, canned_responses: Dict[str, Dict[str, Any]]):
        self.canned = canned_responses

    def query(self, question: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        return self.canned.get(question, {
            "question": question,
            "retrieved_chunks": [],
            "retrieval_status": "no_relevant_context"
        })


def test_evaluator_successful_retrieval():
    """Verifies RecallEvaluator correctly processes successful queries."""
    canned = {
        "What lifestyle changes help hypertension?": {
            "question": "What lifestyle changes help hypertension?",
            "retrieved_chunks": [
                {
                    "chunk_id": "chunk_0",
                    "text": (
                        "General approaches include regular physical activity, maintaining a healthy weight, "
                        "choosing a balanced diet, moderating sodium, avoiding tobacco, limiting alcohol, "
                        "and getting adequate sleep."
                    ),
                    "similarity_score": 0.6903,
                    "metadata": {"filename": "synthetic_hypertension_test.pdf"}
                }
            ],
            "retrieval_status": "success"
        }
    }
    mock_rag = MockRAGService(canned)
    evaluator = RecallEvaluator(rag_service=mock_rag, k_values=[1, 3, 5, 10])

    res = evaluator.evaluate_query(
        query="What lifestyle changes help hypertension?",
        expected_concepts=[
            "regular physical activity",
            "healthy weight",
            "balanced diet",
            "sodium",
            "tobacco",
            "alcohol",
            "sleep"
        ]
    )

    assert res.retrieval_status == "success"
    assert res.recall_at_k[1] == 1.0
    assert res.recall_at_k[3] == 1.0
    assert res.recall_at_k[5] == 1.0
    assert res.recall_at_k[10] == 1.0
    assert len(res.matched_concepts) == 7
    assert len(res.missing_concepts) == 0
    assert res.concept_coverage == 1.0


def test_evaluator_failed_retrieval():
    """Verifies RecallEvaluator correctly handles queries that yield no chunks."""
    mock_rag = MockRAGService({})
    evaluator = RecallEvaluator(rag_service=mock_rag, k_values=[1, 3, 5, 10])

    res = evaluator.evaluate_query(
        query="What non-medication measures help manage hypertension?",
        expected_concepts=["regular physical activity", "healthy weight"]
    )

    assert res.retrieval_status == "no_relevant_context"
    assert res.recall_at_k[1] == 0.0
    assert res.recall_at_k[3] == 0.0
    assert res.recall_at_k[5] == 0.0
    assert res.recall_at_k[10] == 0.0
    assert len(res.matched_concepts) == 0
    assert len(res.missing_concepts) == 2
    assert res.concept_coverage == 0.0


def test_evaluator_deterministic_evaluation():
    """Verifies that running evaluation repeatedly produces identical deterministic metrics."""
    canned = {
        "test query": {
            "question": "test query",
            "retrieved_chunks": [
                {
                    "chunk_id": "chunk_0",
                    "text": "balanced diet and sodium moderation.",
                    "similarity_score": 0.55
                }
            ],
            "retrieval_status": "success"
        }
    }
    mock_rag = MockRAGService(canned)
    evaluator = RecallEvaluator(rag_service=mock_rag, k_values=[1, 3, 5])
    concepts = ["balanced diet", "sodium", "sleep"]

    res1 = evaluator.evaluate_query("test query", expected_concepts=concepts)
    res2 = evaluator.evaluate_query("test query", expected_concepts=concepts)

    assert res1.recall_at_k == res2.recall_at_k
    assert res1.matched_concepts == res2.matched_concepts
    assert res1.missing_concepts == res2.missing_concepts
    assert res1.concept_coverage == res2.concept_coverage


def test_evaluator_report_formatting():
    """Verifies that query and benchmark reports match the required format."""
    mock_rag = MockRAGService({
        "Q1": {
            "question": "Q1",
            "retrieved_chunks": [
                {"chunk_id": "chunk_0", "text": "regular physical activity", "similarity_score": 0.65}
            ],
            "retrieval_status": "success"
        }
    })
    evaluator = RecallEvaluator(rag_service=mock_rag, k_values=[1, 3, 5, 10])
    metrics = evaluator.evaluate_benchmark(
        benchmark=[{"query": "Q1", "expected_concepts": ["regular physical activity", "sleep"]}]
    )

    report = evaluator.format_benchmark_report(metrics)
    assert "Retrieval Recall Evaluation" in report
    assert "Recall@1" in report
    assert "Recall@3" in report
    assert "Recall@5" in report
    assert "Recall@10" in report
    assert "✓ regular physical activity" in report
    assert "✗ sleep" in report
    assert "Aggregate Recall Baseline Summary" in report
    assert "Mean Recall@1" in report


# ==============================================================================
# 4. Live Benchmark Baseline Integration Test
# ==============================================================================

def test_live_hypertension_lifestyle_benchmark_baseline():
    """
    Executes the canonical Phase 2E.1 retrieval benchmark against the live FAISS index.

    Validates:
    - 4 queries evaluated.
    - Queries 1, 2, and 3 retrieve chunk_0 with 100% recall (7/7 concepts).
    - Query 4 ('What non-medication measures help manage hypertension?') fails retrieval
      due to pre-LLM sufficiency gate ('missing_medication_recommendations_in_context').
    - Establishes the ground-truth baseline Mean Recall@K = 0.75 (75.0%).
    """
    from backend.services.vector_store_service import get_vector_store_service
    from backend.rag.rag_service import RAGService

    vs = get_vector_store_service()
    assert vs.count() > 0, "FAISS vector store must be loaded"

    rag = RAGService(vector_store=vs)
    evaluator = RecallEvaluator(rag_service=rag, k_values=[1, 3, 5, 10], default_user_id=2)

    metrics = evaluator.evaluate_benchmark(HYPERTENSION_LIFESTYLE_BENCHMARK)

    assert metrics.total_queries == 4
    assert metrics.successful_queries == 4
    assert metrics.failed_queries == 0

    # Verify individual queries
    q1_res = metrics.query_results[0]
    assert q1_res.retrieval_status == "success"
    assert q1_res.recall_at_k[1] == 1.0
    assert q1_res.recall_at_k[5] == 1.0
    assert "synthetic_hypertension_test.pdf" in q1_res.retrieved_documents[0]

    q2_res = metrics.query_results[1]
    assert q2_res.retrieval_status == "success"
    assert q2_res.recall_at_k[1] == 1.0
    assert q2_res.recall_at_k[5] == 1.0

    q3_res = metrics.query_results[2]
    assert q3_res.retrieval_status == "success"
    assert q3_res.recall_at_k[1] == 1.0
    assert q3_res.recall_at_k[5] == 1.0

    q4_res = metrics.query_results[3]
    # Query 4 ('What non-medication measures help manage hypertension?') succeeds after Phase 2E.2 query expansion
    assert q4_res.retrieval_status == "success"
    assert q4_res.recall_at_k[1] == 1.0
    assert q4_res.recall_at_k[5] == 1.0
    assert len(q4_res.matched_concepts) == 7

    # Aggregate: all 4 queries achieve 1.0 => 1.0
    assert pytest.approx(metrics.mean_recall_at_k[1], rel=1e-3) == 1.0
    assert pytest.approx(metrics.mean_recall_at_k[3], rel=1e-3) == 1.0
    assert pytest.approx(metrics.mean_recall_at_k[5], rel=1e-3) == 1.0
    assert pytest.approx(metrics.mean_recall_at_k[10], rel=1e-3) == 1.0
