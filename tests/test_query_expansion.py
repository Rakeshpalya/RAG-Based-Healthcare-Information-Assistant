"""
Unit and Integration Tests for Phase 2E.2: Medical Query Expansion and Intent Normalization.

Validates:
1. Deterministic intent classification (LIFESTYLE, MEDICATION, MIXED, GENERAL)
2. Strict negation priority (negative pharmaceutical phrases do NOT trigger medication requirements)
3. Canonical lifestyle expansion generation and original query preservation
4. Capitalization and punctuation tolerance
5. Pre-LLM evidence sufficiency gate for lifestyle, medication, and mixed queries
6. End-to-end benchmark recall: 4/4 queries succeed with Mean Recall@1 == 1.0
7. Medical safety fallback preservation for genuine medication queries
"""

import pytest
from backend.rag.query_expansion import (
    QueryIntent,
    QueryIntentResult,
    normalize_medical_query,
    CANONICAL_LIFESTYLE_EXPANSION_TERMS,
)
from backend.rag.rag_service import RAGService
from backend.rag.query_expander import MedicalQueryExpander
from backend.evaluation.recall_evaluator import (
    RecallEvaluator,
    HYPERTENSION_LIFESTYLE_BENCHMARK,
)


# ==============================================================================
# 1. Deterministic Intent Classification Tests (Required Minimum 12 Cases)
# ==============================================================================

@pytest.mark.parametrize("query,expected_intent", [
    ("What lifestyle changes help hypertension?", QueryIntent.LIFESTYLE),
    ("What non-medication measures help manage hypertension?", QueryIntent.LIFESTYLE),
    ("What non medication measures help manage hypertension?", QueryIntent.LIFESTYLE),
    ("What non-pharmacological approaches help hypertension?", QueryIntent.LIFESTYLE),
    ("What non-pharmacologic approaches help hypertension?", QueryIntent.LIFESTYLE),
    ("What non-drug interventions help hypertension?", QueryIntent.LIFESTYLE),
    ("What medications are used for hypertension?", QueryIntent.MEDICATION),
    ("What drugs are used for hypertension?", QueryIntent.MEDICATION),
    ("What pharmaceutical treatment is used for hypertension?", QueryIntent.MEDICATION),
    ("What medication and lifestyle changes help hypertension?", QueryIntent.MIXED),
    ("What are the side effects of hypertension medication?", QueryIntent.MEDICATION),
    ("What lifestyle changes should I make without medication?", QueryIntent.LIFESTYLE),
])
def test_normalize_medical_query_minimum_12_cases(query, expected_intent):
    """Verifies all 12 required canonical benchmark queries classify to their exact expected intent."""
    res = normalize_medical_query(query)
    assert res.intent == expected_intent
    if expected_intent == QueryIntent.LIFESTYLE:
        assert res.is_lifestyle is True
        assert res.is_medication is False
        assert res.is_mixed is False
    elif expected_intent == QueryIntent.MEDICATION:
        assert res.is_lifestyle is False
        assert res.is_medication is True
        assert res.is_mixed is False
    elif expected_intent == QueryIntent.MIXED:
        assert res.is_lifestyle is True
        assert res.is_medication is True
        assert res.is_mixed is True


# ==============================================================================
# 2. Capitalization and Punctuation Tolerance Tests
# ==============================================================================

@pytest.mark.parametrize("query,expected_intent", [
    ("WHAT LIFESTYLE CHANGES HELP HYPERTENSION???", QueryIntent.LIFESTYLE),
    ("What Non-Medication Measures Help Manage Hypertension!", QueryIntent.LIFESTYLE),
    ("what non-pharmacological approaches help hypertension...", QueryIntent.LIFESTYLE),
    ("WHAT MEDICATIONS ARE USED FOR HYPERTENSION?!", QueryIntent.MEDICATION),
    ("What medications, if any, treat hypertension?", QueryIntent.MEDICATION),
    ("Can lifestyle approaches (diet, exercise) control blood pressure without medication?", QueryIntent.LIFESTYLE),
    ("What medications and non-drug interventions help hypertension???", QueryIntent.MIXED),
])
def test_normalize_medical_query_capitalization_and_punctuation(query, expected_intent):
    """Verifies that intent classification is invariant to uppercase, lowercase, and varied punctuation."""
    res = normalize_medical_query(query)
    assert res.intent == expected_intent


# ==============================================================================
# 3. Negation Priority Tests
# ==============================================================================

def test_negation_priority_over_medication_token():
    """
    Verifies that negative pharmaceutical phrases ('non-medication', 'without medication',
    'non-drug', etc.) prevent the token 'medication' or 'drug' from triggering medication intent.
    """
    queries = [
        "What non-medication measures help manage hypertension?",
        "How to lower blood pressure without medication?",
        "Lifestyle measures instead of medications for high blood pressure",
        "Managing hypertension avoiding drugs and pharmaceuticals",
        "Can I stay off medications with diet and exercise?",
        "Non-drug interventions for hypertension",
        "Non-pharmacological management of elevated arterial blood pressure",
    ]
    for q in queries:
        res = normalize_medical_query(q)
        assert res.intent == QueryIntent.LIFESTYLE, f"Query '{q}' was classified as {res.intent}, expected LIFESTYLE"
        assert res.has_negative_pharmaceutical is True
        assert res.is_medication is False
        assert MedicalQueryExpander.has_medication_intent(q) is False


# ==============================================================================
# 4. Query Expansion & Query Preservation Tests
# ==============================================================================

def test_lifestyle_query_expansion_terms():
    """
    Verifies that lifestyle queries generate canonical lifestyle expansion terms
    and preserve the exact original user question at the beginning.
    """
    raw_q = "What non-medication measures help manage hypertension?"
    res = normalize_medical_query(raw_q)

    assert res.raw_query == raw_q
    assert res.expanded_retrieval_query.startswith(raw_q)
    assert len(res.expansion_terms) > 0

    # Ensure key lifestyle terms are in the expansion
    for term in ["diet", "exercise", "physical activity", "healthy weight", "sodium", "sleep"]:
        assert any(term in exp.lower() for exp in res.expansion_terms)

    # Original query returned by MedicalQueryExpander must also preserve front query
    exp_q = MedicalQueryExpander.expand_query(raw_q)
    assert exp_q.startswith(raw_q)


def test_general_intent_for_neutral_queries():
    """Verifies that neutral/overview queries without lifestyle or medication terms resolve to GENERAL."""
    res = normalize_medical_query("What is hypertension?")
    assert res.intent == QueryIntent.GENERAL
    assert res.is_lifestyle is False
    assert res.is_medication is False
    assert res.is_mixed is False


# ==============================================================================
# 5. Sufficiency Gate Unit Tests
# ==============================================================================

def test_sufficiency_gate_lifestyle_query_with_lifestyle_context():
    """Verifies that lifestyle / non-medication queries pass when lifestyle evidence is present."""
    lifestyle_chunks = [{
        "chunk_id": "chunk_0",
        "similarity_score": 0.65,
        "text": (
            "General approaches that may support healthy blood pressure include "
            "regular physical activity, maintaining a healthy weight when appropriate, "
            "choosing a balanced diet rich in vegetables, fruits, whole grains, "
            "moderating sodium intake, avoiding tobacco, limiting alcohol, and getting adequate sleep."
        ),
        "metadata": {"filename": "synthetic_hypertension_test.pdf"}
    }]

    is_sufficient, reason = RAGService.verify_relevance_and_sufficiency(
        question="What non-medication measures help manage hypertension?",
        retrieved_chunks=lifestyle_chunks,
        similarity_threshold=0.25
    )
    assert is_sufficient is True
    assert reason == "context_relevant_and_sufficient"


def test_sufficiency_gate_medication_query_fails_on_lifestyle_context():
    """Verifies that medication queries FAIL when only lifestyle context is available (safety fallback)."""
    lifestyle_chunks = [{
        "chunk_id": "chunk_0",
        "similarity_score": 0.65,
        "text": (
            "General approaches that may support healthy blood pressure include "
            "regular physical activity, maintaining a healthy weight, and moderating sodium."
        ),
        "metadata": {"filename": "synthetic_hypertension_test.pdf"}
    }]

    is_sufficient, reason = RAGService.verify_relevance_and_sufficiency(
        question="What medications are recommended for hypertension?",
        retrieved_chunks=lifestyle_chunks,
        similarity_threshold=0.25
    )
    assert is_sufficient is False
    assert reason == "missing_medication_recommendations_in_context"


def test_sufficiency_gate_mixed_query_fails_when_medication_missing():
    """
    Verifies that mixed queries ('medication and lifestyle') FAIL if medication evidence is missing,
    preventing mixed queries from passing merely because lifestyle evidence exists.
    """
    lifestyle_only_chunks = [{
        "chunk_id": "chunk_0",
        "similarity_score": 0.65,
        "text": "Regular physical activity and low sodium intake reduce blood pressure.",
        "metadata": {"filename": "synthetic_hypertension_test.pdf"}
    }]

    is_sufficient, reason = RAGService.verify_relevance_and_sufficiency(
        question="What medications and lifestyle changes help manage hypertension?",
        retrieved_chunks=lifestyle_only_chunks,
        similarity_threshold=0.25
    )
    assert is_sufficient is False
    assert reason == "missing_medication_recommendations_in_context"


def test_sufficiency_gate_mixed_query_fails_when_lifestyle_missing():
    """Verifies that mixed queries FAIL if lifestyle evidence is missing even if medications are present."""
    medication_only_chunks = [{
        "chunk_id": "chunk_1",
        "similarity_score": 0.65,
        "text": "First-line medication options for hypertension include Lisinopril, Amlodipine, and ACE inhibitors.",
        "metadata": {"filename": "synthetic_hypertension_test.pdf"}
    }]

    is_sufficient, reason = RAGService.verify_relevance_and_sufficiency(
        question="What medications and lifestyle changes help manage hypertension?",
        retrieved_chunks=medication_only_chunks,
        similarity_threshold=0.25
    )
    assert is_sufficient is False
    assert reason == "missing_lifestyle_recommendations_in_context"


def test_sufficiency_gate_mixed_query_passes_with_both_evidences():
    """Verifies that mixed queries PASS when BOTH medication and lifestyle evidence are present."""
    combined_chunks = [{
        "chunk_id": "chunk_both",
        "similarity_score": 0.70,
        "text": (
            "Hypertension management: First-line medication includes Lisinopril and ACE inhibitors. "
            "General lifestyle measures include regular physical activity, diet, and healthy weight."
        ),
        "metadata": {"filename": "synthetic_hypertension_test.pdf"}
    }]

    is_sufficient, reason = RAGService.verify_relevance_and_sufficiency(
        question="What medications and lifestyle changes help manage hypertension?",
        retrieved_chunks=combined_chunks,
        similarity_threshold=0.25
    )
    assert is_sufficient is True
    assert reason == "context_relevant_and_sufficient"


# ==============================================================================
# 6. End-to-End Recall Benchmark Integration (All 4 Queries Must Pass)
# ==============================================================================

def test_e2e_benchmark_all_four_queries_pass():
    """
    Evaluates the complete Phase 2E.1 benchmark suite after Phase 2E.2 query expansion.
    Target: 4/4 successful retrievals, Mean Recall@1 == 1.0.
    """
    evaluator = RecallEvaluator(k_values=[1, 3, 5, 10], default_user_id=2)
    metrics = evaluator.evaluate_benchmark(benchmark=HYPERTENSION_LIFESTYLE_BENCHMARK, user_id=2)

    assert metrics.total_queries == 4
    assert metrics.successful_queries == 4, f"Expected 4 successful retrievals, got {metrics.successful_queries}"
    assert metrics.failed_queries == 0, f"Expected 0 failed retrievals, got {metrics.failed_queries}"

    # Mean Recall across all K values must be 1.0 (100%)
    for k in [1, 3, 5, 10]:
        score = metrics.mean_recall_at_k.get(k, 0.0)
        assert score == pytest.approx(1.0, rel=1e-3), f"Mean Recall@{k} was {score}, expected 1.0"

    # Verify individual Query 4 explicitly
    q4_res = metrics.query_results[3]
    assert q4_res.query == "What non-medication measures help manage hypertension?"
    assert q4_res.retrieval_status == "success"
    assert len(q4_res.retrieved_chunks) > 0
    assert q4_res.recall_at_k[1] == pytest.approx(1.0, rel=1e-3)
    assert len(q4_res.matched_concepts) == 7
    assert len(q4_res.missing_concepts) == 0


# ==============================================================================
# 7. End-to-End Safety Regression: Genuine Medication Queries Fall Back Safely
# ==============================================================================

def test_e2e_safety_genuine_medication_queries_fallback():
    """
    Verifies that genuine medication questions tested against the hypertension document
    (which only contains lifestyle information in chunk_0) continue to safely fall back.
    """
    from backend.services.vector_store_service import get_vector_store_service
    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)
    med_queries = [
        "What medication is recommended for hypertension?",
        "What drugs treat high blood pressure?",
        "What are the side effects of antihypertensive medication?",
    ]

    for q in med_queries:
        res = rag.query(question=q, user_id=2)
        assert res["retrieval_status"] == "no_relevant_context", f"Query '{q}' should have returned no_relevant_context"
        assert len(res["retrieved_chunks"]) == 0
        assert res["timings"].get("gate_reason") == "missing_medication_recommendations_in_context"
