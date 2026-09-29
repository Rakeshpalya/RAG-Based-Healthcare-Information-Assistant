import pytest
from backend.evaluation.answer_evaluator import AnswerEvaluator, AnswerEvaluationResult


def test_answer_evaluator_perfect_grounded_answer():
    """Verify evaluation of a fully grounded and correctly cited answer."""
    question = "What is the recommended blood pressure target for hypertension?"
    answer = "The clinical guideline recommends maintaining blood pressure below 130/80 mmHg [Source 1]."
    sources = [
        {
            "source_index": 1,
            "source_label": "[Source 1]",
            "text": "The recommended target blood pressure is below 130/80 mmHg for high-risk patients.",
            "document_id": "guideline_01"
        }
    ]

    res = AnswerEvaluator.evaluate_answer(
        question=question,
        answer_text=answer,
        retrieved_sources=sources
    )

    assert isinstance(res, AnswerEvaluationResult)
    assert res.citation_correctness == 1.0
    assert res.citation_completeness == 1.0
    assert res.groundedness >= 0.8
    assert res.hallucination_detected is False
    assert res.contradictions_detected == 0
    assert res.is_safe is True
    assert res.final_status == "PASS"


def test_answer_evaluator_missing_citation():
    """Verify that claims with missing citations reduce completeness score."""
    question = "What is metformin used for?"
    answer = "Metformin is a first-line medication for type 2 diabetes."
    sources = [
        {
            "source_index": 1,
            "source_label": "[Source 1]",
            "text": "Metformin hydrochloride is used to treat type 2 diabetes mellitus.",
            "document_id": "doc_diabetes"
        }
    ]

    res = AnswerEvaluator.evaluate_answer(
        question=question,
        answer_text=answer,
        retrieved_sources=sources
    )

    # Missing citation should lower completeness
    assert res.citation_completeness == 0.0
    assert res.citation_correctness == 1.0  # No invalid citation indices used


def test_answer_evaluator_invalid_citation_index():
    """Verify that referencing non-existent [Source 99] is caught as an invalid citation."""
    question = "What is the dose of lisinopril?"
    answer = "The initial dose of lisinopril is 10 mg once daily [Source 99]."
    sources = [
        {
            "source_index": 1,
            "source_label": "[Source 1]",
            "text": "Initial dose is 10 mg daily for hypertension.",
            "document_id": "doc_lisinopril"
        }
    ]

    res = AnswerEvaluator.evaluate_answer(
        question=question,
        answer_text=answer,
        retrieved_sources=sources
    )

    assert res.citation_correctness == 0.0
    assert res.has_invalid_citations is True
    assert "[Source 99]" in res.invalid_citations


def test_answer_evaluator_unsupported_claim_detection():
    """Verify that unsupported clinical claims are flagged as hallucinations."""
    question = "Can metformin cure hypertension?"
    # The source says nothing about metformin curing hypertension
    answer = "Metformin has been proven to permanently cure essential hypertension [Source 1]."
    sources = [
        {
            "source_index": 1,
            "source_label": "[Source 1]",
            "text": "Metformin is prescribed for type 2 diabetes glycemic management.",
            "document_id": "doc_diabetes"
        }
    ]

    res = AnswerEvaluator.evaluate_answer(
        question=question,
        answer_text=answer,
        retrieved_sources=sources
    )

    assert res.hallucination_detected is True
    assert res.unsupported_claims_count > 0
    assert res.is_safe is False
    assert res.final_status == "FAIL"


def test_answer_evaluator_directional_and_negation_contradictions():
    """Verify that clinical contradictions are detected."""
    question = "Does smoking increase hypertension risk?"
    # Source says smoking increases BP, answer says it decreases BP
    answer = "Smoking significantly decreases arterial blood pressure and lowers cardiovascular risk [Source 1]."
    sources = [
        {
            "source_index": 1,
            "source_label": "[Source 1]",
            "text": "Tobacco smoking causes immediate elevation in blood pressure and increases cardiovascular mortality.",
            "document_id": "doc_lifestyle"
        }
    ]

    res = AnswerEvaluator.evaluate_answer(
        question=question,
        answer_text=answer,
        retrieved_sources=sources
    )

    assert res.contradictions_detected > 0 or res.hallucination_detected is True
    assert res.is_safe is False


def test_answer_evaluator_safe_refusal():
    """Verify that safe clinical refusal statements pass without spurious hallucination flags."""
    question = "What is the recommended treatment for rare extraterrestrial disease?"
    answer = (
        "Relevant medical information could not be found in the available reference documents. "
        "To prevent unsupported healthcare answers, generation was halted. "
        "Please refine your query or consult authorized clinical guidelines."
    )
    sources = []

    res = AnswerEvaluator.evaluate_answer(
        question=question,
        answer_text=answer,
        retrieved_sources=sources
    )

    assert res.hallucination_detected is False
    assert res.is_safe is True
    assert res.final_status == "PASS"
