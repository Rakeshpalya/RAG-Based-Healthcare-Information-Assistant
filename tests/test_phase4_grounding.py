"""
Phase 4 Milestone 4.4 — Grounding and Hallucination Evaluation Tests.

Evaluates:
1. Supported claim rate computation on fully grounded responses using HallucinationGuard.guard_answer.
2. Unsupported claim rate computation on partially grounded and ungrounded responses.
3. Hallucination detection across key clinical categories:
   - Dosage hallucination (e.g., 5000mg vs 500mg)
   - Medication substitution / unmentioned medication hallucination
   - Directional contradiction (e.g., increases blood pressure vs lowers blood pressure)
   - Negation contradiction (e.g., indicated vs strictly contraindicated)
4. Source alignment and semantic support scoring.
5. Partial-support handling: selective preservation of supported sentences.
6. Conflicting-source handling: detects contradiction and avoids fabricated synthesis.
7. Safe fallback enforcement: unsupported claims are never silently accepted.
"""

import pytest
from backend.evaluation.hallucination_guard import (
    HallucinationGuard,
    HallucinationType,
    ClaimGuardDetail,
)
from backend.evaluation.citation_validator import (
    CitationValidator,
    ExtractedClaim,
)
from backend.evaluation.grounding_evaluator import GroundingEvaluator


@pytest.fixture
def clinical_sources():
    return [
        {
            "source_index": 1,
            "chunk_id": "chunk_metformin",
            "document_id": "DOC_ENDO_01",
            "text": "Metformin starting dose is 500mg once daily with meals. It lowers fasting plasma glucose.",
        },
        {
            "source_index": 2,
            "chunk_id": "chunk_aspirin_ulcer",
            "document_id": "DOC_GI_01",
            "text": "Aspirin and NSAIDs are strictly contraindicated in active peptic ulcer disease due to gastrointestinal bleeding risk.",
        },
        {
            "source_index": 3,
            "chunk_id": "chunk_hypertension",
            "document_id": "DOC_CARDIO_01",
            "text": "Amlodipine lowers arterial blood pressure through peripheral vasodilation.",
        },
    ]


def test_01_supported_claim_rate_on_grounded_answer(clinical_sources):
    """Verify that a fully grounded answer achieves 100% supported claim rate."""
    answer_text = (
        "Metformin starting dose is 500mg once daily with meals [Source 1]. "
        "Amlodipine lowers arterial blood pressure [Source 3]."
    )
    result = HallucinationGuard.guard_answer(answer_text, clinical_sources)
    assert result.total_claims == 2
    assert result.supported_claims == 2
    assert result.unsupported_claims == 0
    assert result.is_safe is True
    supported_rate = result.supported_claims / result.total_claims
    assert supported_rate == 1.0


def test_02_unsupported_claim_rate_detection(clinical_sources):
    """Verify detection of ungrounded assertions and accurate unsupported claim rate."""
    answer_text = (
        "Metformin starting dose is 500mg once daily [Source 1]. "
        "Metformin also permanently cures chronic viral hepatitis within 48 hours."
    )
    result = HallucinationGuard.guard_answer(answer_text, clinical_sources)
    assert result.total_claims >= 2
    assert result.unsupported_claims >= 1
    unsupported_rate = result.unsupported_claims / result.total_claims
    assert unsupported_rate > 0.0
    assert result.is_safe is False


def test_03_dosage_hallucination_detection(clinical_sources):
    """Verify that fabricated or exaggerated medication dosages are detected as hallucinations."""
    source_map = {s["source_index"]: s for s in clinical_sources}

    # Source 1 specifies 500mg; claim fabricates 5000mg
    dosage_claim = ExtractedClaim(
        claim_text="Metformin starting dose is 5000mg once daily with meals",
        raw_sentence="Metformin starting dose is 5000mg once daily with meals [Source 1].",
        cited_source_indices=[1],
        has_citations=True
    )
    guard_detail = HallucinationGuard.inspect_claim(dosage_claim, source_map)
    assert guard_detail.is_supported is False
    assert guard_detail.hallucination_type == HallucinationType.DOSAGE_HALLUCINATION


def test_04_directional_and_negation_contradiction_detection(clinical_sources):
    """Verify detection of inverted clinical actions (e.g. increase vs decrease, safe vs contraindicated)."""
    source_map = {s["source_index"]: s for s in clinical_sources}

    # Directional contradiction: Source says amlodipine LOWERS blood pressure; claim says it INCREASES it
    directional_claim = ExtractedClaim(
        claim_text="Amlodipine increases arterial blood pressure",
        raw_sentence="Amlodipine increases arterial blood pressure [Source 3].",
        cited_source_indices=[3],
        has_citations=True
    )
    dir_detail = HallucinationGuard.inspect_claim(directional_claim, source_map)
    assert dir_detail.is_supported is False
    assert dir_detail.hallucination_type in (
        HallucinationType.DIRECTIONAL_CONTRADICTION,
        HallucinationType.NEGATION_CONTRADICTION
    )

    # Negation contradiction: Source says aspirin is CONTRAINDICATED; claim says it is RECOMMENDED
    negation_claim = ExtractedClaim(
        claim_text="Aspirin is safe and recommended in active peptic ulcer disease",
        raw_sentence="Aspirin is safe and recommended in active peptic ulcer disease [Source 2].",
        cited_source_indices=[2],
        has_citations=True
    )
    neg_detail = HallucinationGuard.inspect_claim(negation_claim, source_map)
    assert neg_detail.is_supported is False
    assert neg_detail.hallucination_type == HallucinationType.NEGATION_CONTRADICTION


def test_05_partial_support_handling_preserves_supported_facts(clinical_sources):
    """Verify that an answer with mixed supported/unsupported claims prunes the hallucination."""
    answer_text = (
        "Metformin starting dose is 500mg once daily with meals [Source 1]. "
        "Aspirin is recommended for active bleeding peptic ulcers [Source 2]."
    )
    result = HallucinationGuard.guard_answer(answer_text, clinical_sources)
    assert result.total_claims >= 2
    assert result.contradictions_detected >= 1

    # Sanitized answer must retain the true claim and purge the dangerous contradiction
    assert "Metformin starting dose is 500mg" in result.sanitized_answer
    assert "Aspirin is recommended for active bleeding peptic ulcers" not in result.sanitized_answer


def test_06_conflicting_source_handling():
    """Verify that conflicting sources trigger ungrounded detection rather than fabricating agreement."""
    conflicting_sources = [
        {"source_index": 1, "text": "Metformin is indicated for type 2 diabetes management."},
        {"source_index": 2, "text": "Metformin is strictly contraindicated in patients with acute lactic acidosis."},
    ]
    # An answer boldly claiming metformin is safe in acute lactic acidosis
    assertive_claim = ExtractedClaim(
        claim_text="Metformin is completely safe and indicated in acute lactic acidosis",
        raw_sentence="Metformin is completely safe and indicated in acute lactic acidosis [Source 2].",
        cited_source_indices=[2],
        has_citations=True
    )
    source_map = {s["source_index"]: s for s in conflicting_sources}
    detail = HallucinationGuard.inspect_claim(assertive_claim, source_map)
    # Contradiction in source 2 must be flagged
    assert detail.is_supported is False
    assert detail.hallucination_type == HallucinationType.NEGATION_CONTRADICTION


def test_07_unsupported_medical_claims_never_silently_accepted():
    """Verify that purely fabricated answers trigger safe insufficient-evidence fallback."""
    fake_sources = [{"source_index": 1, "text": "Routine checkup notes for blood pressure."}]
    unsupported_answer = "Drinking silver nanoparticles cures type 1 diabetes and regrows pancreatic beta cells [Source 1]."

    # Citation check with claim support verification
    citation_res = CitationValidator.validate_citations(
        unsupported_answer,
        fake_sources,
        check_claim_support=True
    )
    assert citation_res.claims_unsupported >= 1
    assert citation_res.claims_supported == 0

    sanitized = CitationValidator.prune_unsupported_claims(
        unsupported_answer,
        citation_res
    )
    assert "Relevant medical information could not be found" in sanitized
    assert "silver nanoparticles" not in sanitized
