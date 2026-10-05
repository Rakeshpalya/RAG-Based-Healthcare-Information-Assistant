"""
Phase 4 Milestone 4.3 — Citation Evaluation Tests.

Evaluates:
1. Citation presence detection across diverse bracket formats:
   - Single: [Source 1], [1]
   - Grouped: [Source 1, Source 2], [1, 2]
2. Citation correctness & out-of-bounds rejection (e.g., [Source 99]).
3. Citation source alignment (semantic grounding between cited source and claim).
4. Invalid citation stripping & sanitization.
5. Missing citation detection on ungrounded clinical claims.
6. Citation-to-source consistency across valid, invalid, and intentionally tampered citations.
7. Pruning of unsupported claims to preserve clinical safety.
"""

import pytest
from backend.evaluation.citation_validator import (
    CitationValidator,
    CitationValidationResult,
    ExtractedClaim,
)


@pytest.fixture
def sample_sources():
    return [
        {
            "source_index": 1,
            "chunk_id": "chunk_metformin_01",
            "document_id": "DOC_DIABETES",
            "text": "Metformin decreases hepatic glucose production and improves peripheral insulin sensitivity in type 2 diabetes.",
        },
        {
            "source_index": 2,
            "chunk_id": "chunk_hypertension_01",
            "document_id": "DOC_CARDIOLOGY",
            "text": "Lisinopril is an ACE inhibitor indicated for the treatment of essential hypertension and heart failure.",
        },
        {
            "source_index": 3,
            "chunk_id": "chunk_asthma_01",
            "document_id": "DOC_RESPIRATORY",
            "text": "Albuterol is a short-acting beta-2 agonist used for rapid relief of acute bronchospasm in asthma.",
        },
    ]


def test_01_citation_presence_and_extraction():
    """Verify robust extraction of citation indices across standard and grouped formats."""
    text_single = "Metformin reduces hepatic glucose production [Source 1]."
    cits_single = CitationValidator.extract_citations(text_single)
    assert cits_single == [1]

    text_grouped = "Combination therapy includes ACE inhibitors and beta agonists [Source 2, Source 3]."
    cits_grouped = CitationValidator.extract_citations(text_grouped)
    assert cits_grouped == [2, 3]

    text_numeric = "Hypertension management guidelines [2]."
    cits_numeric = CitationValidator.extract_citations(text_numeric)
    assert cits_numeric == [2]

    text_no_citations = "This text has no citations at all."
    cits_none = CitationValidator.extract_citations(text_no_citations)
    assert cits_none == []


def test_02_citation_correctness_and_bounds_validation(sample_sources):
    """Verify that citations within valid bounds are accepted and out-of-bounds are rejected."""
    # Valid citations (1 and 2 exist in sample_sources)
    valid_text = "Metformin lowers glucose [Source 1] and Lisinopril treats hypertension [Source 2]."
    res_valid = CitationValidator.validate_citations(valid_text, sample_sources)
    assert res_valid.is_valid is True
    assert set(res_valid.valid_citations) == {1, 2}
    assert res_valid.invalid_citations == []

    # Invalid citation (Source 99 does not exist)
    invalid_text = "Metformin lowers glucose [Source 1] but cure-all exists [Source 99]."
    res_invalid = CitationValidator.validate_citations(invalid_text, sample_sources)
    assert res_invalid.is_valid is False
    assert 99 in res_invalid.invalid_citations
    assert 1 in res_invalid.valid_citations


def test_03_invalid_citation_stripping():
    """Verify stripping invalid citations preserves remaining valid citations in grouped brackets."""
    text = "Clinical evidence indicates benefits [Source 1, Source 99]."
    cleaned = CitationValidator.strip_invalid_citations(text, invalid_citations=[99])
    assert "[Source 1]" in cleaned
    assert "99" not in cleaned

    text_pure_invalid = "Unverified claim [Source 888]."
    cleaned_pure = CitationValidator.strip_invalid_citations(text_pure_invalid, invalid_citations=[888])
    assert "[Source 888]" not in cleaned_pure


def test_04_citation_source_alignment_semantic_support(sample_sources):
    """Verify semantic claim support check confirms grounded claim matches cited source."""
    source_map = {s["source_index"]: s for s in sample_sources}

    # Grounded claim citing Source 1
    grounded_claim = ExtractedClaim(
        claim_text="Metformin reduces hepatic glucose production and improves insulin sensitivity",
        raw_sentence="Metformin reduces hepatic glucose production and improves insulin sensitivity [Source 1].",
        cited_source_indices=[1],
        has_citations=True
    )
    is_supported = CitationValidator.check_claim_support(grounded_claim, source_map)
    assert is_supported is True
    assert grounded_claim.is_supported is True

    # Hallucinated claim citing Source 1 (contradictory assertion)
    contradictory_claim = ExtractedClaim(
        claim_text="Metformin increases hepatic glucose production and causes immediate kidney failure",
        raw_sentence="Metformin increases hepatic glucose production [Source 1].",
        cited_source_indices=[1],
        has_citations=True
    )
    is_contradicted = CitationValidator.check_claim_support(contradictory_claim, source_map)
    assert is_contradicted is False
    assert contradictory_claim.is_supported is False


def test_05_missing_citation_detection(sample_sources):
    """Verify detection of factual medical assertions lacking required citations."""
    answer_mixed = (
        "Metformin decreases hepatic glucose production [Source 1]. "
        "Aspirin should be taken at 1000mg three times daily without any supervision."
    )
    res = CitationValidator.validate_citations(
        answer_text=answer_mixed,
        retrieved_sources=sample_sources,
        check_claim_support=True
    )
    assert res.claims_checked >= 2
    # The uncited aspirin sentence is detected as unsupported
    assert res.claims_unsupported >= 1
    assert any("Aspirin" in c or "aspirin" in c.lower() for c in res.unsupported_claims)


def test_06_citation_tampering_and_substitution(sample_sources):
    """Verify adversarial citation substitution (citing unrelated source for a claim) is caught."""
    source_map = {s["source_index"]: s for s in sample_sources}

    # Claim about asthma bronchospasm citing Source 2 (which is Lisinopril hypertension)
    tampered_claim = ExtractedClaim(
        claim_text="Albuterol provides rapid relief for acute bronchospasm in asthma",
        raw_sentence="Albuterol provides rapid relief for acute bronchospasm in asthma [Source 2].",
        cited_source_indices=[2],  # Wrong source: Source 2 discusses Lisinopril
        has_citations=True
    )
    supported = CitationValidator.check_claim_support(tampered_claim, source_map)
    assert supported is False
    assert tampered_claim.is_supported is False


def test_07_pruning_unsupported_claims_maintains_safe_envelope(sample_sources):
    """Verify unsupported claims are pruned while valid grounded claims are preserved."""
    answer_text = (
        "Metformin decreases hepatic glucose production [Source 1]. "
        "Quantum crystals eliminate all diabetes symptoms instantly [Source 99]."
    )
    validation = CitationValidator.validate_citations(
        answer_text=answer_text,
        retrieved_sources=sample_sources,
        check_claim_support=True
    )
    pruned = CitationValidator.prune_unsupported_claims(answer_text, validation)
    assert "Metformin decreases hepatic glucose production" in pruned
    assert "Quantum crystals" not in pruned
    assert "Source 99" not in pruned
