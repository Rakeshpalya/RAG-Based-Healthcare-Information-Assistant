"""
Phase 3.3 — Citation Enforcement Test Suite

Validates:
1. Valid citation checking against source content (Case A).
2. Out-of-bounds / nonexistent citation index rejection (Case B).
3. Wrong source citation rejection (Case C).
4. Correct source citation confirmation (Case D).
5. Hallucinated medical claim rejection (Case E).
6. Multi-source citation support (Case F).
7. Missing citation detection on factual claims (Case G).
8. Unsupported claim defense after invalid citation stripping.
9. Citation coverage metric calculation.
10. Multi-claim answers with mixed support (selective pruning).
11. Bracket format compatibility ([Source 1], [1], [Source 1, Source 2], [Source 1] [Source 2]).
12. Backward compatibility with legacy CitationValidator behavior.
13. RAGService end-to-end integration with mock LLM hallucination rejection.
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.evaluation.citation_validator import (
    CitationValidator,
    CitationValidationResult,
    ExtractedClaim
)
from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import VectorStoreService
from backend.services.embedding_service import EmbeddingService


@pytest.fixture
def mock_cardio_and_diabetes_sources():
    return [
        {
            "source_index": 1,
            "chunk_id": "CHUNK_HTN_01",
            "document_id": "DOC_CARDIO_001",
            "page_number": 1,
            "similarity_score": 0.82,
            "text": "Hypertension is defined as persistent elevation of blood pressure. Lifestyle recommendations include physical activity."
        },
        {
            "source_index": 2,
            "chunk_id": "CHUNK_DM_01",
            "document_id": "DOC_ENDO_001",
            "page_number": 3,
            "similarity_score": 0.78,
            "text": "Diabetes management includes blood glucose monitoring and dietary management."
        }
    ]


def test_1_valid_citation(mock_cardio_and_diabetes_sources):
    """CASE A — Valid citation: Claim is directly supported by cited Source 1."""
    answer = "Hypertension involves persistent elevation of blood pressure. [Source 1]"
    res = CitationValidator.validate_grounded_citations(answer, mock_cardio_and_diabetes_sources)

    assert res.is_valid is True
    assert res.claims_checked == 1
    assert res.claims_supported == 1
    assert res.claims_unsupported == 0
    assert res.citation_coverage == 1.0
    assert 1 in res.valid_citations
    assert len(res.invalid_citations) == 0


def test_2_invalid_source_number(mock_cardio_and_diabetes_sources):
    """CASE B — Invalid source number: Citation references nonexistent [Source 99]."""
    answer = "Hypertension involves elevated blood pressure. [Source 99]"
    res = CitationValidator.validate_grounded_citations(answer, mock_cardio_and_diabetes_sources)

    assert res.is_valid is False
    assert 99 in res.invalid_citations
    assert res.claims_unsupported >= 1
    assert res.citation_coverage == 0.0


def test_3_wrong_source_citation(mock_cardio_and_diabetes_sources):
    """CASE C — Wrong source: Glucose monitoring claim cites Source 1 (Hypertension doc).
    Expected: UNSUPPORTED for Source 1.
    """
    answer = "Blood glucose monitoring is part of diabetes management. [Source 1]"
    res = CitationValidator.validate_grounded_citations(answer, mock_cardio_and_diabetes_sources)

    assert res.is_valid is False
    assert res.claims_checked == 1
    assert res.claims_supported == 0
    assert res.claims_unsupported == 1
    assert res.citation_coverage == 0.0
    assert any("unsupported" in err.lower() for err in res.errors)


def test_4_correct_source_citation(mock_cardio_and_diabetes_sources):
    """CASE D — Correct source: Glucose monitoring claim correctly cites Source 2 (Diabetes doc).
    Expected: SUPPORTED.
    """
    answer = "Blood glucose monitoring is part of diabetes management. [Source 2]"
    res = CitationValidator.validate_grounded_citations(answer, mock_cardio_and_diabetes_sources)

    assert res.is_valid is True
    assert res.claims_checked == 1
    assert res.claims_supported == 1
    assert res.claims_unsupported == 0
    assert res.citation_coverage == 1.0


def test_5_hallucinated_claim(mock_cardio_and_diabetes_sources):
    """CASE E — Hallucinated claim:
    Source discusses hypertension lifestyle, but answer claims Metformin is recommended.
    Expected: UNSUPPORTED.
    """
    answer = "Metformin is recommended for hypertension. [Source 1]"
    res = CitationValidator.validate_grounded_citations(answer, mock_cardio_and_diabetes_sources)

    assert res.is_valid is False
    assert res.claims_checked == 1
    assert res.claims_supported == 0
    assert res.claims_unsupported == 1
    assert res.citation_coverage == 0.0
    assert "Metformin is recommended for hypertension" in res.unsupported_claims[0]


def test_6_multiple_source_citation(mock_cardio_and_diabetes_sources):
    """CASE F — Multiple sources:
    Claim references both physical activity (Source 1) and glucose monitoring (Source 2).
    Expected: SUPPORTED when both sources together cover the claim.
    """
    answer = (
        "Physical activity is discussed for hypertension, "
        "while glucose monitoring is discussed for diabetes. [Source 1, Source 2]"
    )
    res = CitationValidator.validate_grounded_citations(answer, mock_cardio_and_diabetes_sources)

    assert res.is_valid is True
    assert res.claims_checked == 1
    assert res.claims_supported == 1
    assert res.claims_unsupported == 0
    assert res.citation_coverage == 1.0


def test_7_missing_citation(mock_cardio_and_diabetes_sources):
    """CASE G — Missing citation:
    Answer contains factual statement with available sources but no citations.
    Expected: Missing citation flagged, claims_unsupported > 0.
    """
    answer = "Hypertension may require lifestyle changes."
    res = CitationValidator.validate_grounded_citations(answer, mock_cardio_and_diabetes_sources)

    assert res.is_valid is False
    assert res.missing_citations is True
    assert res.claims_unsupported >= 1


def test_8_unsupported_claim_defense_after_invalid_stripping(mock_cardio_and_diabetes_sources):
    """Validates that removing an invalid citation tag does NOT leave the unsupported claim in the answer.
    Instead, prune_unsupported_claims removes the claim or converts to safe fallback.
    """
    answer = "Metformin is recommended for hypertension. [Source 99]"
    res = CitationValidator.validate_grounded_citations(answer, mock_cardio_and_diabetes_sources)

    pruned = CitationValidator.prune_unsupported_claims(answer, res)
    # The unsupported claim must NOT survive
    assert "Metformin is recommended" not in pruned
    assert "[Source 99]" not in pruned
    # Safe fallback message returned
    assert "Relevant medical information could not be found" in pruned


def test_9_citation_coverage_calculation(mock_cardio_and_diabetes_sources):
    """Validates precise citation coverage calculation for answers with multiple statements."""
    answer = (
        "Hypertension is defined as persistent elevation of blood pressure [Source 1]. "
        "Blood glucose monitoring is part of diabetes management [Source 2]."
    )
    res = CitationValidator.validate_grounded_citations(answer, mock_cardio_and_diabetes_sources)

    assert res.claims_checked == 2
    assert res.claims_supported == 2
    assert res.claims_unsupported == 0
    assert res.citation_coverage == 1.0


def test_10_multiple_claims_with_mixed_support(mock_cardio_and_diabetes_sources):
    """Validates that in an answer with mixed claims:
    - Supported claim (Source 1) is retained.
    - Unsupported claim (hallucinated Metformin) is pruned.
    """
    answer = (
        "Hypertension is defined as persistent elevation of blood pressure [Source 1].\n"
        "Metformin is recommended for hypertension [Source 1]."
    )
    res = CitationValidator.validate_grounded_citations(answer, mock_cardio_and_diabetes_sources)

    assert res.claims_checked == 2
    assert res.claims_supported == 1
    assert res.claims_unsupported == 1
    assert res.citation_coverage == 0.5
    assert res.is_valid is False

    pruned = CitationValidator.prune_unsupported_claims(answer, res)
    assert "Hypertension is defined as persistent elevation of blood pressure [Source 1]" in pruned
    assert "Metformin" not in pruned


def test_11_existing_citation_format_compatibility():
    """Validates support for diverse bracket formats: [Source 1], [1], [Source 1, Source 2], [Source 1] [Source 2]."""
    text = (
        "Statement one [Source 1]. "
        "Statement two [2]. "
        "Statement three [Source 1, Source 2]. "
        "Statement four [Source 1] [Source 2]."
    )
    extracted = CitationValidator.extract_citations(text)
    assert extracted == [1, 2, 1, 2, 1, 2]


def test_12_legacy_citation_validator_behavior_preserved(mock_cardio_and_diabetes_sources):
    """Verifies that calling validate_citations() without check_claim_support maintains
    100% backward compatibility with legacy syntactic citation tests.
    """
    answer = "Hypertension [Source 1] requires monitoring."
    res = CitationValidator.validate_citations(answer, mock_cardio_and_diabetes_sources)

    assert res.is_valid is True
    assert res.citations_found == [1]
    assert res.valid_citations == [1]
    assert len(res.invalid_citations) == 0


def test_13_rag_service_rejects_hallucinated_answer_with_mock_gemini():
    """End-to-end integration: Verifies RAGService detects unsupported LLM answer
    and safely halts generation, returning the insufficient-evidence response.
    """
    mock_vs = MagicMock(spec=VectorStoreService)
    # Mock retrieval returning hypertension passage
    mock_vs.search.return_value = [
        {
            "chunk_id": "MED_CHUNK_1",
            "document_id": "clinical_htn.pdf",
            "page_number": 1,
            "similarity_score": 0.85,
            "text": "Hypertension is characterized by sustained elevation of blood pressure.",
            "user_id": 1,
            "metadata": {"filename": "clinical_htn.pdf"}
        }
    ]

    rag_service = RAGService(vector_store=mock_vs)

    # Mock Gemini generating a completely unsupported / hallucinated answer
    mock_gemini = MagicMock()
    mock_gemini.generate_answer.return_value = {
        "answer": "Chemotherapy is the first-line curative treatment for hypertension [Source 1].",
        "model": "mock-gemini-3.5-flash",
        "disclaimer": "Medical disclaimer",
        "generation_time_ms": 100.0,
        "status": "success"
    }

    with patch("backend.services.embedding_service.EmbeddingService.embed_query", side_effect=EmbeddingService.embed_query):
        result = rag_service.generate_rag_answer(
            question="What is the treatment for hypertension?",
            gemini_service=mock_gemini
        )

    # Because "Chemotherapy is the first-line curative treatment" is not supported by the HTN chunk,
    # citation enforcement halts generation and returns the safe fallback!
    assert "Relevant medical information could not be found" in result["answer"]
    assert "To prevent unsupported healthcare answers, generation was halted" in result["answer"]
    assert result["timings"]["claims_unsupported"] >= 1
    assert result["timings"]["citation_coverage"] == 0.0
