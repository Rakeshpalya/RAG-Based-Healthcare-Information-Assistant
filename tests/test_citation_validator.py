import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.evaluation.citation_validator import CitationValidator
from backend.evaluation.grounding_evaluator import GroundingEvaluator
from backend.rag.rag_service import RAGService


def get_mock_retrieved_sources():
    return [
        {
            "source_index": 1,
            "chunk_id": "MED_CHUNK_0",
            "document_id": "DOC_CARDIO_001",
            "page_number": 1,
            "similarity_score": 0.75,
            "text": "Hypertension is a chronic medical condition defined as sustained blood pressure above 130/80 mmHg."
        },
        {
            "source_index": 2,
            "chunk_id": "MED_CHUNK_1",
            "document_id": "DOC_ENDO_001",
            "page_number": 2,
            "similarity_score": 0.55,
            "text": "Type 2 diabetes mellitus is characterized by insulin resistance and progressive beta-cell dysfunction."
        }
    ]


def test_valid_single_citation():
    """1. Test that a single valid citation [Source 1] is correctly identified and validated."""
    sources = get_mock_retrieved_sources()
    answer = "According to medical evidence [Source 1], hypertension is defined as BP over 130/80 mmHg."
    res = CitationValidator.validate_citations(answer, sources)

    assert res.is_valid is True
    assert res.has_citations is True
    assert res.citations_found == [1]
    assert res.valid_citations == [1]
    assert len(res.invalid_citations) == 0
    assert res.missing_citations is False
    assert len(res.mapped_sources) == 1
    assert res.mapped_sources[0]["chunk_id"] == "MED_CHUNK_0"
    print("[PASS] test_valid_single_citation passed.")


def test_multiple_valid_citations():
    """2. Test multiple valid citations [Source 1] and [Source 2]."""
    sources = get_mock_retrieved_sources()
    answer = "Hypertension [Source 1] often co-exists with metabolic disorders like diabetes [Source 2]."
    res = CitationValidator.validate_citations(answer, sources)

    assert res.is_valid is True
    assert res.citations_found == [1, 2]
    assert res.valid_citations == [1, 2]
    assert len(res.invalid_citations) == 0
    assert len(res.mapped_sources) == 2
    print("[PASS] test_multiple_valid_citations passed.")


def test_missing_citation_when_sources_available():
    """3. Test missing citation when informative answer is generated with available sources."""
    sources = get_mock_retrieved_sources()
    answer = "Hypertension is defined as elevated blood pressure above 130/80 mmHg without mentioning any source."
    res = CitationValidator.validate_citations(answer, sources)

    assert res.is_valid is False
    assert res.has_citations is False
    assert res.missing_citations is True
    assert "no citations were included" in res.errors[0]
    print("[PASS] test_missing_citation_when_sources_available passed.")


def test_invalid_source_number_out_of_bounds():
    """4. Test that citations to nonexistent sources (e.g., [Source 5]) are flagged as invalid."""
    sources = get_mock_retrieved_sources()  # Only sources 1 and 2 exist
    answer = "Hypertension is defined as blood pressure above 130/80 mmHg [Source 5]."
    res = CitationValidator.validate_citations(answer, sources)

    assert res.is_valid is False
    assert res.citations_found == [5]
    assert 5 in res.invalid_citations
    assert len(res.valid_citations) == 0
    assert len(res.errors) > 0
    print("[PASS] test_invalid_source_number_out_of_bounds passed.")


def test_citation_referencing_unretrieved_source():
    """5. Test mix of valid and unretrieved citation [Source 1] and [Source 99]."""
    sources = get_mock_retrieved_sources()
    answer = "Hypertension [Source 1] can also lead to liver complications [Source 99]."
    res = CitationValidator.validate_citations(answer, sources)

    assert res.is_valid is False
    assert res.citations_found == [1, 99]
    assert res.valid_citations == [1]
    assert res.invalid_citations == [99]
    print("[PASS] test_citation_referencing_unretrieved_source passed.")


def test_duplicate_citations():
    """6. Test detection of duplicate citations [Source 1] ... [Source 1]."""
    sources = get_mock_retrieved_sources()
    answer = "Hypertension [Source 1] requires monitoring. Systolic pressure [Source 1] must be controlled."
    res = CitationValidator.validate_citations(answer, sources)

    assert res.is_valid is True
    assert res.citations_found == [1, 1]
    assert res.duplicate_citations == [1]
    assert res.valid_citations == [1]
    print("[PASS] test_duplicate_citations passed.")


def test_answer_without_citations_safe_fallback():
    """7. Test safe refusal when context is missing does not trigger missing_citations error."""
    sources = []
    answer = "The available documents do not contain enough information to answer this question."
    res = CitationValidator.validate_citations(answer, sources)

    # When no sources were provided and the model safely declined, it is valid
    assert res.is_valid is True
    assert res.missing_citations is False
    assert len(res.citations_found) == 0
    print("[PASS] test_answer_without_citations_safe_fallback passed.")


def test_citation_metadata_preservation():
    """8. Test that citation metadata (chunk_id, doc_id, similarity) is accurately preserved."""
    sources = get_mock_retrieved_sources()
    answer = "Evidence [Source 1] demonstrates chronic hypertension causes vascular changes."
    res = CitationValidator.validate_citations(answer, sources)

    assert len(res.mapped_sources) == 1
    mapping = res.mapped_sources[0]
    assert mapping["citation_index"] == 1
    assert mapping["chunk_id"] == "MED_CHUNK_0"
    assert mapping["document_id"] == "DOC_CARDIO_001"
    assert mapping["similarity_score"] == 0.75
    print("[PASS] test_citation_metadata_preservation passed.")


def test_malformed_llm_outputs_resilience():
    """9. Test that validator never crashes on None, empty, brackets, symbols, or malformed strings."""
    sources = get_mock_retrieved_sources()

    # None
    res_none = CitationValidator.validate_citations(None, sources)
    assert res_none.is_valid is False
    assert len(res_none.citations_found) == 0

    # Empty string
    res_empty = CitationValidator.validate_citations("", sources)
    assert res_empty.is_valid is False

    # Broken bracket expressions
    malformed_text = "Check [Source ], [Source ABC], [Source -1], [Source 99999999999999999999], [Source: 1]."
    res_malformed = CitationValidator.validate_citations(malformed_text, sources)
    # [Source: 1] matches 1 which is valid; huge number is extracted and marked invalid
    assert isinstance(res_malformed.is_valid, bool)
    assert 1 in res_malformed.valid_citations
    print("[PASS] test_malformed_llm_outputs_resilience passed.")


def test_grounding_evaluator_integration():
    """10. Test GroundingEvaluator deterministic checks for both grounded answer and safe fallback."""
    sources = get_mock_retrieved_sources()

    # Grounded answer
    grounded_ans = "Elevated blood pressure above 130/80 mmHg constitutes arterial hypertension [Source 1]."
    g_res = GroundingEvaluator.evaluate_grounding(
        question="What is hypertension?",
        answer=grounded_ans,
        retrieval_status="success",
        retrieved_sources=sources
    )
    assert g_res.is_grounded is True
    assert g_res.answer_present is True
    assert g_res.context_present is True
    assert g_res.lexical_overlap_ratio > 0.0

    # Unsupported fallback answer
    fallback_ans = "The available documents do not contain enough information to answer this question."
    f_res = GroundingEvaluator.evaluate_grounding(
        question="What is sickle cell disease?",
        answer=fallback_ans,
        retrieval_status="no_relevant_context",
        retrieved_sources=[]
    )
    assert f_res.is_grounded is True
    assert f_res.unsupported_fallback_verified is True
    print("[PASS] test_grounding_evaluator_integration passed.")


def test_extract_citations_single():
    """11. Test extracting a single citation [Source 1]."""
    text = "Hypertension is defined as elevated blood pressure [Source 1]."
    cits = CitationValidator.extract_citations(text)
    assert cits == [1]
    print("[PASS] test_extract_citations_single passed.")


def test_extract_citations_grouped_two():
    """12. Test extracting grouped citations [Source 1, Source 2]."""
    text = "Risk factors and lifestyle measures are discussed in clinical guidelines [Source 1, Source 2]."
    cits = CitationValidator.extract_citations(text)
    assert cits == [1, 2]
    print("[PASS] test_extract_citations_grouped_two passed.")


def test_extract_citations_grouped_three():
    """13. Test extracting grouped citations [Source 1, Source 2, Source 3]."""
    text = "Hypertension risk factors include X and Y [Source 1, Source 2, Source 3]."
    cits = CitationValidator.extract_citations(text)
    assert cits == [1, 2, 3]
    print("[PASS] test_extract_citations_grouped_three passed.")


def test_extract_citations_grouped_nonconsecutive():
    """14. Test extracting grouped non-consecutive citations [Source 1, Source 3]."""
    text = "Clinical evidence links sodium intake and genetics to vascular resistance [Source 1, Source 3]."
    cits = CitationValidator.extract_citations(text)
    assert cits == [1, 3]
    print("[PASS] test_extract_citations_grouped_nonconsecutive passed.")


def test_extract_citations_bare_digits():
    """15. Test extracting bare grouped citations [1, 2, 3]."""
    text = "Evidence summary [1, 2, 3]."
    cits = CitationValidator.extract_citations(text)
    assert cits == [1, 2, 3]
    print("[PASS] test_extract_citations_bare_digits passed.")


def test_invalid_citation_source_99_and_stripping():
    """16. Test validation and stripping of invalid citations such as [Source 99]."""
    sources = [
        {"source_index": 1, "chunk_id": "c1", "document_id": "doc1", "text": "Passage 1"},
        {"source_index": 2, "chunk_id": "c2", "document_id": "doc1", "text": "Passage 2"},
    ]
    # Single invalid citation [Source 99]
    ans_single = "Claim with invalid source [Source 99]."
    res_single = CitationValidator.validate_citations(ans_single, sources)
    assert res_single.is_valid is False
    assert res_single.invalid_citations == [99]
    stripped_single = CitationValidator.strip_invalid_citations(ans_single, res_single.invalid_citations)
    assert "[Source 99]" not in stripped_single

    # Grouped citation with valid and invalid: [Source 1, Source 99]
    ans_grouped = "Claim supported by real and hallucinated sources [Source 1, Source 99]."
    res_grouped = CitationValidator.validate_citations(ans_grouped, sources)
    assert res_grouped.is_valid is False
    assert res_grouped.valid_citations == [1]
    assert res_grouped.invalid_citations == [99]
    stripped_grouped = CitationValidator.strip_invalid_citations(ans_grouped, res_grouped.invalid_citations)
    assert "[Source 99]" not in stripped_grouped
    assert "[Source 1]" in stripped_grouped
    print("[PASS] test_invalid_citation_source_99_and_stripping passed.")


def test_source_card_selection_grouped_citations():
    """
    17. Test that when an answer contains [Source 1, Source 2, Source 3],
    the pipeline returns 3 corresponding source cards with matching source indices.
    """
    retrieved_sources = [
        {"source_index": 1, "source_label": "[Source 1]", "chunk_id": "c1", "document_id": "doc1"},
        {"source_index": 2, "source_label": "[Source 2]", "chunk_id": "c2", "document_id": "doc1"},
        {"source_index": 3, "source_label": "[Source 3]", "chunk_id": "c3", "document_id": "doc1"},
    ]
    answer = "Hypertension risk factors include X and Y [Source 1, Source 2, Source 3]."
    val_res = CitationValidator.validate_citations(answer, retrieved_sources)
    assert val_res.is_valid is True
    assert val_res.citations_found == [1, 2, 3]
    assert val_res.valid_citations == [1, 2, 3]

    selected_sources = RAGService.select_cited_sources(
        retrieved_sources=retrieved_sources,
        citations_found=val_res.citations_found,
        valid_citations=val_res.valid_citations,
    )

    assert len(selected_sources) == 3
    assert selected_sources[0]["source_index"] == 1
    assert selected_sources[0]["source_label"] == "[Source 1]"
    assert selected_sources[1]["source_index"] == 2
    assert selected_sources[1]["source_label"] == "[Source 2]"
    assert selected_sources[2]["source_index"] == 3
    assert selected_sources[2]["source_label"] == "[Source 3]"
    print("[PASS] test_source_card_selection_grouped_citations passed.")


def test_source_card_selection_nonconsecutive_preserves_indices():
    """
    18. Test that when an answer contains [Source 1, Source 3],
    Source 1 and Source 3 are returned without renaming Source 3 to Source 2.
    """
    retrieved_sources = [
        {"source_index": 1, "source_label": "[Source 1]", "chunk_id": "c1", "document_id": "doc1"},
        {"source_index": 2, "source_label": "[Source 2]", "chunk_id": "c2", "document_id": "doc1"},
        {"source_index": 3, "source_label": "[Source 3]", "chunk_id": "c3", "document_id": "doc1"},
    ]
    answer = "Hypertension lifestyle factors [Source 1, Source 3]."
    val_res = CitationValidator.validate_citations(answer, retrieved_sources)
    assert val_res.is_valid is True
    assert val_res.citations_found == [1, 3]
    assert val_res.valid_citations == [1, 3]

    selected_sources = RAGService.select_cited_sources(
        retrieved_sources=retrieved_sources,
        citations_found=val_res.citations_found,
        valid_citations=val_res.valid_citations,
    )

    assert len(selected_sources) == 2
    assert selected_sources[0]["source_index"] == 1
    assert selected_sources[0]["source_label"] == "[Source 1]"
    assert selected_sources[1]["source_index"] == 3
    assert selected_sources[1]["source_label"] == "[Source 3]"
    print("[PASS] test_source_card_selection_nonconsecutive_preserves_indices passed.")


if __name__ == "__main__":
    print("Running CitationValidator and GroundingEvaluator Tests...")
    test_valid_single_citation()
    test_multiple_valid_citations()
    test_missing_citation_when_sources_available()
    test_invalid_source_number_out_of_bounds()
    test_citation_referencing_unretrieved_source()
    test_duplicate_citations()
    test_answer_without_citations_safe_fallback()
    test_citation_metadata_preservation()
    test_malformed_llm_outputs_resilience()
    test_grounding_evaluator_integration()
    test_extract_citations_single()
    test_extract_citations_grouped_two()
    test_extract_citations_grouped_three()
    test_extract_citations_grouped_nonconsecutive()
    test_extract_citations_bare_digits()
    test_invalid_citation_source_99_and_stripping()
    test_source_card_selection_grouped_citations()
    test_source_card_selection_nonconsecutive_preserves_indices()
    print("\n[SUCCESS] All 18 CitationValidator and GroundingEvaluator tests passed!")
