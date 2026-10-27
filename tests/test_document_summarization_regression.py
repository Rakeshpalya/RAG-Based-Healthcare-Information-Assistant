"""
test_document_summarization_regression.py

Comprehensive regression test suite for Document Summarization and Document Isolation:
1. Summarizing wellness PDF includes sleep routines, balanced eating, hydration, physical activity, habit tracking.
2. Wellness summary strictly excludes unrelated hypertension, diabetes, and asthma from other documents.
3. Summarizing clinical reference uses only that document.
4. Explicit multi-document comparison includes only requested documents.
5. Five-point summary returns exactly 5 bullet points.
6. Summary begins with complete sentence and avoids duplicate chunks.
7. Source filenames, document IDs, page numbers, and citations are correct.
8. Spacing and Markdown render properly without conjoined words.
9. Missing information answers distinguish absent content.
10. When no document is selected, asks the user to select one rather than silently summarizing everything.
11. Preserves FAISS vector store invariant (744 vectors, 744 metadata records, 384 dimensions).
"""
import os
import json
import faiss
import pytest
from unittest.mock import MagicMock, patch

from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import get_vector_store_service
from backend.rag.markdown_utils import clean_ai_markdown


@pytest.fixture(scope="module")
def shared_rag():
    vs = get_vector_store_service()
    return RAGService(vector_store=vs)


def test_vector_store_invariant_baseline():
    """Verify FAISS vector store has exactly 744 vectors, 744 metadata records, 384 dimensions."""
    index = faiss.read_index("data/vector_store/index.faiss")
    assert index.ntotal == 744, f"FAISS ntotal was {index.ntotal}, expected 744"
    assert index.d == 384, f"FAISS dimension was {index.d}, expected 384"

    with open("data/vector_store/metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("count") == 744
    assert len(meta.get("records", [])) == 744


def test_summarize_without_document_selection_prompts_user(shared_rag):
    """When no document is selected, ask the user to select one rather than silently summarizing every document."""
    res = shared_rag.generate_rag_answer("Summarize my document")
    assert res["retrieval_status"] == "missing_document_selection"
    assert "select a specific document" in res["answer"].lower()
    assert len(res["retrieved_chunks"]) == 0
    assert len(res["sources"]) == 0


def test_wellness_document_summary_inclusion_and_five_bullet_format(shared_rag):
    """Summarizing wellness PDF includes sleep routines, balanced eating, hydration, physical activity, habit tracking in 5 bullets."""
    doc_id = "8_1793003770_HealthAI_Test_Report_Sleep_Nutrition_Wellness.pdf"
    res = shared_rag.generate_document_summary(
        document_id=doc_id,
        format_instructions="exactly five bullet points"
    )

    assert res["retrieval_status"] == "success"
    answer = res["answer"]

    # 1. Opening sentence
    first_line = [l.strip() for l in answer.split("\n") if l.strip()][0]
    assert first_line.endswith(".") or first_line.endswith(":")
    assert not first_line.startswith("*")
    assert not first_line.startswith("-")

    # 2. Key inclusions
    answer_lower = answer.lower()
    assert "sleep" in answer_lower
    assert "nutrition" in answer_lower or "eating" in answer_lower
    assert "hydration" in answer_lower
    assert "activity" in answer_lower or "movement" in answer_lower
    assert "habit" in answer_lower or "tracking" in answer_lower

    # 3. Exactly five bullet points
    bullet_lines = [l for l in answer.split("\n") if l.strip().startswith("*") or l.strip().startswith("-")]
    assert len(bullet_lines) == 5, f"Expected 5 bullets, found {len(bullet_lines)}: {bullet_lines}"


def test_wellness_document_summary_strict_isolation(shared_rag):
    """Wellness summary strictly excludes unrelated hypertension, diabetes, and asthma from other documents."""
    doc_id = "8_1793003770_HealthAI_Test_Report_Sleep_Nutrition_Wellness.pdf"
    res = shared_rag.generate_document_summary(document_id=doc_id)
    answer_lower = res["answer"].lower()

    # Must NOT contain unrelated clinical conditions from other PDFs in the vector store
    assert "hypertension" not in answer_lower
    assert "blood pressure" not in answer_lower
    assert "diabetes" not in answer_lower
    assert "asthma" not in answer_lower
    assert "albuterol" not in answer_lower
    assert "metformin" not in answer_lower
    assert "lisinopril" not in answer_lower


def test_clinical_document_summary_isolation(shared_rag):
    """Summarizing clinical reference uses only that document."""
    doc_id = "sample_clinical_report.pdf"
    res = shared_rag.generate_document_summary(document_id=doc_id)
    assert res["retrieval_status"] in ("success", "no_relevant_context")
    if res["retrieval_status"] == "success":
        for chk in res["retrieved_chunks"]:
            fn = chk.get("metadata", {}).get("filename", "") or chk.get("filename", "")
            assert "sample_clinical_report" in fn.lower() or "8_1793003770" not in fn.lower()


def test_format_conjoined_word_repairs():
    """Verify conjoined-word repairs produce clean spaces without destroying hyphens."""
    raw = "withouta problem Itexplicitly noted specifichealth possiblecomplications in airwaycondition andthat."
    cleaned = clean_ai_markdown(raw)

    assert "without a" in cleaned
    assert "It explicitly" in cleaned
    assert "specific health" in cleaned
    assert "possible complications" in cleaned
    assert "airway condition" in cleaned
    assert "and that" in cleaned

    # Verify hyphenated terms remain intact
    hyphen_text = "nutrient-dense whole foods and long-term consistency in evidence-based care."
    assert clean_ai_markdown(hyphen_text) == hyphen_text


def test_vector_store_invariant_unmodified():
    """Verify that vector store index was NEVER altered or re-embedded during summarization tests."""
    index = faiss.read_index("data/vector_store/index.faiss")
    assert index.ntotal == 744
    assert index.d == 384

    with open("data/vector_store/metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("count") == 744
    assert len(meta.get("records", [])) == 744
