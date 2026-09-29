"""
Focused regression tests for Phase 2A Retrieval Evidence Deduplication.

Verifies:
1. 3 identical PDFs -> 1 unique evidence source
2. 3 different PDFs -> 3 unique evidence sources
3. 2 identical PDFs + 1 different PDF -> 2 unique evidence sources
4. Two documents with different content MUST NOT be merged even if their filenames are identical
5. Highest similarity score and corresponding metadata are preserved
6. Sequential renumbering of sources [Source 1], [Source 2], ...
"""

import pytest
from unittest.mock import MagicMock
from backend.rag.rag_service import RAGService


def test_three_identical_pdfs_yield_one_evidence_source():
    """
    Case 1: User uploaded the same PDF 3 times.
    All 3 chunks contain identical normalized content across different document IDs.
    Expectation: Exactly 1 unique evidence source with highest similarity score preserved.
    """
    identical_text = (
        "HealthAI RAG Test Document. Synthetic educational healthcare document "
        "for testing document-grounded question answering. 1. Hypertension is high blood pressure."
    )
    chunks = [
        {
            "document_id": "18",
            "chunk_id": "chunk_0",
            "similarity_score": 0.395,
            "text": identical_text,
            "page_number": 1,
            "metadata": {"filename": "HealthAI_RAG_Test_Document.pdf", "user_id": 8}
        },
        {
            "document_id": "17",
            "chunk_id": "chunk_0",
            "similarity_score": 0.394,
            "text": identical_text,
            "page_number": 1,
            "metadata": {"filename": "HealthAI_RAG_Test_Document.pdf", "user_id": 8}
        },
        {
            "document_id": "16",
            "chunk_id": "chunk_0",
            "similarity_score": 0.392,
            "text": identical_text,
            "page_number": 1,
            "metadata": {"filename": "HealthAI_RAG_Test_Document.pdf", "user_id": 8}
        },
    ]

    deduped = RAGService.deduplicate_chunks(chunks)

    assert len(deduped) == 1, f"Expected 1 unique evidence chunk, got {len(deduped)}"
    assert deduped[0]["document_id"] == "18"
    assert deduped[0]["similarity_score"] == 0.395
    assert deduped[0]["metadata"]["filename"] == "HealthAI_RAG_Test_Document.pdf"

    # Verify build_sources sequentially numbers the remaining unique source as [Source 1]
    rag = RAGService(vector_store=MagicMock())
    sources = rag.build_sources(deduped)
    assert len(sources) == 1
    assert sources[0]["source_index"] == 1
    assert sources[0]["source_label"] == "[Source 1]"


def test_three_different_pdfs_yield_three_evidence_sources():
    """
    Case 2: 3 genuinely different PDFs.
    Chunks have distinct content.
    Expectation: All 3 unique evidence sources are preserved.
    """
    chunks = [
        {
            "document_id": "doc_htn",
            "chunk_id": "chunk_0",
            "similarity_score": 0.88,
            "text": "Hypertension is chronically elevated arterial blood pressure above 130/80 mmHg.",
            "page_number": 1,
            "metadata": {"filename": "hypertension_guide.pdf"}
        },
        {
            "document_id": "doc_dm",
            "chunk_id": "chunk_0",
            "similarity_score": 0.82,
            "text": "Type 2 diabetes mellitus is characterized by insulin resistance and hyperglycemia.",
            "page_number": 1,
            "metadata": {"filename": "diabetes_overview.pdf"}
        },
        {
            "document_id": "doc_asthma",
            "chunk_id": "chunk_0",
            "similarity_score": 0.75,
            "text": "Asthma is a chronic inflammatory disorder of the airways causing reversible obstruction.",
            "page_number": 1,
            "metadata": {"filename": "asthma_management.pdf"}
        },
    ]

    deduped = RAGService.deduplicate_chunks(chunks)

    assert len(deduped) == 3, f"Expected 3 unique evidence chunks, got {len(deduped)}"
    scores = [c["similarity_score"] for c in deduped]
    assert scores == [0.88, 0.82, 0.75]

    rag = RAGService(vector_store=MagicMock())
    sources = rag.build_sources(deduped)
    assert len(sources) == 3
    assert [s["source_label"] for s in sources] == ["[Source 1]", "[Source 2]", "[Source 3]"]
    assert [s["source_index"] for s in sources] == [1, 2, 3]


def test_two_identical_plus_one_different_pdf_yield_two_evidence_sources():
    """
    Case 3: 2 identical PDFs + 1 different PDF.
    Expectation: Exactly 2 unique evidence sources.
    Highest similarity score is preserved for the duplicate pair.
    """
    shared_text = "Cardiovascular disease encompasses coronary artery disease, heart failure, and stroke."
    distinct_text = "Chronic kidney disease is defined as persistent abnormalities of kidney structure or function."

    chunks = [
        {
            "document_id": "doc_cvd_copy1",
            "chunk_id": "chunk_0",
            "similarity_score": 0.81,
            "text": shared_text,
            "page_number": 1,
            "metadata": {"filename": "cvd_report.pdf"}
        },
        {
            "document_id": "doc_cvd_copy2",
            "chunk_id": "chunk_0",
            "similarity_score": 0.89,  # higher score
            "text": shared_text,
            "page_number": 1,
            "metadata": {"filename": "cvd_report.pdf"}
        },
        {
            "document_id": "doc_ckd",
            "chunk_id": "chunk_0",
            "similarity_score": 0.76,
            "text": distinct_text,
            "page_number": 1,
            "metadata": {"filename": "ckd_report.pdf"}
        },
    ]

    deduped = RAGService.deduplicate_chunks(chunks)

    assert len(deduped) == 2, f"Expected 2 unique evidence chunks, got {len(deduped)}"
    
    # CVD chunk should preserve the higher score 0.89 and doc_cvd_copy2
    cvd_rep = next(c for c in deduped if "Cardiovascular" in c["text"])
    assert cvd_rep["similarity_score"] == 0.89
    assert cvd_rep["document_id"] == "doc_cvd_copy2"

    ckd_rep = next(c for c in deduped if "kidney" in c["text"])
    assert ckd_rep["similarity_score"] == 0.76
    assert ckd_rep["document_id"] == "doc_ckd"

    rag = RAGService(vector_store=MagicMock())
    sources = rag.build_sources(deduped)
    assert len(sources) == 2
    assert [s["source_label"] for s in sources] == ["[Source 1]", "[Source 2]"]


def test_identical_filenames_different_content_must_not_be_merged():
    """
    Case 4: Two different documents that happen to have the identical filename.
    Content is genuinely different.
    Expectation: MUST NOT be merged.
    """
    chunks = [
        {
            "document_id": "doc_101",
            "chunk_id": "chunk_0",
            "similarity_score": 0.84,
            "text": "Oncology clinical trials phase 3 results for immunotherapy agent.",
            "page_number": 1,
            "metadata": {"filename": "research_report.pdf"}
        },
        {
            "document_id": "doc_102",
            "chunk_id": "chunk_0",
            "similarity_score": 0.81,
            "text": "Neurology diagnostic criteria for early-stage Alzheimer's disease.",
            "page_number": 1,
            "metadata": {"filename": "research_report.pdf"}
        }
    ]

    deduped = RAGService.deduplicate_chunks(chunks)
    assert len(deduped) == 2, "Different content with same filename must remain distinct."
