import pytest
from unittest.mock import MagicMock, patch
from backend.rag.rag_service import RAGService
from backend.rag.prompt_builder import build_rag_prompt, HEALTHCARE_SYSTEM_INSTRUCTIONS
from backend.rag.markdown_utils import clean_ai_markdown
from backend.services.vector_store_service import VectorStoreService
from backend.evaluation.citation_validator import CitationValidator


def test_deduplication_and_sorting():
    chunks = [
        {"document_id": "doc1", "chunk_id": "c1", "similarity_score": 0.65, "text": "Chunk 1 lower score"},
        {"document_id": "doc1", "chunk_id": "c1", "similarity_score": 0.82, "text": "Chunk 1 higher score"},
        {"document_id": "doc1", "chunk_id": "c2", "similarity_score": 0.70, "text": "Chunk 2 distinct"},
        {"document_id": "doc2", "chunk_id": "c1", "similarity_score": 0.40, "text": "Different document chunk"},
    ]
    deduped = RAGService.deduplicate_chunks(chunks)
    assert len(deduped) == 3
    # Check that highest score was preserved for doc1, c1
    c1 = next(c for c in deduped if c["document_id"] == "doc1" and c["chunk_id"] == "c1")
    assert c1["similarity_score"] == 0.82
    # Verify descending sort order
    scores = [c["similarity_score"] for c in deduped]
    assert scores == sorted(scores, reverse=True)


def test_followup_query_resolution():
    history = [
        {"role": "user", "content": "What is hypertension and what are its common risk factors?"},
        {"role": "assistant", "content": "Hypertension is high blood pressure. Risk factors include age and sodium intake. [Source 1]"}
    ]
    query = "What lifestyle changes can help with it?"
    effective_q, conv_ctx = RAGService.resolve_followup_query(query, history)
    assert "hypertension" in effective_q.lower()
    assert conv_ctx is not None
    assert "What is hypertension" in conv_ctx


def test_followup_query_without_pronouns_unchanged():
    history = [
        {"role": "user", "content": "What is hypertension?"},
        {"role": "assistant", "content": "Hypertension is high blood pressure."}
    ]
    query = "What are the symptoms of asthma?"
    effective_q, conv_ctx = RAGService.resolve_followup_query(query, history)
    assert effective_q == "What are the symptoms of asthma?"
    assert conv_ctx is not None


def test_prompt_builder_includes_conversation_context():
    prompt = build_rag_prompt(
        question="What lifestyle changes can help with it?",
        context="[SOURCE 1]\nDocument: doc1\n\nLifestyle measures include physical activity.",
        conversation_context="User asked previously: \"What is hypertension?\""
    )
    assert "=== SYSTEM INSTRUCTIONS ===" in prompt
    assert "=== PREVIOUS CONVERSATION CONTEXT ===" in prompt
    assert "What is hypertension?" in prompt
    assert "=== RETRIEVED MEDICAL CONTEXT ===" in prompt
    assert "=== USER QUESTION ===" in prompt
    assert "=== GROUNDED ANSWER ===" in prompt


def test_no_relevant_context_does_not_call_gemini():
    mock_vs = MagicMock(spec=VectorStoreService)
    mock_vs.search.return_value = []  # No chunks retrieved

    rag_service = RAGService(vector_store=mock_vs)
    mock_gemini = MagicMock()

    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        res = rag_service.generate_rag_answer(
            question="What are the symptoms of diabetes?",
            gemini_service=mock_gemini
        )

    assert res["retrieval_status"] == "no_relevant_context"
    assert res["sources"] == []
    assert res["retrieved_chunks"] == []
    assert res["timings"]["llm_generation_time_ms"] == 0.0
    assert res["timings"]["generation_time_ms"] == 0.0
    assert res["timings"]["gemini_calls_count"] == 0
    # Crucial: Gemini service must NEVER be called
    mock_gemini.generate_answer.assert_not_called()


def test_deterministic_citation_mapping_and_evidence_pruning():
    mock_vs = MagicMock(spec=VectorStoreService)
    # Return 3 chunks passing threshold
    mock_vs.search.return_value = [
        {"document_id": "doc1", "chunk_id": "c1", "page_number": 1, "similarity_score": 0.80, "text": "Hypertension definition", "metadata": {"filename": "doc1.pdf"}},
        {"document_id": "doc1", "chunk_id": "c2", "page_number": 2, "similarity_score": 0.75, "text": "Risk factors info", "metadata": {"filename": "doc1.pdf"}},
        {"document_id": "doc1", "chunk_id": "c3", "page_number": 3, "similarity_score": 0.70, "text": "Lifestyle changes", "metadata": {"filename": "doc1.pdf"}},
    ]

    rag_service = RAGService(vector_store=mock_vs)
    mock_gemini = MagicMock()
    # Gemini only uses and cites Source 1 and Source 3; Source 2 was not used
    mock_gemini.generate_answer.return_value = {
        "answer": "Hypertension is high blood pressure [Source 1]. Recommended lifestyle measures include diet [Source 3].",
        "generation_time_ms": 1250.0,
        "api_request_time_ms": 1250.0,
        "gemini_calls_count": 1,
        "model": "gemini-3.5-flash-lite",
        "input_tokens": 400,
        "output_tokens": 50,
        "status": "success"
    }

    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        res = rag_service.generate_rag_answer(
            question="What is hypertension and what lifestyle changes help?",
            gemini_service=mock_gemini
        )

    assert res["retrieval_status"] == "success"
    # Unused Source 2 must NOT be in final sources list!
    assert len(res["sources"]) == 2
    assert res["sources"][0]["chunk_id"] == "c1"
    assert res["sources"][0]["source_label"] == "[Source 1]"
    assert res["sources"][0]["source_index"] == 1
    assert res["sources"][1]["chunk_id"] == "c3"
    assert res["sources"][1]["source_label"] == "[Source 3]"
    assert res["sources"][1]["source_index"] == 3


def test_hallucinated_citation_pruning():
    mock_vs = MagicMock(spec=VectorStoreService)
    # Only 1 chunk retrieved
    mock_vs.search.return_value = [
        {"document_id": "doc1", "chunk_id": "c1", "page_number": 1, "similarity_score": 0.80, "text": "Only one chunk available", "metadata": {"filename": "doc1.pdf"}}
    ]

    rag_service = RAGService(vector_store=mock_vs)
    mock_gemini = MagicMock()
    # Gemini hallucinates [Source 5]
    mock_gemini.generate_answer.return_value = {
        "answer": "Based on the evidence [Source 1], and additional notes [Source 5].",
        "generation_time_ms": 1100.0,
        "api_request_time_ms": 1100.0,
        "gemini_calls_count": 1,
        "model": "gemini-3.5-flash-lite",
        "input_tokens": 300,
        "output_tokens": 40,
        "status": "success"
    }

    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        res = rag_service.generate_rag_answer(
            question="Tell me about the document",
            gemini_service=mock_gemini
        )

    # Hallucinated [Source 5] must be stripped from the answer text
    assert "[Source 5]" not in res["answer"]
    assert "[Source 1]" in res["answer"]
    assert len(res["sources"]) == 1
    assert res["sources"][0]["source_label"] == "[Source 1]"


def test_markdown_sanitization_comprehensive():
    dirty_markdown = (
        "### What Is Hypertension?\n"
        "[svg](http://localhost:8501/#what-is-hypertension)\n"
        "Hypertension is chronically elevated blood pressure [Source 1].\n"
        "<a class=\"anchor\" href=\"#link\"><svg></svg></a>\n"
        "- High sodium diet\n"
        "- Lack of exercise\n"
        "<span style=\"color:red\">Consult doctor</span>\n"
        "[svg](http://127.0.0.1:8000/api)\n"
        "[broken]()\n"
        "svg\n"
        "[svg]: http://localhost:8501/#common-risk-factors"
    )
    cleaned = clean_ai_markdown(dirty_markdown)

    assert "### What Is Hypertension?" in cleaned
    assert "[Source 1]" in cleaned
    assert "- High sodium diet" in cleaned
    assert "- Lack of exercise" in cleaned
    assert "Consult doctor" in cleaned
    assert "localhost" not in cleaned
    assert "127.0.0.1" not in cleaned
    assert "<svg" not in cleaned
    assert "[svg]" not in cleaned
    assert "[broken]()" not in cleaned
    assert not cleaned.endswith("svg")


def test_regression_case_a_hypertension_risk_factors():
    mock_vs = MagicMock(spec=VectorStoreService)
    mock_vs.search.return_value = [
        {
            "document_id": "doc1",
            "chunk_id": "c1",
            "page_number": 1,
            "similarity_score": 0.82,
            "text": "Hypertension common risk factors include increasing age, family history, excess body weight, physical inactivity, a diet high in sodium, tobacco use, excessive alcohol consumption.",
            "metadata": {"filename": "synthetic_hypertension_test.pdf"}
        }
    ]
    rag_service = RAGService(vector_store=mock_vs)
    mock_gemini = MagicMock()
    mock_gemini.generate_answer.return_value = {
        "answer": "Common risk factors for hypertension include age, family history, excess weight, and high sodium diet [Source 1].",
        "generation_time_ms": 100.0,
        "api_request_time_ms": 100.0,
        "gemini_calls_count": 1,
        "model": "gemini-3.5-flash-lite",
        "input_tokens": 150,
        "output_tokens": 40,
        "status": "success"
    }

    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        res = rag_service.generate_rag_answer(
            question="What are the risk factors for hypertension?",
            gemini_service=mock_gemini
        )

    assert res["retrieval_status"] == "success"
    assert len(res["sources"]) == 1
    assert res["sources"][0]["source_label"] == "[Source 1]"
    assert "[Source 1]" in res["answer"]
    assert mock_gemini.generate_answer.called


def test_regression_case_b_modifiable_risk_factors_followup_no_unsupported_classification():
    mock_vs = MagicMock(spec=VectorStoreService)
    mock_vs.search.return_value = [
        {
            "document_id": "doc1",
            "chunk_id": "c1",
            "page_number": 1,
            "similarity_score": 0.80,
            "text": "Hypertension: 2. Common Risk Factors: increasing age, family history, excess body weight, physical inactivity, diet high in sodium. 3. General Lifestyle Measures: regular physical activity, maintaining healthy weight, moderating sodium intake.",
            "metadata": {"filename": "synthetic_hypertension_test.pdf"}
        }
    ]
    rag_service = RAGService(vector_store=mock_vs)
    mock_gemini = MagicMock()
    # Mock LLM attempt to classify modifiable risk factors across sections
    mock_gemini.generate_answer.return_value = {
        "answer": "Excess body weight and physical inactivity can be modified through lifestyle changes [Source 1].",
        "generation_time_ms": 90.0,
        "api_request_time_ms": 90.0,
        "gemini_calls_count": 1,
        "model": "gemini-3.5-flash-lite",
        "input_tokens": 150,
        "output_tokens": 35,
        "status": "success"
    }

    history = [
        {"role": "user", "content": "What are the risk factors for hypertension?"},
        {"role": "assistant", "content": "Common risk factors include age and high sodium diet [Source 1]."}
    ]

    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        res = rag_service.generate_rag_answer(
            question="Which of them can be modified?",
            conversation_history=history,
            gemini_service=mock_gemini
        )

    assert res["retrieval_status"] == "success"
    assert len(res["sources"]) == 1
    assert res["sources"][0]["source_label"] == "[Source 1]"
    assert "[Source 1]" in res["answer"]
    # Verify grounded limitation is enforced and unsupported classification is eliminated
    assert "does not explicitly identify which specific risk factors are modifiable" in res["answer"].lower()
    assert "can be modified through lifestyle changes" not in res["answer"].lower()
    assert mock_gemini.generate_answer.called


def test_regression_case_c_unrelated_diabetes_risk_factors():
    mock_vs = MagicMock(spec=VectorStoreService)
    mock_vs.search.return_value = [
        {
            "document_id": "doc1",
            "chunk_id": "c1",
            "page_number": 1,
            "similarity_score": 0.65,
            "text": "Hypertension is high blood pressure. Risk factors include age and family history.",
            "metadata": {"filename": "synthetic_hypertension_test.pdf"}
        }
    ]
    rag_service = RAGService(vector_store=mock_vs)
    mock_gemini = MagicMock()

    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        res = rag_service.generate_rag_answer(
            question="What are the main risk factors for diabetes?",
            gemini_service=mock_gemini
        )

    assert res["retrieval_status"] == "no_relevant_context"
    assert res["sources"] == []
    assert res["timings"]["llm_called"] is False
    assert "[Source 1]" not in res["answer"]
    mock_gemini.generate_answer.assert_not_called()


def test_regression_case_d_diabetes_complications_doc_preamble():
    mock_vs = MagicMock(spec=VectorStoreService)
    mock_vs.search.return_value = [
        {
            "document_id": "doc1",
            "chunk_id": "c1",
            "page_number": 1,
            "similarity_score": 0.70,
            "text": "Hypertension educational document...",
            "metadata": {"filename": "synthetic_hypertension_test.pdf"}
        }
    ]
    rag_service = RAGService(vector_store=mock_vs)
    mock_gemini = MagicMock()

    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        res = rag_service.generate_rag_answer(
            question="According to the hypertension document, what are the complications of diabetes?",
            gemini_service=mock_gemini
        )

    assert res["retrieval_status"] == "no_relevant_context"
    assert res["sources"] == []
    assert res["timings"]["llm_called"] is False
    assert "[Source 1]" not in res["answer"]
    mock_gemini.generate_answer.assert_not_called()


def test_regression_case_e_medications_recommended():
    mock_vs = MagicMock(spec=VectorStoreService)
    mock_vs.search.return_value = [
        {
            "document_id": "doc1",
            "chunk_id": "c1",
            "page_number": 1,
            "similarity_score": 0.72,
            "text": "Hypertension lifestyle measures include regular physical activity, maintaining healthy weight, and moderating sodium intake.",
            "metadata": {"filename": "synthetic_hypertension_test.pdf"}
        }
    ]
    rag_service = RAGService(vector_store=mock_vs)
    mock_gemini = MagicMock()

    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        res = rag_service.generate_rag_answer(
            question="What medications are recommended for treating hypertension?",
            gemini_service=mock_gemini
        )

    assert res["retrieval_status"] == "no_relevant_context"
    assert res["sources"] == []
    assert res["timings"]["llm_called"] is False
    assert "[Source 1]" not in res["answer"]
    mock_gemini.generate_answer.assert_not_called()


def test_phase2_grounded_document_boundary_malaria():
    """Phase 2: Explicit document boundary for malaria returns grounded_boundary status with preserved answer and source card."""
    mock_vs = MagicMock(spec=VectorStoreService)
    mock_vs.search.return_value = [
        {
            "document_id": "doc_boundary_1",
            "chunk_id": "chunk_0",
            "page_number": 1,
            "similarity_score": 0.3949,
            "text": (
                "HealthAI RAG Test Document. 1. Hypertension. 2. Diabetes. 3. Asthma. "
                "4. Testing Boundary: This document does not contain information about tuberculosis, malaria, cancer, or their treatments. "
                "Questions about those topics should be treated as unsupported by this document."
            ),
            "metadata": {"filename": "HealthAI_RAG_Test_Document.pdf"}
        }
    ]
    rag_service = RAGService(vector_store=mock_vs)
    mock_gemini = MagicMock()
    mock_gemini.generate_answer.return_value = {
        "answer": (
            "According to your uploaded PDF, malaria and its symptoms are not covered in the document. "
            "The document explicitly states that questions about malaria should be treated as unsupported by this document. [Source 1]"
        ),
        "generation_time_ms": 120.0,
        "api_request_time_ms": 120.0,
        "gemini_calls_count": 1,
        "model": "gemini-3.5-flash-lite",
        "input_tokens": 200,
        "output_tokens": 45,
        "status": "success"
    }

    with patch("backend.services.embedding_service.EmbeddingService.embed_query", return_value=[0.1] * 384):
        res = rag_service.generate_rag_answer(
            question="What are the symptoms of malaria according to my uploaded PDF?",
            gemini_service=mock_gemini
        )

    assert res["retrieval_status"] == "grounded_boundary"
    assert len(res["sources"]) == 1
    assert res["sources"][0]["source_label"] == "[Source 1]"
    assert "[Source 1]" in res["answer"]
    assert "Relevant medical information could not be found" not in res["answer"]
    assert "malaria" in res["answer"].lower()
    assert mock_gemini.generate_answer.called

