import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.agents.agent_state import AgentState
from backend.agents.document_agent import DocumentAgent
from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiService


def test_document_agent_summary_workflow():
    """20. Test DocumentAgent structured summary generation."""
    mock_rag = MagicMock(spec=RAGService)
    mock_rag.query.return_value = {
        "question": "Summarize patient report",
        "retrieval_status": "success",
        "context": "[SOURCE 1]\nDocument: DOC_HOSP_001\nChunk ID: CHUNK_1\n\nPatient was admitted for acute heart failure, treated with furosemide, discharged on lisinopril.",
        "retrieved_chunks": [{"chunk_id": "CHUNK_1", "similarity_score": 0.82}],
        "sources": [{"source_index": 1, "chunk_id": "CHUNK_1", "document_id": "DOC_HOSP_001"}],
        "timings": {"total_retrieval_time_ms": 11.2}
    }

    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.generate_answer.return_value = {
        "answer": (
            "1. CLINICAL BACKGROUND: Patient admitted with acute decompensated heart failure [Source 1].\n"
            "2. KEY FINDINGS: Responded positively to loop diuretics with resolution of edema [Source 1].\n"
            "3. CARE PLAN: Discharged on oral lisinopril with outpatient follow-up in 7 days [Source 1]."
        ),
        "model": "gemini-2.5-flash"
    }

    agent = DocumentAgent(rag_service=mock_rag, gemini_service=mock_gemini)
    state = AgentState(user_question="Summarize patient report")
    result_state = agent.run(state)

    assert result_state.final_answer is not None
    assert "CLINICAL BACKGROUND" in result_state.final_answer
    assert "CARE PLAN" in result_state.final_answer
    assert result_state.citations["is_valid"] is True
    print("[PASS] test_document_agent_summary_workflow passed.")


def test_document_agent_missing_context_fallback():
    """21. Test DocumentAgent safe handling when requested document context is missing."""
    mock_rag = MagicMock(spec=RAGService)
    mock_rag.query.return_value = {
        "question": "Summarize unindexed document",
        "retrieval_status": "no_relevant_context",
        "context": "",
        "retrieved_chunks": [],
        "sources": []
    }

    mock_gemini = MagicMock(spec=GeminiService)

    agent = DocumentAgent(rag_service=mock_rag, gemini_service=mock_gemini)
    state = AgentState(user_question="Summarize unindexed document")
    result_state = agent.run(state)

    mock_gemini.generate_answer.assert_not_called()
    assert "could not be found" in result_state.final_answer
    print("[PASS] test_document_agent_missing_context_fallback passed.")


if __name__ == "__main__":
    print("Running DocumentAgent Unit Tests...")
    test_document_agent_summary_workflow()
    test_document_agent_missing_context_fallback()
    print("\n[SUCCESS] All 2 DocumentAgent unit tests passed successfully!")
