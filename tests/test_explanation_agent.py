import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.agents.agent_state import AgentState
from backend.agents.explanation_agent import ExplanationAgent
from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiService


def get_mock_rag_for_explanation():
    rag = MagicMock(spec=RAGService)
    rag.query.return_value = {
        "question": "Explain hypertension simply",
        "retrieval_status": "success",
        "context": "[SOURCE 1]\nDocument: DOC_CARDIO_001\nChunk ID: MED_CHUNK_0\n\nHypertension is blood pressure above 130/80 mmHg.",
        "retrieved_chunks": [{"chunk_id": "MED_CHUNK_0", "similarity_score": 0.88}],
        "sources": [{"source_index": 1, "chunk_id": "MED_CHUNK_0", "document_id": "DOC_CARDIO_001"}],
        "timings": {"total_retrieval_time_ms": 14.5}
    }
    return rag


def test_simple_explanation_tier():
    """17. Test ExplanationAgent using 'simple' readability level."""
    mock_rag = get_mock_rag_for_explanation()
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.generate_answer.return_value = {
        "answer": "High blood pressure [Source 1] means your heart has to work too hard to push blood through your body.",
        "model": "gemini-2.5-flash"
    }

    agent = ExplanationAgent(rag_service=mock_rag, gemini_service=mock_gemini)
    state = AgentState(user_question="Explain hypertension simply", explanation_level="simple")
    result_state = agent.run(state)

    assert result_state.final_answer is not None
    assert "High blood pressure [Source 1]" in result_state.final_answer
    assert mock_gemini.generate_answer.called
    call_args = mock_gemini.generate_answer.call_args[1]
    assert "EXPLANATION TIER: SIMPLE" in call_args["question"]
    print("[PASS] test_simple_explanation_tier passed.")


def test_technical_explanation_tier():
    """18. Test ExplanationAgent using 'technical' readability level."""
    mock_rag = get_mock_rag_for_explanation()
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.generate_answer.return_value = {
        "answer": "Systemic arterial hypertension [Source 1] involves persistent elevation above 130/80 mmHg with vascular remodeling.",
        "model": "gemini-2.5-flash"
    }

    agent = ExplanationAgent(rag_service=mock_rag, gemini_service=mock_gemini)
    state = AgentState(user_question="Explain hypertension pathophysiology", explanation_level="technical")
    result_state = agent.run(state)

    assert result_state.final_answer is not None
    call_args = mock_gemini.generate_answer.call_args[1]
    assert "EXPLANATION TIER: TECHNICAL" in call_args["question"]
    print("[PASS] test_technical_explanation_tier passed.")


def test_explanation_agent_citation_preservation():
    """19. Test that ExplanationAgent preserves and validates citations."""
    mock_rag = get_mock_rag_for_explanation()
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.generate_answer.return_value = {
        "answer": "Medical guidelines [Source 1] state that normal blood pressure should be kept below 130/80 mmHg.",
        "model": "gemini-2.5-flash"
    }

    agent = ExplanationAgent(rag_service=mock_rag, gemini_service=mock_gemini)
    state = AgentState(user_question="Explain hypertension", explanation_level="simple")
    result_state = agent.run(state)

    assert result_state.citations["is_valid"] is True
    assert 1 in result_state.citations["valid_citations"]
    assert len(result_state.sources) == 1
    print("[PASS] test_explanation_agent_citation_preservation passed.")


if __name__ == "__main__":
    print("Running ExplanationAgent Unit Tests...")
    test_simple_explanation_tier()
    test_technical_explanation_tier()
    test_explanation_agent_citation_preservation()
    print("\n[SUCCESS] All 3 ExplanationAgent unit tests passed successfully!")
