import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.agents.agent_state import AgentState, SafetyDecision, OrchestratorDecision
from backend.agents.research_agent import ResearchAgent
from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiService


def get_mock_rag_service(success: bool = True):
    rag = MagicMock(spec=RAGService)
    if success:
        rag.query.return_value = {
            "question": "What is hypertension?",
            "retrieval_status": "success",
            "context": "[SOURCE 1]\nDocument: DOC_CARDIO_001\nChunk ID: MED_CHUNK_0\n\nHypertension is blood pressure above 130/80 mmHg.",
            "retrieved_chunks": [
                {
                    "chunk_id": "MED_CHUNK_0",
                    "document_id": "DOC_CARDIO_001",
                    "similarity_score": 0.85
                }
            ],
            "sources": [
                {
                    "source_index": 1,
                    "chunk_id": "MED_CHUNK_0",
                    "document_id": "DOC_CARDIO_001",
                    "similarity_score": 0.85
                }
            ],
            "timings": {"total_retrieval_time_ms": 15.2}
        }
    else:
        rag.query.return_value = {
            "question": "What is an unknown rare condition?",
            "retrieval_status": "no_relevant_context",
            "context": "",
            "retrieved_chunks": [],
            "sources": [],
            "timings": {"total_retrieval_time_ms": 12.0}
        }
    return rag


def get_mock_gemini_service():
    gemini = MagicMock(spec=GeminiService)
    gemini.generate_answer.return_value = {
        "answer": "According to the evidence [Source 1], hypertension is defined as blood pressure above 130/80 mmHg.",
        "model": "gemini-2.5-flash",
        "timings": {"generation_time_ms": 120.0}
    }
    return gemini


def test_research_agent_rag_integration():
    """13. Test that ResearchAgent integrates with RAG and Gemini successfully."""
    mock_rag = get_mock_rag_service(success=True)
    mock_gemini = get_mock_gemini_service()

    agent = ResearchAgent(rag_service=mock_rag, gemini_service=mock_gemini)
    state = AgentState(
        user_question="What is hypertension?",
        safety_decision=SafetyDecision(category="SAFE_INFORMATIONAL", allowed=True),
        orchestrator_decision=OrchestratorDecision(intent="RESEARCH_QUESTION", agent="research_agent")
    )

    result_state = agent.run(state)

    assert result_state.retrieval_status == "success"
    assert len(result_state.sources) == 1
    assert "hypertension is defined as blood pressure" in result_state.final_answer.lower()
    mock_gemini.generate_answer.assert_called_once()
    print("[PASS] test_research_agent_rag_integration passed.")


def test_research_agent_no_context_fallback():
    """14. Test that ResearchAgent bypasses Gemini completely when no context is retrieved."""
    mock_rag = get_mock_rag_service(success=False)
    mock_gemini = get_mock_gemini_service()

    agent = ResearchAgent(rag_service=mock_rag, gemini_service=mock_gemini)
    state = AgentState(
        user_question="What is an unknown rare condition?",
        safety_decision=SafetyDecision(category="SAFE_INFORMATIONAL", allowed=True),
        orchestrator_decision=OrchestratorDecision(intent="RESEARCH_QUESTION", agent="research_agent")
    )

    result_state = agent.run(state)

    # Assert Gemini was NEVER called
    mock_gemini.generate_answer.assert_not_called()
    assert result_state.retrieval_status == "no_relevant_context"
    assert "do not contain sufficient evidence" in result_state.final_answer
    print("[PASS] test_research_agent_no_context_fallback passed.")


def test_research_agent_source_preservation():
    """15. Test that authoritative FAISS source metadata is preserved in state."""
    mock_rag = get_mock_rag_service(success=True)
    mock_gemini = get_mock_gemini_service()

    agent = ResearchAgent(rag_service=mock_rag, gemini_service=mock_gemini)
    state = AgentState(user_question="What is hypertension?")
    result_state = agent.run(state)

    assert len(result_state.sources) == 1
    src = result_state.sources[0]
    assert src["chunk_id"] == "MED_CHUNK_0"
    assert src["document_id"] == "DOC_CARDIO_001"
    assert src["similarity_score"] == 0.85
    print("[PASS] test_research_agent_source_preservation passed.")


def test_research_agent_citation_validation():
    """16. Test that CitationValidator evaluates the LLM output in ResearchAgent."""
    mock_rag = get_mock_rag_service(success=True)
    mock_gemini = get_mock_gemini_service()

    agent = ResearchAgent(rag_service=mock_rag, gemini_service=mock_gemini)
    state = AgentState(user_question="What is hypertension?")
    result_state = agent.run(state)

    assert result_state.citations is not None
    assert result_state.citations["is_valid"] is True
    assert 1 in result_state.citations["valid_citations"]
    print("[PASS] test_research_agent_citation_validation passed.")


if __name__ == "__main__":
    print("Running ResearchAgent Unit Tests...")
    test_research_agent_rag_integration()
    test_research_agent_no_context_fallback()
    test_research_agent_source_preservation()
    test_research_agent_citation_validation()
    print("\n[SUCCESS] All 4 ResearchAgent unit tests passed successfully!")
