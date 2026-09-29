"""
Unit and Integration Tests for the HealthAI Intelligent Query Router.

Verifies the 10 exact mandatory test cases, conversation-context handling,
session state isolation, and safety/relevance gate constraints:

Test 1: "What are the risk factors for hypertension according to my uploaded PDF?" -> RAG
Test 2: "What is hypertension?" -> AGNO
Test 3: "What are its symptoms?" after Test 2 -> AGNO ("its" = hypertension)
Test 4: "What are the risk factors for hypertension according to my uploaded PDF?" -> RAG
Test 5: "Which of them can be modified?" after Test 4 -> RAG
Test 6: "What are the symptoms of diabetes?" -> AGNO
Test 7: "What is asthma?" -> AGNO
Test 8: "What are its symptoms?" after "What is asthma?" -> AGNO ("its" = asthma)
Test 9: "According to my uploaded PDF, what are the complications of hypertension?" -> RAG
Test 10: "What are its complications?" after Test 9 -> RAG

Additional Verification:
- Document existence does not route general questions to RAG.
- Session routing state tracking (conversation_mode and last_topic).
- Preservation of RAG relevance gate and citations.
- API endpoints POST /api/query behavior.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.main import app
from backend.router.query_router import (
    classify_query,
    route_and_execute_query,
    extract_query_topic,
    resolve_ambiguous_message,
    is_ambiguous_followup,
    routing_state_manager,
    get_routing_state,
    reset_routing_state,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_routing_state():
    """Ensure clean state before each test run."""
    reset_routing_state()
    yield
    reset_routing_state()


# ==============================================================================
# 10 EXACT REQUIRED SCENARIOS
# ==============================================================================


def test_scenario_1_risk_factors_hypertension_uploaded_pdf():
    """
    Test 1:
    'What are the risk factors for hypertension according to my uploaded PDF?'
    Expected route: RAG
    """
    query = "What are the risk factors for hypertension according to my uploaded PDF?"
    classification = classify_query(query)
    assert classification["route"] == "rag"
    assert "uploaded" in classification["reason"].lower() or "document" in classification["reason"].lower()
    assert classification["topic"] == "hypertension"


def test_scenario_2_what_is_hypertension():
    """
    Test 2:
    'What is hypertension?'
    Expected route: AGNO
    (Must route to Agno even if indexed PDFs exist!)
    """
    query = "What is hypertension?"
    classification = classify_query(query)
    assert classification["route"] == "agno"
    assert "general medical" in classification["reason"].lower()
    assert classification["topic"] == "hypertension"


def test_scenario_3_what_are_its_symptoms_after_test_2():
    """
    Test 3:
    'What are its symptoms?' after Test 2 ('What is hypertension?')
    Expected route: AGNO
    Expected interpretation: 'its' = hypertension
    """
    session_id = "test-session-sc3"

    # Turn 1: Test 2 query executed
    turn1_res = route_and_execute_query(
        message="What is hypertension?",
        session_id=session_id,
    )
    assert turn1_res["route"] == "agno"

    # Verify conversation state is tracked
    state_after_turn1 = get_routing_state(session_id)
    assert state_after_turn1.get("conversation_mode") == "agno"
    assert state_after_turn1.get("last_topic") == "hypertension"

    # Turn 2: Ambiguous follow-up
    classification = classify_query(
        message="What are its symptoms?",
        session_id=session_id,
    )
    assert classification["route"] == "agno"
    assert classification["is_followup"] is True
    assert classification["topic"] == "hypertension"

    # Verify pronoun resolution
    resolved = resolve_ambiguous_message("What are its symptoms?", classification["topic"])
    assert "hypertension" in resolved.lower()

    # Execute and verify response
    turn2_res = route_and_execute_query(
        message="What are its symptoms?",
        session_id=session_id,
    )
    assert turn2_res["route"] == "agno"
    assert "hypertension" in turn2_res["answer"].lower() or "condition" in turn2_res["answer"].lower()


def test_scenario_4_risk_factors_hypertension_uploaded_pdf():
    """
    Test 4:
    'What are the risk factors for hypertension according to my uploaded PDF?'
    Expected route: RAG
    """
    query = "What are the risk factors for hypertension according to my uploaded PDF?"
    classification = classify_query(query)
    assert classification["route"] == "rag"
    assert classification["topic"] == "hypertension"


def test_scenario_5_which_of_them_can_be_modified_after_test_4():
    """
    Test 5:
    'Which of them can be modified?' after Test 4
    Expected route: RAG
    """
    session_id = "test-session-sc5"

    mock_rag = MagicMock()
    mock_rag.generate_rag_answer.return_value = {
        "question": "Which of them can be modified?",
        "answer": "According to the uploaded paper, modifiable risk factors include diet and physical inactivity. [Source 1]",
        "sources": [{"chunk_id": "c1", "text": "diet and inactivity"}],
        "retrieval_status": "success",
    }

    # Turn 1: Test 4 query executed via RAG
    turn1_res = route_and_execute_query(
        message="What are the risk factors for hypertension according to my uploaded PDF?",
        session_id=session_id,
        rag_service=mock_rag,
    )
    assert turn1_res["route"] == "rag"

    state_after_turn1 = get_routing_state(session_id)
    assert state_after_turn1.get("conversation_mode") == "rag"
    assert state_after_turn1.get("last_topic") == "hypertension"

    # Turn 2: Ambiguous follow-up
    classification = classify_query(
        message="Which of them can be modified?",
        session_id=session_id,
    )
    assert classification["route"] == "rag"
    assert classification["is_followup"] is True
    assert classification["topic"] == "hypertension"

    turn2_res = route_and_execute_query(
        message="Which of them can be modified?",
        session_id=session_id,
        rag_service=mock_rag,
    )
    assert turn2_res["route"] == "rag"
    assert "modifiable" in turn2_res["answer"]


def test_scenario_6_symptoms_of_diabetes():
    """
    Test 6:
    'What are the symptoms of diabetes?'
    Expected route: AGNO
    """
    query = "What are the symptoms of diabetes?"
    classification = classify_query(query)
    assert classification["route"] == "agno"
    assert "general medical" in classification["reason"].lower()
    assert classification["topic"] == "diabetes"


def test_scenario_7_what_is_asthma():
    """
    Test 7:
    'What is asthma?'
    Expected route: AGNO
    """
    query = "What is asthma?"
    classification = classify_query(query)
    assert classification["route"] == "agno"
    assert "general medical" in classification["reason"].lower()
    assert classification["topic"] == "asthma"


def test_scenario_8_what_are_its_symptoms_after_what_is_asthma():
    """
    Test 8:
    'What are its symptoms?' after 'What is asthma?'
    Expected route: AGNO
    Expected interpretation: 'its' = asthma
    """
    session_id = "test-session-sc8"

    # Turn 1: 'What is asthma?'
    turn1_res = route_and_execute_query(
        message="What is asthma?",
        session_id=session_id,
    )
    assert turn1_res["route"] == "agno"

    state_after_turn1 = get_routing_state(session_id)
    assert state_after_turn1.get("conversation_mode") == "agno"
    assert state_after_turn1.get("last_topic") == "asthma"

    # Turn 2: 'What are its symptoms?'
    classification = classify_query(
        message="What are its symptoms?",
        session_id=session_id,
    )
    assert classification["route"] == "agno"
    assert classification["is_followup"] is True
    assert classification["topic"] == "asthma"

    # Verify pronoun resolution resolves 'its' = asthma
    resolved = resolve_ambiguous_message("What are its symptoms?", classification["topic"])
    assert "asthma" in resolved.lower()

    # Execute Turn 2
    turn2_res = route_and_execute_query(
        message="What are its symptoms?",
        session_id=session_id,
    )
    assert turn2_res["route"] == "agno"
    # Should discuss asthma symptoms (wheezing, shortness of breath, etc.)
    assert "asthma" in turn2_res["answer"].lower() or "breath" in turn2_res["answer"].lower()


def test_scenario_9_according_to_pdf_complications_of_hypertension():
    """
    Test 9:
    'According to my uploaded PDF, what are the complications of hypertension?'
    Expected route: RAG
    """
    query = "According to my uploaded PDF, what are the complications of hypertension?"
    classification = classify_query(query)
    assert classification["route"] == "rag"
    assert classification["topic"] == "hypertension"


def test_scenario_10_what_are_its_complications_after_test_9():
    """
    Test 10:
    'What are its complications?' after Test 9
    Expected route: RAG
    """
    session_id = "test-session-sc10"

    mock_rag = MagicMock()
    mock_rag.generate_rag_answer.return_value = {
        "question": "What are its complications?",
        "answer": "According to the uploaded document, complications include stroke and kidney damage. [Source 1]",
        "sources": [{"chunk_id": "c2", "text": "stroke, kidney damage"}],
        "retrieval_status": "success",
    }

    # Turn 1: Test 9 executed
    turn1_res = route_and_execute_query(
        message="According to my uploaded PDF, what are the complications of hypertension?",
        session_id=session_id,
        rag_service=mock_rag,
    )
    assert turn1_res["route"] == "rag"

    state_after_turn1 = get_routing_state(session_id)
    assert state_after_turn1.get("conversation_mode") == "rag"
    assert state_after_turn1.get("last_topic") == "hypertension"

    # Turn 2: 'What are its complications?'
    classification = classify_query(
        message="What are its complications?",
        session_id=session_id,
    )
    assert classification["route"] == "rag"
    assert classification["is_followup"] is True
    assert classification["topic"] == "hypertension"

    turn2_res = route_and_execute_query(
        message="What are its complications?",
        session_id=session_id,
        rag_service=mock_rag,
    )
    assert turn2_res["route"] == "rag"
    assert "complications" in turn2_res["answer"].lower()


# ==============================================================================
# ADDITIONAL REQUIREMENTS & SAFETY CONSTRAINTS
# ==============================================================================


def test_general_query_not_routed_to_rag_merely_because_pdf_exists():
    """
    Verifies requirement D & F:
    The presence of an uploaded/indexed PDF or previous RAG turn must NOT cause
    a fresh, standalone general medical query like 'What is hypertension?' to route to RAG.
    """
    session_id = "session-pdf-exists"

    # Simulate that user previously queried a document
    routing_state_manager.update_state(
        session_id=session_id,
        conversation_mode="rag",
        topic="leukemia",
        last_query="What does the uploaded PDF say about leukemia?",
    )

    # Now user asks a standalone general medical question without document reference
    result = classify_query(
        message="What is hypertension?",
        session_id=session_id,
    )
    # MUST route to AGNO, NOT RAG!
    assert result["route"] == "agno"
    assert "general medical" in result["reason"].lower()


def test_topic_extraction():
    """Verifies that clinical topics are extracted accurately."""
    assert extract_query_topic("What are the risk factors for hypertension according to my uploaded PDF?") == "hypertension"
    assert extract_query_topic("What is asthma?") == "asthma"
    assert extract_query_topic("What are the symptoms of diabetes?") == "diabetes"
    assert extract_query_topic("What causes high blood pressure?") == "high blood pressure"
    assert extract_query_topic("What does the uploaded paper say about pediatric leukemia?") == "pediatric leukemia"


def test_ambiguous_followup_detection():
    """Verifies ambiguous vs. standalone queries."""
    assert is_ambiguous_followup("What are its symptoms?") is True
    assert is_ambiguous_followup("Which of them can be modified?") is True
    assert is_ambiguous_followup("What are its complications?") is True
    assert is_ambiguous_followup("How is it treated?") is True
    assert is_ambiguous_followup("Why does it happen?") is True
    assert is_ambiguous_followup("What causes it?") is True

    # Standalone queries should NOT be considered ambiguous follow-ups
    assert is_ambiguous_followup("What is hypertension?") is False
    assert is_ambiguous_followup("What is asthma?") is False
    assert is_ambiguous_followup("What are the symptoms of diabetes?") is False
    assert is_ambiguous_followup("What causes high blood pressure?") is False


def test_pronoun_resolution():
    """Verifies that pronouns are resolved to the active clinical topic."""
    assert resolve_ambiguous_message("What are its symptoms?", "hypertension") == "What are the symptoms of hypertension?"
    assert resolve_ambiguous_message("What are its symptoms?", "asthma") == "What are the symptoms of asthma?"
    assert resolve_ambiguous_message("What are its complications?", "hypertension") == "What are the complications of hypertension?"
    assert resolve_ambiguous_message("Which of them can be modified?", "hypertension") == "Which risk factors of hypertension can be modified?"
    assert resolve_ambiguous_message("How is it treated?", "asthma") == "How is asthma treated?"


def test_rag_relevance_gate_preserved():
    """
    Verifies that when RAG is selected and no relevant evidence is found,
    the existing relevance gate refusal is preserved without hallucinating.
    """
    mock_rag_service = MagicMock()
    mock_rag_service.generate_rag_answer.return_value = {
        "question": "What does the uploaded paper say about pediatric leukemia?",
        "answer": "I could not find any relevant information in your uploaded documents regarding this topic.",
        "retrieval_status": "no_relevant_context",
        "sources": [],
        "context": "",
        "disclaimer": "Grounding refusal: No evidence found.",
        "timings": {"total_time_ms": 15.0}
    }

    result = route_and_execute_query(
        message="What does the uploaded paper say about pediatric leukemia?",
        rag_service=mock_rag_service
    )

    assert result["route"] == "rag"
    assert result["retrieval_status"] == "no_relevant_context"
    assert result["sources"] == []
    assert "could not find any relevant information" in result["answer"]


def test_api_query_endpoint_success_rag():
    """Verifies that POST /api/query correctly returns RAG response schema."""
    mock_rag_res = {
        "question": "According to the uploaded PDF, what is hypertension?",
        "answer": "According to the document [Source 1], hypertension is high blood pressure.",
        "retrieval_status": "success",
        "sources": [{"chunk_id": "c1", "source_name": "guidelines.pdf", "text": "...", "page": 1}],
        "context": "[SOURCE 1] ...",
        "disclaimer": "Educational use.",
        "timings": {"total_time_ms": 120.0},
        "route": "rag",
        "route_reason": "User explicitly refers to uploaded document",
        "status": "success"
    }

    with patch("backend.api.query_api.route_and_execute_query", return_value=mock_rag_res):
        response = client.post(
            "/api/query",
            json={"message": "According to the uploaded PDF, what is hypertension?"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "rag"
    assert len(data["sources"]) == 1
    assert data["status"] == "success"
    assert "session_id" in data


def test_api_query_endpoint_success_agno():
    """Verifies that POST /api/query correctly returns Agno response schema."""
    mock_agno_res = {
        "question": "What is asthma?",
        "answer": "Asthma is a chronic inflammatory disorder of the airways.",
        "retrieval_status": "agno_agent",
        "sources": [],
        "session_id": "session-test-endpoint",
        "disclaimer": "General educational info.",
        "route": "agno",
        "route_reason": "General medical question without document reference",
        "status": "success",
        "agent": "HealthAI Medical Assistant"
    }

    with patch("backend.api.query_api.route_and_execute_query", return_value=mock_agno_res):
        response = client.post(
            "/api/query",
            json={"message": "What is asthma?", "session_id": "session-test-endpoint"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "agno"
    assert data["session_id"] == "session-test-endpoint"
    assert data["sources"] == []


def test_api_query_endpoint_empty_message_validation():
    """Verifies that empty message raises HTTP 400."""
    response = client.post(
        "/api/query",
        json={"message": "   "}
    )
    assert response.status_code == 400
    assert "empty" in response.json().get("detail", "").lower()
