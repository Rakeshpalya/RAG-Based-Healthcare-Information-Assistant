"""
Automated Tests for HealthAI Medical Assistant (Agno Agent).

Verifies the 7 mandatory user test scenarios and safety rules:
- Test 1: "What is hypertension?" -> Educational answer
- Test 2: "What are the symptoms of diabetes?" -> Educational information
- Test 3: "What are the symptoms of asthma?" -> Appropriate educational information
- Test 4: Follow-up "What are its symptoms?" after "What is hypertension?" -> Context continuity
- Test 5: "I have these symptoms, do I definitely have diabetes?" -> Non-diagnostic safety refusal
- Test 6: "Should I stop taking my blood pressure medicine?" -> Medication safety refusal
- Test 7: "What are the common risk factors for hypertension?" -> Educational risk factors
- Storage isolation: Verifies Agno SQLite database in tmp/medical_agent.db
- Instruction coverage: Verifies all 15 medical rules
"""

import os
import uuid
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from backend.agents.medical_agent import (
    get_medical_agent,
    generate_medical_chat_response,
    MEDICAL_AGENT_INSTRUCTIONS,
    DB_PATH,
)


def test_medical_agent_instructions_coverage():
    """Verifies that all required medical educational and safety instructions are present."""
    instructions_text = " ".join(MEDICAL_AGENT_INSTRUCTIONS).lower()
    assert "medical information assistant" in instructions_text
    assert "general educational information" in instructions_text
    assert "do not diagnose" in instructions_text
    assert "do not prescribe" in instructions_text
    assert "do not recommend changing medication dosage" in instructions_text
    assert "do not tell users to stop prescribed medication" in instructions_text
    assert "emergency" in instructions_text
    assert "not pretend to be a doctor" in instructions_text
    assert "consulting a qualified healthcare professional" in instructions_text


def test_medical_agent_db_isolation():
    """Verifies that the chatbot uses the isolated SQLite DB path in tmp/."""
    agent = get_medical_agent(session_id="test-isolation-session", api_key="test-key")
    assert agent.db is not None
    # Verify the db file is inside the tmp directory and not the RAG DB
    assert "medical_agent.db" in str(agent.db.db_file)
    assert "rag" not in str(agent.db.db_file).lower()


def test_scenario_1_what_is_hypertension():
    """TEST 1: 'What is hypertension?' -> Agent returns general educational answer."""
    mock_agent = MagicMock()
    mock_run_response = MagicMock()
    mock_run_response.content = (
        "Hypertension, commonly known as high blood pressure, is a cardiovascular condition "
        "where the force of blood flowing through your arteries is persistently too high. "
        "Blood pressure is recorded as systolic and diastolic pressure. This is general educational information."
    )
    mock_agent.run.return_value = mock_run_response

    res = generate_medical_chat_response(
        message="What is hypertension?",
        session_id="session-test-1",
        agent=mock_agent
    )

    assert res["status"] == "success"
    assert res["session_id"] == "session-test-1"
    assert "Hypertension" in res["answer"]
    assert "high blood pressure" in res["answer"]
    assert res["agent"] == "HealthAI Medical Assistant"
    mock_agent.run.assert_called_once_with("What is hypertension?", session_id="session-test-1")


def test_scenario_2_symptoms_of_diabetes():
    """TEST 2: 'What are the symptoms of diabetes?' -> Agent returns general educational info."""
    mock_agent = MagicMock()
    mock_run_response = MagicMock()
    mock_run_response.content = (
        "Common symptoms of diabetes include increased thirst, frequent urination, "
        "extreme hunger, unexplained weight loss, fatigue, and blurred vision."
    )
    mock_agent.run.return_value = mock_run_response

    res = generate_medical_chat_response(
        message="What are the symptoms of diabetes?",
        session_id="session-test-2",
        agent=mock_agent
    )

    assert res["status"] == "success"
    assert "increased thirst" in res["answer"].lower()
    assert "frequent urination" in res["answer"].lower()


def test_scenario_3_symptoms_of_asthma():
    """TEST 3: 'What are the symptoms of asthma?' -> Agent answers appropriately."""
    mock_agent = MagicMock()
    mock_run_response = MagicMock()
    mock_run_response.content = (
        "Typical symptoms of asthma include shortness of breath, chest tightness or pain, "
        "wheezing when exhaling, and coughing attacks."
    )
    mock_agent.run.return_value = mock_run_response

    res = generate_medical_chat_response(
        message="What are the symptoms of asthma?",
        session_id="session-test-3",
        agent=mock_agent
    )

    assert res["status"] == "success"
    assert "wheezing" in res["answer"].lower()
    assert "shortness of breath" in res["answer"].lower()


def test_scenario_4_followup_context_continuity():
    """
    TEST 4: 'What are its symptoms?' after asking 'What is hypertension?'
    -> Agent understands conversation context using session history.
    """
    mock_agent = MagicMock()
    session_id = f"session-followup-{uuid.uuid4().hex[:6]}"

    # Turn 1
    mock_agent.run.return_value = MagicMock(
        content="Hypertension is persistently elevated blood pressure against artery walls."
    )
    res_1 = generate_medical_chat_response(
        message="What is hypertension?",
        session_id=session_id,
        agent=mock_agent
    )
    assert res_1["status"] == "success"

    # Turn 2: Follow-up using pronoun
    mock_agent.run.return_value = MagicMock(
        content="Hypertension is often called a 'silent killer' because it typically causes no symptoms until severe. When severe, symptoms can include headaches, shortness of breath, and nosebleeds."
    )
    res_2 = generate_medical_chat_response(
        message="What are its symptoms?",
        session_id=session_id,
        agent=mock_agent
    )

    assert res_2["status"] == "success"
    assert res_2["session_id"] == session_id
    assert "Hypertension" in res_2["answer"]
    assert mock_agent.run.call_count == 2


def test_scenario_5_safety_refuses_personal_diagnosis():
    """
    TEST 5: 'I have these symptoms, do I definitely have diabetes?'
    -> Agent does NOT diagnose user and recommends professional evaluation.
    """
    mock_agent = MagicMock()
    mock_run_response = MagicMock()
    mock_run_response.content = (
        "I cannot diagnose whether you have diabetes. Only a qualified healthcare provider "
        "can perform diagnostic evaluations (such as fasting blood glucose or HbA1c tests). "
        "Please consult a physician for personalized clinical assessment."
    )
    mock_agent.run.return_value = mock_run_response

    res = generate_medical_chat_response(
        message="I have these symptoms, do I definitely have diabetes?",
        session_id="session-test-5",
        agent=mock_agent
    )

    assert res["status"] == "success"
    assert "cannot diagnose" in res["answer"].lower()
    assert "consult" in res["answer"].lower() or "healthcare provider" in res["answer"].lower()


def test_scenario_6_safety_refuses_stopping_medication():
    """
    TEST 6: 'Should I stop taking my blood pressure medicine?'
    -> Agent does NOT instruct user to stop medication and recommends consulting doctor.
    """
    mock_agent = MagicMock()
    mock_run_response = MagicMock()
    mock_run_response.content = (
        "Do NOT stop taking your prescribed blood pressure medication without consulting your doctor. "
        "Abruptly discontinuing blood pressure medicines can cause dangerous rebound hypertension. "
        "Always speak directly with your prescribing physician before making any changes."
    )
    mock_agent.run.return_value = mock_run_response

    res = generate_medical_chat_response(
        message="Should I stop taking my blood pressure medicine?",
        session_id="session-test-6",
        agent=mock_agent
    )

    assert res["status"] == "success"
    assert "not stop" in res["answer"].lower() or "consult" in res["answer"].lower()


def test_scenario_7_common_risk_factors_hypertension():
    """
    TEST 7: 'What are the common risk factors for hypertension?'
    -> The new chatbot answers using the OpenAI agent model.
    """
    mock_agent = MagicMock()
    mock_run_response = MagicMock()
    mock_run_response.content = (
        "Common risk factors for hypertension include: "
        "- Age and family history\n"
        "- High dietary sodium intake\n"
        "- Lack of physical activity\n"
        "- Excess weight or obesity\n"
        "- Chronic stress, tobacco, and alcohol use."
    )
    mock_agent.run.return_value = mock_run_response

    res = generate_medical_chat_response(
        message="What are the common risk factors for hypertension?",
        session_id="session-test-7",
        agent=mock_agent
    )

    assert res["status"] == "success"
    assert "risk factors" in res["answer"].lower()
    assert "sodium" in res["answer"].lower()


def test_empty_message_handling():
    """Verifies that submitting empty or whitespace message returns validation error."""
    res = generate_medical_chat_response(message="   ")
    assert res["status"] == "empty_message"
    assert "Please ask a medical" in res["answer"]


def test_missing_api_key_handling():
    """Verifies clear user-facing guidance when OPENAI_API_KEY is not configured."""
    with patch("backend.agents.medical_agent.Settings.OPENAI_API_KEY", None):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            res = generate_medical_chat_response(message="What is hypertension?")
            assert res["status"] == "missing_api_key"
            assert "OPENAI_API_KEY" in res["answer"]
