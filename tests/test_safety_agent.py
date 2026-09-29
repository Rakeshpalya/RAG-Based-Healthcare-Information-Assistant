import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.agents.safety_agent import SafetyAgent


def test_normal_health_question_allowed():
    """1. Normal educational question is classified as SAFE_INFORMATIONAL and allowed."""
    q = "What is hypertension and how does it affect the cardiovascular system?"
    decision = SafetyAgent.evaluate_pre_check(q)

    assert decision.allowed is True
    assert decision.category == "SAFE_INFORMATIONAL"
    assert decision.requires_clinician is False
    assert decision.requires_emergency_guidance is False
    assert decision.fallback_response == ""
    print("[PASS] test_normal_health_question_allowed passed.")


def test_diagnosis_request_blocked():
    """2. Direct diagnostic inquiry is blocked and recommends clinician consultation."""
    queries = [
        "Do I have hypertension?",
        "What disease do I have based on these symptoms?",
        "Can you diagnose me with diabetes?"
    ]
    for q in queries:
        decision = SafetyAgent.evaluate_pre_check(q)
        assert decision.allowed is False
        assert decision.category == "DIAGNOSIS_REQUEST"
        assert decision.requires_clinician is True
        assert "cannot provide personal medical diagnoses" in decision.fallback_response
    print("[PASS] test_diagnosis_request_blocked passed.")


def test_medication_request_blocked():
    """3. Personal medication and dosing requests are blocked."""
    queries = [
        "What medicine should I take for my blood pressure?",
        "What dosage should I take of furosemide?",
        "Can you prescribe me medication?"
    ]
    for q in queries:
        decision = SafetyAgent.evaluate_pre_check(q)
        assert decision.allowed is False
        assert decision.category == "MEDICATION_REQUEST"
        assert decision.requires_clinician is True
        assert "not authorized to prescribe pharmaceutical drugs" in decision.fallback_response
    print("[PASS] test_medication_request_blocked passed.")


def test_treatment_alteration_blocked():
    """4. Inquiries asking about stopping or changing prescribed treatments are blocked."""
    queries = [
        "Should I stop taking my prescribed blood pressure pills?",
        "Can I change my dosage of lisinopril on my own?",
        "Should I skip my dose today?"
    ]
    for q in queries:
        decision = SafetyAgent.evaluate_pre_check(q)
        assert decision.allowed is False
        assert decision.category == "TREATMENT_REQUEST"
        assert decision.requires_clinician is True
        assert "never stop, pause, or alter the dosage" in decision.fallback_response
    print("[PASS] test_treatment_alteration_blocked passed.")


def test_emergency_symptom_immediate_guidance():
    """5. Acute emergency symptoms trigger emergency advisory and bypass LLM."""
    queries = [
        "I have severe chest pain and difficulty breathing. What should I do?",
        "My left arm is numb and my face is drooping, what is happening?",
        "Patient has severe bleeding from wound."
    ]
    for q in queries:
        decision = SafetyAgent.evaluate_pre_check(q)
        assert decision.allowed is False
        assert decision.category == "EMERGENCY_SYMPTOM"
        assert decision.requires_emergency_guidance is True
        assert "EMERGENCY ADVISORY" in decision.fallback_response
        assert "911" in decision.fallback_response
    print("[PASS] test_emergency_symptom_immediate_guidance passed.")


def test_empty_question_handling():
    """6. Empty or whitespace question is blocked cleanly."""
    for empty_q in ["", "   ", None]:
        decision = SafetyAgent.evaluate_pre_check(empty_q)
        assert decision.allowed is False
        assert decision.category == "EMPTY_QUERY"
        assert "cannot be empty" in decision.reason
    print("[PASS] test_empty_question_handling passed.")


def test_safety_post_check():
    """7. Post-generation safety check validates safe response and flags forbidden claims."""
    # Safe output
    safe_output = "Hypertension is defined as blood pressure above 130/80 mmHg [Source 1]."
    res_safe = SafetyAgent.evaluate_post_check("What is hypertension?", safe_output)
    assert res_safe["post_check_passed"] is True
    assert len(res_safe["warnings"]) == 0

    # Output with prohibited diagnostic phrasing
    unsafe_diagnostic = "Based on what you said, you have been diagnosed with chronic kidney disease."
    res_diag = SafetyAgent.evaluate_post_check("My back hurts", unsafe_diagnostic)
    assert res_diag["post_check_passed"] is False
    assert len(res_diag["warnings"]) > 0

    # Output with prohibited prescribing phrasing
    unsafe_prescription = "I prescribe that you must take this medication twice daily."
    res_rx = SafetyAgent.evaluate_post_check("What should I take?", unsafe_prescription)
    assert res_rx["post_check_passed"] is False
    assert len(res_rx["warnings"]) > 0
    print("[PASS] test_safety_post_check passed.")


if __name__ == "__main__":
    print("Running SafetyAgent Unit Tests...")
    test_normal_health_question_allowed()
    test_diagnosis_request_blocked()
    test_medication_request_blocked()
    test_treatment_alteration_blocked()
    test_emergency_symptom_immediate_guidance()
    test_empty_question_handling()
    test_safety_post_check()
    print("\n[SUCCESS] All 7 SafetyAgent unit tests passed successfully!")
