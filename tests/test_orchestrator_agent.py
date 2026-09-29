import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.agents.orchestrator_agent import OrchestratorAgent


def test_document_summary_routing():
    """8. Inquiries asking for document summaries are routed to document_agent."""
    queries = [
        "Summarize this discharge report",
        "Can you provide an overview of this document?",
        "What are the key findings in the report?",
        "What does the uploaded document say about the patient?"
    ]
    for q in queries:
        decision = OrchestratorAgent.classify_intent(q)
        assert decision.intent == "DOCUMENT_SUMMARY"
        assert decision.agent == "document_agent"
        assert decision.requires_retrieval is True
    print("[PASS] test_document_summary_routing passed.")


def test_medical_explanation_routing():
    """9. Inquiries asking for simplification or explanations are routed to explanation_agent."""
    queries = [
        "Explain hypertension in simple terms",
        "What does this lab value mean in plain english?",
        "Can you break this down for a patient?",
        "Explain diabetic nephropathy simply"
    ]
    for q in queries:
        decision = OrchestratorAgent.classify_intent(q)
        assert decision.intent == "MEDICAL_EXPLANATION"
        assert decision.agent == "explanation_agent"
        assert decision.requires_retrieval is True
    print("[PASS] test_medical_explanation_routing passed.")


def test_research_question_routing():
    """10. Technical medical inquiries are routed to research_agent."""
    queries = [
        "What is the mechanism of action of loop diuretics?",
        "How does chronic hypertension alter arterial elasticity?",
        "What clinical trials support DASH dietary guidelines?"
    ]
    for q in queries:
        decision = OrchestratorAgent.classify_intent(q)
        assert decision.intent == "RESEARCH_QUESTION"
        assert decision.agent == "research_agent"
        assert decision.requires_retrieval is True
    print("[PASS] test_research_question_routing passed.")


def test_safety_sensitive_routing():
    """11. Questions with emergency, diagnosis, or prescription keywords are routed to safety_agent."""
    queries = [
        "I have severe chest pain and cannot breathe",
        "Do I have congestive heart failure?",
        "What dose of medication should I take?"
    ]
    for q in queries:
        decision = OrchestratorAgent.classify_intent(q)
        assert decision.intent == "SAFETY_SENSITIVE"
        assert decision.agent == "safety_agent"
        assert decision.requires_retrieval is False
    print("[PASS] test_safety_sensitive_routing passed.")


def test_general_health_fallback_routing():
    """12. General health queries without explicit explanation or summary keywords default safely."""
    q = "Aerobic physical activity and vascular health"
    decision = OrchestratorAgent.classify_intent(q)
    assert decision.intent in ("RESEARCH_QUESTION", "GENERAL_HEALTH_INFORMATION")
    assert decision.agent == "research_agent"
    print("[PASS] test_general_health_fallback_routing passed.")


if __name__ == "__main__":
    print("Running OrchestratorAgent Unit Tests...")
    test_document_summary_routing()
    test_medical_explanation_routing()
    test_research_question_routing()
    test_safety_sensitive_routing()
    test_general_health_fallback_routing()
    print("\n[SUCCESS] All 5 OrchestratorAgent unit tests passed successfully!")
