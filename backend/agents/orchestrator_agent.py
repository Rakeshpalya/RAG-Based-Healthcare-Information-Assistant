import re
from typing import Optional, Dict, Any

from backend.agents.agent_state import OrchestratorDecision, SafetyDecision
from backend.agents.safety_agent import SafetyAgent


class OrchestratorAgent:
    """
    Controlled Orchestrator Agent responsible for classifying user intent
    and routing inquiries to specialized downstream domain agents.

    DESIGN PRINCIPLES:
    - Bounded workflow routing; no arbitrary code execution or unrestricted loops.
    - Intent mapping connects user query patterns to specialized sub-agents.
    - Safety pre-check is unconditionally enforced before agent invocation.
    """

    # Keyword patterns for Intent Classification
    SUMMARY_PATTERNS = [
        r"\bsummariz\w*\b",
        r"\bsummary\b",
        r"\boverview\s+of\s+(?:this|the)\s+document\b",
        r"\bkey\s+findings\b",
        r"\bdigest\s+of\s+the\s+report\b",
        r"\bwhat\s+does\s+the\s+uploaded\s+document\s+say\b"
    ]

    EXPLANATION_PATTERNS = [
        r"\bexplain\b",
        r"\bin\s+simple\s+(?:terms|language|words)\b",
        r"\bin\s+plain\s+english\b",
        r"\bwhat\s+does\s+(?:this|that)\s+mean\b",
        r"\bsimplify\b",
        r"\bbreak\s+(?:this|it)\s+down\b",
        r"\bfor\s+a\s+patient\b",
        r"\bpatient-friendly\b"
    ]

    _re_summary = [re.compile(p, re.IGNORECASE) for p in SUMMARY_PATTERNS]
    _re_explanation = [re.compile(p, re.IGNORECASE) for p in EXPLANATION_PATTERNS]

    @classmethod
    def classify_intent(
        cls,
        user_question: str,
        safety_decision: Optional[SafetyDecision] = None
    ) -> OrchestratorDecision:
        """
        Determines the appropriate agent workflow for a given inquiry.

        Args:
            user_question: The patient or researcher's prompt.
            safety_decision: Optional pre-computed SafetyDecision. If not provided,
                             SafetyAgent.evaluate_pre_check is executed.

        Returns:
            OrchestratorDecision specifying the intent, assigned agent, and routing flags.
        """
        # 1. Evaluate safety if not already provided
        safety = safety_decision or SafetyAgent.evaluate_pre_check(user_question)

        # 2. If safety pre-check blocked the query, route immediately to safety_agent
        if not safety.allowed:
            return OrchestratorDecision(
                intent="SAFETY_SENSITIVE",
                agent="safety_agent",
                confidence=0.99,
                requires_retrieval=False,
                requires_safety_check=True
            )

        clean_q = (user_question or "").strip()

        # 3. Check for Document Summary Intent
        for pat in cls._re_summary:
            if pat.search(clean_q):
                return OrchestratorDecision(
                    intent="DOCUMENT_SUMMARY",
                    agent="document_agent",
                    confidence=0.92,
                    requires_retrieval=True,
                    requires_safety_check=True
                )

        # 4. Check for Medical Explanation Intent
        for pat in cls._re_explanation:
            if pat.search(clean_q):
                return OrchestratorDecision(
                    intent="MEDICAL_EXPLANATION",
                    agent="explanation_agent",
                    confidence=0.90,
                    requires_retrieval=True,
                    requires_safety_check=True
                )

        # 5. Check if query is a broad general health or research query
        if any(w in clean_q.lower() for w in ["what is", "how does", "mechanism", "pathology", "symptoms of", "guidelines"]):
            return OrchestratorDecision(
                intent="RESEARCH_QUESTION",
                agent="research_agent",
                confidence=0.88,
                requires_retrieval=True,
                requires_safety_check=True
            )

        # Default fallback intent: GENERAL_HEALTH_INFORMATION handled by research_agent
        return OrchestratorDecision(
            intent="GENERAL_HEALTH_INFORMATION",
            agent="research_agent",
            confidence=0.80,
            requires_retrieval=True,
            requires_safety_check=True
        )
