from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SafetyDecision:
    """Structured decision produced by the SafetyAgent."""
    category: str  # SAFE_INFORMATIONAL, DIAGNOSIS_REQUEST, TREATMENT_REQUEST, MEDICATION_REQUEST, EMERGENCY_SYMPTOM, EMPTY_QUERY, OTHER_SENSITIVE
    allowed: bool
    requires_clinician: bool = False
    requires_emergency_guidance: bool = False
    reason: str = ""
    fallback_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "allowed": self.allowed,
            "requires_clinician": self.requires_clinician,
            "requires_emergency_guidance": self.requires_emergency_guidance,
            "reason": self.reason
        }


@dataclass
class OrchestratorDecision:
    """Structured routing decision produced by OrchestratorAgent."""
    intent: str  # DOCUMENT_SUMMARY, MEDICAL_EXPLANATION, RESEARCH_QUESTION, GENERAL_HEALTH_INFORMATION, SAFETY_SENSITIVE
    agent: str   # document_agent, explanation_agent, research_agent, safety_agent
    confidence: float = 1.0
    requires_retrieval: bool = True
    requires_safety_check: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "agent": self.agent,
            "confidence": self.confidence,
            "requires_retrieval": self.requires_retrieval,
            "requires_safety_check": self.requires_safety_check
        }


@dataclass
class AgentState:
    """
    Immutable structured state object tracking the execution lifecycle of a query
    across the multi-agent orchestration pipeline.
    """
    user_question: str
    explanation_level: Optional[str] = "simple"  # simple, intermediate, technical
    top_k: int = 5
    safety_decision: Optional[SafetyDecision] = None
    orchestrator_decision: Optional[OrchestratorDecision] = None
    retrieval_status: str = "pending"
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    context: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    raw_answer: Optional[str] = None
    final_answer: Optional[str] = None
    citations: Optional[Dict[str, Any]] = None
    timings: Dict[str, float] = field(default_factory=dict)
    disclaimer: str = ""

    def to_response_dict(self) -> Dict[str, Any]:
        """Formats the final state into a client-ready API response."""
        return {
            "question": self.user_question,
            "intent": self.orchestrator_decision.intent if self.orchestrator_decision else "UNKNOWN",
            "agent": self.orchestrator_decision.agent if self.orchestrator_decision else "safety_agent",
            "safety": self.safety_decision.to_dict() if self.safety_decision else {
                "category": "UNKNOWN",
                "allowed": False,
                "requires_clinician": False,
                "requires_emergency_guidance": False,
                "reason": "Safety check uninitialized"
            },
            "answer": self.final_answer or "",
            "sources": self.sources,
            "citations": self.citations or {},
            "timings": self.timings,
            "disclaimer": self.disclaimer
        }
