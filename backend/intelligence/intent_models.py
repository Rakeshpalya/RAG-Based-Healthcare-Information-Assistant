"""
Clinical Intent Models & Taxonomy for AI-Healthcare-Agent (Phase 6.1).

Defines structured clinical intents, routing strategies, safety priority levels,
and classification result schemas for intelligent query routing.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


class ClinicalIntent(str, Enum):
    """
    Standardized clinical intent taxonomy for healthcare RAG.
    """
    # Critical Safety & Emergency Intents
    EMERGENCY = "EMERGENCY"
    SELF_HARM = "SELF_HARM"
    POISONING = "POISONING"

    # Clinical Consultation & Inquiry Intents
    SYMPTOM_QUERY = "SYMPTOM_QUERY"
    DIAGNOSIS_QUERY = "DIAGNOSIS_QUERY"
    MEDICATION_QUERY = "MEDICATION_QUERY"
    DOSAGE_QUERY = "DOSAGE_QUERY"
    LAB_RESULT_QUERY = "LAB_RESULT_QUERY"
    TREATMENT_QUERY = "TREATMENT_QUERY"
    PREVENTION_QUERY = "PREVENTION_QUERY"
    GENERAL_HEALTH = "GENERAL_HEALTH"

    # Document-Centric Intents
    DOCUMENT_SUMMARY = "DOCUMENT_SUMMARY"
    DOCUMENT_COMPARISON = "DOCUMENT_COMPARISON"

    # Non-Clinical / Boundary Intents
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNCERTAIN = "UNCERTAIN"


class SafetyPriority(str, Enum):
    """Safety priority level directing pipeline interception order."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    NONE = "none"


class ClinicalRoutingStrategy(str, Enum):
    """Routing strategies for downstream retrieval and processing."""
    # Safety Interceptions
    EMERGENCY_SAFETY = "emergency_safety"
    SELF_HARM_SAFETY = "self_harm_safety"
    POISONING_SAFETY = "poisoning_safety"

    # Specialized RAG Strategies
    MEDICATION_RAG = "medication_rag"
    DOSAGE_RAG = "dosage_rag"
    LAB_RAG = "lab_rag"
    TREATMENT_RAG = "treatment_rag"
    PREVENTION_RAG = "prevention_rag"
    SYMPTOM_RAG = "symptom_rag"
    DIAGNOSIS_RAG = "diagnosis_rag"
    GENERAL_HEALTH_RAG = "general_health_rag"

    # Document Scoped Strategies
    DOCUMENT_SUMMARY_RAG = "document_summary_rag"
    DOCUMENT_COMPARISON_RAG = "document_comparison_rag"

    # Fallback / Boundary Strategies
    OUT_OF_SCOPE_RESPONSE = "out_of_scope_response"
    STANDARD_RAG = "standard_rag"


@dataclass
class IntentClassificationResult:
    """
    Structured outcome of the clinical intent classification process.
    """
    intent: ClinicalIntent
    confidence: float
    matched_signals: List[str] = field(default_factory=list)
    requires_retrieval: bool = True
    requires_document_context: bool = False
    safety_priority: SafetyPriority = SafetyPriority.NORMAL
    routing_strategy: ClinicalRoutingStrategy = ClinicalRoutingStrategy.STANDARD_RAG
    latency_ms: float = 0.0
    recommended_top_k: int = 5
    recommended_similarity_threshold: float = 0.25
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the result to a JSON-compatible dictionary."""
        return {
            "intent": self.intent.value if isinstance(self.intent, Enum) else str(self.intent),
            "confidence": round(float(self.confidence), 3),
            "matched_signals": list(self.matched_signals),
            "requires_retrieval": bool(self.requires_retrieval),
            "requires_document_context": bool(self.requires_document_context),
            "safety_priority": self.safety_priority.value if isinstance(self.safety_priority, Enum) else str(self.safety_priority),
            "routing_strategy": self.routing_strategy.value if isinstance(self.routing_strategy, Enum) else str(self.routing_strategy),
            "latency_ms": round(float(self.latency_ms), 3),
            "recommended_top_k": int(self.recommended_top_k),
            "recommended_similarity_threshold": float(self.recommended_similarity_threshold),
            "metadata": dict(self.metadata)
        }
