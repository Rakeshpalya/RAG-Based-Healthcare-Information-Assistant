from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


class SafetyCategory(str, Enum):
    """
    Standardized medical safety categories defined in Phase 4.1.
    Cover the complete taxonomy from emergency symptoms to informational inquiries.
    """
    NORMAL_MEDICAL_INFORMATION = "NORMAL_MEDICAL_INFORMATION"
    EMERGENCY_SYMPTOMS = "EMERGENCY_SYMPTOMS"
    DIAGNOSIS_REQUEST = "DIAGNOSIS_REQUEST"
    TREATMENT_REQUEST = "TREATMENT_REQUEST"
    MEDICATION_REQUEST = "MEDICATION_REQUEST"
    DOSAGE_REQUEST = "DOSAGE_REQUEST"
    DRUG_INTERACTION_REQUEST = "DRUG_INTERACTION_REQUEST"
    CONTRAINDICATION_REQUEST = "CONTRAINDICATION_REQUEST"
    SELF_HARM_OR_SUICIDE = "SELF_HARM_OR_SUICIDE"
    POISONING_OR_OVERDOSE = "POISONING_OR_OVERDOSE"
    PREGNANCY_HIGH_RISK = "PREGNANCY_HIGH_RISK"
    PEDIATRIC_HIGH_RISK = "PEDIATRIC_HIGH_RISK"
    IMMEDIATE_DANGER = "IMMEDIATE_DANGER"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNSAFE_OR_UNSUPPORTED_REQUEST = "UNSAFE_OR_UNSUPPORTED_REQUEST"


class RiskLevel(str, Enum):
    """Clinical risk classification levels for safety routing."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class SafetyAssessment:
    """
    Comprehensive structured assessment produced by the Phase 4 safety engine.
    
    Attributes:
        category: Primary SafetyCategory identified.
        risk_level: Risk classification (CRITICAL, HIGH, MEDIUM, LOW, INFO).
        requires_escalation: True if inquiry requires clinical evaluation or emergency attention.
        allow_normal_rag: True if query is permitted to invoke retrieval and LLM synthesis.
        allow_medication_information: True if general evidence about medicines can be presented.
        allow_dosage_information: True if general evidence about dosages can be presented.
        emergency_message: Immediate advisory if query involves critical emergency/harm.
        reason: Clinical explanation of the decision.
        matched_rules: Specific deterministic rule IDs matched during evaluation.
        guidance_message: Targeted disclaimer or advisory appended to responses.
    """
    category: SafetyCategory
    risk_level: str
    requires_escalation: bool
    allow_normal_rag: bool
    allow_medication_information: bool = True
    allow_dosage_information: bool = True
    emergency_message: Optional[str] = None
    reason: str = ""
    matched_rules: List[str] = field(default_factory=list)
    guidance_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value if isinstance(self.category, Enum) else str(self.category),
            "risk_level": self.risk_level.value if isinstance(self.risk_level, Enum) else str(self.risk_level),
            "requires_escalation": self.requires_escalation,
            "allow_normal_rag": self.allow_normal_rag,
            "allow_medication_information": self.allow_medication_information,
            "allow_dosage_information": self.allow_dosage_information,
            "emergency_message": self.emergency_message,
            "reason": self.reason,
            "matched_rules": self.matched_rules,
            "guidance_message": self.guidance_message
        }
