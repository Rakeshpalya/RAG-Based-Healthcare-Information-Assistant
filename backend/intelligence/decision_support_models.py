"""
Clinical Decision Support, Uncertainty Calibration & Care Pathway Data Models (Phase 6.7).

Defines strongly typed, JSON-serializable dataclasses and enums for:
1. Clinical uncertainty level & source categorization (Aleatoric vs Epistemic)
2. Clinical risk stratification tiers (Minimal, Low, Moderate, High, Critical)
3. Actionable guideline-concordant recommendations
4. Clinical red flag warnings & escalation triggers
5. Agentic follow-up inquiries
6. Provider SBAR clinical handoff summaries
7. Composite decision support results
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class ClinicalUncertaintyLevel(str, Enum):
    """Calibrated level of clinical uncertainty for synthesized guidance."""
    LOW = "LOW"                      # Well-corroborated by guidelines, high similarity, zero conflicts
    MODERATE = "MODERATE"            # Minor evidence gaps or medium similarity without contradiction
    HIGH = "HIGH"                    # Document conflicts, ungrounded claims, or low similarity
    INDETERMINATE = "INDETERMINATE"  # No relevant reference evidence available or fallback triggered


class ClinicalRiskTier(str, Enum):
    """Clinical risk stratification based on clinical intent, safety category, and evidence."""
    MINIMAL = "MINIMAL"    # General health education, prevention, or document summary
    LOW = "LOW"            # Stable chronic health management, lifestyle education
    MODERATE = "MODERATE"  # Symptom evaluation, diagnostic staging, lab interpretations
    HIGH = "HIGH"          # Pharmacotherapy changes, dosage modifications, contraindications
    CRITICAL = "CRITICAL"  # Acute emergency, red-flag symptoms, poisoning, self-harm, or severe contradictions


class UncertaintySourceType(str, Enum):
    """Classification of root cause of clinical uncertainty."""
    ALEATORIC = "ALEATORIC"                        # Inherent variability or conflicting medical guidelines
    EPISTEMIC = "EPISTEMIC"                        # Missing literature or insufficient retrieval coverage
    VERIFICATION_DEFICIT = "VERIFICATION_DEFICIT"  # Claims pruned or ungrounded during synthesis
    NONE = "NONE"                                  # Highly certain, negligible uncertainty


class ActionRecommendationType(str, Enum):
    """Functional category of actionable clinical recommendation."""
    LIFESTYLE_MODIFICATION = "LIFESTYLE_MODIFICATION"
    DIAGNOSTIC_MONITORING = "DIAGNOSTIC_MONITORING"
    CLINICIAN_CONSULTATION = "CLINICIAN_CONSULTATION"
    MEDICATION_REVIEW = "MEDICATION_REVIEW"
    EMERGENCY_ESCALATION = "EMERGENCY_ESCALATION"
    PREVENTIVE_MEASURE = "PREVENTIVE_MEASURE"


@dataclass
class ActionableRecommendation:
    """Guideline-concordant non-prescriptive action item for patient or clinician."""
    recommendation_id: str
    category: ActionRecommendationType
    text: str
    urgency: str = "ROUTINE"  # ROUTINE, PROMPT, IMMEDIATE
    evidence_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "category": self.category.value if hasattr(self.category, "value") else str(self.category),
            "text": self.text,
            "urgency": self.urgency,
            "evidence_source": self.evidence_source
        }


@dataclass
class RedFlagTrigger:
    """Clinical warning sign requiring immediate emergency or prompt physician escalation."""
    flag_id: str
    symptom_or_sign: str
    clinical_rationale: str
    action_required: str
    is_critical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flag_id": self.flag_id,
            "symptom_or_sign": self.symptom_or_sign,
            "clinical_rationale": self.clinical_rationale,
            "action_required": self.action_required,
            "is_critical": bool(self.is_critical)
        }


@dataclass
class ClinicalHandoffSummary:
    """Structured SBAR (Situation, Background, Assessment, Recommendation) provider handoff."""
    situation: str
    background: str
    assessment: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "situation": self.situation,
            "background": self.background,
            "assessment": self.assessment,
            "recommendation": self.recommendation
        }


@dataclass
class ClinicalDecisionSupportResult:
    """Comprehensive decision support, uncertainty calibration, and care pathway result."""
    calibrated_confidence: float
    uncertainty_level: ClinicalUncertaintyLevel
    primary_uncertainty_source: UncertaintySourceType
    risk_tier: ClinicalRiskTier
    escalation_required: bool
    escalation_reason: Optional[str]
    actionable_recommendations: List[ActionableRecommendation] = field(default_factory=list)
    red_flag_triggers: List[RedFlagTrigger] = field(default_factory=list)
    suggested_follow_up_inquiries: List[str] = field(default_factory=list)
    clinical_handoff: Optional[ClinicalHandoffSummary] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibrated_confidence": round(float(self.calibrated_confidence), 4),
            "uncertainty_level": self.uncertainty_level.value if hasattr(self.uncertainty_level, "value") else str(self.uncertainty_level),
            "primary_uncertainty_source": self.primary_uncertainty_source.value if hasattr(self.primary_uncertainty_source, "value") else str(self.primary_uncertainty_source),
            "risk_tier": self.risk_tier.value if hasattr(self.risk_tier, "value") else str(self.risk_tier),
            "escalation_required": bool(self.escalation_required),
            "escalation_reason": self.escalation_reason,
            "actionable_recommendations": [r.to_dict() for r in self.actionable_recommendations],
            "red_flag_triggers": [f.to_dict() for f in self.red_flag_triggers],
            "suggested_follow_up_inquiries": list(self.suggested_follow_up_inquiries),
            "clinical_handoff": self.clinical_handoff.to_dict() if self.clinical_handoff else None,
            "latency_ms": round(float(self.latency_ms), 3),
            "metadata": dict(self.metadata)
        }
