"""
Clinical Grounding Verification & Hallucination Guardrail Models (Phase 6.6).
Defines strongly typed, JSON-serializable enums and dataclasses for post-synthesis
claim-level clinical verification, directional contradiction detection, entity grounding,
and clinical safety post-screening.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class GroundingVerificationStatus(str, Enum):
    """Grounding outcome for a clinical assertion evaluated against reference evidence."""
    GROUNDED = "GROUNDED"                            # Fully substantiated by cited reference passages
    PARTIALLY_GROUNDED = "PARTIALLY_GROUNDED"        # Supported with minor nuance or multiple valid sources
    UNGROUNDED = "UNGROUNDED"                        # Not substantiated by cited or available evidence
    DIRECTIONAL_CONTRADICTION = "DIRECTIONAL_CONTRADICTION"  # Clinical trajectory inverted (e.g. increase vs decrease)
    NUMERICAL_DISCREPANCY = "NUMERICAL_DISCREPANCY"  # Dosage, lab target, or measurement conflict
    UNSUBSTANTIATED_ENTITY = "UNSUBSTANTIATED_ENTITY"  # Novel medication or diagnosis not present in evidence
    PROHIBITED_DIRECTIVE = "PROHIBITED_DIRECTIVE"    # Unauthorized prescriptive or definitive diagnostic pronouncement
    EXEMPT_STRUCTURAL = "EXEMPT_STRUCTURAL"          # Section header, disclaimer, or non-factual transitional prose


class ClinicalHallucinationType(str, Enum):
    """Taxonomy of clinical hallucinations and medical inaccuracies."""
    NONE = "NONE"
    MEDICATION_FABRICATION = "MEDICATION_FABRICATION"          # Drug name not found in evidence
    DOSAGE_DISCREPANCY = "DOSAGE_DISCREPANCY"                  # Dose amount/unit deviates from evidence
    DIRECTIONAL_INVERSION = "DIRECTIONAL_INVERSION"            # Inversion of effect (e.g. lowers vs raises)
    NEGATION_CONFLICT = "NEGATION_CONFLICT"                    # Indicating a contraindicated intervention or vice versa
    UNGROUNDED_DIAGNOSTIC_CLAIM = "UNGROUNDED_DIAGNOSIS"       # Diagnostic assertion unsupported by source
    UNGROUNDED_RECOMMENDATION = "UNGROUNDED_RECOMMENDATION"    # Prescriptive recommendation unsupported by source
    OUT_OF_BOUNDS_EXTRAPOLATION = "OUT_OF_BOUNDS"              # Asserting facts outside document testing boundaries
    UNSUBSTANTIATED_ASSERTION = "UNSUBSTANTIATED_ASSERTION"    # General factual assertion lacking evidence backing


class SafetyPostScreenAction(str, Enum):
    """Action taken by the verification and safety guardrail engine."""
    ALLOW = "ALLOW"                                            # Answer verified safe and grounded
    SANITIZE_PRESCRIPTION = "SANITIZE_PRESCRIPTION"            # Prescriptive imperatives rewritten to informational framing
    SANITIZE_DIAGNOSIS = "SANITIZE_DIAGNOSIS"                  # Definitive diagnoses rewritten to exploratory clinical framing
    ATTACH_DISCLAIMER = "ATTACH_DISCLAIMER"                    # Mandatory clinical disclaimer attached
    PRUNE_UNSUPPORTED = "PRUNE_UNSUPPORTED"                    # Ungrounded claims pruned while retaining grounded core
    TRIGGER_FALLBACK = "TRIGGER_FALLBACK"                      # Complete fallback triggered due to severe hallucination/lack of support
    QUARANTINE = "QUARANTINE"                                  # Critical safety violation isolated


@dataclass
class ExtractedClinicalEntities:
    """Entities extracted from a clinical claim for grounding verification."""
    medications: List[str] = field(default_factory=list)
    dosages: List[str] = field(default_factory=list)
    diagnoses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    has_negation: bool = False
    negation_terms: List[str] = field(default_factory=list)
    direction: Optional[str] = None  # 'UP', 'DOWN', 'MIXED', or None
    direction_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "medications": self.medications,
            "dosages": self.dosages,
            "diagnoses": self.diagnoses,
            "recommendations": self.recommendations,
            "has_negation": self.has_negation,
            "negation_terms": self.negation_terms,
            "direction": self.direction,
            "direction_terms": self.direction_terms,
        }


@dataclass
class ClinicalVerificationClaim:
    """Detailed clinical grounding audit for an individual claim."""
    claim_id: str
    claim_text: str
    raw_sentence: str
    verification_status: GroundingVerificationStatus
    hallucination_type: ClinicalHallucinationType
    is_grounded: bool
    confidence_score: float = 0.0
    discrepancy_details: List[str] = field(default_factory=list)
    supporting_sources: List[int] = field(default_factory=list)
    entities: Optional[ExtractedClinicalEntities] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "raw_sentence": self.raw_sentence,
            "verification_status": self.verification_status.value if hasattr(self.verification_status, "value") else str(self.verification_status),
            "hallucination_type": self.hallucination_type.value if hasattr(self.hallucination_type, "value") else str(self.hallucination_type),
            "is_grounded": self.is_grounded,
            "confidence_score": round(self.confidence_score, 4),
            "discrepancy_details": self.discrepancy_details,
            "supporting_sources": self.supporting_sources,
            "entities": self.entities.to_dict() if self.entities else None,
        }


@dataclass
class ClinicalVerificationResult:
    """Comprehensive post-synthesis verification audit result."""
    is_verified_safe: bool
    overall_grounding_score: float
    total_claims_analyzed: int
    grounded_claims_count: int
    ungrounded_claims_count: int
    contradictions_count: int
    hallucinations_detected: int
    claim_verifications: List[ClinicalVerificationClaim]
    action_taken: SafetyPostScreenAction
    sanitized_answer: str
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    negative_boundary_enforced: bool = False
    disclaimer_enforced: bool = True
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_verified_safe": self.is_verified_safe,
            "overall_grounding_score": round(self.overall_grounding_score, 4),
            "total_claims_analyzed": self.total_claims_analyzed,
            "grounded_claims_count": self.grounded_claims_count,
            "ungrounded_claims_count": self.ungrounded_claims_count,
            "contradictions_count": self.contradictions_count,
            "hallucinations_detected": self.hallucinations_detected,
            "claim_verifications": [c.to_dict() for c in self.claim_verifications],
            "action_taken": self.action_taken.value if hasattr(self.action_taken, "value") else str(self.action_taken),
            "sanitized_answer": self.sanitized_answer,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason,
            "negative_boundary_enforced": self.negative_boundary_enforced,
            "disclaimer_enforced": self.disclaimer_enforced,
            "latency_ms": round(self.latency_ms, 3),
            "metadata": self.metadata,
        }
