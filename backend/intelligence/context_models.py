"""
Strongly Typed Data Models for Longitudinal Clinical Context & Multi-Turn Interaction Memory (Phase 6.9).

Provides structured schemas for clinical entity tracking across conversation turns,
cumulative patient profile aggregation, contextual query reformulation, and dialogue state auditing.
"""

from enum import Enum
import hashlib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ClinicalEntityType(str, Enum):
    """Enumeration of clinical entity categories."""
    CONDITION = "CONDITION"
    SYMPTOM = "SYMPTOM"
    MEDICATION = "MEDICATION"
    ALLERGY = "ALLERGY"
    DEMOGRAPHIC = "DEMOGRAPHIC"
    RISK_FACTOR = "RISK_FACTOR"


class EntityTemporalState(str, Enum):
    """Temporal or certainty qualification of an extracted entity."""
    ACTIVE = "ACTIVE"
    HISTORICAL = "HISTORICAL"
    NEGATED = "NEGATED"
    SUSPECTED = "SUSPECTED"


class ClinicalEntity(BaseModel):
    """Individual clinical entity extracted from conversational dialogue."""
    name: str
    entity_type: ClinicalEntityType
    temporal_state: EntityTemporalState = EntityTemporalState.ACTIVE
    turn_index: int = 0
    turn_introduced: int = 0
    negated: bool = False
    source_text: str = ""
    confidence: float = 1.0
    entity_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        eid = self.entity_id or f"ent_{self.name.lower().replace(' ', '_')}"
        return {
            "entity_id": eid,
            "name": self.name,
            "entity_type": self.entity_type.value if hasattr(self.entity_type, "value") else str(self.entity_type),
            "temporal_state": self.temporal_state.value if hasattr(self.temporal_state, "value") else str(self.temporal_state),
            "turn_index": self.turn_index,
            "turn_introduced": self.turn_introduced or self.turn_index,
            "negated": self.negated,
            "source_text": self.source_text,
            "confidence": round(self.confidence, 3)
        }


class ContraindicationAlert(BaseModel):
    """Specific contraindication alert evaluated from cumulative profile."""
    contraindication_id: str
    reason: str
    severity: str = "HIGH"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contraindication_id": self.contraindication_id,
            "reason": self.reason,
            "severity": self.severity
        }


class CumulativeClinicalProfile(BaseModel):
    """Aggregated clinical profile accumulated across conversational turns."""
    active_conditions: List[str] = Field(default_factory=list)
    active_symptoms: List[str] = Field(default_factory=list)
    active_medications: List[str] = Field(default_factory=list)
    confirmed_allergies: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    demographics: Dict[str, Any] = Field(default_factory=dict)
    entities: List[ClinicalEntity] = Field(default_factory=list)
    total_entities_extracted: int = 0
    profile_hash: str = ""

    def add_entity(self, entity: ClinicalEntity) -> None:
        self.entities.append(entity)
        name = entity.name
        if entity.entity_type == ClinicalEntityType.CONDITION:
            if entity.negated:
                if name in self.active_conditions:
                    self.active_conditions.remove(name)
            elif name not in self.active_conditions:
                self.active_conditions.append(name)
        elif entity.entity_type == ClinicalEntityType.MEDICATION:
            if entity.negated:
                if name in self.active_medications:
                    self.active_medications.remove(name)
            elif name not in self.active_medications:
                self.active_medications.append(name)
        elif entity.entity_type == ClinicalEntityType.ALLERGY:
            if name not in self.confirmed_allergies:
                self.confirmed_allergies.append(name)
            if name in self.active_medications:
                self.active_medications.remove(name)
        elif entity.entity_type == ClinicalEntityType.RISK_FACTOR:
            if name not in self.risk_factors:
                self.risk_factors.append(name)
        self.total_entities_extracted = len(self.entities)
        hash_seed = f"C:{'|'.join(sorted(self.active_conditions))};M:{'|'.join(sorted(self.active_medications))};A:{'|'.join(sorted(self.confirmed_allergies))};R:{'|'.join(sorted(self.risk_factors))}"
        self.profile_hash = hashlib.sha256(hash_seed.encode("utf-8")).hexdigest()[:16]

    def get_summary(self) -> Dict[str, Any]:
        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_conditions": list(self.active_conditions),
            "active_symptoms": list(self.active_symptoms),
            "active_medications": list(self.active_medications),
            "confirmed_allergies": list(self.confirmed_allergies),
            "risk_factors": list(self.risk_factors),
            "demographics": dict(self.demographics),
            "entities": [e.to_dict() for e in self.entities],
            "total_entities_extracted": self.total_entities_extracted,
            "profile_hash": self.profile_hash
        }


class DialogueStateAuditRecord(BaseModel):
    """Audit record capturing dialogue context state for Phase 6.8 orchestration inclusion."""
    turn_count: int = 0
    prior_turn_count: int = 0
    is_follow_up: bool = False
    resolved_topic: Optional[str] = None
    entities_count: int = 0
    active_conditions_count: int = 0
    active_medications_count: int = 0
    allergies_count: int = 0
    profile_hash: str = ""
    resolution_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_count": self.turn_count or self.prior_turn_count,
            "prior_turn_count": self.prior_turn_count or self.turn_count,
            "is_follow_up": self.is_follow_up,
            "resolved_topic": self.resolved_topic,
            "entities_count": self.entities_count,
            "active_conditions_count": self.active_conditions_count,
            "active_medications_count": self.active_medications_count,
            "allergies_count": self.allergies_count,
            "profile_hash": self.profile_hash,
            "resolution_latency_ms": round(self.resolution_latency_ms, 3)
        }


class TurnContextResolution(BaseModel):
    """Complete outcome of the Phase 6.9 longitudinal context resolution stage."""
    original_query: str
    effective_query: str
    is_follow_up: bool = False
    resolved_topic: Optional[str] = None
    prior_turn_count: int = 0
    cumulative_profile: CumulativeClinicalProfile = Field(default_factory=CumulativeClinicalProfile)
    inherited_contraindications: List[str] = Field(default_factory=list)
    contraindication_alerts: List[ContraindicationAlert] = Field(default_factory=list)
    resolution_confidence: float = 1.0
    latency_ms: float = 0.0

    @property
    def resolution_latency_ms(self) -> float:
        return self.latency_ms

    @resolution_latency_ms.setter
    def resolution_latency_ms(self, val: float) -> None:
        self.latency_ms = val

    def to_audit_record(self) -> DialogueStateAuditRecord:
        """Converts resolution into a lightweight audit record for Phase 6.8 orchestrator."""
        return DialogueStateAuditRecord(
            turn_count=self.prior_turn_count,
            prior_turn_count=self.prior_turn_count,
            is_follow_up=self.is_follow_up,
            resolved_topic=self.resolved_topic,
            entities_count=self.cumulative_profile.total_entities_extracted,
            active_conditions_count=len(self.cumulative_profile.active_conditions),
            active_medications_count=len(self.cumulative_profile.active_medications),
            allergies_count=len(self.cumulative_profile.confirmed_allergies),
            profile_hash=self.cumulative_profile.profile_hash,
            resolution_latency_ms=self.latency_ms
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "effective_query": self.effective_query,
            "is_follow_up": self.is_follow_up,
            "resolved_topic": self.resolved_topic,
            "prior_turn_count": self.prior_turn_count,
            "cumulative_profile": self.cumulative_profile.to_dict(),
            "inherited_contraindications": list(self.inherited_contraindications),
            "contraindication_alerts": [a.to_dict() for a in self.contraindication_alerts],
            "resolution_confidence": round(self.resolution_confidence, 3),
            "latency_ms": round(self.latency_ms, 3)
        }
