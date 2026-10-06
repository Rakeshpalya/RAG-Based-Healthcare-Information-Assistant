"""
Strongly Typed Data Models for Clinical Intelligence Orchestration & Auditability (Phase 6.8).

Provides structured schemas for pipeline stage auditing, execution provenance,
safety guardianship verification, data isolation tracking, and cryptographic audit checksums.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone


class PipelineStage(str, Enum):
    """Enumeration of clinical reasoning pipeline stages."""
    SAFETY_PRE_SCREEN = "SAFETY_PRE_SCREEN"
    DIALOGUE_CONTEXT = "DIALOGUE_CONTEXT"
    INTENT_CLASSIFICATION = "INTENT_CLASSIFICATION"
    QUERY_PLANNING = "QUERY_PLANNING"
    EVIDENCE_FUSION = "EVIDENCE_FUSION"
    ANSWER_SYNTHESIS = "ANSWER_SYNTHESIS"
    CITATION_ATTRIBUTION = "CITATION_ATTRIBUTION"
    CLINICAL_VERIFICATION = "CLINICAL_VERIFICATION"
    DECISION_SUPPORT = "DECISION_SUPPORT"
    AUDIT_ORCHESTRATION = "AUDIT_ORCHESTRATION"


class StageExecutionStatus(str, Enum):
    """Execution status for a pipeline stage."""
    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    DEGRADED = "DEGRADED"
    INTERCEPTED = "INTERCEPTED"
    FAILED = "FAILED"


@dataclass
class StageAuditRecord:
    """Individual stage audit record detailing execution health and flags."""
    stage: PipelineStage
    status: StageExecutionStatus
    execution_latency_ms: float = 0.0
    summary: str = ""
    flags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value if hasattr(self.stage, "value") else str(self.stage),
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "execution_latency_ms": round(self.execution_latency_ms, 3),
            "summary": self.summary,
            "flags": self.flags
        }


@dataclass
class ClinicalSafetyProvenance:
    """Consolidated safety posture across pre-screen, verification, and decision support."""
    emergency_intercepted: bool = False
    self_harm_intercepted: bool = False
    poisoning_intercepted: bool = False
    requires_escalation: bool = False
    risk_tier: str = "MINIMAL"
    uncertainty_level: str = "LOW"
    red_flags_count: int = 0
    contradictions_detected: int = 0
    citations_verified: bool = True
    grounding_verified: bool = True
    safety_assessment_category: str = "GENERAL_INQUIRY"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataIsolationProvenance:
    """Multi-tenant isolation and user boundary provenance."""
    tenant_id: Optional[str] = None
    user_id: Optional[int] = None
    documents_accessed_count: int = 0
    chunks_accessed_count: int = 0
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    cross_tenant_contamination_check_passed: bool = True
    cache_isolated: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClinicalIntelligenceOrchestrationResult:
    """Root structured outcome of Phase 6.8 Clinical Intelligence Orchestration."""
    trace_id: str
    timestamp_iso: str
    pipeline_stages: List[StageAuditRecord]
    stage_status_map: Dict[str, str]
    safety_provenance: ClinicalSafetyProvenance
    isolation_provenance: DataIsolationProvenance
    audit_checksum: str
    is_concordant: bool
    orchestration_latency_ms: float
    total_pipeline_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp_iso": self.timestamp_iso,
            "pipeline_stages": [s.to_dict() for s in self.pipeline_stages],
            "stage_status_map": self.stage_status_map,
            "safety_provenance": self.safety_provenance.to_dict(),
            "isolation_provenance": self.isolation_provenance.to_dict(),
            "audit_checksum": self.audit_checksum,
            "is_concordant": self.is_concordant,
            "orchestration_latency_ms": round(self.orchestration_latency_ms, 3),
            "total_pipeline_latency_ms": round(self.total_pipeline_latency_ms, 3)
        }
