"""
Clinical Evidence Fusion Data Models for AI-Healthcare-Agent (Phase 6.3).

Defines strongly typed data structures for multi-document evidence aggregation,
conflict detection, evidence coverage evaluation, and source attribution.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


class ConflictType(str, Enum):
    """Categories of clinical contradictions across evidence sources."""
    NONE = "none"
    DOSAGE_DISCREPANCY = "dosage_discrepancy"
    CONTRAINDICATION_CONFLICT = "contraindication_conflict"
    TREATMENT_RECOMMENDATION_CONFLICT = "treatment_recommendation_conflict"
    DIAGNOSTIC_CRITERIA_CONFLICT = "diagnostic_criteria_conflict"
    CLINICAL_OUTCOME_CONFLICT = "clinical_outcome_conflict"


class ConflictSeverity(str, Enum):
    """Clinical risk severity of detected contradictions."""
    LOW = "low"            # Minor phrasing or guideline variation
    MODERATE = "moderate"  # Differing recommendations or dosing schedules
    HIGH = "high"          # Direct safety contradiction (e.g. contraindicated vs recommended)


class CoverageStatus(str, Enum):
    """Completeness of retrieved evidence against user inquiry requirements."""
    FULL = "full"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


@dataclass
class EvidenceConflict:
    """
    Represents a detected clinical contradiction between two evidence documents.
    """
    topic: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    doc_a_id: str
    doc_b_id: str
    doc_a_name: str
    doc_b_name: str
    statement_a: str
    statement_b: str
    resolution_guidance: str

    def to_dict(self) -> Dict[str, Any]:
        """Serializes conflict record to dictionary."""
        return {
            "topic": self.topic,
            "conflict_type": self.conflict_type.value if isinstance(self.conflict_type, Enum) else str(self.conflict_type),
            "severity": self.severity.value if isinstance(self.severity, Enum) else str(self.severity),
            "doc_a_id": str(self.doc_a_id),
            "doc_b_id": str(self.doc_b_id),
            "doc_a_name": str(self.doc_a_name),
            "doc_b_name": str(self.doc_b_name),
            "statement_a": str(self.statement_a),
            "statement_b": str(self.statement_b),
            "resolution_guidance": str(self.resolution_guidance)
        }


@dataclass
class EvidenceCoverage:
    """
    Evaluates whether retrieved evidence covers all required clinical aspects of the query.
    """
    coverage_score: float
    status: CoverageStatus
    query_aspects: List[str] = field(default_factory=list)
    covered_aspects: List[str] = field(default_factory=list)
    missing_aspects: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes coverage assessment to dictionary."""
        return {
            "coverage_score": round(float(self.coverage_score), 3),
            "status": self.status.value if isinstance(self.status, Enum) else str(self.status),
            "query_aspects": list(self.query_aspects),
            "covered_aspects": list(self.covered_aspects),
            "missing_aspects": list(self.missing_aspects)
        }


@dataclass
class FusedEvidenceChunk:
    """
    Represents a single fused evidence item with explicit provenance, ranking, and weighting.
    """
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int]
    text: str
    similarity_score: float
    weighting_boost: float
    fused_score: float
    rank: int
    source_index: int
    aspects: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes fused chunk to dictionary."""
        return {
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "document_name": str(self.document_name),
            "page_number": self.page_number,
            "text": str(self.text),
            "similarity_score": round(float(self.similarity_score), 4),
            "weighting_boost": round(float(self.weighting_boost), 4),
            "fused_score": round(float(self.fused_score), 4),
            "rank": int(self.rank),
            "source_index": int(self.source_index),
            "aspects": list(self.aspects),
            "metadata": dict(self.metadata)
        }


@dataclass
class FusedContextResult:
    """
    Complete outcome of multi-document evidence fusion and clinical reasoning.
    """
    fused_chunks: List[FusedEvidenceChunk]
    contributing_documents: List[str]
    contributing_documents_count: int
    conflicts: List[EvidenceConflict]
    has_conflicts: bool
    coverage: EvidenceCoverage
    formatted_context: str
    sources: List[Dict[str, Any]]
    deduped_count: int
    is_sufficient: bool
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes complete fusion result to dictionary."""
        return {
            "fused_chunks": [c.to_dict() for c in self.fused_chunks],
            "contributing_documents": list(self.contributing_documents),
            "contributing_documents_count": int(self.contributing_documents_count),
            "conflicts": [c.to_dict() for c in self.conflicts],
            "has_conflicts": bool(self.has_conflicts),
            "coverage": self.coverage.to_dict(),
            "formatted_context": str(self.formatted_context),
            "sources": list(self.sources),
            "deduped_count": int(self.deduped_count),
            "is_sufficient": bool(self.is_sufficient),
            "latency_ms": round(float(self.latency_ms), 3),
            "metadata": dict(self.metadata)
        }
