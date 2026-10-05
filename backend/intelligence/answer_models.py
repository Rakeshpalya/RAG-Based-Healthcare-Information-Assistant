"""
Clinical Answer Synthesis Data Models for AI-Healthcare-Agent (Phase 6.4).

Defines strongly typed, JSON-serializable data models and enums for clinical answer
sections, confidence scoring, evidence support levels, and synthesis results.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from backend.intelligence.intent_models import ClinicalIntent


class AnswerSectionType(str, Enum):
    """Semantic types for structured clinical answer sections."""
    DIRECT_ANSWER = "DIRECT_ANSWER"
    KEY_POINTS = "KEY_POINTS"
    EVIDENCE = "EVIDENCE"
    DOSAGE_INFORMATION = "DOSAGE_INFORMATION"
    CONTRAINDICATIONS = "CONTRAINDICATIONS"
    WARNINGS = "WARNINGS"
    COMPARISON = "COMPARISON"
    DIAGNOSTIC_CONTEXT = "DIAGNOSTIC_CONTEXT"
    NEXT_STEPS = "NEXT_STEPS"
    SAFETY_NOTICE = "SAFETY_NOTICE"
    LIMITATIONS = "LIMITATIONS"


class AnswerConfidence(str, Enum):
    """Deterministic confidence level based on evidence quality and coverage."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


class EvidenceSupportLevel(str, Enum):
    """Categorical assessment of how well evidence supports the synthesized answer."""
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"
    CONFLICTING = "CONFLICTING"


@dataclass
class ClinicalAnswerSection:
    """
    A single typed section within a structured clinical answer.
    """
    section_type: AnswerSectionType
    title: str
    content: str
    source_indices: List[int] = field(default_factory=list)
    is_limitation_or_warning: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes section to a JSON-compatible dictionary."""
        return {
            "section_type": self.section_type.value if isinstance(self.section_type, Enum) else str(self.section_type),
            "title": str(self.title),
            "content": str(self.content),
            "source_indices": [int(idx) for idx in self.source_indices],
            "is_limitation_or_warning": bool(self.is_limitation_or_warning)
        }


@dataclass
class ClinicalAnswer:
    """
    Structured clinical answer consisting of an assembled text and individual typed sections.
    """
    text: str
    sections: List[ClinicalAnswerSection] = field(default_factory=list)
    citations: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes clinical answer to dictionary."""
        return {
            "text": str(self.text),
            "sections": [s.to_dict() for s in self.sections],
            "citations": [int(c) for c in self.citations]
        }


@dataclass
class AnswerSynthesisResult:
    """
    Complete outcome of the clinical answer synthesis process.
    Preserves intent, confidence, coverage status, conflict information, and citations.
    """
    intent: str
    confidence: AnswerConfidence
    support_level: EvidenceSupportLevel
    coverage_status: str
    answer: str
    sections: List[ClinicalAnswerSection] = field(default_factory=list)
    conflicts_present: bool = False
    conflict_summary: Optional[str] = None
    limitations_noted: bool = False
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    cited_sources: List[int] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes synthesis result to a JSON-compatible dictionary."""
        return {
            "intent": str(self.intent),
            "confidence": self.confidence.value if isinstance(self.confidence, Enum) else str(self.confidence),
            "support_level": self.support_level.value if isinstance(self.support_level, Enum) else str(self.support_level),
            "coverage_status": str(self.coverage_status),
            "answer": str(self.answer),
            "sections": [s.to_dict() for s in self.sections],
            "conflicts_present": bool(self.conflicts_present),
            "conflict_summary": str(self.conflict_summary) if self.conflict_summary else None,
            "limitations_noted": bool(self.limitations_noted),
            "is_fallback": bool(self.is_fallback),
            "fallback_reason": str(self.fallback_reason) if self.fallback_reason else None,
            "cited_sources": [int(s) for s in self.cited_sources],
            "latency_ms": round(float(self.latency_ms), 3),
            "metadata": dict(self.metadata)
        }
