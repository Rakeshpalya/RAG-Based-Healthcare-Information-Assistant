"""
Clinical Citation & Attribution Data Models for AI-Healthcare-Agent (Phase 6.5).

Defines strongly typed, JSON-serializable data models and enums for claim segmentation,
evidence provenance attribution, citation verification status, spoofing detection,
and attribution reporting.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


class CitationVerificationStatus(str, Enum):
    """Verification outcome of an inline citation against retrieved authoritative evidence."""
    VERIFIED = "VERIFIED"                      # Directly substantiated by cited document passage
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"  # Partially supported by cited source (some facets confirmed)
    UNSUPPORTED = "UNSUPPORTED"                # Cited source does not substantiate the clinical assertion
    INVALID_SOURCE = "INVALID_SOURCE"          # Cited index does not correspond to any retrieved source
    UNSPECIFIED_CITATION = "UNSPECIFIED"       # Factual assertion requires citation but none was provided
    STRUCTURAL_OR_DISCLAIMER = "STRUCTURAL"    # Heading, formatting, or disclaimer (no citation required)


class ClinicalClaimType(str, Enum):
    """Categorization of clinical assertions extracted from synthesized answers."""
    FACTUAL_MEDICAL = "FACTUAL_MEDICAL"                      # General medical pathophysiology, symptom, definition
    DOSAGE_INSTRUCTION = "DOSAGE_INSTRUCTION"                # Specific dosing quantity, schedule, or titration
    CONTRAINDICATION_OR_WARNING = "CONTRAINDICATION"         # Adverse effect, black-box warning, risk factor
    TREATMENT_RECOMMENDATION = "TREATMENT_RECOMMENDATION"    # Recommended clinical therapy or medication choice
    COMPARATIVE_CLAIM = "COMPARATIVE_CLAIM"                  # Head-to-head comparison between drugs or interventions
    LIMITATION_OR_DISCLAIMER = "LIMITATION_OR_DISCLAIMER"    # Explicit limitation statement or medical disclaimer
    STRUCTURAL = "STRUCTURAL"                                # Section title, bullet prefix, or transitional phrase


@dataclass
class AttributedEvidenceSpan:
    """
    Detailed evidence provenance linking a specific claim to a retrieved chunk.
    """
    source_index: int
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int]
    passage_snippet: str
    similarity_score: float
    support_score: float
    matched_entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes attributed evidence span to dictionary."""
        return {
            "source_index": int(self.source_index),
            "chunk_id": str(self.chunk_id),
            "document_id": str(self.document_id),
            "document_name": str(self.document_name),
            "page_number": int(self.page_number) if self.page_number is not None else None,
            "passage_snippet": str(self.passage_snippet),
            "similarity_score": round(float(self.similarity_score), 4),
            "support_score": round(float(self.support_score), 4),
            "matched_entities": list(self.matched_entities)
        }


@dataclass
class ClinicalClaimAttribution:
    """
    A segmented clinical assertion with evidence attribution and verification status.
    """
    claim_id: str
    claim_text: str
    raw_sentence: str
    claim_type: ClinicalClaimType
    cited_source_indices: List[int] = field(default_factory=list)
    verification_status: CitationVerificationStatus = CitationVerificationStatus.UNSPECIFIED_CITATION
    best_support_score: float = 0.0
    attributed_sources: List[AttributedEvidenceSpan] = field(default_factory=list)
    unsupported_reasons: List[str] = field(default_factory=list)
    is_supported: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes claim attribution record to dictionary."""
        return {
            "claim_id": str(self.claim_id),
            "claim_text": str(self.claim_text),
            "raw_sentence": str(self.raw_sentence),
            "claim_type": self.claim_type.value if isinstance(self.claim_type, Enum) else str(self.claim_type),
            "cited_source_indices": [int(idx) for idx in self.cited_source_indices],
            "verification_status": self.verification_status.value if isinstance(self.verification_status, Enum) else str(self.verification_status),
            "best_support_score": round(float(self.best_support_score), 4),
            "attributed_sources": [s.to_dict() for s in self.attributed_sources],
            "unsupported_reasons": list(self.unsupported_reasons),
            "is_supported": bool(self.is_supported)
        }


@dataclass
class CitationAttributionReport:
    """
    Comprehensive audit report of citation correctness, claim grounding,
    and anti-spoofing analysis for a clinical answer.
    """
    is_valid: bool
    claims: List[ClinicalClaimAttribution] = field(default_factory=list)
    total_claims_count: int = 0
    factual_claims_count: int = 0
    verified_claims_count: int = 0
    unsupported_claims_count: int = 0
    citations_found: List[int] = field(default_factory=list)
    valid_citations: List[int] = field(default_factory=list)
    invalid_citations: List[int] = field(default_factory=list)
    duplicate_citations: List[int] = field(default_factory=list)
    citation_precision: float = 1.0
    claim_attribution_coverage: float = 1.0
    spoofed_citations_detected: bool = False
    spoofed_citation_tags: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    cleaned_attributed_answer: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes citation attribution report to a JSON-compatible dictionary."""
        return {
            "is_valid": bool(self.is_valid),
            "claims": [c.to_dict() for c in self.claims],
            "total_claims_count": int(self.total_claims_count),
            "factual_claims_count": int(self.factual_claims_count),
            "verified_claims_count": int(self.verified_claims_count),
            "unsupported_claims_count": int(self.unsupported_claims_count),
            "citations_found": [int(c) for c in self.citations_found],
            "valid_citations": [int(c) for c in self.valid_citations],
            "invalid_citations": [int(c) for c in self.invalid_citations],
            "duplicate_citations": [int(c) for c in self.duplicate_citations],
            "citation_precision": round(float(self.citation_precision), 4),
            "claim_attribution_coverage": round(float(self.claim_attribution_coverage), 4),
            "spoofed_citations_detected": bool(self.spoofed_citations_detected),
            "spoofed_citation_tags": list(self.spoofed_citation_tags),
            "unsupported_claims": list(self.unsupported_claims),
            "cleaned_attributed_answer": str(self.cleaned_attributed_answer) if self.cleaned_attributed_answer is not None else None,
            "latency_ms": round(float(self.latency_ms), 3),
            "metadata": dict(self.metadata)
        }
