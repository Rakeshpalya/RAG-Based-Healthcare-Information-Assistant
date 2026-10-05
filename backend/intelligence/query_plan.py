"""
Clinical Query Plan Data Models for AI-Healthcare-Agent (Phase 6.2).

Defines strongly typed QueryPlan data models, weighting strategies,
document aggregation modes, and execution parameters for adaptive RAG.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from backend.intelligence.intent_models import (
    ClinicalIntent,
    SafetyPriority,
    ClinicalRoutingStrategy
)


class ChunkWeightingStrategy(str, Enum):
    """Strategies for reranking and prioritizing retrieved context chunks."""
    STANDARD = "standard"
    MEDICATION_PRIORITY = "medication_priority"
    DOSAGE_PRIORITY = "dosage_priority"
    LAB_PRIORITY = "lab_priority"
    DIAGNOSTIC_PRIORITY = "diagnostic_priority"
    SYMPTOM_PRIORITY = "symptom_priority"
    TREATMENT_PRIORITY = "treatment_priority"
    PREVENTION_PRIORITY = "prevention_priority"
    BROAD_COVERAGE = "broad_coverage"
    BALANCED_MULTI_DOCUMENT = "balanced_multi_document"


class DocumentFilterStrategy(str, Enum):
    """Document scoping and filtering modes."""
    ALL_AVAILABLE = "all_available"
    USER_DOCUMENTS_ONLY = "user_documents_only"
    SCOPED_DOCUMENT = "scoped_document"
    EXCLUDE_NON_CLINICAL = "exclude_non_clinical"


class MultiDocumentStrategy(str, Enum):
    """Multi-document aggregation strategies."""
    SINGLE_DOCUMENT = "single_document"
    MULTI_DOCUMENT = "multi_document"
    MAP_REDUCE = "map_reduce"
    BALANCED_DOCUMENT_RETRIEVAL = "balanced_document_retrieval"


class GenerationStrategy(str, Enum):
    """Downstream generative synthesis directives."""
    STANDARD_GROUNDED = "standard_grounded"
    HIGH_CONFIDENCE_GROUNDED = "high_confidence_grounded"
    MULTI_SOURCE_SYNTHESIS = "multi_source_synthesis"
    DOCUMENT_SUMMARY_SYNTHESIS = "document_summary_synthesis"
    COMPARATIVE_SYNTHESIS = "comparative_synthesis"
    SAFETY_REFUSAL = "safety_refusal"
    OUT_OF_SCOPE_REFUSAL = "out_of_scope_refusal"


@dataclass
class QueryPlan:
    """
    Strongly typed execution plan for clinical query processing.
    """
    intent: ClinicalIntent
    retrieval_required: bool
    retrieval_strategy: ClinicalRoutingStrategy
    top_k: int = 5
    similarity_threshold: float = 0.25
    max_chunks: int = 8
    chunk_weighting_strategy: ChunkWeightingStrategy = ChunkWeightingStrategy.STANDARD
    document_filter_strategy: DocumentFilterStrategy = DocumentFilterStrategy.ALL_AVAILABLE
    query_expansion_enabled: bool = False
    query_expansions: List[str] = field(default_factory=list)
    multi_document_strategy: MultiDocumentStrategy = MultiDocumentStrategy.MULTI_DOCUMENT
    context_budget: int = 3500
    requires_high_confidence_evidence: bool = False
    safety_priority: SafetyPriority = SafetyPriority.NORMAL
    generation_strategy: GenerationStrategy = GenerationStrategy.STANDARD_GROUNDED
    scoped_document_name: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes QueryPlan into a dictionary for JSON responses, logs, and caching."""
        return {
            "intent": self.intent.value if isinstance(self.intent, Enum) else str(self.intent),
            "retrieval_required": bool(self.retrieval_required),
            "retrieval_strategy": self.retrieval_strategy.value if isinstance(self.retrieval_strategy, Enum) else str(self.retrieval_strategy),
            "top_k": int(self.top_k),
            "similarity_threshold": float(self.similarity_threshold),
            "max_chunks": int(self.max_chunks),
            "chunk_weighting_strategy": self.chunk_weighting_strategy.value if isinstance(self.chunk_weighting_strategy, Enum) else str(self.chunk_weighting_strategy),
            "document_filter_strategy": self.document_filter_strategy.value if isinstance(self.document_filter_strategy, Enum) else str(self.document_filter_strategy),
            "query_expansion_enabled": bool(self.query_expansion_enabled),
            "query_expansions": list(self.query_expansions),
            "multi_document_strategy": self.multi_document_strategy.value if isinstance(self.multi_document_strategy, Enum) else str(self.multi_document_strategy),
            "context_budget": int(self.context_budget),
            "requires_high_confidence_evidence": bool(self.requires_high_confidence_evidence),
            "safety_priority": self.safety_priority.value if isinstance(self.safety_priority, Enum) else str(self.safety_priority),
            "generation_strategy": self.generation_strategy.value if isinstance(self.generation_strategy, Enum) else str(self.generation_strategy),
            "scoped_document_name": self.scoped_document_name,
            "latency_ms": round(float(self.latency_ms), 3),
            "metadata": dict(self.metadata)
        }
