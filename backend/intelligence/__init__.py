"""
Clinical Intelligence & Agentic RAG Module (Phase 6).
Provides intent classification, query routing, and clinical reasoning abstractions.
"""

from backend.intelligence.intent_models import (
    ClinicalIntent,
    SafetyPriority,
    ClinicalRoutingStrategy,
    IntentClassificationResult
)
from backend.intelligence.intent_classifier import ClinicalIntentClassifier
from backend.intelligence.query_plan import (
    QueryPlan,
    ChunkWeightingStrategy,
    DocumentFilterStrategy,
    MultiDocumentStrategy,
    GenerationStrategy
)
from backend.intelligence.query_planner import ClinicalQueryPlanner
from backend.intelligence.evidence_models import (
    ConflictType,
    ConflictSeverity,
    CoverageStatus,
    EvidenceConflict,
    EvidenceCoverage,
    FusedEvidenceChunk,
    FusedContextResult
)
from backend.intelligence.evidence_fusion import ClinicalEvidenceFusionEngine
from backend.intelligence.answer_models import (
    AnswerSectionType,
    AnswerConfidence,
    EvidenceSupportLevel,
    ClinicalAnswerSection,
    ClinicalAnswer,
    AnswerSynthesisResult
)
from backend.intelligence.answer_synthesis import ClinicalAnswerSynthesisEngine
from backend.intelligence.citation_models import (
    CitationVerificationStatus,
    ClinicalClaimType,
    AttributedEvidenceSpan,
    ClinicalClaimAttribution,
    CitationAttributionReport
)
from backend.intelligence.citation_attribution import ClinicalCitationAttributionEngine
from backend.intelligence.verification_models import (
    GroundingVerificationStatus,
    ClinicalHallucinationType,
    SafetyPostScreenAction,
    ExtractedClinicalEntities,
    ClinicalVerificationClaim,
    ClinicalVerificationResult
)
from backend.intelligence.clinical_verification import ClinicalVerificationEngine
from backend.intelligence.decision_support_models import (
    ClinicalUncertaintyLevel,
    ClinicalRiskTier,
    UncertaintySourceType,
    ActionRecommendationType,
    ActionableRecommendation,
    RedFlagTrigger,
    ClinicalHandoffSummary,
    ClinicalDecisionSupportResult
)
from backend.intelligence.clinical_decision_support import ClinicalDecisionSupportEngine

__all__ = [
    "ClinicalIntent",
    "SafetyPriority",
    "ClinicalRoutingStrategy",
    "IntentClassificationResult",
    "ClinicalIntentClassifier",
    "QueryPlan",
    "ChunkWeightingStrategy",
    "DocumentFilterStrategy",
    "MultiDocumentStrategy",
    "GenerationStrategy",
    "ClinicalQueryPlanner",
    "ConflictType",
    "ConflictSeverity",
    "CoverageStatus",
    "EvidenceConflict",
    "EvidenceCoverage",
    "FusedEvidenceChunk",
    "FusedContextResult",
    "ClinicalEvidenceFusionEngine",
    "AnswerSectionType",
    "AnswerConfidence",
    "EvidenceSupportLevel",
    "ClinicalAnswerSection",
    "ClinicalAnswer",
    "AnswerSynthesisResult",
    "ClinicalAnswerSynthesisEngine",
    "CitationVerificationStatus",
    "ClinicalClaimType",
    "AttributedEvidenceSpan",
    "ClinicalClaimAttribution",
    "CitationAttributionReport",
    "ClinicalCitationAttributionEngine",
    "GroundingVerificationStatus",
    "ClinicalHallucinationType",
    "SafetyPostScreenAction",
    "ExtractedClinicalEntities",
    "ClinicalVerificationClaim",
    "ClinicalVerificationResult",
    "ClinicalVerificationEngine",
    "ClinicalUncertaintyLevel",
    "ClinicalRiskTier",
    "UncertaintySourceType",
    "ActionRecommendationType",
    "ActionableRecommendation",
    "RedFlagTrigger",
    "ClinicalHandoffSummary",
    "ClinicalDecisionSupportResult",
    "ClinicalDecisionSupportEngine",
    # Phase 6.8 Orchestration & Auditability
    "PipelineStage",
    "StageExecutionStatus",
    "StageAuditRecord",
    "ClinicalSafetyProvenance",
    "DataIsolationProvenance",
    "ClinicalIntelligenceOrchestrationResult",
    "ClinicalIntelligenceOrchestrator",
]

from backend.intelligence.orchestration_models import (
    PipelineStage,
    StageExecutionStatus,
    StageAuditRecord,
    ClinicalSafetyProvenance,
    DataIsolationProvenance,
    ClinicalIntelligenceOrchestrationResult
)
from backend.intelligence.clinical_orchestrator import ClinicalIntelligenceOrchestrator
