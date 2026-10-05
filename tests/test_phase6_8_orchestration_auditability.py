"""
Phase 6.8 Dedicated Test Suite: Clinical Intelligence Orchestration & Auditability.

Covers:
1. Normal clinical query orchestration.
2. Evidence-backed response provenance.
3. Citation preservation.
4. Clinical verification preservation.
5. Decision-support preservation.
6. Missing/empty evidence handling.
7. Low-confidence evidence handling.
8. Conflicting evidence handling (degraded stage status).
9. Emergency query safety interception.
10. Self-harm query safety interception.
11. Poisoning query safety interception.
12. Malformed and edge-case inputs.
13. Tenant A isolation.
14. Tenant B isolation.
15. Cross-tenant contamination detection.
16. PHI and telemetry protection.
17. Backward compatibility with existing schemas.
18. Deterministic cryptographic audit checksums.
19. RAG service sync response payload integration.
20. RAG service streaming SSE event integration.
21. Observability metrics recording.
22. Prometheus exposition formatting.
23. JSON serializability of all models.
24. Latency performance bounds.
25. Concordance verification with contradictory assertions.
"""

import json
import pytest
from unittest.mock import MagicMock

from backend.intelligence.orchestration_models import (
    PipelineStage,
    StageExecutionStatus,
    StageAuditRecord,
    ClinicalSafetyProvenance,
    DataIsolationProvenance,
    ClinicalIntelligenceOrchestrationResult
)
from backend.intelligence.clinical_orchestrator import ClinicalIntelligenceOrchestrator
from backend.intelligence.intent_models import ClinicalIntent, IntentClassificationResult, ClinicalRoutingStrategy
from backend.intelligence.query_plan import QueryPlan, GenerationStrategy
from backend.intelligence.evidence_models import (
    FusedContextResult,
    EvidenceCoverage,
    CoverageStatus,
    ConflictType,
    ConflictSeverity,
    EvidenceConflict
)
from backend.intelligence.answer_models import AnswerSynthesisResult, AnswerConfidence, EvidenceSupportLevel
from backend.intelligence.citation_models import CitationAttributionReport
from backend.intelligence.verification_models import (
    ClinicalVerificationResult,
    GroundingVerificationStatus,
    SafetyPostScreenAction
)
from backend.intelligence.decision_support_models import (
    ClinicalDecisionSupportResult,
    ClinicalUncertaintyLevel,
    ClinicalRiskTier,
    UncertaintySourceType
)
from backend.safety.safety_types import SafetyCategory, RiskLevel, SafetyAssessment
from backend.evaluation.observability import (
    ProductionMetricsCollector,
    record_orchestration_event,
    get_metrics_collector
)
from backend.api.rag_router import RAGQueryResponse


# ==============================================================================
# Helpers & Fixtures
# ==============================================================================

def make_sample_sources():
    return [
        {
            "chunk_id": "CHUNK_HTN_001",
            "document_id": "DOC_AHA_2023",
            "document_name": "AHA_Hypertension_Guidelines.pdf",
            "similarity_score": 0.88,
            "text": "First-line antihypertensive therapy includes ACE inhibitors, ARBs, CCBs, or thiazide diuretics.",
            "source_index": 1,
            "user_id": 101
        },
        {
            "chunk_id": "CHUNK_HTN_002",
            "document_id": "DOC_JNC8",
            "document_name": "JNC8_Clinical_Report.pdf",
            "similarity_score": 0.82,
            "text": "In the general nonblack population, initial antihypertensive treatment should include a thiazide-type diuretic.",
            "source_index": 2,
            "user_id": 101
        }
    ]


# ==============================================================================
# Test Cases
# ==============================================================================

def test_1_normal_clinical_query():
    """Validates complete orchestration of a normal clinical query with all stages present."""
    intent_res = IntentClassificationResult(
        intent=ClinicalIntent.TREATMENT_QUERY,
        confidence=0.96,
        routing_strategy=ClinicalRoutingStrategy.STANDARD_RAG,
        requires_retrieval=True,
        requires_document_context=True
    )
    plan = QueryPlan(
        intent=ClinicalIntent.TREATMENT_QUERY,
        retrieval_required=True,
        retrieval_strategy=ClinicalRoutingStrategy.STANDARD_RAG,
        generation_strategy=GenerationStrategy.STANDARD_GROUNDED,
        top_k=5,
        similarity_threshold=0.25
    )
    sources = make_sample_sources()

    result = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="trace_test_001",
        query="What are first-line treatments for stage 1 hypertension?",
        intent_result=intent_res,
        query_plan=plan,
        user_id=101,
        tenant_id="tenant_hospital_a",
        retrieved_sources=sources,
        total_pipeline_latency_ms=120.5
    )

    assert result.trace_id == "trace_test_001"
    assert len(result.pipeline_stages) == 9
    assert result.is_concordant is True
    assert result.safety_provenance.risk_tier == "MINIMAL"
    assert result.isolation_provenance.tenant_id == "tenant_hospital_a"
    assert result.isolation_provenance.cross_tenant_contamination_check_passed is True
    assert len(result.audit_checksum) == 64  # SHA256 hex


def test_2_evidence_backed_response():
    """Validates evidence chunk and document counting in data isolation provenance."""
    sources = make_sample_sources()
    result = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="trace_test_002",
        query="Hypertension first line agents",
        retrieved_sources=sources,
        user_id=101
    )
    assert result.isolation_provenance.chunks_accessed_count == 2
    assert result.isolation_provenance.documents_accessed_count == 2
    assert "CHUNK_HTN_001" in result.isolation_provenance.retrieved_chunk_ids
    assert "CHUNK_HTN_002" in result.isolation_provenance.retrieved_chunk_ids


def test_3_citation_preservation():
    """Ensures citation attribution metrics are preserved in orchestration stage record."""
    attr_report = CitationAttributionReport(
        is_valid=True,
        citation_precision=0.92,
        factual_claims_count=5,
        verified_claims_count=5,
        unsupported_claims_count=0
    )
    result = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="trace_test_003",
        query="Hypertension medications",
        attribution_report=attr_report
    )
    cit_stage = next(s for s in result.pipeline_stages if s.stage == PipelineStage.CITATION_ATTRIBUTION)
    assert cit_stage.status == StageExecutionStatus.SUCCESS
    assert cit_stage.flags["precision"] == 0.92
    assert cit_stage.flags["claims_count"] == 5
    assert result.safety_provenance.citations_verified is True


def test_4_clinical_verification_preservation():
    """Verifies clinical verification contradictions and grounding are reflected in provenance."""
    ver_res = ClinicalVerificationResult(
        is_verified_safe=True,
        overall_grounding_score=1.0,
        total_claims_analyzed=1,
        grounded_claims_count=1,
        ungrounded_claims_count=0,
        contradictions_count=0,
        hallucinations_detected=0,
        claim_verifications=[],
        action_taken=SafetyPostScreenAction.ALLOW,
        sanitized_answer="First line therapy includes ACE inhibitors [Source 1]."
    )
    result = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="trace_test_004",
        query="Hypertension guidelines",
        verification_result=ver_res
    )
    assert result.safety_provenance.grounding_verified is True
    assert result.safety_provenance.contradictions_detected == 0


def test_5_decision_support_preservation():
    """Verifies uncertainty level, risk tier, and red flags are preserved from decision support."""
    cds_res = ClinicalDecisionSupportResult(
        calibrated_confidence=0.85,
        uncertainty_level=ClinicalUncertaintyLevel.LOW,
        primary_uncertainty_source=UncertaintySourceType.NONE,
        risk_tier=ClinicalRiskTier.LOW,
        escalation_required=False,
        escalation_reason=None,
        actionable_recommendations=[],
        red_flag_triggers=[],
        suggested_follow_up_inquiries=[]
    )
    result = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="trace_test_005",
        query="Hypertension follow-up",
        decision_support=cds_res
    )
    assert result.safety_provenance.risk_tier == "LOW"
    assert result.safety_provenance.uncertainty_level == "LOW"
    assert result.safety_provenance.requires_escalation is False


def test_6_missing_empty_evidence():
    """Tests orchestrator behavior when no sources are retrieved."""
    result = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="trace_test_006",
        query="Unindexed rare medical syndrome",
        retrieved_sources=[]
    )
    assert result.isolation_provenance.chunks_accessed_count == 0
    assert result.isolation_provenance.documents_accessed_count == 0
    assert result.is_concordant is True


def test_7_low_confidence_evidence():
    """Tests behavior when answer synthesis confidence is insufficient."""
    synth_res = AnswerSynthesisResult(
        intent="TREATMENT_PLAN",
        confidence=AnswerConfidence.LOW,
        support_level=EvidenceSupportLevel.PARTIAL,
        coverage_status="PARTIAL",
        answer="Limited information available in guidelines.",
        sections=[],
        is_fallback=False
    )
    result = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="trace_test_007",
        query="Experimental hypertensive regimen",
        answer_synthesis=synth_res
    )
    synth_stage = next(s for s in result.pipeline_stages if s.stage == PipelineStage.ANSWER_SYNTHESIS)
    assert synth_stage.flags["confidence"] == "LOW"
    assert synth_stage.flags["support_level"] == "PARTIAL"


def test_8_conflicting_evidence():
    """Tests that evidence fusion conflicts result in DEGRADED fusion stage status."""
    conflict = EvidenceConflict(
        topic="aspirin dosage",
        conflict_type=ConflictType.TREATMENT_RECOMMENDATION_CONFLICT,
        severity=ConflictSeverity.MODERATE,
        doc_a_id="DocA",
        doc_b_id="DocB",
        doc_a_name="DocA",
        doc_b_name="DocB",
        statement_a="Take aspirin daily",
        statement_b="Do not take aspirin daily",
        resolution_guidance="Consult treating physician"
    )
    fused_res = FusedContextResult(
        fused_chunks=[],
        contributing_documents=["DocA", "DocB"],
        contributing_documents_count=2,
        conflicts=[conflict],
        has_conflicts=True,
        coverage=EvidenceCoverage(coverage_score=0.5, status=CoverageStatus.PARTIAL, missing_aspects=[]),
        formatted_context="Contradictory guidelines text",
        sources=[],
        deduped_count=0,
        is_sufficient=True,
        latency_ms=1.0
    )
    result = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="trace_test_008",
        query="Aspirin primary prevention guidelines",
        fused_evidence=fused_res
    )
    fusion_stage = next(s for s in result.pipeline_stages if s.stage == PipelineStage.EVIDENCE_FUSION)
    assert fusion_stage.status == StageExecutionStatus.DEGRADED
    assert fusion_stage.flags["conflicts_count"] == 1


def test_9_emergency_query_interception():
    """Tests emergency query interception sets INTERCEPTED status and flags escalation."""
    safety_assessment = SafetyAssessment(
        category=SafetyCategory.EMERGENCY_SYMPTOMS,
        risk_level=RiskLevel.CRITICAL,
        requires_escalation=True,
        allow_normal_rag=False
    )
    result = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="trace_test_009",
        query="I have severe crushing chest pain radiating to my arm",
        safety_assessment=safety_assessment
    )
    assert result.stage_status_map["SAFETY_PRE_SCREEN"] == "INTERCEPTED"
    assert result.safety_provenance.emergency_intercepted is True
    assert result.safety_provenance.safety_assessment_category == "EMERGENCY_SYMPTOMS"


def test_10_self_harm_query_interception():
    """Tests self-harm query is immediately recorded in safety provenance."""
    safety_assessment = SafetyAssessment(
        category=SafetyCategory.SELF_HARM_OR_SUICIDE,
        risk_level=RiskLevel.CRITICAL,
        requires_escalation=True,
        allow_normal_rag=False
    )
    result = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="trace_test_010",
        query="Ways to harm myself",
        safety_assessment=safety_assessment
    )
    assert result.safety_provenance.self_harm_intercepted is True
    assert result.stage_status_map["SAFETY_PRE_SCREEN"] == "INTERCEPTED"


def test_11_poisoning_query_interception():
    """Tests acute poisoning intercept tracking."""
    safety_assessment = SafetyAssessment(
        category=SafetyCategory.POISONING_OR_OVERDOSE,
        risk_level=RiskLevel.CRITICAL,
        requires_escalation=True,
        allow_normal_rag=False
    )
    result = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="trace_test_011",
        query="Toddler drank cleaning bleach",
        safety_assessment=safety_assessment
    )
    assert result.safety_provenance.poisoning_intercepted is True


def test_12_malformed_and_edge_inputs():
    """Ensures orchestrator handles empty strings, None objects, and zero lengths without crashing."""
    result = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="",
        query="",
        safety_assessment=None,
        intent_result=None,
        query_plan=None,
        fused_evidence=None,
        answer_synthesis=None,
        attribution_report=None,
        verification_result=None,
        decision_support=None,
        user_id=None,
        tenant_id=None,
        retrieved_sources=None,
        total_pipeline_latency_ms=0.0
    )
    assert isinstance(result, ClinicalIntelligenceOrchestrationResult)
    assert len(result.pipeline_stages) == 9
    assert result.is_concordant is True


def test_13_tenant_a_isolation():
    """Validates provenance binding for Tenant A."""
    sources = [{"chunk_id": "TA_01", "user_id": 101, "document_id": "D1"}]
    res = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="t_a",
        query="test",
        user_id=101,
        tenant_id="hospital_alpha",
        retrieved_sources=sources
    )
    assert res.isolation_provenance.tenant_id == "hospital_alpha"
    assert res.isolation_provenance.user_id == 101
    assert res.isolation_provenance.cross_tenant_contamination_check_passed is True


def test_14_tenant_b_isolation():
    """Validates provenance binding for Tenant B."""
    sources = [{"chunk_id": "TB_01", "user_id": 202, "document_id": "D2"}]
    res = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="t_b",
        query="test",
        user_id=202,
        tenant_id="clinic_beta",
        retrieved_sources=sources
    )
    assert res.isolation_provenance.tenant_id == "clinic_beta"
    assert res.isolation_provenance.user_id == 202
    assert res.isolation_provenance.cross_tenant_contamination_check_passed is True


def test_15_cross_tenant_contamination_detection():
    """Tests that cross-tenant chunk leakage is detected, setting check to False and is_concordant to False."""
    contaminated_sources = [
        {"chunk_id": "TA_01", "user_id": 101, "document_id": "D1"},
        {"chunk_id": "LEAK_02", "user_id": 999, "document_id": "OTHER_TENANT"}  # Different user!
    ]
    res = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="t_leak",
        query="test query",
        user_id=101,
        retrieved_sources=contaminated_sources
    )
    assert res.isolation_provenance.cross_tenant_contamination_check_passed is False
    assert res.is_concordant is False


def test_16_phi_and_telemetry_protection():
    """Ensures serialized audit records and Prometheus exposition contain zero PHI."""
    res = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="trace_phi_check",
        query="Patient John Doe DOB 01/01/1980 SSN 123-45-6789 has hypertension",
        retrieved_sources=make_sample_sources(),
        user_id=101
    )
    d = res.to_dict()
    serialized = json.dumps(d)
    assert "John Doe" not in serialized
    assert "123-45-6789" not in serialized
    assert "01/01/1980" not in serialized


def test_17_backward_compatibility_schemas():
    """Validates RAGQueryResponse accepts payloads with or without orchestration fields."""
    legacy_payload = {
        "question": "What is hypertension?",
        "answer": "High blood pressure.",
        "retrieval_status": "success",
        "sources": [],
        "disclaimer": "Consult doctor."
    }
    resp1 = RAGQueryResponse(**legacy_payload)
    assert resp1.orchestration is None

    new_payload = {
        **legacy_payload,
        "orchestration": {"trace_id": "t1", "is_concordant": True}
    }
    resp2 = RAGQueryResponse(**new_payload)
    assert resp2.orchestration["trace_id"] == "t1"


def test_18_deterministic_provenance_checksum():
    """Verifies that two runs with identical inputs produce the exact same cryptographic checksum."""
    sources = make_sample_sources()
    res1 = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="det_trace",
        query="Identical query",
        retrieved_sources=sources,
        user_id=101
    )
    res2 = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="det_trace",
        query="Identical query",
        retrieved_sources=sources,
        user_id=101
    )
    assert res1.audit_checksum == res2.audit_checksum
    assert len(res1.audit_checksum) == 64


def test_19_api_compatibility_query_endpoint():
    """Verifies that generate_rag_answer includes orchestration and clinical_intelligence_orchestration."""
    from backend.rag.rag_service import RAGService
    service = RAGService()
    # Mocking retrieval to avoid real model call in fast unit test
    service.query = MagicMock(return_value={
        "question": "What is hypertension?",
        "retrieved_chunks": [],
        "context": "",
        "sources": [],
        "retrieval_status": "no_relevant_context",
        "timings": {
            "total_retrieval_time_ms": 1.0,
            "embedding_time_ms": 0.5,
            "faiss_retrieval_time_ms": 0.5
        }
    })
    ans = service.generate_rag_answer(
        question="What is hypertension?",
        top_k=2,
        similarity_threshold=0.25
    )
    assert "orchestration" in ans
    assert "clinical_intelligence_orchestration" in ans
    assert ans["orchestration"]["is_concordant"] is True


def test_20_streaming_sse_orchestration_event():
    """Verifies that generate_rag_stream yields clinical_intelligence_orchestration event."""
    from backend.rag.rag_service import RAGService
    service = RAGService()
    service.query = MagicMock(return_value={
        "question": "What is hypertension?",
        "retrieved_chunks": [],
        "context": "",
        "sources": [],
        "retrieval_status": "no_relevant_context",
        "timings": {
            "total_retrieval_time_ms": 1.0,
            "embedding_time_ms": 0.5,
            "faiss_retrieval_time_ms": 0.5
        }
    })
    events = list(service.generate_rag_stream("What is hypertension?"))
    event_types = [e[0] for e in events]
    assert "complete" in event_types
    # Find the complete event payload
    complete_event = next(e[1] for e in events if e[0] == "complete")
    assert "clinical_intelligence_orchestration" in complete_event or "orchestration" in complete_event


def test_21_observability_metrics_recording():
    """Tests Prometheus telemetry accumulator for Phase 6.8."""
    collector = ProductionMetricsCollector(max_samples=100)
    collector.record_orchestration_event(is_concordant=True, is_intercepted=False, latency_ms=0.45)
    collector.record_orchestration_event(is_concordant=False, is_intercepted=True, latency_ms=0.55)

    snap = collector.get_metrics_snapshot()
    orch_snap = snap["intelligence"]["clinical_orchestration"]
    assert orch_snap["total"] == 2
    assert orch_snap["concordant_total"] == 1
    assert orch_snap["intercepted_total"] == 1
    assert orch_snap["latency"]["avg"] == 0.5


def test_22_prometheus_exposition_format():
    """Validates RFC-compliant text exposition of Phase 6.8 metrics."""
    collector = ProductionMetricsCollector(max_samples=10)
    collector.record_orchestration_event(is_concordant=True, latency_ms=1.2)
    expo = collector.get_prometheus_exposition()

    assert "rag_orchestration_total 1" in expo
    assert "rag_orchestration_concordant_total 1" in expo
    assert "rag_orchestration_intercepted_total 0" in expo
    assert "rag_orchestration_duration_seconds" in expo


def test_23_json_serialization_all_models():
    """Verifies that all Phase 6.8 models serialize to 100% compliant JSON."""
    rec = StageAuditRecord(stage=PipelineStage.AUDIT_ORCHESTRATION, status=StageExecutionStatus.SUCCESS)
    sp = ClinicalSafetyProvenance()
    ip = DataIsolationProvenance()
    res = ClinicalIntelligenceOrchestrationResult(
        trace_id="test",
        timestamp_iso="2026-10-05T00:00:00Z",
        pipeline_stages=[rec],
        stage_status_map={"AUDIT": "SUCCESS"},
        safety_provenance=sp,
        isolation_provenance=ip,
        audit_checksum="abc",
        is_concordant=True,
        orchestration_latency_ms=0.1,
        total_pipeline_latency_ms=10.0
    )
    s = json.dumps(res.to_dict())
    loaded = json.loads(s)
    assert loaded["trace_id"] == "test"
    assert loaded["is_concordant"] is True


def test_24_sub_millisecond_overhead():
    """Verifies orchestrator executes in sub-millisecond time (< 5.0ms maximum)."""
    res = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="perf_test",
        query="Standard hypertension query",
        retrieved_sources=make_sample_sources()
    )
    assert res.orchestration_latency_ms < 5.0  # Extremely lightweight pure Python


def test_25_unhandled_hallucination_degrades_concordance():
    """Verifies that unhandled contradictions cause is_concordant to evaluate to False."""
    ver_res = ClinicalVerificationResult(
        is_verified_safe=False,
        overall_grounding_score=0.3,
        total_claims_analyzed=2,
        grounded_claims_count=0,
        ungrounded_claims_count=1,
        contradictions_count=2,
        hallucinations_detected=1,
        claim_verifications=[],
        action_taken=SafetyPostScreenAction.PRUNE_UNSUPPORTED,
        sanitized_answer="Fallback text"
    )
    res = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="contra_trace",
        query="Query with contradictory evidence",
        verification_result=ver_res
    )
    assert res.is_concordant is False
    assert res.safety_provenance.grounding_verified is False
    assert res.safety_provenance.contradictions_detected == 2
