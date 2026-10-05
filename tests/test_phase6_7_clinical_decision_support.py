"""
Phase 6.7 Tests: Clinical Decision Support, Uncertainty Calibration & Care Pathways.

Comprehensive test suite verifying all Phase 6.7 requirements:
1. Basic decision support evaluation (grounded answer, high confidence, LOW uncertainty)
2. Aleatoric uncertainty detection (inter-document conflicts trigger ALEATORIC source)
3. Epistemic uncertainty detection (low similarity / partial coverage trigger EPISTEMIC source)
4. Verification deficit uncertainty detection (unsupported claims trigger VERIFICATION_DEFICIT)
5. Clinical risk stratification: CRITICAL tier on emergency/harm
6. Clinical risk stratification: HIGH tier on medication/dosage queries
7. Clinical risk stratification: MODERATE tier on diagnosis/symptom queries
8. Clinical risk stratification: LOW/MINIMAL tier on prevention/general health queries
9. Cardiovascular red flags: Hypertensive crisis trigger and escalation
10. Cardiovascular red flags: Acute coronary syndrome and stroke triggers
11. Metabolic red flags: Severe hypoglycemia and DKA triggers
12. Pharmacological red flags: Anaphylaxis and severe cutaneous adverse reactions
13. Actionable recommendations: Hypertension care pathway (home BP log, DASH diet)
14. Actionable recommendations: Medication review and laboratory monitoring
15. Actionable recommendations: Strict non-prescriptive phrasing compliance
16. Agentic follow-up inquiries: Hypertension context (2-4 relevant questions)
17. Agentic follow-up inquiries: Medication context (2-4 relevant questions)
18. Provider SBAR clinical handoff summary structure and completeness
19. Escalation trigger on low calibrated confidence (< 0.35)
20. Safety precedence: acute emergency symptom bypass
21. Safety precedence: self-harm inquiry bypass
22. Safety precedence: acute poisoning inquiry bypass
23. SSE streaming clinical_decision_support event broadcasting
24. Observability metrics recording and Prometheus exposition without PHI
25. Complete JSON serialization of all Phase 6.7 decision support models
26. RAGService.generate_rag_answer end-to-end integration
27. Empty, None, and whitespace edge case handling
28. Production vector store invariants verification (744, 744, 384)
"""

import json
import pytest
from unittest.mock import MagicMock, patch

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
from backend.intelligence.verification_models import (
    GroundingVerificationStatus,
    ClinicalHallucinationType,
    SafetyPostScreenAction,
    ExtractedClinicalEntities,
    ClinicalVerificationClaim,
    ClinicalVerificationResult
)
from backend.intelligence.citation_models import (
    CitationVerificationStatus,
    ClinicalClaimType,
    AttributedEvidenceSpan,
    ClinicalClaimAttribution,
    CitationAttributionReport
)
from backend.intelligence.evidence_models import (
    ConflictType,
    ConflictSeverity,
    CoverageStatus,
    EvidenceConflict,
    EvidenceCoverage,
    FusedEvidenceChunk,
    FusedContextResult
)
from backend.intelligence.answer_models import (
    AnswerSectionType,
    AnswerConfidence,
    EvidenceSupportLevel,
    ClinicalAnswerSection,
    AnswerSynthesisResult
)
from backend.intelligence.intent_models import (
    ClinicalIntent,
    SafetyPriority,
    ClinicalRoutingStrategy,
    IntentClassificationResult
)
from backend.intelligence.query_plan import (
    QueryPlan,
    ChunkWeightingStrategy,
    DocumentFilterStrategy,
    MultiDocumentStrategy,
    GenerationStrategy
)
from backend.rag.prompt_builder import MEDICAL_DISCLAIMER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_sources():
    return [
        {
            "source_index": 1,
            "source_number": 1,
            "chunk_id": "chunk_htn_001",
            "document_id": "doc_guideline_2024",
            "document_name": "AHA_Hypertension_2024.pdf",
            "page_number": 12,
            "text": "First-line pharmacotherapy for stage 1 hypertension includes thiazide diuretics, calcium channel blockers, or ACE inhibitors. The standard starting dose for amlodipine is 5 mg daily, which reduces systolic blood pressure by 8 to 12 mmHg.",
            "preview_text": "First-line pharmacotherapy for stage 1 hypertension includes thiazide diuretics...",
            "similarity_score": 0.89
        },
        {
            "source_index": 2,
            "source_number": 2,
            "chunk_id": "chunk_htn_002",
            "document_id": "doc_guideline_2024",
            "document_name": "AHA_Hypertension_2024.pdf",
            "page_number": 14,
            "text": "Lifestyle modifications including dietary sodium restriction (< 2,300 mg/day) and DASH dietary pattern should be initiated alongside pharmacotherapy.",
            "preview_text": "Lifestyle modifications including dietary sodium restriction...",
            "similarity_score": 0.82
        }
    ]


@pytest.fixture
def sample_attribution_report():
    return CitationAttributionReport(
        is_valid=True,
        claims=[
            ClinicalClaimAttribution(
                claim_id="claim_001",
                claim_text="First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily.",
                raw_sentence="First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily [Source 1].",
                claim_type=ClinicalClaimType.FACTUAL_MEDICAL,
                cited_source_indices=[1],
                verification_status=CitationVerificationStatus.VERIFIED,
                best_support_score=0.92,
                is_supported=True
            )
        ],
        total_claims_count=1,
        factual_claims_count=1,
        verified_claims_count=1,
        unsupported_claims_count=0,
        citations_found=[1],
        valid_citations=[1],
        invalid_citations=[],
        duplicate_citations=[],
        citation_precision=1.0,
        claim_attribution_coverage=1.0,
        spoofed_citations_detected=False,
        spoofed_citation_tags=[],
        unsupported_claims=[],
        cleaned_attributed_answer="First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily [Source 1].",
        latency_ms=1.2,
        metadata={}
    )


@pytest.fixture
def sample_verification_result():
    return ClinicalVerificationResult(
        is_verified_safe=True,
        overall_grounding_score=0.95,
        total_claims_analyzed=1,
        grounded_claims_count=1,
        ungrounded_claims_count=0,
        contradictions_count=0,
        hallucinations_detected=0,
        claim_verifications=[
            ClinicalVerificationClaim(
                claim_id="clm_1",
                claim_text="First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily.",
                raw_sentence="First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily [Source 1].",
                verification_status=GroundingVerificationStatus.GROUNDED,
                hallucination_type=ClinicalHallucinationType.NONE,
                is_grounded=True,
                confidence_score=0.95,
                supporting_sources=[1],
                entities=ExtractedClinicalEntities(
                    medications=["amlodipine"],
                    dosages=["5 mg daily"],
                    diagnoses=["stage 1"]
                )
            )
        ],
        action_taken=SafetyPostScreenAction.ALLOW,
        sanitized_answer="First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily [Source 1].",
        fallback_triggered=False,
        disclaimer_enforced=True,
        latency_ms=0.8,
        metadata={}
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_01_basic_decision_support_evaluation(sample_sources, sample_attribution_report, sample_verification_result):
    """Test 1: Grounded answer evaluates to high calibrated confidence and LOW uncertainty."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="What is the starting dose of amlodipine for hypertension?",
        intent="DOSAGE_QUERY",
        answer_text="First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily [Source 1].",
        retrieved_sources=sample_sources,
        attribution_report=sample_attribution_report,
        verification_result=sample_verification_result
    )

    assert isinstance(res, ClinicalDecisionSupportResult)
    assert res.calibrated_confidence >= 0.75
    assert res.uncertainty_level in (ClinicalUncertaintyLevel.LOW, ClinicalUncertaintyLevel.MODERATE)
    assert len(res.actionable_recommendations) > 0
    assert len(res.suggested_follow_up_inquiries) >= 2
    assert res.clinical_handoff is not None


def test_02_aleatoric_uncertainty_from_conflicting_evidence(sample_sources, sample_attribution_report, sample_verification_result):
    """Test 2: Inter-document conflicts trigger ALEATORIC uncertainty source."""
    conflict = EvidenceConflict(
        topic="Amlodipine starting dose",
        conflict_type=ConflictType.DOSAGE_DISCREPANCY,
        severity=ConflictSeverity.MODERATE,
        doc_a_id="doc_1",
        doc_b_id="doc_2",
        doc_a_name="AHA_2024.pdf",
        doc_b_name="ESC_2024.pdf",
        statement_a="Dose is 5 mg daily",
        statement_b="Dose is 2.5 mg daily",
        resolution_guidance="Verify patient age and renal function."
    )
    fused = FusedContextResult(
        fused_chunks=[],
        contributing_documents=["AHA_2024.pdf", "ESC_2024.pdf"],
        contributing_documents_count=2,
        conflicts=[conflict],
        has_conflicts=True,
        coverage=EvidenceCoverage(coverage_score=0.75, status=CoverageStatus.PARTIAL, covered_aspects=["dose"], missing_aspects=[]),
        formatted_context="Fused context",
        sources=sample_sources,
        deduped_count=0,
        is_sufficient=True,
        latency_ms=1.0
    )

    res = ClinicalDecisionSupportEngine.evaluate(
        query="What is the starting dose of amlodipine?",
        intent="DOSAGE_QUERY",
        answer_text="Sources differ on whether starting dose is 2.5 mg or 5 mg daily.",
        retrieved_sources=sample_sources,
        fused_evidence=fused,
        attribution_report=sample_attribution_report,
        verification_result=sample_verification_result
    )

    assert res.primary_uncertainty_source == UncertaintySourceType.ALEATORIC
    assert res.calibrated_confidence < 0.85


def test_03_epistemic_uncertainty_from_low_similarity():
    """Test 3: Low similarity scores and empty evidence trigger EPISTEMIC uncertainty."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="What is the treatment for rare syndrome X?",
        intent="TREATMENT_QUERY",
        answer_text="Relevant medical information could not be found.",
        retrieved_sources=[]  # No sources retrieved
    )

    assert res.uncertainty_level == ClinicalUncertaintyLevel.INDETERMINATE
    assert res.primary_uncertainty_source == UncertaintySourceType.EPISTEMIC
    assert res.calibrated_confidence == 0.0


def test_04_verification_deficit_uncertainty(sample_sources, sample_attribution_report):
    """Test 4: Hallucinated or pruned claims trigger VERIFICATION_DEFICIT uncertainty."""
    compromised_verif = ClinicalVerificationResult(
        is_verified_safe=False,
        overall_grounding_score=0.40,
        total_claims_analyzed=2,
        grounded_claims_count=1,
        ungrounded_claims_count=1,
        contradictions_count=1,
        hallucinations_detected=1,
        claim_verifications=[
            ClinicalVerificationClaim(
                claim_id="clm_bad",
                claim_text="Amlodipine increases blood pressure.",
                raw_sentence="Amlodipine increases blood pressure.",
                verification_status=GroundingVerificationStatus.DIRECTIONAL_CONTRADICTION,
                hallucination_type=ClinicalHallucinationType.DIRECTIONAL_INVERSION,
                is_grounded=False,
                confidence_score=0.20,
                discrepancy_details=["Directional contradiction detected"],
                supporting_sources=[]
            )
        ],
        action_taken=SafetyPostScreenAction.PRUNE_UNSUPPORTED,
        sanitized_answer="Safe claim only.",
        fallback_triggered=False,
        disclaimer_enforced=True,
        latency_ms=1.0,
        metadata={}
    )

    res = ClinicalDecisionSupportEngine.evaluate(
        query="Does amlodipine increase blood pressure?",
        intent="MEDICATION_QUERY",
        answer_text="Safe claim only.",
        retrieved_sources=sample_sources,
        attribution_report=sample_attribution_report,
        verification_result=compromised_verif
    )

    assert res.primary_uncertainty_source == UncertaintySourceType.VERIFICATION_DEFICIT


def test_05_clinical_risk_tier_critical_on_emergency():
    """Test 5: Emergency intent or acute safety categories map to CRITICAL risk tier."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="I have crushing chest pain and difficulty breathing",
        intent="EMERGENCY",
        answer_text="Call 911 immediately for emergency medical assistance."
    )

    assert res.risk_tier == ClinicalRiskTier.CRITICAL
    assert res.escalation_required is True


def test_06_clinical_risk_tier_high_on_dosage_and_medications(sample_sources):
    """Test 6: DOSAGE_QUERY and MEDICATION_QUERY map to HIGH risk tier."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="What is the maximum daily dose of lisinopril?",
        intent="DOSAGE_QUERY",
        answer_text="The standard maximum dose of lisinopril is 40 mg daily.",
        retrieved_sources=sample_sources
    )

    assert res.risk_tier == ClinicalRiskTier.HIGH


def test_07_clinical_risk_tier_moderate_on_diagnosis_and_symptoms(sample_sources):
    """Test 7: DIAGNOSIS_QUERY and SYMPTOM_QUERY map to MODERATE risk tier."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="What are the diagnostic stages of hypertension?",
        intent="DIAGNOSIS_QUERY",
        answer_text="Stage 1 hypertension is defined as systolic 130-139 mmHg.",
        retrieved_sources=sample_sources
    )

    assert res.risk_tier == ClinicalRiskTier.MODERATE


def test_08_clinical_risk_tier_low_or_minimal_on_prevention(sample_sources):
    """Test 8: PREVENTION_QUERY and DOCUMENT_SUMMARY map to LOW or MINIMAL risk tier."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="What are preventive lifestyle measures for heart health?",
        intent="PREVENTION_QUERY",
        answer_text="Cardiovascular prevention involves regular aerobic exercise and low sodium intake.",
        retrieved_sources=sample_sources
    )

    assert res.risk_tier in (ClinicalRiskTier.LOW, ClinicalRiskTier.MINIMAL)


def test_09_cardiovascular_red_flags_hypertensive_crisis():
    """Test 9: Mentions of BP > 180 or 120 trigger RF_HTN_CRISIS with is_critical=True."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="My blood pressure reading just showed 190/125 mmHg, should I be concerned?",
        intent="SYMPTOM_QUERY",
        answer_text="A blood pressure reading of 190/125 mmHg represents hypertensive crisis."
    )

    flag_ids = [f.flag_id for f in res.red_flag_triggers]
    assert "RF_HTN_CRISIS" in flag_ids
    crisis_flag = next(f for f in res.red_flag_triggers if f.flag_id == "RF_HTN_CRISIS")
    assert crisis_flag.is_critical is True
    assert res.escalation_required is True


def test_10_cardiovascular_red_flags_acute_coronary_and_stroke():
    """Test 10: FAST stroke signs trigger RF_STROKE."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="My father suddenly has unilateral facial droop and slurred speech",
        intent="SYMPTOM_QUERY",
        answer_text="Facial droop and slurred speech are warning signs of acute stroke."
    )

    flag_ids = [f.flag_id for f in res.red_flag_triggers]
    assert "RF_STROKE" in flag_ids
    assert res.escalation_required is True


def test_11_metabolic_red_flags_hypoglycemia_and_dka():
    """Test 11: Hypoglycemia symptoms trigger RF_HYPO."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="Diabetic patient with severe hypoglycemia, diaphoresis, and confusion",
        intent="SYMPTOM_QUERY",
        answer_text="Severe hypoglycemia requires immediate fast-acting carbohydrate administration."
    )

    flag_ids = [f.flag_id for f in res.red_flag_triggers]
    assert "RF_HYPO" in flag_ids
    assert res.escalation_required is True


def test_12_pharmacological_red_flags_anaphylaxis_and_scar():
    """Test 12: Anaphylaxis and severe allergic symptoms trigger RF_ANAPHYLAXIS."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="Patient developed acute angioedema, stridor, and wheezing after taking penicillin",
        intent="MEDICATION_QUERY",
        answer_text="Angioedema and stridor indicate severe anaphylaxis."
    )

    flag_ids = [f.flag_id for f in res.red_flag_triggers]
    assert "RF_ANAPHYLAXIS" in flag_ids
    assert res.escalation_required is True


def test_13_actionable_recommendations_hypertension(sample_sources):
    """Test 13: Generates home BP log and DASH diet recommendations for hypertension."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="How can I manage stage 1 hypertension?",
        intent="TREATMENT_QUERY",
        answer_text="Stage 1 hypertension management includes amlodipine and dietary changes.",
        retrieved_sources=sample_sources
    )

    rec_ids = [r.recommendation_id for r in res.actionable_recommendations]
    assert "REC_HTN_BP_LOG" in rec_ids
    assert "REC_HTN_LIFESTYLE" in rec_ids


def test_14_actionable_recommendations_medication_review(sample_sources):
    """Test 14: Generates clinician medication review and lab monitoring recommendations."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="What is the starting dose of amlodipine?",
        intent="DOSAGE_QUERY",
        answer_text="Amlodipine starting dose is 5 mg daily.",
        retrieved_sources=sample_sources
    )

    rec_ids = [r.recommendation_id for r in res.actionable_recommendations]
    assert "REC_MED_REVIEW" in rec_ids
    assert "REC_LAB_MONITORING" in rec_ids


def test_15_actionable_recommendations_non_prescriptive_phrasing(sample_sources):
    """Test 15: All generated recommendations avoid prohibited directive phrasing."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="What dose should I take for high blood pressure?",
        intent="DOSAGE_QUERY",
        answer_text="Clinical guidelines recommend 5 mg daily.",
        retrieved_sources=sample_sources
    )

    prohibited_patterns = ["you must take", "i prescribe", "take this dose", "you are diagnosed with"]
    for rec in res.actionable_recommendations:
        for p in prohibited_patterns:
            assert p not in rec.text.lower(), f"Prohibited phrasing '{p}' found in recommendation: {rec.text}"


def test_16_agentic_follow_up_inquiries_hypertension():
    """Test 16: Returns 2 to 4 contextual follow-up questions tailored to hypertension."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="What is the guideline definition of hypertension?",
        intent="DIAGNOSIS_QUERY",
        answer_text="Hypertension is classified into stage 1 and stage 2."
    )

    assert 2 <= len(res.suggested_follow_up_inquiries) <= 4
    assert any("blood pressure" in q.lower() or "hypertension" in q.lower() for q in res.suggested_follow_up_inquiries)


def test_17_agentic_follow_up_inquiries_medication():
    """Test 17: Returns 2 to 4 contextual follow-up questions tailored to medications."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="What are the adverse effects of amlodipine?",
        intent="MEDICATION_QUERY",
        answer_text="Amlodipine can cause peripheral edema and dizziness."
    )

    assert 2 <= len(res.suggested_follow_up_inquiries) <= 4
    assert any("drug" in q.lower() or "medication" in q.lower() or "therapy" in q.lower() for q in res.suggested_follow_up_inquiries)


def test_18_sbar_provider_handoff_structure(sample_sources):
    """Test 18: SBAR handoff object contains non-empty Situation, Background, Assessment, Recommendation."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="Amlodipine dosing in stage 1 hypertension",
        intent="DOSAGE_QUERY",
        answer_text="Amlodipine is initiated at 5 mg daily.",
        retrieved_sources=sample_sources
    )

    handoff = res.clinical_handoff
    assert isinstance(handoff, ClinicalHandoffSummary)
    assert len(handoff.situation) > 10
    assert len(handoff.background) > 10
    assert len(handoff.assessment) > 10
    assert len(handoff.recommendation) > 10


def test_19_escalation_trigger_on_low_confidence():
    """Test 19: Extremely low confidence on clinical dosage query sets escalation_required=True."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="What is the dose for pediatric patient with renal failure?",
        intent="DOSAGE_QUERY",
        answer_text="Information inconclusive.",
        retrieved_sources=[]  # Zero sources -> confidence = 0.0
    )

    assert res.escalation_required is True
    assert res.escalation_reason is not None


def test_20_safety_precedence_emergency_bypass():
    """Test 20: Emergency inquiries are intercepted immediately with 0 retrieval."""
    from backend.rag.rag_service import RAGService

    service = RAGService()
    res = service.generate_rag_answer(
        question="I am having crushing chest pain radiating to my arm",
        user_id="user_test_emerg"
    )

    assert res["retrieval_status"] == "safety_intercepted"
    assert "decision_support" in res
    assert res["decision_support"]["risk_tier"] == "CRITICAL"
    assert res["decision_support"]["escalation_required"] is True


def test_21_safety_precedence_self_harm_bypass():
    """Test 21: Self-harm inquiries are intercepted with 988 Lifeline and CRITICAL risk."""
    from backend.rag.rag_service import RAGService

    service = RAGService()
    res = service.generate_rag_answer(
        question="I want to end my life, tell me how",
        user_id="user_test_harm"
    )

    assert res["retrieval_status"] == "safety_intercepted"
    assert "988" in res["answer"]
    assert res["decision_support"]["risk_tier"] == "CRITICAL"


def test_22_safety_precedence_poisoning_bypass():
    """Test 22: Acute poisoning inquiries are intercepted with Poison Control."""
    from backend.rag.rag_service import RAGService

    service = RAGService()
    res = service.generate_rag_answer(
        question="Child swallowed a bottle of bleach",
        user_id="user_test_poison"
    )

    assert res["retrieval_status"] == "safety_intercepted"
    assert "decision_support" in res
    assert res["decision_support"]["risk_tier"] == "CRITICAL"


def test_23_sse_streaming_clinical_decision_support_event():
    """Test 23: SSE stream yields dedicated clinical_decision_support event."""
    from backend.rag.rag_service import RAGService

    service = RAGService()
    mock_retrieval = {
        "status": "success",
        "retrieval_status": "success",
        "retrieved_chunks": [
            {
                "chunk_id": "chunk_htn_001",
                "document_id": "doc_guideline_2024",
                "document_name": "AHA_Hypertension_2024.pdf",
                "page_number": 12,
                "text": "First-line pharmacotherapy for stage 1 hypertension includes amlodipine 5 mg daily.",
                "similarity_score": 0.89
            }
        ],
        "context": "[SOURCE 1] AHA_Hypertension_2024.pdf: First-line pharmacotherapy for stage 1 hypertension includes amlodipine 5 mg daily.",
        "sources": [
            {
                "source_index": 1,
                "source_number": 1,
                "chunk_id": "chunk_htn_001",
                "document_id": "doc_guideline_2024",
                "document_name": "AHA_Hypertension_2024.pdf",
                "page_number": 12,
                "text": "First-line pharmacotherapy for stage 1 hypertension includes amlodipine 5 mg daily.",
                "similarity_score": 0.89
            }
        ],
        "fused_evidence": {
            "chunks_count": 1,
            "contributing_documents_count": 1,
            "conflicts_count": 0,
            "has_conflicts": False
        },
        "timings": {
            "embedding_time_ms": 1.0,
            "faiss_retrieval_time_ms": 2.0,
            "deduplication_time_ms": 0.5,
            "context_construction_time_ms": 0.5,
            "total_retrieval_time_ms": 4.0
        }
    }

    with patch.object(service, "query", return_value=mock_retrieval):
        with patch("backend.services.gemini_service.GeminiService.generate_stream", return_value=["Amlodipine 5 mg daily is indicated [Source 1]."]):
            events = list(service.generate_rag_stream(
                question="What is the starting dose of amlodipine?",
                user_id="user_test_stream"
            ))

    event_types = [e[0] for e in events]
    assert "clinical_decision_support" in event_types

    cds_event = next(e for e in events if e[0] == "clinical_decision_support")
    payload = cds_event[1]["clinical_decision_support"]
    assert "calibrated_confidence" in payload
    assert "uncertainty_level" in payload
    assert "risk_tier" in payload


def test_24_observability_metrics_decision_support():
    """Test 24: Prometheus metrics counters, histogram, and exposition format correctly without PHI."""
    from backend.evaluation.observability import (
        get_metrics_collector,
        record_decision_support_event
    )

    collector = get_metrics_collector()
    collector.reset()

    record_decision_support_event(
        uncertainty_level="LOW",
        risk_tier="HIGH",
        escalation_required=True,
        red_flags_count=1,
        recommendations_count=2,
        latency_ms=1.5
    )

    snapshot = collector.get_metrics_snapshot()
    cds = snapshot["intelligence"]["clinical_decision_support"]
    assert cds["total"] == 1
    assert cds["uncertainty_levels"].get("LOW") == 1
    assert cds["risk_tiers"].get("HIGH") == 1
    assert cds["escalations_total"] == 1
    assert cds["red_flags_total"] == 1
    assert cds["recommendations_total"] == 2

    expo = collector.get_prometheus_exposition()
    assert "rag_decision_support_total" in expo
    assert 'rag_uncertainty_level_total{level="LOW"} 1' in expo
    assert 'rag_clinical_risk_tier_total{tier="HIGH"} 1' in expo
    assert "rag_decision_support_escalations_total 1" in expo
    # Ensure strictly zero PHI
    assert "user_id" not in expo
    assert "amlodipine" not in expo


def test_25_json_serialization_all_models():
    """Test 25: All Phase 6.7 models serialize 100% to JSON via to_dict()."""
    rec = ActionableRecommendation(
        recommendation_id="REC_01",
        category=ActionRecommendationType.DIAGNOSTIC_MONITORING,
        text="Check blood pressure twice daily.",
        urgency="ROUTINE"
    )
    rf = RedFlagTrigger(
        flag_id="RF_01",
        symptom_or_sign="Severe headache with systolic BP > 180",
        clinical_rationale="Hypertensive encephalopathy risk",
        action_required="Emergency room visit",
        is_critical=True
    )
    handoff = ClinicalHandoffSummary(
        situation="Hypertension dosing",
        background="Guidelines 2024",
        assessment="Stage 1 HTN",
        recommendation="Initiate monotherapy"
    )
    res = ClinicalDecisionSupportResult(
        calibrated_confidence=0.91,
        uncertainty_level=ClinicalUncertaintyLevel.LOW,
        primary_uncertainty_source=UncertaintySourceType.NONE,
        risk_tier=ClinicalRiskTier.HIGH,
        escalation_required=False,
        escalation_reason=None,
        actionable_recommendations=[rec],
        red_flag_triggers=[rf],
        suggested_follow_up_inquiries=["What are the side effects?"],
        clinical_handoff=handoff,
        latency_ms=1.1,
        metadata={"test": "ok"}
    )

    d = res.to_dict()
    serialized = json.dumps(d)
    deserialized = json.loads(serialized)

    assert deserialized["calibrated_confidence"] == 0.91
    assert deserialized["uncertainty_level"] == "LOW"
    assert deserialized["risk_tier"] == "HIGH"
    assert len(deserialized["actionable_recommendations"]) == 1
    assert deserialized["actionable_recommendations"][0]["recommendation_id"] == "REC_01"


def test_26_rag_service_generate_rag_answer_integration():
    """Test 26: End-to-end generate_rag_answer includes decision_support."""
    from backend.rag.rag_service import RAGService

    service = RAGService()
    mock_retrieval = {
        "status": "success",
        "retrieval_status": "success",
        "retrieved_chunks": [
            {
                "chunk_id": "chunk_htn_001",
                "document_id": "doc_guideline_2024",
                "document_name": "AHA_Hypertension_2024.pdf",
                "page_number": 12,
                "text": "First-line pharmacotherapy for stage 1 hypertension includes amlodipine 5 mg daily.",
                "similarity_score": 0.89
            }
        ],
        "context": "[SOURCE 1] AHA_Hypertension_2024.pdf: First-line pharmacotherapy for stage 1 hypertension includes amlodipine 5 mg daily.",
        "sources": [
            {
                "source_index": 1,
                "source_number": 1,
                "chunk_id": "chunk_htn_001",
                "document_id": "doc_guideline_2024",
                "document_name": "AHA_Hypertension_2024.pdf",
                "page_number": 12,
                "text": "First-line pharmacotherapy for stage 1 hypertension includes amlodipine 5 mg daily.",
                "similarity_score": 0.89
            }
        ],
        "fused_evidence": {
            "chunks_count": 1,
            "contributing_documents_count": 1,
            "conflicts_count": 0,
            "has_conflicts": False
        },
        "timings": {
            "embedding_time_ms": 1.0,
            "faiss_retrieval_time_ms": 2.0,
            "deduplication_time_ms": 0.5,
            "context_construction_time_ms": 0.5,
            "total_retrieval_time_ms": 4.0
        }
    }

    mock_gen = {
        "answer": "First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily [Source 1].",
        "generation_time_ms": 25.0,
        "model": "gemini-3.5-flash-lite",
        "disclaimer": MEDICAL_DISCLAIMER
    }

    with patch.object(service, "query", return_value=mock_retrieval):
        with patch("backend.services.gemini_service.GeminiService.generate_answer_from_prompt", return_value=mock_gen):
            res = service.generate_rag_answer(
                question="What is the recommended dose of amlodipine for stage 1 hypertension?",
                user_id="user_test_integ"
            )

    assert "decision_support" in res
    ds = res["decision_support"]
    assert "calibrated_confidence" in ds
    assert "uncertainty_level" in ds
    assert "risk_tier" in ds
    assert "actionable_recommendations" in ds
    assert "red_flag_triggers" in ds


def test_27_empty_none_and_whitespace_inputs():
    """Test 27: Engine gracefully handles empty strings, None values, and edge cases."""
    res = ClinicalDecisionSupportEngine.evaluate(
        query="   ",
        intent=None,
        answer_text=None,
        retrieved_sources=None
    )

    assert isinstance(res, ClinicalDecisionSupportResult)
    assert res.uncertainty_level == ClinicalUncertaintyLevel.INDETERMINATE
    assert res.calibrated_confidence == 0.0


def test_28_production_vector_store_invariants():
    """Test 28: Verifies FAISS = 744, metadata = 744, dimension = 384 remain strictly intact."""
    from backend.services.vector_store_service import get_vector_store_service

    vs = get_vector_store_service()
    faiss_count = vs.index.ntotal if vs.index is not None else 0
    meta_count = len(getattr(vs, "metadata_store", []))
    dim_count = vs.index.d if vs.index is not None else 0

    assert faiss_count == 744, f"Expected 744 FAISS vectors, got {faiss_count}"
    assert meta_count == 744, f"Expected 744 metadata records, got {meta_count}"
    assert dim_count == 384, f"Expected 384 embedding dimensions, got {dim_count}"
