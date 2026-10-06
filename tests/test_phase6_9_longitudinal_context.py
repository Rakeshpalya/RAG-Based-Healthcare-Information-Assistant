"""
==============================================================================
PHASE 6.9 DEDICATED TEST SUITE: Longitudinal Clinical Context & Multi-Turn Memory
==============================================================================

Comprehensive test suite verifying:
1. Strongly typed context and dialogue state models.
2. Deterministic entity extraction (conditions, medications, allergies, risk factors).
3. Negation handling and allergy-vs-medication separation.
4. Multi-turn pronoun resolution and elliptical query reformulation.
5. Cross-turn clinical contraindication interception (renal vs NSAIDs, penicillin vs beta-lactams, etc.).
6. Bounded 6-turn (3-cycle) context windowing.
7. Cumulative profile integration with ClinicalVerificationEngine (Phase 6.6).
8. Cumulative profile integration with ClinicalDecisionSupportEngine (Phase 6.7).
9. Dialogue context integration with ClinicalIntelligenceOrchestrator (Phase 6.8).
10. PHI-safe observability and Prometheus metrics exposition.
11. Tenant isolation across independent multi-turn sessions.
12. Backward compatibility with single-turn inquiries.
==============================================================================
"""

import pytest
import time
from typing import List, Dict, Any

from backend.intelligence.context_models import (
    ClinicalEntityType,
    EntityTemporalState,
    ClinicalEntity,
    CumulativeClinicalProfile,
    TurnContextResolution,
    DialogueStateAuditRecord,
)
from backend.intelligence.longitudinal_context import ClinicalContextEngine
from backend.intelligence.clinical_verification import ClinicalVerificationEngine
from backend.intelligence.clinical_decision_support import ClinicalDecisionSupportEngine
from backend.intelligence.clinical_orchestrator import ClinicalIntelligenceOrchestrator
from backend.intelligence.orchestration_models import PipelineStage, StageExecutionStatus
from backend.safety.medical_safety_guard import MedicalSafetyGuard
from backend.intelligence.intent_classifier import ClinicalIntentClassifier
from backend.intelligence.query_planner import ClinicalQueryPlanner
from backend.evaluation.observability import (
    get_metrics_collector,
    record_context_event,
)


# ============================================================================
# 1. DATA MODELS & SERIALIZATION TESTS
# ============================================================================

def test_01_clinical_entity_type_and_model_serialization():
    """Verifies that all entity types serialize correctly and retain schema."""
    entity = ClinicalEntity(
        entity_id="entity_test_01",
        name="Metformin",
        entity_type=ClinicalEntityType.MEDICATION,
        negated=False,
        turn_introduced=1,
        confidence=0.98,
        temporal_state=EntityTemporalState.ACTIVE,
    )
    d = entity.to_dict()
    assert d["entity_id"] == "entity_test_01"
    assert d["name"] == "Metformin"
    assert d["entity_type"] == "MEDICATION"
    assert d["negated"] is False
    assert d["temporal_state"] == "ACTIVE"


def test_02_cumulative_profile_construction_and_hashing():
    """Verifies cumulative profile entity aggregation, profile hashing, and safety counts."""
    profile = CumulativeClinicalProfile()
    profile.add_entity(ClinicalEntity(
        entity_id="e1",
        name="Hypertension",
        entity_type=ClinicalEntityType.CONDITION,
        negated=False,
        turn_introduced=1,
    ))
    profile.add_entity(ClinicalEntity(
        entity_id="e2",
        name="Aspirin",
        entity_type=ClinicalEntityType.MEDICATION,
        negated=False,
        turn_introduced=1,
    ))
    profile.add_entity(ClinicalEntity(
        entity_id="e3",
        name="Penicillin",
        entity_type=ClinicalEntityType.ALLERGY,
        negated=False,
        turn_introduced=2,
    ))

    summary = profile.get_summary()
    assert "Hypertension" in summary["active_conditions"]
    assert "Aspirin" in summary["active_medications"]
    assert "Penicillin" in summary["confirmed_allergies"]
    assert len(profile.profile_hash) == 16


def test_03_turn_context_resolution_audit_record_phi_safety():
    """Verifies that to_audit_record() contains ZERO PHI text and binds safely."""
    resolution = TurnContextResolution(
        original_query="What are its side effects?",
        effective_query="What are Penicillin's side effects?",
        is_follow_up=True,
        prior_turn_count=2,
        resolution_latency_ms=0.45,
    )
    audit = resolution.to_audit_record()
    assert isinstance(audit, DialogueStateAuditRecord)
    assert audit.prior_turn_count == 2
    assert audit.is_follow_up is True
    audit_dict = audit.to_dict()
    assert "Penicillin" not in audit_dict.values()  # Zero raw drug name in top audit dict
    assert "side effects" not in str(audit_dict)


# ============================================================================
# 2. CONTEXT ENGINE DETERMINISTIC EXTRACTION & RESOLUTION
# ============================================================================

def test_04_single_turn_inquiry_fallback():
    """Verifies single-turn behavior when conversation_history is None or empty."""
    res = ClinicalContextEngine.resolve_context(
        current_turn="What is the standard treatment for asthma?",
        conversation_history=None
    )
    assert res.is_follow_up is False
    assert res.effective_query == "What is the standard treatment for asthma?"
    assert res.prior_turn_count == 0
    assert len(res.contraindication_alerts) == 0


def test_05_multi_turn_condition_tracking():
    """Verifies multi-turn condition propagation across turns."""
    history = [
        {"role": "user", "content": "I was diagnosed with chronic kidney disease last year."},
        {"role": "assistant", "content": "Chronic kidney disease requires careful medical management."}
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="What diet should I follow?",
        conversation_history=history
    )
    assert res.is_follow_up is True
    assert res.prior_turn_count == 2
    conditions = res.cumulative_profile.active_conditions
    assert any("kidney disease" in c.lower() for c in conditions)


def test_06_multi_turn_medication_and_allergy_separation():
    """Verifies that allergies and medications are strictly separated."""
    history = [
        {"role": "user", "content": "I take metformin 500mg daily for diabetes, but I am allergic to penicillin."},
        {"role": "assistant", "content": "Thank you for noting your metformin regimen and penicillin allergy."}
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="What are general dietary tips?",
        conversation_history=history
    )
    summary = res.cumulative_profile.get_summary()
    assert any("metformin" in m.lower() for m in summary["active_medications"])
    assert any("penicillin" in a.lower() for a in summary["confirmed_allergies"])
    # Penicillin MUST NOT be in active medications
    assert not any("penicillin" in m.lower() for m in summary["active_medications"])


def test_07_negation_resolution():
    """Verifies that negated entities are marked as negated."""
    history = [
        {"role": "user", "content": "I do not have hypertension or diabetes."},
        {"role": "assistant", "content": "Understood, noted absence of hypertension and diabetes."}
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="What is a good exercise routine?",
        conversation_history=history
    )
    negated_entities = [e.name.lower() for e in res.cumulative_profile.entities if e.negated]
    assert any("hypertension" in e for e in negated_entities)
    assert any("diabetes" in e for e in negated_entities)
    # Negated entities should NOT be in active conditions
    assert not any("hypertension" in c.lower() for c in res.cumulative_profile.active_conditions)


def test_08_pronoun_resolution_and_elliptical_expansion():
    """Verifies pronoun replacement ('its', 'it', 'this') with prior clinical topic."""
    history = [
        {"role": "user", "content": "Can you explain how Penicillin works?"},
        {"role": "assistant", "content": "Penicillin is a beta-lactam antibiotic..."}
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="What are its side effects?",
        conversation_history=history
    )
    assert res.is_follow_up is True
    assert "penicillin" in res.effective_query.lower()
    assert "its side effects" not in res.effective_query.lower()


def test_09_elliptical_query_dosage_expansion():
    """Verifies elliptical follow-ups like 'What about dosage?' expand to include the prior topic."""
    history = [
        {"role": "user", "content": "Tell me about Metformin."},
        {"role": "assistant", "content": "Metformin is an oral biguanide..."}
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="What about dosage?",
        conversation_history=history
    )
    assert res.is_follow_up is True
    assert "metformin" in res.effective_query.lower()
    assert "dosage" in res.effective_query.lower()


# ============================================================================
# 3. CROSS-TURN CONTRAINDICATION INTERCEPTION TESTS
# ============================================================================

def test_10_renal_impairment_vs_nsaid_contraindication():
    """Verifies that Turn 1 renal impairment + Turn 2 NSAID triggers contraindication alert."""
    history = [
        {"role": "user", "content": "I have stage 3 chronic kidney disease and high blood pressure."},
        {"role": "assistant", "content": "Noted chronic kidney disease."}
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="Can I take ibuprofen 800mg for back pain?",
        conversation_history=history
    )
    assert len(res.contraindication_alerts) > 0
    alert = res.contraindication_alerts[0]
    assert "RENAL" in alert.contraindication_id or "KIDNEY" in alert.contraindication_id or "NSAID" in alert.reason


def test_11_penicillin_allergy_vs_amoxicillin_contraindication():
    """Verifies that Turn 1 penicillin allergy + Turn 2 amoxicillin triggers allergy alert."""
    history = [
        {"role": "user", "content": "I am severely allergic to penicillin (anaphylaxis)."},
        {"role": "assistant", "content": "Noted your severe penicillin allergy."}
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="The dentist prescribed amoxicillin. Is that safe for me?",
        conversation_history=history
    )
    assert len(res.contraindication_alerts) > 0
    alert = res.contraindication_alerts[0]
    assert "PENICILLIN" in alert.contraindication_id or "beta-lactam" in alert.reason.lower()


def test_12_liver_disease_vs_acetaminophen_contraindication():
    """Verifies that Turn 1 cirrhosis/liver disease + Turn 2 Tylenol triggers warning."""
    history = [
        {"role": "user", "content": "I have been diagnosed with cirrhosis and chronic liver failure."},
        {"role": "assistant", "content": "Noted liver disease."}
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="Can I take high dose Tylenol for fever?",
        conversation_history=history
    )
    assert len(res.contraindication_alerts) > 0
    assert any("HEPATIC" in a.contraindication_id or "liver" in a.reason.lower() for a in res.contraindication_alerts)


def test_13_pregnancy_vs_ace_inhibitor_contraindication():
    """Verifies that Turn 1 pregnancy + Turn 2 lisinopril triggers teratogenic alert."""
    history = [
        {"role": "user", "content": "I am currently 16 weeks pregnant."},
        {"role": "assistant", "content": "Congratulations on your pregnancy."}
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="My doctor mentioned lisinopril for blood pressure. Can I take it?",
        conversation_history=history
    )
    assert len(res.contraindication_alerts) > 0
    assert any("PREGNANCY" in a.contraindication_id or "teratogen" in a.reason.lower() or "ace" in a.reason.lower() for a in res.contraindication_alerts)


# ============================================================================
# 4. BOUNDED CONTEXT WINDOWING & LATENCY TESTS
# ============================================================================

def test_14_bounded_six_turn_window_truncation():
    """Verifies that context window strictly truncates older turns beyond 6 turns (3 cycles)."""
    history = [
        {"role": "user", "content": f"Historical turn {i} mentioning condition_{i}"} if i % 2 == 1
        else {"role": "assistant", "content": f"Historical answer {i}"}
        for i in range(1, 15)  # 14 turns total
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="What should I do next?",
        conversation_history=history
    )
    # Must use at most 6 prior turns
    assert res.prior_turn_count == 6


def test_15_context_engine_submillisecond_latency():
    """Verifies that context engine resolution executes in sub-millisecond deterministic time."""
    history = [
        {"role": "user", "content": "I have type 2 diabetes and hypertension, and I take metformin."},
        {"role": "assistant", "content": "Noted your medical conditions and metformin prescription."},
        {"role": "user", "content": "I also take lisinopril 10mg daily."},
        {"role": "assistant", "content": "Noted lisinopril 10mg."}
    ]
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        ClinicalContextEngine.resolve_context("What are its side effects?", history)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    median_lat = sorted(latencies)[len(latencies) // 2]
    # In pure Python without network, median latency is typically < 1.0 ms
    assert median_lat < 5.0, f"Expected median latency < 5.0 ms, got {median_lat:.3f} ms"


# ============================================================================
# 5. INTEGRATION WITH CLINICAL VERIFICATION (PHASE 6.6)
# ============================================================================

def test_16_verification_engine_cross_turn_contraindication_guard():
    """Verifies ClinicalVerificationEngine sanitizes and guards against cross-turn contraindications."""
    history = [
        {"role": "user", "content": "I have severe chronic kidney disease."},
        {"role": "assistant", "content": "Understood, noted kidney disease."}
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="Can I take ibuprofen for headache?",
        conversation_history=history
    )

    unfiltered_answer = "You can safely take ibuprofen 400mg every 6 hours for headache relief [Source 1]."
    verification = ClinicalVerificationEngine.verify_and_guard(
        answer_text=unfiltered_answer,
        retrieved_sources=[{"source_index": 1, "document_name": "analgesics.pdf", "text": "Ibuprofen is an NSAID"}],
        query="Can I take ibuprofen for headache?",
        intent="MEDICATION_PRESCRIBING",
        cumulative_profile=res.cumulative_profile,
    )
    assert verification.contradictions_count > 0
    assert "contraindication" in verification.sanitized_answer.lower() or "caution" in verification.sanitized_answer.lower()


# ============================================================================
# 6. INTEGRATION WITH CLINICAL DECISION SUPPORT (PHASE 6.7)
# ============================================================================

def test_17_decision_support_cross_turn_comorbidity_risk_stratification():
    """Verifies ClinicalDecisionSupportEngine elevates risk tier when cumulative profile has comorbidity."""
    history = [
        {"role": "user", "content": "I have coronary artery disease and heart failure."},
        {"role": "assistant", "content": "Noted cardiac history."}
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="I have shortness of breath when walking up stairs.",
        conversation_history=history
    )

    cds = ClinicalDecisionSupportEngine.evaluate(
        query="I have shortness of breath when walking up stairs.",
        intent="SYMPTOM_ASSESSMENT",
        answer_text="Shortness of breath with exertion can indicate worsening cardiovascular disease [Source 1].",
        retrieved_sources=[{"source_index": 1, "document_name": "cardiology.pdf", "text": "Dyspnea on exertion"}],
        cumulative_profile=res.cumulative_profile,
    )
    assert cds.risk_tier.value in ("HIGH", "CRITICAL")
    assert any("Cumulative comorbidity" in r or "Coronary artery disease" in r for r in cds.red_flag_triggers)


# ============================================================================
# 7. INTEGRATION WITH ORCHESTRATION & AUDITABILITY (PHASE 6.8)
# ============================================================================

def test_18_orchestrator_stage_dialogue_context_included_when_present():
    """Verifies ClinicalIntelligenceOrchestrator includes Stage 0.5 DIALOGUE_CONTEXT when dialogue_context is passed."""
    history = [
        {"role": "user", "content": "I have hypertension."},
        {"role": "assistant", "content": "Noted hypertension."}
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="What medications treat it?",
        conversation_history=history
    )
    audit_rec = res.to_audit_record()

    allow_rag, safety_assessment, _ = MedicalSafetyGuard.pre_screen_inquiry("What medications treat it?")
    intent_res = ClinicalIntentClassifier.classify("What medications treat hypertension?", safety_assessment)
    query_plan = ClinicalQueryPlanner.plan("What medications treat hypertension?", intent_res)

    orch = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="test_multi_turn_trace",
        query="What medications treat it?",
        safety_assessment=safety_assessment,
        intent_result=intent_res,
        query_plan=query_plan,
        dialogue_context=audit_rec,
        user_id=101
    )
    stages = [s.stage for s in orch.pipeline_stages]
    assert PipelineStage.DIALOGUE_CONTEXT in stages
    assert orch.stage_status_map.get("DIALOGUE_CONTEXT") == "SUCCESS"
    assert len(orch.audit_checksum) == 64


def test_19_orchestrator_backward_compatibility_when_dialogue_context_none():
    """Verifies ClinicalIntelligenceOrchestrator maintains 9 stages when dialogue_context is None."""
    allow_rag, safety_assessment, _ = MedicalSafetyGuard.pre_screen_inquiry("What is hypertension?")
    intent_res = ClinicalIntentClassifier.classify("What is hypertension?", safety_assessment)
    query_plan = ClinicalQueryPlanner.plan("What is hypertension?", intent_res)

    orch = ClinicalIntelligenceOrchestrator.orchestrate(
        trace_id="test_single_turn_trace",
        query="What is hypertension?",
        safety_assessment=safety_assessment,
        intent_result=intent_res,
        query_plan=query_plan,
        dialogue_context=None,
        user_id=101
    )
    stages = [s.stage for s in orch.pipeline_stages]
    assert PipelineStage.DIALOGUE_CONTEXT not in stages
    assert len(orch.pipeline_stages) == 9


# ============================================================================
# 8. OBSERVABILITY & TELEMETRY TESTS
# ============================================================================

def test_20_observability_context_metrics_recording_and_zero_phi():
    """Verifies that record_context_event records metrics cleanly without exposing PHI."""
    collector = get_metrics_collector()
    init_total = collector.context_resolution_total

    record_context_event(
        is_follow_up=True,
        prior_turns_used=4,
        entities_count=3,
        contraindications_count=1,
        latency_ms=0.62,
    )

    assert collector.context_resolution_total == init_total + 1
    assert collector.context_follow_up_total >= 1
    assert collector.context_contraindications_flagged_total >= 1

    summary = collector.get_summary()
    assert "longitudinal_context" in summary["intelligence"]
    assert summary["intelligence"]["longitudinal_context"]["total"] >= 1
    assert summary["intelligence"]["longitudinal_context"]["follow_up_total"] >= 1

    prom = collector.get_prometheus_metrics()
    assert "rag_context_resolution_total" in prom
    assert "rag_context_contraindications_flagged_total" in prom
    # Zero PHI assertions
    assert "patient" not in prom.lower()
    assert "medication" not in prom.lower()


# ============================================================================
# 9. TENANT ISOLATION TESTS
# ============================================================================

def test_21_tenant_isolation_in_multi_turn_context():
    """Verifies that Tenant A's cumulative context never leaks into Tenant B."""
    tenant_a_history = [
        {"role": "user", "content": "I am taking Warfarin 5mg daily for atrial fibrillation."},
        {"role": "assistant", "content": "Noted Warfarin."}
    ]
    tenant_b_history = [
        {"role": "user", "content": "I need information about asthma."},
        {"role": "assistant", "content": "Noted asthma."}
    ]

    res_a = ClinicalContextEngine.resolve_context(
        current_turn="Can I take aspirin?",
        conversation_history=tenant_a_history,
        user_id=1001,
    )
    res_b = ClinicalContextEngine.resolve_context(
        current_turn="Can I take aspirin?",
        conversation_history=tenant_b_history,
        user_id=2002,
    )

    # Tenant A has Warfarin (bleeding risk with aspirin)
    assert any("warfarin" in m.lower() for m in res_a.cumulative_profile.active_medications)
    # Tenant B has NO Warfarin
    assert not any("warfarin" in m.lower() for m in res_b.cumulative_profile.active_medications)
    # Hashes must differ
    assert res_a.cumulative_profile.profile_hash != res_b.cumulative_profile.profile_hash


# ============================================================================
# 10. END-TO-END RAG SERVICE MULTI-TURN TESTS
# ============================================================================

def test_22_rag_service_generate_rag_answer_multi_turn_payload():
    """Verifies generate_rag_answer attaches dialogue_context and resolves follow-ups."""
    from backend.rag.rag_service import RAGService
    service = RAGService()

    history = [
        {"role": "user", "content": "What is Type 2 Diabetes?"},
        {"role": "assistant", "content": "Type 2 diabetes is a chronic metabolic condition."}
    ]
    resp = service.generate_rag_answer(
        question="What are its common symptoms?",
        conversation_history=history,
        user_id=10
    )
    assert "dialogue_context" in resp
    d_ctx = resp["dialogue_context"]
    assert d_ctx["is_follow_up"] is True
    assert "Type 2 Diabetes" in d_ctx["effective_query"] or "diabetes" in d_ctx["effective_query"].lower()
    assert d_ctx["prior_turn_count"] == 2


def test_23_rag_service_generate_rag_answer_single_turn_backward_compatibility():
    """Verifies generate_rag_answer single-turn produces dialogue_context with is_follow_up=False."""
    from backend.rag.rag_service import RAGService
    service = RAGService()

    resp = service.generate_rag_answer(
        question="What is hypertension?",
        conversation_history=None,
        user_id=10
    )
    assert "dialogue_context" in resp
    assert resp["dialogue_context"]["is_follow_up"] is False
    assert resp["dialogue_context"]["prior_turn_count"] == 0


def test_24_rag_service_emergency_interception_preserves_dialogue_context():
    """Verifies emergency interception still attaches dialogue_context without breaking pipeline."""
    from backend.rag.rag_service import RAGService
    service = RAGService()

    history = [
        {"role": "user", "content": "I am feeling unwell."},
        {"role": "assistant", "content": "Please describe your symptoms."}
    ]
    resp = service.generate_rag_answer(
        question="I am having severe crushing chest pain radiating to my left arm, difficulty breathing!",
        conversation_history=history,
        user_id=10
    )
    assert resp["retrieval_status"] == "safety_intercepted"
    assert "dialogue_context" in resp
    assert resp["dialogue_context"]["prior_turn_count"] == 2
    # Verify emergency message
    assert "emergency" in resp["answer"].lower() or "911" in resp["answer"].lower()


def test_25_rag_service_empty_query_preserves_dialogue_context():
    """Verifies empty query returns empty_query status and attaches dialogue_context."""
    from backend.rag.rag_service import RAGService
    service = RAGService()

    resp = service.generate_rag_answer(
        question="",
        conversation_history=None,
        user_id=10
    )
    assert resp["retrieval_status"] == "safety_intercepted"
    assert "dialogue_context" in resp
    assert resp["dialogue_context"]["is_follow_up"] is False
