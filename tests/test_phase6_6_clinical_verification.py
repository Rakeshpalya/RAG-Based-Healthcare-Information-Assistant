"""
Phase 6.6 Tests: Clinical Grounding Verification & Hallucination Guardrails.

Comprehensive test suite verifying all Phase 6.6 requirements:
1. Basic grounded answer verification (high grounding score, GROUNDED status)
2. Directional contradiction detection (UP vs DOWN inversion)
3. Directional contradiction detection (DOWN vs UP inversion)
4. Negation conflict detection (contraindicated vs indicated)
5. Negation conflict detection (indicated vs contraindicated)
6. Numerical dosage discrepancy detection (deviating milligram amounts)
7. Unsubstantiated medication entity detection (fabricated drug names)
8. Unsubstantiated diagnostic entity detection (fabricated clinical stages/types)
9. Prohibited prescriptive phrasing sanitization ("you must take" -> safe framing)
10. Prohibited diagnostic pronouncement sanitization ("you have stage 2" -> safe framing)
11. Explicit negative document scope & boundary enforcement
12. Medical disclaimer enforcement (mandatory disclaimer attached)
13. Unsupported claim pruning on mixed-grounded answers
14. All-claims-unsupported conservative fallback halting
15. Insufficient clinical content after pruning triggers fallback
16. Standard refusal & safety advisory bypass (zero false-positive hallucination flags)
17. Structural and disclaimer exemption from verification penalties
18. Multi-tenant isolation verification across distinct user_ids
19. Cache isolation verification preserving verification metadata
20. Safety precedence: emergency symptom bypass
21. Safety precedence: self-harm inquiry bypass
22. Safety precedence: poisoning inquiry bypass
23. SSE streaming clinical_verification event broadcasting
24. Observability metrics and Prometheus exposition without PHI
25. Complete JSON serialization of all Phase 6.6 verification models
26. RAGService.generate_rag_answer end-to-end integration
27. Empty, None, and whitespace edge case handling
28. Production vector store invariants verification (744, 744, 384)
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from backend.intelligence.verification_models import (
    GroundingVerificationStatus,
    ClinicalHallucinationType,
    SafetyPostScreenAction,
    ExtractedClinicalEntities,
    ClinicalVerificationClaim,
    ClinicalVerificationResult
)
from backend.intelligence.clinical_verification import ClinicalVerificationEngine
from backend.intelligence.citation_models import (
    CitationVerificationStatus,
    ClinicalClaimType,
    AttributedEvidenceSpan,
    ClinicalClaimAttribution,
    CitationAttributionReport
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
            "text": "First-line pharmacotherapy for stage 1 hypertension includes thiazide diuretics, calcium channel blockers, or ACE inhibitors. The standard starting dose for amlodipine is 5 mg daily, which effectively reduces systolic blood pressure by 8 to 12 mmHg.",
            "preview_text": "First-line pharmacotherapy for stage 1 hypertension includes thiazide diuretics, calcium channel blockers...",
            "similarity_score": 0.89
        },
        {
            "source_index": 2,
            "source_number": 2,
            "chunk_id": "chunk_htn_002",
            "document_id": "doc_guideline_2024",
            "document_name": "AHA_Hypertension_2024.pdf",
            "page_number": 15,
            "text": "ACE inhibitors such as lisinopril are strictly contraindicated during pregnancy due to fetal renal toxicity. Beta-blockers are not recommended as first-line therapy for uncomplicated hypertension.",
            "preview_text": "ACE inhibitors such as lisinopril are strictly contraindicated during pregnancy...",
            "similarity_score": 0.84
        },
        {
            "source_index": 3,
            "source_number": 3,
            "chunk_id": "chunk_scope_003",
            "document_id": "doc_boundary_manual",
            "document_name": "Clinical_Trial_Scope.pdf",
            "page_number": 3,
            "text": "Testing boundary: This clinical guideline does not contain information about pediatric dosing or rare pheochromocytoma management; these conditions are outside the scope of this document.",
            "preview_text": "Testing boundary: This clinical guideline does not contain information about pediatric dosing...",
            "similarity_score": 0.78
        }
    ]


# ---------------------------------------------------------------------------
# Unit & Functional Tests
# ---------------------------------------------------------------------------

def test_01_basic_grounded_answer_verification(sample_sources):
    """Test 1: Fully substantiated clinical answer passes verification with high grounding score."""
    answer = "Amlodipine 5 mg daily effectively reduces systolic blood pressure [Source 1]."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="What is the starting dose and effect of amlodipine?"
    )

    assert result.is_verified_safe is True
    assert result.overall_grounding_score == 1.0
    assert result.grounded_claims_count >= 1
    assert result.ungrounded_claims_count == 0
    assert result.contradictions_count == 0
    assert result.hallucinations_detected == 0
    assert result.fallback_triggered is False
    assert result.action_taken in (SafetyPostScreenAction.ALLOW, SafetyPostScreenAction.ATTACH_DISCLAIMER)
    assert MEDICAL_DISCLAIMER in result.sanitized_answer


def test_02_directional_contradiction_up_vs_down(sample_sources):
    """Test 2: Claim asserts blood pressure increases when evidence states it reduces."""
    answer = "Amlodipine 5 mg daily increases blood pressure significantly [Source 1]."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="How does amlodipine affect blood pressure?"
    )

    assert result.contradictions_count >= 1
    claim = result.claim_verifications[0]
    assert claim.verification_status == GroundingVerificationStatus.DIRECTIONAL_CONTRADICTION
    assert claim.hallucination_type == ClinicalHallucinationType.DIRECTIONAL_INVERSION
    assert claim.is_grounded is False
    assert result.fallback_triggered is True


def test_03_directional_contradiction_down_vs_up():
    """Test 3: Claim asserts risk factor decreases risk when evidence states it elevates risk."""
    sources = [{
        "source_index": 1,
        "source_number": 1,
        "text": "Chronic smoking elevates cardiovascular disease risk and increases arterial stiffness.",
        "similarity_score": 0.85
    }]
    answer = "Chronic smoking reduces cardiovascular risk [Source 1]."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sources,
        query="How does smoking affect cardiovascular risk?"
    )

    assert result.contradictions_count >= 1
    assert result.claim_verifications[0].hallucination_type == ClinicalHallucinationType.DIRECTIONAL_INVERSION
    assert result.fallback_triggered is True


def test_04_negation_conflict_contraindicated_vs_indicated(sample_sources):
    """Test 4: Claim falsely asserts drug is contraindicated when guideline states it is first-line."""
    answer = "Amlodipine is contraindicated as first-line therapy for hypertension [Source 1]."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="Is amlodipine contraindicated?"
    )

    assert result.contradictions_count >= 1
    assert result.claim_verifications[0].hallucination_type == ClinicalHallucinationType.NEGATION_CONFLICT


def test_05_negation_conflict_indicated_vs_contraindicated(sample_sources):
    """Test 5: Claim recommends a drug during pregnancy when guideline states it is strictly contraindicated."""
    answer = "Lisinopril is safe and recommended during pregnancy [Source 2]."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="Can pregnant patients take lisinopril?"
    )

    assert result.contradictions_count >= 1
    assert result.claim_verifications[0].hallucination_type == ClinicalHallucinationType.NEGATION_CONFLICT
    assert result.fallback_triggered is True


def test_06_dosage_discrepancy_detection(sample_sources):
    """Test 6: Claim asserts 150 mg starting dose when evidence specifies 5 mg."""
    answer = "The standard starting dose for amlodipine is 150 mg daily [Source 1]."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="What is the starting dose of amlodipine?"
    )

    assert result.hallucinations_detected >= 1
    claim = result.claim_verifications[0]
    assert claim.verification_status == GroundingVerificationStatus.NUMERICAL_DISCREPANCY
    assert claim.hallucination_type == ClinicalHallucinationType.DOSAGE_DISCREPANCY
    assert result.fallback_triggered is True


def test_07_unsubstantiated_medication_entity(sample_sources):
    """Test 7: Model hallucinates a medication completely absent from reference passages."""
    answer = "Ozempic and Wegovy are first-line treatments for stage 1 hypertension [Source 1]."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="What drugs are used?"
    )

    assert result.hallucinations_detected >= 1
    claim = result.claim_verifications[0]
    assert claim.verification_status == GroundingVerificationStatus.UNSUBSTANTIATED_ENTITY
    assert claim.hallucination_type == ClinicalHallucinationType.MEDICATION_FABRICATION
    assert result.fallback_triggered is True


def test_08_unsubstantiated_diagnostic_entity(sample_sources):
    """Test 8: Model introduces an ungrounded clinical diagnosis stage absent from evidence."""
    answer = "Patients diagnosed with stage 4 glioblastoma require immediate craniotomy [Source 1]."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="What about glioblastoma?"
    )

    assert result.hallucinations_detected >= 1
    assert result.fallback_triggered is True


def test_09_prohibited_prescriptive_sanitization(sample_sources):
    """Test 9: Imperative prescriptive statements are rewritten to objective informational phrasing."""
    answer = "You must take 5 mg of amlodipine daily to lower your blood pressure [Source 1]."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="How much amlodipine should I take?"
    )

    assert result.is_verified_safe is True
    assert "you must take" not in result.sanitized_answer.lower()
    assert "clinical guidelines note that physicians may prescribe" in result.sanitized_answer.lower()
    assert result.action_taken == SafetyPostScreenAction.SANITIZE_PRESCRIPTION


def test_10_prohibited_diagnostic_sanitization(sample_sources):
    """Test 10: Definitive diagnostic assertions are rewritten to exploratory evidence phrasing."""
    answer = "You have stage 1 hypertension based on your symptoms [Source 1]."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="Do I have hypertension?"
    )

    assert result.is_verified_safe is True
    assert "you have stage 1" not in result.sanitized_answer.lower()
    assert "clinical evidence discusses" in result.sanitized_answer.lower()
    assert result.action_taken == SafetyPostScreenAction.SANITIZE_DIAGNOSIS


def test_11_explicit_negative_boundary_enforcement(sample_sources):
    """Test 11: Recognizes and preserves grounded negative document scope boundary statements."""
    answer = "This clinical guideline does not contain information about pediatric dosing, as it is outside the scope of this document [Source 3]."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="What is the pediatric dosing according to this document?"
    )

    assert result.negative_boundary_enforced is True
    assert result.is_verified_safe is True
    assert result.fallback_triggered is False
    assert "[Source 3]" in result.sanitized_answer


def test_12_medical_disclaimer_enforcement(sample_sources):
    """Test 12: Automatically enforces mandatory medical disclaimer in verified responses."""
    answer = "Amlodipine starting dose is 5 mg daily [Source 1]."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="What is the amlodipine dose?"
    )

    assert result.disclaimer_enforced is True
    assert MEDICAL_DISCLAIMER in result.sanitized_answer


def test_13_unsupported_claim_pruning_mixed_answer(sample_sources):
    """Test 13: Mixed answer with 2 grounded claims and 1 hallucinated sentence prunes the hallucination."""
    answer = (
        "First-line pharmacotherapy for stage 1 hypertension includes calcium channel blockers [Source 1]. "
        "The standard starting dose for amlodipine is 5 mg daily [Source 1]. "
        "Patients should also take 100 mg of Dapagliflozin immediately [Source 1]."
    )
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="What are the medications and starting dose for stage 1 hypertension?"
    )

    assert result.is_verified_safe is True
    assert result.grounded_claims_count >= 2
    assert result.ungrounded_claims_count == 1
    assert "Dapagliflozin" not in result.sanitized_answer
    assert "Amlodipine" in result.sanitized_answer or "calcium channel blockers" in result.sanitized_answer
    assert result.action_taken == SafetyPostScreenAction.PRUNE_UNSUPPORTED


def test_14_all_claims_unsupported_triggers_fallback(sample_sources):
    """Test 14: Answer where all factual assertions are unsupported triggers conservative fallback."""
    answer = "Take 250 mg of Metformin and 50 mg of Atorvastatin for immediate hypertension cure [Source 1]."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="How to cure hypertension?"
    )

    assert result.fallback_triggered is True
    assert result.is_verified_safe is False
    assert result.action_taken == SafetyPostScreenAction.TRIGGER_FALLBACK
    assert ClinicalVerificationEngine.SAFE_FALLBACK_TEXT in result.sanitized_answer


def test_15_insufficient_content_after_pruning_triggers_fallback(sample_sources):
    """Test 15: Pruning leaving trivial content (< 25 characters) halts with safe fallback."""
    # A single sentence with a hallucinated dosage
    answer = "Dose is 250 mg [Source 1]."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="What is the dose?"
    )

    assert result.fallback_triggered is True
    assert ClinicalVerificationEngine.SAFE_FALLBACK_TEXT in result.sanitized_answer


def test_16_standard_refusal_advisory_bypass(sample_sources):
    """Test 16: Existing safe fallback, refusal, or emergency advisories pass through cleanly."""
    refusal = "Relevant medical information could not be found in the available reference documents."
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=refusal,
        retrieved_sources=sample_sources,
        query="What is the dose of a non-existent drug?"
    )

    assert result.is_verified_safe is True
    assert result.fallback_triggered is False
    assert result.action_taken == SafetyPostScreenAction.ALLOW
    assert result.sanitized_answer == refusal


def test_17_structural_and_disclaimer_exemption(sample_sources):
    """Test 17: Headings, bullet points, and disclaimers are exempt from verification penalties."""
    answer = (
        "### Clinical Summary\n\n"
        "Amlodipine 5 mg daily reduces blood pressure [Source 1].\n\n"
        "Please consult your personal physician before starting any treatment."
    )
    result = ClinicalVerificationEngine.verify_and_guard(
        answer_text=answer,
        retrieved_sources=sample_sources,
        query="Summary of amlodipine"
    )

    assert result.is_verified_safe is True
    assert any(c.verification_status == GroundingVerificationStatus.EXEMPT_STRUCTURAL for c in result.claim_verifications)


def test_18_multi_tenant_isolation_verification(sample_sources):
    """Test 18: Verifies document signature and multi-tenant user_id isolation."""
    from backend.rag.rag_service import RAGService
    rag_service = RAGService()
    sig_user1 = rag_service.compute_document_signature(user_id=101, document_id="doc_htn_001")
    sig_user2 = rag_service.compute_document_signature(user_id=202, document_id="doc_htn_001")

    assert sig_user1 != sig_user2
    assert len(sig_user1) == 16
    assert len(sig_user2) == 16


def test_19_cache_isolation_verification():
    """Test 19: Cache key generation isolates strategies and preserves verification integrity."""
    from backend.services.llm_cache_service import get_llm_cache_service
    cache = get_llm_cache_service()

    key_standard = cache.generate_cache_key(
        normalized_query="hypertension dosage",
        user_scope="user_101",
        document_signature="sig_alpha",
        synthesis_strategy="standard_rag"
    )
    key_adaptive = cache.generate_cache_key(
        normalized_query="hypertension dosage",
        user_scope="user_101",
        document_signature="sig_alpha",
        synthesis_strategy="multi_doc_compare"
    )

    assert key_standard != key_adaptive
    assert len(key_standard) == 64
    assert len(key_adaptive) == 64


def test_20_safety_precedence_emergency_bypass():
    """Test 20: Emergency symptoms bypass retrieval, LLM generation, attribution, and verification."""
    from backend.rag.rag_service import RAGService
    service = RAGService()

    resp = service.generate_rag_answer(
        question="I am experiencing crushing chest pain radiating to my left jaw",
        user_id=101
    )

    assert resp["retrieval_status"] == "safety_intercepted"
    assert "EMERGENCY ADVISORY" in resp["answer"]
    assert resp["timings"]["llm_called"] is False
    assert len(resp["sources"]) == 0


def test_21_safety_precedence_self_harm_bypass():
    """Test 21: Self-harm inquiries bypass retrieval, generation, attribution, and verification."""
    from backend.rag.rag_service import RAGService
    service = RAGService()

    resp = service.generate_rag_answer(
        question="I want to end my life, what pills can I take?",
        user_id=101
    )

    assert resp["retrieval_status"] == "safety_intercepted"
    assert "988" in resp["answer"]
    assert resp["timings"]["llm_called"] is False


def test_22_safety_precedence_poisoning_bypass():
    """Test 22: Poisoning inquiries bypass retrieval and generation."""
    from backend.rag.rag_service import RAGService
    service = RAGService()

    resp = service.generate_rag_answer(
        question="My child just swallowed a bottle of bleach, help",
        user_id=101
    )

    assert resp["retrieval_status"] == "safety_intercepted"
    assert "Poison Help" in resp["answer"] or "1-800-222-1222" in resp["answer"]
    assert resp["timings"]["llm_called"] is False


def test_23_sse_streaming_clinical_verification_event(sample_sources):
    """Test 23: SSE streaming yields clinical_verification event with structured payload."""
    from backend.rag.rag_service import RAGService

    service = RAGService()
    mock_llm = MagicMock()
    mock_llm.generate_answer.return_value = {
        "answer": "Amlodipine 5 mg daily effectively reduces blood pressure [Source 1].",
        "model": "gemini-1.5-flash",
        "api_request_time_ms": 120.0,
        "input_tokens": 100,
        "output_tokens": 50,
        "disclaimer": MEDICAL_DISCLAIMER
    }

    with patch.object(service, "query") as mock_query:
        mock_query.return_value = {
            "retrieval_status": "success",
            "retrieved_chunks": sample_sources,
            "sources": sample_sources,
            "context": sample_sources[0]["text"],
            "status": "success",
            "timings": {
                "embedding_time_ms": 1.0,
                "faiss_retrieval_time_ms": 2.0,
                "deduplication_time_ms": 0.5,
                "context_construction_time_ms": 0.5,
                "total_retrieval_time_ms": 4.0,
                "similarity_scores": [0.89],
                "retrieved_chunk_ids": ["c1"],
                "retrieved_document_names": ["AHA.pdf"]
            }
        }

        events = list(service.generate_rag_stream(
            question="What is the starting dose of amlodipine?",
            gemini_service=mock_llm,
            user_id=101
        ))

        event_types = [e[0] for e in events]
        assert "start" in event_types
        assert "citation_attribution" in event_types
        assert "clinical_verification" in event_types
        assert "answer_synthesis" in event_types
        assert "token" in event_types
        assert "complete" in event_types

        # Verify clinical_verification event payload
        verif_event = next(e[1] for e in events if e[0] == "clinical_verification")
        assert "clinical_verification" in verif_event
        verif_data = verif_event["clinical_verification"]
        assert verif_data["is_verified_safe"] is True
        assert verif_data["overall_grounding_score"] == 1.0

        # Verify complete event has clinical_verification
        complete_event = next(e[1] for e in events if e[0] == "complete")
        assert "clinical_verification" in complete_event


def test_24_observability_metrics_clinical_verification():
    """Test 24: Verifies low-cardinality Prometheus metrics emission with zero PHI."""
    from backend.evaluation.observability import (
        get_metrics_collector,
        record_clinical_verification_event
    )

    collector = get_metrics_collector()
    init_verif = collector.clinical_verifications_total
    init_grounded = collector.verification_grounded_claims_total

    record_clinical_verification_event(
        grounded_claims=3,
        ungrounded_claims=1,
        contradictions=1,
        hallucinations=1,
        fallback_triggered=False,
        latency_ms=2.15
    )

    assert collector.clinical_verifications_total == init_verif + 1
    assert collector.verification_grounded_claims_total == init_grounded + 3

    expo = collector.get_prometheus_exposition()
    assert "rag_clinical_verifications_total" in expo
    assert "rag_verification_grounded_claims_total" in expo
    assert "rag_verification_contradictions_total" in expo
    assert "rag_clinical_verification_duration_seconds" in expo

    # PHI verification: Ensure no query text or medical names are present in metric names or labels
    for line in expo.splitlines():
        if line.startswith("rag_verification_") or line.startswith("rag_clinical_verification"):
            assert "amlodipine" not in line.lower()
            assert "lisinopril" not in line.lower()
            assert "user_101" not in line.lower()


def test_25_json_serialization_all_models():
    """Test 25: Verifies 100% JSON serializability of all Phase 6.6 verification models."""
    claim = ClinicalVerificationClaim(
        claim_id="clm_001",
        claim_text="Amlodipine starting dose is 5 mg daily.",
        raw_sentence="Amlodipine starting dose is 5 mg daily [Source 1].",
        verification_status=GroundingVerificationStatus.GROUNDED,
        hallucination_type=ClinicalHallucinationType.NONE,
        is_grounded=True,
        confidence_score=0.92,
        discrepancy_details=[],
        supporting_sources=[1],
        entities=ExtractedClinicalEntities(
            medications=["amlodipine"],
            dosages=["5 mg"],
            direction="DOWN",
            direction_terms=["reduces"]
        )
    )

    res = ClinicalVerificationResult(
        is_verified_safe=True,
        overall_grounding_score=1.0,
        total_claims_analyzed=1,
        grounded_claims_count=1,
        ungrounded_claims_count=0,
        contradictions_count=0,
        hallucinations_detected=0,
        claim_verifications=[claim],
        action_taken=SafetyPostScreenAction.ALLOW,
        sanitized_answer="Amlodipine starting dose is 5 mg daily [Source 1].\n\n" + MEDICAL_DISCLAIMER,
        fallback_triggered=False,
        latency_ms=1.45,
        metadata={"test": "ok"}
    )

    serialized = json.dumps(res.to_dict())
    deserialized = json.loads(serialized)

    assert deserialized["is_verified_safe"] is True
    assert deserialized["overall_grounding_score"] == 1.0
    assert deserialized["action_taken"] == "ALLOW"
    assert deserialized["claim_verifications"][0]["verification_status"] == "GROUNDED"
    assert deserialized["claim_verifications"][0]["entities"]["medications"] == ["amlodipine"]


def test_26_rag_service_generate_rag_answer_integration(sample_sources):
    """Test 26: Verifies full integration of clinical verification in generate_rag_answer."""
    from backend.rag.rag_service import RAGService

    service = RAGService()
    mock_llm = MagicMock()
    mock_llm.generate_answer.return_value = {
        "answer": "Amlodipine starting dose is 5 mg daily [Source 1].",
        "model": "gemini-1.5-flash",
        "api_request_time_ms": 110.0,
        "input_tokens": 100,
        "output_tokens": 40,
        "disclaimer": MEDICAL_DISCLAIMER
    }

    with patch.object(service, "query") as mock_query:
        mock_query.return_value = {
            "retrieval_status": "success",
            "retrieved_chunks": sample_sources,
            "sources": sample_sources,
            "context": sample_sources[0]["text"],
            "status": "success",
            "timings": {
                "embedding_time_ms": 1.0,
                "faiss_retrieval_time_ms": 2.0,
                "deduplication_time_ms": 0.5,
                "context_construction_time_ms": 0.5,
                "total_retrieval_time_ms": 4.0,
                "similarity_scores": [0.89],
                "retrieved_chunk_ids": ["c1"],
                "retrieved_document_names": ["AHA.pdf"]
            }
        }

        resp = service.generate_rag_answer(
            question="What is the starting dose for amlodipine?",
            gemini_service=mock_llm,
            user_id=101
        )

        assert "clinical_verification" in resp
        verif = resp["clinical_verification"]
        assert verif["is_verified_safe"] is True
        assert verif["overall_grounding_score"] == 1.0
        assert "citation_attribution" in resp
        assert "answer_synthesis" in resp
        assert resp["answer_synthesis"]["metadata"]["clinical_verification"]["is_verified_safe"] is True


def test_27_empty_and_whitespace_edge_cases(sample_sources):
    """Test 27: Handles None, empty, and whitespace strings gracefully without exception."""
    for empty_input in [None, "", "   ", "\n\t"]:
        res = ClinicalVerificationEngine.verify_and_guard(
            answer_text=empty_input,
            retrieved_sources=sample_sources
        )
        assert res.is_verified_safe is False
        assert res.fallback_triggered is True
        assert res.action_taken == SafetyPostScreenAction.TRIGGER_FALLBACK
        assert ClinicalVerificationEngine.SAFE_FALLBACK_TEXT in res.sanitized_answer


def test_28_production_vector_store_invariants():
    """Test 28: Verifies FAISS=744, metadata=744, dimension=384 are strictly preserved."""
    from backend.services.vector_store_service import get_vector_store_service

    vs = get_vector_store_service()
    faiss_count = vs.index.ntotal if vs.index is not None else 0
    meta_count = len(getattr(vs, "metadata_store", []))
    dim_count = vs.index.d if vs.index is not None else 0

    assert faiss_count == 744, f"Expected 744 FAISS vectors, got {faiss_count}"
    assert meta_count == 744, f"Expected 744 metadata records, got {meta_count}"
    assert dim_count == 384, f"Expected 384 embedding dimensions, got {dim_count}"
