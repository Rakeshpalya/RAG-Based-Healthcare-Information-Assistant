"""
Comprehensive Unit & Integration Test Suite for Phase 3.4 — Hallucination Protection.

Validates the multi-layer medical grounding defense:
1. Medication entity hallucination detection (drug not in cited evidence).
2. Dosage hallucination detection (fabricated or mismatched dose/unit).
3. Negation contradiction detection (claim asserts what source negates).
4. Contraindication contradiction detection (claim recommends what source contraindicates).
5. Directional contradiction detection (claim asserts increase when source states decrease).
6. Unsupported diagnosis detection.
7. Unsupported medical recommendation detection.
8. Wrong-source attribution detection.
9. Partial / mixed support (prunes invalid claims, preserves grounded claims).
10. Complete hallucination rejection with safe fallback.
11. Preserving valid grounded medical statements with real dosages and medications.
12. Integration with RAGService (mock Gemini generating hallucination -> halted safely).
13. Vector store read-only invariance verification.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.evaluation.hallucination_guard import (
    HallucinationGuard,
    HallucinationType,
    HallucinationGuardResult
)
from backend.evaluation.citation_validator import (
    CitationValidator,
    CitationValidationResult
)
from backend.rag.rag_service import RAGService


@pytest.fixture
def mock_cardio_source():
    return [{
        "source_index": 1,
        "chunk_id": "CHUNK_CARDIO_01",
        "document": "Hypertension Clinical Guidelines",
        "filename": "hypertension_guidelines.pdf",
        "page": 1,
        "similarity_score": 0.88,
        "text": (
            "Hypertension is defined as persistent elevation of blood pressure. "
            "Lifestyle recommendations include regular physical activity and dietary sodium reduction. "
            "Beta-blockers are not recommended as first-line therapy for uncomplicated hypertension. "
            "Regular aerobic exercise reduces resting systolic blood pressure."
        )
    }]


@pytest.fixture
def mock_pharma_source():
    return [{
        "source_index": 1,
        "chunk_id": "CHUNK_PHARMA_01",
        "document": "Cardiology Pharmacology Compendium",
        "filename": "cardio_pharma.pdf",
        "page": 5,
        "similarity_score": 0.91,
        "text": (
            "Amlodipine 5 mg daily is indicated for the treatment of Stage 2 essential hypertension. "
            "Metformin is contraindicated in patients with severe renal impairment due to lactic acidosis risk. "
            "ACE inhibitors lower systemic vascular resistance."
        )
    }]


@pytest.fixture
def mock_multi_topic_sources():
    return [
        {
            "source_index": 1,
            "chunk_id": "CHUNK_HTN_01",
            "document": "Cardiovascular Health Handbook",
            "filename": "cardio_health.pdf",
            "page": 1,
            "similarity_score": 0.85,
            "text": "Hypertension lifestyle recommendations include physical activity and low dietary sodium."
        },
        {
            "source_index": 2,
            "chunk_id": "CHUNK_DM_01",
            "document": "Diabetes Care Standards",
            "filename": "diabetes_care.pdf",
            "page": 3,
            "similarity_score": 0.89,
            "text": "Diabetes management includes blood glucose monitoring and glycemic index control."
        }
    ]


def test_1_medication_hallucination_detection(mock_cardio_source):
    """Case 1: Claim introduces a medication (Metformin) absent from the hypertension source."""
    answer = "Metformin is recommended for hypertension. [Source 1]"
    res = HallucinationGuard.guard_answer(answer, mock_cardio_source)

    assert res.is_safe is False
    assert res.medication_hallucinations >= 1
    assert res.total_claims == 1
    assert res.supported_claims == 0
    assert res.unsupported_claims == 1
    assert res.fallback_triggered is True

    claim_audit = res.claims[0]
    assert claim_audit.is_supported is False
    assert claim_audit.hallucination_type == HallucinationType.MEDICATION_HALLUCINATION
    assert any("metformin" in r.lower() for r in claim_audit.reasons)


def test_2_dosage_hallucination_detection(mock_pharma_source):
    """Case 2: Claim cites Amlodipine with a fabricated dosage (50 mg instead of 5 mg)."""
    answer = "Amlodipine 50 mg daily is indicated for the treatment of hypertension. [Source 1]"
    res = HallucinationGuard.guard_answer(answer, mock_pharma_source)

    assert res.is_safe is False
    assert res.dosage_hallucinations >= 1
    assert res.total_claims == 1
    assert res.supported_claims == 0
    assert res.fallback_triggered is True

    claim_audit = res.claims[0]
    assert claim_audit.is_supported is False
    assert claim_audit.hallucination_type == HallucinationType.DOSAGE_HALLUCINATION
    assert any("50 mg" in r for r in claim_audit.reasons)


def test_3_negation_contradiction_detection(mock_cardio_source):
    """Case 3: Source states beta-blockers are NOT recommended; claim affirms they ARE recommended."""
    answer = "Beta-blockers are recommended as first-line therapy for uncomplicated hypertension. [Source 1]"
    res = HallucinationGuard.guard_answer(answer, mock_cardio_source)

    assert res.is_safe is False
    assert res.contradictions_detected >= 1
    assert res.fallback_triggered is True

    claim_audit = res.claims[0]
    assert claim_audit.is_supported is False
    assert claim_audit.hallucination_type == HallucinationType.NEGATION_CONTRADICTION
    assert any("negation contradiction" in r.lower() for r in claim_audit.reasons)


def test_4_contraindication_contradiction_detection(mock_pharma_source):
    """Case 4: Source states metformin is contraindicated; claim claims it is recommended and indicated."""
    answer = "Metformin is recommended and indicated for patients with severe renal impairment. [Source 1]"
    res = HallucinationGuard.guard_answer(answer, mock_pharma_source)

    assert res.is_safe is False
    assert res.contradictions_detected >= 1

    claim_audit = res.claims[0]
    assert claim_audit.is_supported is False
    assert claim_audit.hallucination_type == HallucinationType.NEGATION_CONTRADICTION
    assert any("negates assertion" in r.lower() or "contraindicat" in r.lower() for r in claim_audit.reasons)


def test_5_directional_contradiction_detection(mock_cardio_source):
    """Case 5: Source states aerobic exercise REDUCES blood pressure; claim asserts it INCREASES it."""
    answer = "Regular aerobic exercise increases resting systolic blood pressure. [Source 1]"
    res = HallucinationGuard.guard_answer(answer, mock_cardio_source)

    assert res.is_safe is False
    assert res.contradictions_detected >= 1

    claim_audit = res.claims[0]
    assert claim_audit.is_supported is False
    assert claim_audit.hallucination_type == HallucinationType.DIRECTIONAL_CONTRADICTION
    assert any("directional contradiction" in r.lower() for r in claim_audit.reasons)


def test_6_unsupported_diagnosis_detection(mock_cardio_source):
    """Case 6: Claim diagnoses pheochromocytoma when document only discusses general hypertension."""
    answer = "The patient has been diagnosed with pheochromocytoma. [Source 1]"
    res = HallucinationGuard.guard_answer(answer, mock_cardio_source)

    assert res.is_safe is False
    claim_audit = res.claims[0]
    assert claim_audit.is_supported is False
    assert claim_audit.hallucination_type in (
        HallucinationType.SEMANTIC_MISMATCH,
        HallucinationType.UNSUPPORTED_DIAGNOSIS
    )


def test_7_unsupported_medical_recommendation_detection(mock_cardio_source):
    """Case 7: Prescriptive recommendation for surgery/unmentioned intervention."""
    answer = "The patient is prescribed immediate renal denervation surgery for hypertension. [Source 1]"
    res = HallucinationGuard.guard_answer(answer, mock_cardio_source)

    assert res.is_safe is False
    assert res.unsupported_claims >= 1
    assert res.fallback_triggered is True


def test_8_wrong_source_attribution_detection(mock_multi_topic_sources):
    """Case 8: Claim cites Source 1 (Cardio) for diabetes glucose monitoring (found in Source 2)."""
    answer = "Blood glucose monitoring is recommended for diabetes management. [Source 1]"
    res = HallucinationGuard.guard_answer(answer, mock_multi_topic_sources)

    assert res.is_safe is False
    claim_audit = res.claims[0]
    assert claim_audit.is_supported is False
    assert claim_audit.hallucination_type == HallucinationType.WRONG_SOURCE
    assert any("wrong source" in r.lower() for r in claim_audit.reasons)


def test_9_partial_mixed_support_pruning(mock_cardio_source):
    """Case 9: Mixed answer containing 1 grounded claim and 1 hallucinated medication.
    Expected: Hallucinated claim is pruned; grounded claim is preserved.
    """
    answer = (
        "Lifestyle recommendations include regular physical activity [Source 1]. "
        "Metformin 500 mg daily should also be taken [Source 1]."
    )
    res = HallucinationGuard.guard_answer(answer, mock_cardio_source)

    assert res.is_safe is False
    assert res.total_claims == 2
    assert res.supported_claims == 1
    assert res.unsupported_claims == 1
    assert res.fallback_triggered is False

    # The sanitized answer retains physical activity and prunes Metformin
    assert "physical activity" in res.sanitized_answer
    assert "Metformin" not in res.sanitized_answer
    assert "[Source 1]" in res.sanitized_answer


def test_10_complete_hallucination_fallback(mock_cardio_source):
    """Case 10: Answer consisting entirely of hallucinated drugs and dosages.
    Expected: Safe fallback triggered, no hallucinated content survives.
    """
    answer = "Ozempic 2 mg weekly and Lisinopril 80 mg daily cure hypertension. [Source 1]"
    res = HallucinationGuard.guard_answer(answer, mock_cardio_source)

    assert res.is_safe is False
    assert res.supported_claims == 0
    assert res.fallback_triggered is True
    assert "insufficient evidence" in res.sanitized_answer.lower()
    assert "Ozempic" not in res.sanitized_answer


def test_11_valid_grounded_medical_statement(mock_pharma_source):
    """Case 11: Valid grounded answer with real medication and correct dosage.
    Expected: 100% supported, 0 hallucinations, is_safe == True.
    """
    answer = "Amlodipine 5 mg daily is indicated for the treatment of Stage 2 essential hypertension. [Source 1]"
    res = HallucinationGuard.guard_answer(answer, mock_pharma_source)

    assert res.is_safe is True
    assert res.total_claims == 1
    assert res.supported_claims == 1
    assert res.unsupported_claims == 0
    assert res.medication_hallucinations == 0
    assert res.dosage_hallucinations == 0
    assert res.contradictions_detected == 0
    assert res.fallback_triggered is False
    assert "Amlodipine 5 mg daily" in res.sanitized_answer


def test_12_rag_service_rejects_hallucination_with_mock_gemini():
    """Case 12: End-to-end integration test with RAGService.
    Mock Gemini generates a hallucinated answer -> RAGService halts generation safely.
    """
    mock_vs = MagicMock()
    mock_vs.search.return_value = [{
        "chunk_id": "HTN_01",
        "document_name": "Cardio.pdf",
        "document_id": "DOC_HTN_01",
        "page_number": 1,
        "score": 0.88,
        "text": "Hypertension guidelines emphasize dietary sodium restriction."
    }]

    rag_service = RAGService(vector_store=mock_vs)

    mock_gemini = MagicMock()
    mock_gemini.generate_answer.return_value = {
        "answer": "Atorvastatin 80 mg is required for immediate hypertension cure. [Source 1]",
        "model": "mock-gemini-3.5-flash",
        "disclaimer": "Medical disclaimer.",
        "generation_time_ms": 110.0,
        "status": "success"
    }

    result = rag_service.generate_rag_answer(
        question="How to cure hypertension?",
        gemini_service=mock_gemini
    )

    # Generation should have halted or safe fallback returned
    assert "Atorvastatin" not in result["answer"]
    assert any(p in result["answer"].lower() for p in ("could not be found", "insufficient evidence", "halted"))
    assert result["sources"] == []


def test_13_vector_store_read_only_invariance(mock_cardio_source):
    """Case 13: Verifies that Phase 3.4 hallucination protection is 100% read-only."""
    import faiss
    import json

    faiss_path = Path("data/vector_store/index.faiss")
    meta_path = Path("data/vector_store/metadata.json")

    idx_before = faiss.read_index(str(faiss_path)).ntotal
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_before = json.load(f)["count"]

    # Execute hallucination guard on multiple candidate answers
    HallucinationGuard.guard_answer(
        "Hypertension is defined as persistent elevation of blood pressure. [Source 1]",
        mock_cardio_source
    )
    HallucinationGuard.guard_answer(
        "Metformin 500 mg is prescribed for hypertension. [Source 1]",
        mock_cardio_source
    )

    idx_after = faiss.read_index(str(faiss_path)).ntotal
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_after = json.load(f)["count"]

    assert idx_after == idx_before
    assert meta_after == meta_before
