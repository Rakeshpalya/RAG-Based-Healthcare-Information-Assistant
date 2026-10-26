"""
Phase 6: Generalizable RAG Retrieval and Grounded Answer Validation Test Suite.

Automated verification covering:
1. Supported facts expressed with varied wording / natural language synonyms.
2. Risk factors, symptoms, and lifestyle recommendations retrieval and grounding.
3. Document summaries and overview queries without false pruning.
4. Document comparisons across multiple documents and single-document boundaries.
5. Multiple unrelated medical documents and scoping.
6. Deduplication of identical documents preserving max score.
7. Genuinely unsupported questions returning safe refusal (no_relevant_context).
8. Documents with no relevant answer returning structured refusal.
9. Detection and stripping of invalid, hallucinated, or ungrounded citations.
10. Empty bullet points and malformed Markdown normalization.
11. Cross-user and cross-tenant isolation enforcement.
12. Safety-sensitive questions and clinical emergency guardrails.
13. Newly uploaded synthetic documents with novel clinical conditions.
"""

import pytest
import re
from unittest.mock import MagicMock, patch

from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import VectorStoreService
from backend.evaluation.citation_validator import CitationValidator
from backend.evaluation.hallucination_guard import HallucinationGuard, HallucinationType
from backend.intelligence.intent_classifier import ClinicalIntentClassifier
from backend.intelligence.intent_models import ClinicalIntent
from backend.intelligence.query_planner import ClinicalQueryPlanner
from backend.intelligence.query_plan import MultiDocumentStrategy, GenerationStrategy
from backend.safety.medical_safety_guard import MedicalSafetyGuard


# Synthetic newly uploaded documents representing novel clinical scenarios
SYNTHETIC_NEW_UPLOAD_CHUNKS = [
    {
        "chunk_id": "NEW_HYP_01",
        "document_id": "DOC_NEW_HYP_77",
        "page_number": 1,
        "text": "Essential hypertension is persistently elevated systemic arterial pressure above 130/80 mmHg. Risk factors include sedentary lifestyle, excessive dietary sodium, obesity, chronic psychological stress, and family history.",
        "user_id": 101,
        "metadata": {"filename": "cardiology_practice_guide_2026.pdf", "user_id": 101}
    },
    {
        "chunk_id": "NEW_HYP_02",
        "document_id": "DOC_NEW_HYP_77",
        "page_number": 2,
        "text": "Lifestyle modifications for arterial hypertension include aerobic physical activity of 150 minutes weekly, the DASH diet rich in potassium, sodium restriction to under 2 grams daily, and smoking cessation.",
        "user_id": 101,
        "metadata": {"filename": "cardiology_practice_guide_2026.pdf", "user_id": 101}
    },
    {
        "chunk_id": "NEW_PULM_01",
        "document_id": "DOC_NEW_PULM_88",
        "page_number": 1,
        "text": "Idiopathic pulmonary fibrosis causes progressive dyspnea on exertion, bibasilar inspiratory crackles, and digital clubbing. High-resolution computed tomography shows subpleural reticulation with honeycombing.",
        "user_id": 101,
        "metadata": {"filename": "pulmonary_interstitial_review.pdf", "user_id": 101}
    },
    {
        "chunk_id": "NEW_DIAB_01",
        "document_id": "DOC_NEW_DIAB_99",
        "page_number": 1,
        "text": "Type 2 diabetes management involves glycemic monitoring with HbA1c target below 7.0%, metformin pharmacotherapy, and routine screening for diabetic retinopathy and peripheral nephropathy.",
        "user_id": 101,
        "metadata": {"filename": "endocrinology_standards_2026.pdf", "user_id": 101}
    },
    # Document belonging strictly to another user (User 202)
    {
        "chunk_id": "USER2_DOC_01",
        "document_id": "DOC_USER2_SECRET",
        "page_number": 1,
        "text": "Patient Robert Miller: Confidential oncology pathology report indicating localized prostate adenocarcinoma Gleason score 3+4=7.",
        "user_id": 202,
        "metadata": {"filename": "robert_miller_oncology_confidential.pdf", "user_id": 202}
    }
]


@pytest.fixture(scope="module")
def synthetic_rag():
    """Build an isolated VectorStoreService populated with synthetic new documents."""
    store = VectorStoreService()
    store.add_chunks(SYNTHETIC_NEW_UPLOAD_CHUNKS)
    rag = RAGService(vector_store=store, default_top_k=5, default_similarity_threshold=0.25)
    return rag


# ==============================================================================
# TEST 1: Varied natural language phrasing retrieves the same clinical ground truth
# ==============================================================================
def test_varied_wording_retrieves_facts(synthetic_rag):
    """Test that varied natural-language queries retrieve the correct clinical chunk."""
    variations = [
        "What causes elevated blood pressure?",
        "Why does systemic arterial pressure rise?",
        "What are the predisposing conditions for hypertension?",
        "Predisposing risk factors for high BP"
    ]
    for q in variations:
        chunks = synthetic_rag.retrieve_context(q, top_k=3, user_id=101)
        assert len(chunks) > 0, f"Query '{q}' returned no chunks"
        assert chunks[0]["chunk_id"] in ("NEW_HYP_01", "NEW_HYP_02"), f"Query '{q}' retrieved wrong chunk: {chunks[0]['chunk_id']}"
        assert chunks[0]["similarity_score"] >= 0.25, f"Score below threshold for query '{q}'"


# ==============================================================================
# TEST 2: Risk factors, symptoms, and lifestyle recommendations
# ==============================================================================
def test_lifestyle_and_risk_factors_retrieval(synthetic_rag):
    """Verify queries about lifestyle, exercise, and diet retrieve lifestyle recommendations."""
    lifestyle_queries = [
        "What lifestyle measures are recommended for hypertension?",
        "Can physical activity and diet lower blood pressure?",
        "What are non-pharmacological interventions for high blood pressure?"
    ]
    for q in lifestyle_queries:
        chunks = synthetic_rag.retrieve_context(q, top_k=3, user_id=101)
        assert len(chunks) > 0, f"Query '{q}' returned no chunks"
        retrieved_texts = " ".join(c["text"] for c in chunks)
        assert "DASH diet" in retrieved_texts or "aerobic physical activity" in retrieved_texts or "sedentary lifestyle" in retrieved_texts


# ==============================================================================
# TEST 3: Document summary and overview queries
# ==============================================================================
def test_document_summary_and_overview(synthetic_rag):
    """Verify document overview / summary queries classify and retrieve without being pruned."""
    summary_queries = [
        "Summarize the uploaded document",
        "Give me an overview of the document",
        "What conditions are discussed in the uploaded document?",
        "What does the pulmonary document cover?"
    ]
    for q in summary_queries:
        intent_res = ClinicalIntentClassifier.classify(q)
        assert intent_res.intent == ClinicalIntent.DOCUMENT_SUMMARY, f"Query '{q}' classified as {intent_res.intent}"

        plan = ClinicalQueryPlanner.plan(q, intent_res, user_id=101)
        assert plan.retrieval_required is True
        assert plan.multi_document_strategy == MultiDocumentStrategy.MAP_REDUCE

        # Retrieval should not be dropped by candidate precision filtering
        chunks = synthetic_rag.retrieve_context(q, top_k=5, user_id=101, query_plan=plan)
        assert len(chunks) > 0, f"Summary query '{q}' was unexpectedly pruned to 0 chunks"


# ==============================================================================
# TEST 4: Comparison queries across multiple documents
# ==============================================================================
def test_document_comparison_planning_and_guidelines():
    """Verify comparison queries trigger balanced retrieval and have proper guidelines."""
    comparison_q = "Compare the cardiology guide and the endocrinology standards"
    intent_res = ClinicalIntentClassifier.classify(comparison_q)
    assert intent_res.intent == ClinicalIntent.DOCUMENT_COMPARISON

    plan = ClinicalQueryPlanner.plan(comparison_q, intent_res, user_id=101)
    assert plan.retrieval_strategy.value == "document_comparison_rag"
    assert plan.generation_strategy == GenerationStrategy.COMPARATIVE_SYNTHESIS
    assert plan.multi_document_strategy == MultiDocumentStrategy.BALANCED_DOCUMENT_RETRIEVAL


# ==============================================================================
# TEST 5: Multiple unrelated documents and scoping
# ==============================================================================
def test_unrelated_documents_scoping(synthetic_rag):
    """Ensure querying a specific condition retrieves only the relevant document."""
    # Pulmonary query should not return cardiology or diabetes chunks
    pulm_chunks = synthetic_rag.retrieve_context("What are symptoms of idiopathic pulmonary fibrosis?", top_k=3, user_id=101)
    assert len(pulm_chunks) > 0
    assert pulm_chunks[0]["chunk_id"] == "NEW_PULM_01"
    assert "fibrosis" in pulm_chunks[0]["text"].lower()

    # Diabetes query should return diabetes standards
    diab_chunks = synthetic_rag.retrieve_context("What is the HbA1c target for type 2 diabetes?", top_k=3, user_id=101)
    assert len(diab_chunks) > 0
    assert diab_chunks[0]["chunk_id"] == "NEW_DIAB_01"
    assert "hba1c" in diab_chunks[0]["text"].lower()


# ==============================================================================
# TEST 6: Duplicate documents deduplication
# ==============================================================================
def test_duplicate_document_deduplication():
    """Verify 3 identical copies are collapsed into exactly 1 unique source card with maximum score."""
    dup_chunks = [
        {"document_id": "copy_1", "chunk_id": "c1", "similarity_score": 0.72, "text": "Identical medical content.", "metadata": {"filename": "test.pdf"}},
        {"document_id": "copy_2", "chunk_id": "c1", "similarity_score": 0.88, "text": "Identical medical content.", "metadata": {"filename": "test.pdf"}},
        {"document_id": "copy_3", "chunk_id": "c1", "similarity_score": 0.65, "text": "Identical medical content.", "metadata": {"filename": "test.pdf"}}
    ]
    deduped = RAGService.deduplicate_chunks(dup_chunks)
    assert len(deduped) == 1
    assert deduped[0]["similarity_score"] == 0.88

    rag = RAGService(vector_store=MagicMock())
    sources = rag.build_sources(deduped)
    assert len(sources) == 1
    assert sources[0]["source_label"] == "[Source 1]"


# ==============================================================================
# TEST 7: Genuinely unsupported queries return safe refusal (no_relevant_context)
# ==============================================================================
def test_unsupported_questions_refuse_safely(synthetic_rag):
    """Verify genuinely unsupported questions (e.g. unindexed rare disease or non-medical) return no_relevant_context."""
    unsupported_queries = [
        "What are the diagnostic criteria for pheochromocytoma?",
        "What is the airspeed velocity of an unladen swallow?",
        "How do you repair a Toyota Camry transmission?"
    ]
    for q in unsupported_queries:
        res = synthetic_rag.generate_rag_answer(question=q, user_id=101)
        assert res["retrieval_status"] in ("no_relevant_context", "out_of_scope"), f"Expected refusal for '{q}', got {res['retrieval_status']}"
        assert len(res["sources"]) == 0, f"Expected 0 sources for '{q}', got {len(res['sources'])}"


# ==============================================================================
# TEST 8: Invalid or hallucinated citations are stripped
# ==============================================================================
def test_invalid_citations_stripped():
    """Verify that hallucinated citations (e.g. [Source 99]) are stripped while valid citations are preserved."""
    simulated_answer = "Hypertension elevates arterial pressure [Source 1], and glomangioma is vascular [Source 99]."
    sources = [{"source_index": 1, "source_label": "[Source 1]", "document_id": "DOC_1", "text": "Hypertension elevates arterial pressure."}]
    val = CitationValidator.validate_grounded_citations(simulated_answer, sources)

    assert 1 in val.valid_citations
    assert 99 in val.invalid_citations
    assert val.cleaned_grounded_answer is not None
    assert "[Source 99]" not in val.cleaned_grounded_answer
    assert "[Source 1]" in val.cleaned_grounded_answer


# ==============================================================================
# TEST 9: Conversational disclaimer recognition
# ==============================================================================
def test_conversational_disclaimers_not_treated_as_unsupported_claims():
    """Verify assistant notices and disclaimers are not treated as uncited factual claims."""
    disclaimer_sentence = "*Note: As an informational research assistant, I am not a physician or doctor. For personalized clinical advice, please consult a qualified healthcare professional.*"
    assert CitationValidator.is_structural_or_meta(disclaimer_sentence) is True


# ==============================================================================
# TEST 10: Cross-user and cross-tenant isolation
# ==============================================================================
def test_cross_user_isolation(synthetic_rag):
    """User 101 cannot access User 202's confidential oncology document."""
    # User 101 searches for oncology report
    u1_chunks = synthetic_rag.retrieve_context("Robert Miller oncology pathology Gleason score", top_k=5, user_id=101)
    for c in u1_chunks:
        assert c.get("user_id") != 202
        assert "Robert Miller" not in c.get("text", "")

    # User 202 querying their own document succeeds
    u2_chunks = synthetic_rag.retrieve_context("Robert Miller oncology pathology Gleason score", top_k=5, user_id=202)
    assert len(u2_chunks) > 0
    assert u2_chunks[0]["chunk_id"] == "USER2_DOC_01"
    assert u2_chunks[0]["user_id"] == 202


# ==============================================================================
# TEST 11: Safety-sensitive clinical emergency queries
# ==============================================================================
def test_safety_emergency_interception():
    """Verify acute emergencies trigger safety intervention without vector search."""
    emergency_queries = [
        "I am having severe crushing chest pain radiating to my jaw and left arm",
        "I swallowed a bottle of bleach and cannot breathe",
        "I want to end my life right now"
    ]
    for q in emergency_queries:
        is_blocked, assessment, reply = MedicalSafetyGuard.pre_screen_inquiry(q)
        assert assessment.risk_level in ("CRITICAL", "HIGH")
        assert assessment.allow_normal_rag is False
        assert assessment.requires_escalation is True
        assert reply is not None

        intent_res = ClinicalIntentClassifier.classify(q)
        assert intent_res.intent in (ClinicalIntent.EMERGENCY, ClinicalIntent.POISONING, ClinicalIntent.SELF_HARM)

        plan = ClinicalQueryPlanner.plan(q, intent_res)
        assert plan.retrieval_required is False
        assert plan.top_k == 0
