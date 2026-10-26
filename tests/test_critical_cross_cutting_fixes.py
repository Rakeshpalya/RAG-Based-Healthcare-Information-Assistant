"""
==============================================================================
CRITICAL CROSS-CUTTING REGRESSION TEST SUITE: Priorities 1 - 5
==============================================================================

Verifies:
1. Priority 1: Contraindication alerts only fire when patient has active condition
   AND query context references the conflicting therapy. Documents discussing
   asthma do not leak into patient clinical profiles or trigger spurious warnings.
2. Priority 2: Chemical ingestion, acute caustic exposure, and overdose inquiries
   are intercepted by deterministic safety pre-screening before FAISS or LLM.
   Poisoning advisory explicitly instructs not to induce vomiting.
3. Priority 3: Document summaries without disease-name anchors return grounded
   evidence without being overwritten by generic refusal fallbacks.
4. Priority 4: Markdown formatting cleanly separates conjoined sentences,
   brackets ([Source 1]. Regarding), normalized Source numbers (Source 1),
   and stripped-tag artifacts (Excess body).
5. Priority 5: Follow-up queries resolve prior turn context accurately, and
   return polite clarifying questions when references cannot be resolved.
==============================================================================
"""

import pytest
import re
from typing import List, Dict, Any

from backend.safety.safety_classifier import SafetyClassifier
from backend.safety.safety_types import SafetyCategory, RiskLevel
from backend.safety.medical_safety_guard import MedicalSafetyGuard
from backend.intelligence.longitudinal_context import ClinicalContextEngine
from backend.rag.markdown_utils import clean_ai_markdown
from backend.rag.rag_service import RAGService


# ============================================================================
# PRIORITY 1: Unrelated Asthma & Contraindication Alert Tests
# ============================================================================

def test_priority1_document_mention_of_asthma_does_not_leak_into_patient_profile():
    """Verifies that assistant literature summary discussing asthma does NOT attribute asthma to patient."""
    history = [
        {"role": "user", "content": "What does the uploaded medical document discuss?"},
        {"role": "assistant", "content": "The document discusses asthma symptoms such as wheezing and cough [Source 1]."}
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="What are the risk factors for Type 2 Diabetes?",
        conversation_history=history
    )
    # The patient does not have asthma; it was merely an assistant document response
    assert not any("asthma" in c.lower() for c in res.cumulative_profile.active_conditions)
    assert len(res.contraindication_alerts) == 0


def test_priority1_objective_asthma_query_does_not_trigger_contraindication_on_unrelated_queries():
    """Verifies that an objective user query about asthma does not create an active condition."""
    history = [
        {"role": "user", "content": "What are the common clinical signs of bronchial asthma?"},
        {"role": "assistant", "content": "Bronchial asthma presents with wheezing and shortness of breath."}
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="What diet is recommended for hypertension?",
        conversation_history=history
    )
    assert len(res.contraindication_alerts) == 0


def test_priority1_genuine_asthma_only_fires_when_conflicting_therapy_is_discussed():
    """Verifies that a disclosed asthma diagnosis ONLY alerts when a beta-blocker is discussed."""
    history = [
        {"role": "user", "content": "I was diagnosed with asthma and take an albuterol rescue inhaler."},
        {"role": "assistant", "content": "Understood, noting your history of asthma and albuterol use."}
    ]

    # Query 1: Unrelated therapy (diet/metformin) -> NO alert
    res_unrelated = ClinicalContextEngine.resolve_context(
        current_turn="Can I take metformin for blood sugar management?",
        conversation_history=history
    )
    assert len(res_unrelated.contraindication_alerts) == 0

    # Query 2: Conflicting therapy (carvedilol / propranolol) -> ALERT FIRES
    res_conflicting = ClinicalContextEngine.resolve_context(
        current_turn="My doctor suggested carvedilol for blood pressure. Is this safe?",
        conversation_history=history
    )
    assert len(res_conflicting.contraindication_alerts) > 0
    alert = res_conflicting.contraindication_alerts[0]
    assert alert.contraindication_id == "ASTHMA_BETA_BLOCKER"
    assert "bronchospasm" in alert.reason.lower()


def test_priority1_user_switching_isolation():
    """Verifies that switching users isolates cumulative patient profiles."""
    user1_history = [
        {"role": "user", "content": "I have severe asthma."},
        {"role": "assistant", "content": "Noted."}
    ]
    res_user1 = ClinicalContextEngine.resolve_context(
        current_turn="Can I take propranolol?",
        conversation_history=user1_history,
        user_id=101
    )
    assert len(res_user1.contraindication_alerts) == 1

    # User 2 with clean history has NO alerts
    res_user2 = ClinicalContextEngine.resolve_context(
        current_turn="Can I take propranolol?",
        conversation_history=[],
        user_id=102
    )
    assert len(res_user2.contraindication_alerts) == 0


# ============================================================================
# PRIORITY 2: Chemical Ingestion & Emergency Screening Tests
# ============================================================================

@pytest.mark.parametrize("query,expected_category", [
    ("I swallowed a dangerous chemical. What should I do immediately?", SafetyCategory.POISONING_OR_OVERDOSE),
    ("Accidental or intentional drain-cleaner ingestion", SafetyCategory.POISONING_OR_OVERDOSE),
    ("A toddler swallowing a caustic household cleaner", SafetyCategory.POISONING_OR_OVERDOSE),
    ("Bleach ingestion", SafetyCategory.POISONING_OR_OVERDOSE),
    ("Medication overdose", SafetyCategory.POISONING_OR_OVERDOSE),
    ("Ingestion of acidic battery fluid", SafetyCategory.POISONING_OR_OVERDOSE),
    ("Severe chest pain with shortness of breath", SafetyCategory.EMERGENCY_SYMPTOMS),
    ("Explicit immediate self-harm intent", SafetyCategory.SELF_HARM_OR_SUICIDE),
    ("I took an entire bottle of sleeping pills", SafetyCategory.POISONING_OR_OVERDOSE),
])
def test_priority2_emergency_screening_intercepts_varied_phrasings(query, expected_category):
    """Verifies that all emergency and acute poisoning phrasings are intercepted with allow_normal_rag=False."""
    assessment = SafetyClassifier.classify_question(query)
    assert assessment.category == expected_category
    assert assessment.allow_normal_rag is False
    assert assessment.requires_escalation is True


def test_priority2_poisoning_advisory_contents():
    """Verifies that poisoning advisory directs to emergency/poison control and forbids inducing vomiting."""
    assessment = SafetyClassifier.classify_question("Accidental bleach ingestion by infant")
    assert assessment.emergency_message is not None
    msg = assessment.emergency_message.lower()
    assert "poison" in msg
    assert "not induce vomiting" in msg or "do not induce vomiting" in msg
    assert "emergency" in msg


def test_priority2_medical_safety_guard_prescreen():
    """Verifies MedicalSafetyGuard pre_screen_inquiry stops query before RAG retrieval."""
    allow_rag, assessment, msg = MedicalSafetyGuard.pre_screen_inquiry("Accidental or intentional drain-cleaner ingestion")
    assert allow_rag is False
    assert assessment.category == SafetyCategory.POISONING_OR_OVERDOSE
    assert "not induce vomiting" in msg.lower()


# ============================================================================
# PRIORITY 3: Inconsistent Document Summaries & Grounding Tests
# ============================================================================

def test_priority3_document_summary_query_planning():
    """Verifies that broad document summary inquiries without disease anchors plan document summary RAG."""
    from backend.intelligence.query_planner import ClinicalQueryPlanner
    from backend.intelligence.intent_classifier import ClinicalIntentClassifier

    q = "Provide a comprehensive summary of the uploaded reference document"
    intent_res = ClinicalIntentClassifier.classify(q)
    plan = ClinicalQueryPlanner.plan(q, intent_res)

    assert plan.retrieval_strategy.value == "document_summary_rag"
    assert plan.query_expansion_enabled is True
    assert len(plan.query_expansions) > 0


def test_priority3_absence_gate_does_not_clobber_grounded_answer():
    """Verifies that answers with valid citations mentioning unaddressed topics are not clobbered."""
    rag = RAGService()
    # Check that filter_candidate_precision preserves chunks for summary queries
    test_chunks = [
        {"document_id": "doc1", "chunk_id": "c1", "similarity_score": 0.45, "text": "Hypertension is high blood pressure."},
        {"document_id": "doc1", "chunk_id": "c2", "similarity_score": 0.40, "text": "Type 2 diabetes involves insulin resistance."}
    ]
    res = rag.filter_candidate_precision(
        query="Summarize what this document covers",
        chunks=test_chunks,
        threshold=0.25
    )
    assert len(res) == 2


# ============================================================================
# PRIORITY 4: Response Formatting & Whitespace Tests
# ============================================================================

def test_priority4_citation_and_punctuation_spacing():
    """Verifies that [Source 1].Regarding is cleanly split into [Source 1]. Regarding."""
    raw = "The primary risk factor is obesity [Source 1].Regarding complications, stroke is prevalent."
    cleaned = clean_ai_markdown(raw)
    assert "[Source 1]. Regarding" in cleaned
    assert "obesity [Source 1]" in cleaned


def test_priority4_source_number_normalization():
    """Verifies that 'Source1' is normalized to 'Source 1'."""
    raw = "As noted in Source1 and [Source2], blood pressure goals should be personalized."
    cleaned = clean_ai_markdown(raw)
    assert "Source 1" in cleaned
    assert "[Source 2]" in cleaned


def test_priority4_excess_body_joined_words_repair():
    """Verifies that stripped tag artifacts like 'Excessbody' are resolved to 'Excess body'."""
    raw = "Risk factors include Excessbody weight and physical inactivity."
    cleaned = clean_ai_markdown(raw)
    assert "Excess body weight" in cleaned
    assert "Excessbody" not in cleaned


def test_priority4_html_stripping_preserves_word_boundaries():
    """Verifies that HTML tags stripped from responses insert spaces rather than merging words."""
    raw = "<p>Excess</p><p>body</p><span>weight</span>"
    cleaned = clean_ai_markdown(raw)
    assert "Excess body weight" in cleaned


# ============================================================================
# PRIORITY 5: Follow-Up Context & Clarification Tests
# ============================================================================

def test_priority5_ordinal_recommendation_resolution_with_history():
    """Verifies that 'Explain the second recommendation in simpler language' resolves prior point."""
    history = [
        {"role": "user", "content": "What are the lifestyle measures for hypertension?"},
        {
            "role": "assistant",
            "content": "Recommended lifestyle measures [Source 1]:\n1. Moderating dietary sodium intake [Source 1].\n2. Maintaining a healthy weight through balanced nutrition [Source 1].\n3. Engaging in regular physical activity [Source 1]."
        }
    ]
    res = ClinicalContextEngine.resolve_context(
        current_turn="Explain the second recommendation in simpler language",
        conversation_history=history
    )
    assert res.is_follow_up is True
    assert "Maintaining a healthy weight" in res.effective_query


def test_priority5_ordinal_recommendation_without_history_returns_clarification():
    """Verifies that ordinal follow-ups with NO prior history ask a clarifying question."""
    res = ClinicalContextEngine.resolve_context(
        current_turn="Explain the second recommendation in simpler language",
        conversation_history=[]
    )
    assert res.is_follow_up is True
    assert "CLARIFICATION_REQUIRED:" in res.effective_query
    assert "specify which recommendation" in res.effective_query.lower()


def test_priority5_recap_without_history_returns_clarification():
    """Verifies that recap queries with NO prior history ask a clarifying question."""
    res = ClinicalContextEngine.resolve_context(
        current_turn="Summarize your previous answer",
        conversation_history=[]
    )
    assert res.is_follow_up is True
    assert "CLARIFICATION_REQUIRED:" in res.effective_query
    assert "no previous messages" in res.effective_query.lower()


def test_priority5_rag_service_resolves_ordinal_recommendation():
    """Verifies that RAGService successfully resolves and simplifies the referenced recommendation."""
    rag = RAGService()
    history = [
        {"role": "user", "content": "What are the lifestyle measures for hypertension?"},
        {
            "role": "assistant",
            "content": "Recommended lifestyle measures [Source 1]:\n1. Moderating dietary sodium intake [Source 1].\n2. Maintaining a healthy weight through balanced nutrition [Source 1].\n3. Engaging in regular physical activity [Source 1]."
        }
    ]
    res = rag.generate_rag_answer(
        question="Explain the second recommendation in simpler language",
        conversation_history=history,
        user_id=1
    )
    assert res["retrieval_status"] == "success"
    assert len(res["sources"]) > 0
    assert "healthy weight" in res["answer"].lower() or "weight" in res["answer"].lower()


def test_priority5_rag_service_ordinal_without_target_asks_clarification():
    """Verifies that RAGService asks for clarification when prior message has only 1 point."""
    rag = RAGService()
    history = [
        {"role": "user", "content": "What is hypertension?"},
        {
            "role": "assistant",
            "content": "Hypertension is persistently elevated blood pressure [Source 1]."
        }
    ]
    res = rag.generate_rag_answer(
        question="Explain the second recommendation in simpler language",
        conversation_history=history,
        user_id=1
    )
    assert res["retrieval_status"] in ("no_relevant_context", "clarification_required")
    assert "specify which recommendation" in res["answer"].lower()

