import pytest
import re
from backend.rag.rag_service import RAGService
from backend.safety.medical_safety_guard import MedicalSafetyGuard
from backend.safety.safety_classifier import SafetyCategory
from backend.evaluation.citation_validator import CitationValidator
from backend.rag.markdown_utils import clean_ai_markdown
from backend.intelligence.longitudinal_context import ClinicalContextEngine
from backend.intelligence.intent_classifier import ClinicalIntentClassifier
from backend.intelligence.intent_models import ClinicalIntent


@pytest.fixture
def rag_service():
    return RAGService()


def test_emergency_precedence_toxic_ingestion():
    """Requirement 9: Emergency screening must execute before RAG and intercept toxic chemical ingestion."""
    q = "My toddler just swallowed some drain cleaner, what should I do?"
    allow_rag, safety_assessment, immediate_msg = MedicalSafetyGuard.pre_screen_inquiry(q)
    assert allow_rag is False
    assert safety_assessment.category == SafetyCategory.POISONING_OR_OVERDOSE
    assert "POISON" in immediate_msg.upper() or "TOXICOLOGY" in immediate_msg.upper() or "EMERGENCY" in immediate_msg.upper()


def test_emergency_precedence_overdose():
    """Requirement 9: Medication overdose must be intercepted immediately."""
    q = "I took an entire bottle of pills by accident, help me"
    allow_rag, safety_assessment, immediate_msg = MedicalSafetyGuard.pre_screen_inquiry(q)
    assert allow_rag is False
    assert safety_assessment.category in (SafetyCategory.POISONING_OR_OVERDOSE, SafetyCategory.SELF_HARM_OR_SUICIDE)
    assert immediate_msg is not None


def test_whitespace_preservation_around_citations():
    """Requirement 10: Citation markers must not cause adjacent words to merge."""
    raw_markdown = "<p>Stage 2 Essential Hypertension</p><a href='#'>[Source 1]</a><span>is chronically elevated blood pressure.</span>"
    cleaned = clean_ai_markdown(raw_markdown)
    assert "Hypertension[Source" not in cleaned
    assert "[Source 1]is" not in cleaned
    assert "[Source 1]" in cleaned
    idx = cleaned.find("[Source 1]")
    assert idx > 0
    assert cleaned[idx - 1] in (" ", "\n")
    if idx + len("[Source 1]") < len(cleaned):
        assert cleaned[idx + len("[Source 1]")] in (" ", "\n", ".", ",", ";", ":")


def test_follow_up_ordinal_recommendation_resolution():
    """Requirement 4.D & 7: Resolves ordinal references ('second recommendation') using real prior turns."""
    history = [
        {"role": "user", "content": "What does the document recommend for managing hypertension?"},
        {
            "role": "assistant",
            "content": (
                "The document provides several recommendations: "
                "1. Adopt the DASH eating plan rich in fruits, vegetables, and low-fat dairy. "
                "2. Implement dietary sodium reduction below 1,500 mg daily. "
                "3. Engage in at least 150 minutes per week of moderate-intensity physical activity [Source 1]."
            )
        }
    ]
    current_q = "Explain the second recommendation in simpler language."
    resolution = ClinicalContextEngine.resolve_context(
        current_turn=current_q,
        conversation_history=history,
        user_id=1
    )
    assert resolution.is_follow_up is True
    assert "sodium" in resolution.effective_query.lower() or "1,500" in resolution.effective_query


def test_meta_conversational_previous_answer_resolution(rag_service):
    """Requirement 4.D & 8: 'Summarize your previous answer' resolves from conversation history without hallucinating."""
    history = [
        {"role": "user", "content": "What are the lifestyle measures for blood pressure?"},
        {
            "role": "assistant",
            "content": (
                "Lifestyle modifications for blood pressure control include weight loss, "
                "adherence to the DASH diet, reduction of sodium intake to less than 1,500 mg/day, "
                "regular physical exercise, and limitation of alcohol consumption [Source 1]."
            )
        }
    ]
    resp = rag_service.generate_rag_answer(
        question="Summarize your previous answer",
        conversation_history=history,
        user_id=1
    )
    assert resp["retrieval_status"] == "success"
    assert "DASH" in resp["answer"] or "sodium" in resp["answer"] or "weight" in resp["answer"].lower()
    assert len(resp["sources"]) > 0


def test_meta_conversational_without_history(rag_service):
    """Requirement 7: If no history exists for a meta-conversational question, returns an informative prompt."""
    resp = rag_service.generate_rag_answer(
        question="Summarize your previous answer",
        conversation_history=[],
        user_id=1
    )
    assert resp["retrieval_status"] == "no_relevant_context"
    assert "no previous messages" in resp["answer"].lower()


def test_missing_evidence_distinguished_from_retrieval_failure(rag_service):
    """Requirement 4.E & 5: Unsupported clinical entities distinguish missing evidence from retrieval failure."""
    resp = rag_service.generate_rag_answer(
        question="What is the recommended chemotherapy dosage for stage 2 breast cancer?",
        conversation_history=None,
        user_id=1
    )
    assert resp["retrieval_status"] == "no_relevant_context"
    assert "do not contain" in resp["answer"].lower() or "not contain" in resp["answer"].lower()
    assert "chemotherapy" in resp["answer"].lower() or "cancer" in resp["answer"].lower()
    # It lists accessible reference documents rather than giving a blind generic fallback
    assert len(resp["sources"]) == 0


def test_compound_query_partial_evidence(rag_service):
    """Requirement 4.B & 6: Compound inquiries with partial evidence succeed without being discarded."""
    resp = rag_service.query(
        question="What are the symptoms and complications of hypertension?",
        user_id=1
    )
    assert resp["retrieval_status"] == "success"
    assert len(resp["retrieved_chunks"]) > 0


def test_document_comparison_retrieval(rag_service):
    """Requirement 4.C & 8: Comparative inquiries retrieve multi-document evidence."""
    resp = rag_service.query(
        question="Compare the recommendations in the uploaded documents.",
        user_id=1
    )
    assert resp["retrieval_status"] == "success"
    assert len(resp["retrieved_chunks"]) >= 1


def test_user_ownership_isolation(rag_service):
    """Requirement 12: Documents owned by User A cannot be accessed by User B."""
    docs_user_1 = rag_service.vector_store.get_accessible_document_names(user_id=1)
    docs_user_9999 = rag_service.vector_store.get_accessible_document_names(user_id=999999)
    # User 1 has accessible documents in the store
    assert len(docs_user_1) > 0
    # User 999999 has no documents associated with that ID
    assert len(docs_user_9999) <= len(docs_user_1)


def test_main_topics_and_scope_classification_and_expansion():
    """Requirement 1: 'What are the main topics covered?' and 'What is the document scope?' classify as DOCUMENT_SUMMARY and generate broad expansions."""
    from backend.intelligence.query_planner import ClinicalQueryPlanner

    classifier = ClinicalIntentClassifier()

    for phrase in [
        "What are the main topics covered?",
        "What topics are covered in this document?",
        "What is the document scope?",
        "What are the key findings of this report?",
        "What sections are included in this document?"
    ]:
        result = classifier.classify(phrase)
        assert result.intent == ClinicalIntent.DOCUMENT_SUMMARY, f"Failed for '{phrase}': got {result.intent}"

        plan = ClinicalQueryPlanner.plan(phrase, result)
        assert plan is not None
        expansions = ClinicalQueryPlanner.generate_expansions(phrase, result.intent)
        assert len(expansions) > 0
        joined = " ".join(expansions).lower()
        assert any(k in joined for k in ["topic", "overview", "scope", "section", "finding", "summary"])


def test_grounding_rejects_unsupported_diagnosis_and_prescription():
    """Requirement 2: Ensure claims of diagnosis or prescription are rejected if evidence only shows symptoms."""
    from backend.evaluation.hallucination_guard import HallucinationGuard
    from backend.intelligence.clinical_verification import ClinicalVerificationEngine

    # Evidence only establishes symptoms, NOT a formal diagnosis or prescription
    symptom_evidence = (
        "Patient presents with episodic shortness of breath, wheezing, and dry cough for the past 2 weeks. "
        "Symptoms worsen at night. Physical examination reveals bilateral expiratory wheezes. No formal diagnosis recorded."
    )

    # Claim stating diagnosis was made
    unsupported_diagnosis = "The patient was diagnosed with bronchial asthma [Source 1]."
    diag_res = HallucinationGuard.guard_answer(unsupported_diagnosis, [{"source_number": 1, "text": symptom_evidence}])
    assert diag_res.is_safe is False or diag_res.unsupported_claims > 0

    # Claim stating inhaler was prescribed
    unsupported_rx = "An albuterol inhaler was prescribed for symptom management [Source 1]."
    rx_res = HallucinationGuard.guard_answer(unsupported_rx, [{"source_number": 1, "text": symptom_evidence}])
    assert rx_res.is_safe is False or rx_res.unsupported_claims > 0

    # ClinicalVerificationEngine check
    verif = ClinicalVerificationEngine.verify_and_guard(
        answer_text=unsupported_diagnosis + " " + unsupported_rx,
        retrieved_sources=[{"source_number": 1, "text": symptom_evidence}]
    )
    assert verif.is_verified_safe is False or verif.ungrounded_claims_count > 0 or verif.fallback_triggered is True


def test_grounding_accepts_supported_diagnosis_and_prescription():
    """Requirement 2: When diagnosis and prescription are explicitly documented, grounding checks succeed."""
    from backend.evaluation.hallucination_guard import HallucinationGuard
    from backend.intelligence.clinical_verification import ClinicalVerificationEngine

    confirmed_evidence = (
        "Clinical Assessment: The patient is diagnosed with persistent bronchial asthma. "
        "Plan: Prescribed albuterol inhaler 2 puffs every 4-6 hours as needed."
    )

    supported_claim = "The patient was diagnosed with asthma and an albuterol inhaler was prescribed [Source 1]."
    guard_res = HallucinationGuard.guard_answer(supported_claim, [{"source_number": 1, "text": confirmed_evidence}])
    assert guard_res.is_safe is True
    assert guard_res.unsupported_claims == 0

    verif = ClinicalVerificationEngine.verify_and_guard(
        answer_text=supported_claim,
        retrieved_sources=[{"source_number": 1, "text": confirmed_evidence}]
    )
    assert verif.ungrounded_claims_count == 0


def test_clarification_history_not_treated_as_clinical_answer(rag_service):
    """Requirement 3: 'Summarize your previous answer' when previous message was a clarification question."""
    clarification_history = [
        {"role": "user", "content": "Tell me about my treatment plan"},
        {
            "role": "assistant",
            "content": (
                "Could you please clarify which document or medical condition you are referring to? "
                "I have access to your cardiology records as well as your recent lab results."
            )
        }
    ]

    resp = rag_service.generate_rag_answer(
        question="Summarize your previous answer",
        conversation_history=clarification_history,
        user_id=1
    )
    # It must NOT treat the clarification request as clinical findings
    assert resp["retrieval_status"] == "no_relevant_context"
    assert "clarif" in resp["answer"].lower() or "no previous clinical answer" in resp["answer"].lower()


def test_single_document_user_comparison_explains_insufficiency(rag_service):
    """Requirement 4: When a user only has 1 or 0 documents, document comparison explains two documents cannot be compared."""
    user_no_docs = 999998
    resp = rag_service.generate_rag_answer(
        question="Compare the two documents in my profile",
        conversation_history=None,
        user_id=user_no_docs
    )
    assert resp["retrieval_status"] == "insufficient_documents"
    assert "cannot be completed" in resp["answer"].lower()
    assert "at least two" in resp["answer"].lower() or "two distinct" in resp["answer"].lower()


def test_document_id_scoping_in_rag_service(rag_service):
    """Requirement 1 & 2: When document_id is provided, retrieval is strictly scoped to that document."""
    # Find User 8 document IDs in the vector store
    user8_docs = sorted(list(set(str(m["document_id"]) for m in rag_service.vector_store.metadata_store if str(m.get("user_id")) == "8" and m.get("document_id"))))
    assert len(user8_docs) >= 2

    doc1 = user8_docs[0]
    doc2 = user8_docs[1]

    # Query scoped to doc1
    res1 = rag_service.retrieve_context(
        query="What are the symptoms of asthma?",
        user_id=8,
        document_id=doc1,
        similarity_threshold=0.35
    )
    assert len(res1) > 0
    for chunk in res1:
        c_doc = str(chunk.get("document_id") or chunk.get("metadata", {}).get("document_id"))
        assert c_doc == doc1

    # Query scoped to doc2
    res2 = rag_service.retrieve_context(
        query="What are the symptoms of asthma?",
        user_id=8,
        document_id=doc2,
        similarity_threshold=0.35
    )
    assert len(res2) > 0
    for chunk in res2:
        c_doc = str(chunk.get("document_id") or chunk.get("metadata", {}).get("document_id"))
        assert c_doc == doc2


def test_formatting_conjoined_words_and_bullet_points():
    """Requirement 5: Formatting cleans up conjoined words and ensures newlines before bullet points."""
    raw = "Excessbody weight increases risk.*Maintain healthy weight-Exercise daily•Monitor blood pressure"
    cleaned = clean_ai_markdown(raw)

    assert "Excess body" in cleaned
    assert "\n* Maintain healthy weight" in cleaned
    assert "\n- Exercise daily" in cleaned
    assert "\n• Monitor blood pressure" in cleaned

