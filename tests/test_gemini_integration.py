"""
Integration and regression test suite for Google Gemini API integration in HealthAI.
Tests cover:
- Lightweight greeting handling (zero retrieval, zero LLM)
- Educational non-document questions (NLP, Tumkur, AI, Python, FastAPI, RAG)
- Authentication and credential validation
- Error handling for invalid keys, quota limits, and timeouts
- Medical document grounding and citation preservation
- Emergency safety screening precedence independent of Gemini
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiService, GeminiServiceError
from backend.intelligence.intent_classifier import ClinicalIntentClassifier
from backend.intelligence.intent_models import ClinicalIntent
from backend.safety.safety_classifier import SafetyClassifier, SafetyCategory


@pytest.fixture
def rag_service():
    return RAGService()


@pytest.fixture
def gemini_service():
    return GeminiService()


# -----------------------------------------------------------------------------
# 1. Greeting Handling (Zero Retrieval & Zero LLM Overhead)
# -----------------------------------------------------------------------------

def test_greeting_uses_lightweight_handler(rag_service):
    """Verifies that greetings route immediately to direct_conversation without retrieval or LLM."""
    query = "hi"
    intent_res = ClinicalIntentClassifier.classify(query)
    assert intent_res.intent == ClinicalIntent.GREETING
    assert not intent_res.requires_retrieval

    resp = rag_service.generate_rag_answer(query, user_id=8)
    assert resp["retrieval_status"] == "direct_conversation"
    assert resp["sources"] == []
    assert resp["timings"]["gemini_calls_count"] == 0
    assert "HealthAI assistant" in resp["answer"]


# -----------------------------------------------------------------------------
# 2. General Educational Questions Powered by Gemini
# -----------------------------------------------------------------------------

def test_explain_nlp_uses_gemini(rag_service):
    """Verifies that 'explain NLP' routes to educational_response with appropriate labeling."""
    query = "explain NLP"
    intent_res = ClinicalIntentClassifier.classify(query)
    assert intent_res.intent == ClinicalIntent.EDUCATIONAL_QUERY
    assert not intent_res.requires_retrieval

    resp = rag_service.generate_rag_answer(query, user_id=8)
    assert resp["retrieval_status"] == "educational_response"
    assert "[General Educational Information - Not from uploaded clinical documents]" in resp["answer"]
    ans_lower = resp["answer"].lower()
    assert "nlp" in ans_lower or "language" in ans_lower


def test_tumkur_query_returns_general_overview(rag_service):
    """Verifies that 'give some info about Tumkur' routes to educational_response without document refusal."""
    query = "give some info about Tumkur"
    intent_res = ClinicalIntentClassifier.classify(query)
    assert intent_res.intent == ClinicalIntent.EDUCATIONAL_QUERY
    assert not intent_res.requires_retrieval

    resp = rag_service.generate_rag_answer(query, user_id=8)
    assert resp["retrieval_status"] == "educational_response"
    assert "[General Educational Information - Not from uploaded clinical documents]" in resp["answer"]
    ans_lower = resp["answer"].lower()
    assert "tumkur" in ans_lower or "tumakuru" in ans_lower or "karnataka" in ans_lower


def test_single_word_tumkur_routes_to_gemini(rag_service):
    """Verifies that single-word 'tumkur' routes to educational_response without medical document refusal."""
    query = "tumkur"
    intent_res = ClinicalIntentClassifier.classify(query)
    assert intent_res.intent == ClinicalIntent.EDUCATIONAL_QUERY
    assert not intent_res.requires_retrieval

    resp = rag_service.generate_rag_answer(query, user_id=8)
    assert resp["retrieval_status"] == "educational_response"
    assert "Relevant medical information could not be found" not in resp["answer"]
    assert "[General Educational Information - Not from uploaded clinical documents]" in resp["answer"]
    ans_lower = resp["answer"].lower()
    assert "tumkur" in ans_lower or "tumakuru" in ans_lower or "karnataka" in ans_lower


def test_multiturn_followup_preserves_conversation_context(rag_service):
    """Verifies that follow-up questions like 'explain it with an example' preserve dialogue history."""
    conv_history = [
        {"role": "user", "content": "tumkur"},
        {"role": "assistant", "content": "Tumkur is an educational and industrial city in Karnataka, India."}
    ]
    follow_up = "explain it with an example of its universities"
    resp = rag_service.generate_rag_answer(follow_up, user_id=8, conversation_history=conv_history)
    assert resp["retrieval_status"] == "educational_response"
    assert "Relevant medical information could not be found" not in resp["answer"]
    assert "[General Educational Information - Not from uploaded clinical documents]" in resp["answer"]


@pytest.mark.parametrize("concept,expected_token", [
    ("what is artificial intelligence", "intelligence"),
    ("explain Python", "python"),
    ("what is machine learning", "learning"),
    ("explain FastAPI", "fastapi"),
    ("what is RAG", "rag"),
    ("explain photosynthesis", "photosynthesis"),
    ("tell me about Karnataka", "karnataka"),
    ("answer general educational questions in ordinary language", "educational"),
])
def test_general_educational_concepts(rag_service, concept, expected_token):
    """Verifies that core tech/educational concepts return labeled answers without retrieval."""
    resp = rag_service.generate_rag_answer(concept, user_id=8)
    assert resp["retrieval_status"] == "educational_response"
    assert "[General Educational Information - Not from uploaded clinical documents]" in resp["answer"]
    assert expected_token in resp["answer"].lower()


# -----------------------------------------------------------------------------
# 3. Authentication & Error Handling
# -----------------------------------------------------------------------------

def test_invalid_credentials_raise_clear_auth_error():
    """Verifies that invalid credentials fast-abort with a clear authentication exception."""
    service = GeminiService(api_key="invalid-bogus-key")
    with patch.object(service, "get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API key not valid. Please pass a valid API key.")
        mock_get_client.return_value = mock_client

        with pytest.raises(GeminiServiceError) as exc_info:
            service.generate("Test prompt")
        
        assert "authentication" in str(exc_info.value).lower()
        assert exc_info.value.error_type == "auth_error"


def test_quota_limits_handled_distinctly():
    """Verifies that quota / 429 exhaustion raises clear quota error."""
    service = GeminiService(api_key="valid-mock-key")
    with patch.object(service, "get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 Resource exhausted: quota exceeded")
        mock_get_client.return_value = mock_client

        with pytest.raises(GeminiServiceError) as exc_info:
            service.generate("Test prompt")
        
        assert "quota" in str(exc_info.value).lower() or "rate limit" in str(exc_info.value).lower()
        assert exc_info.value.error_type == "quota_error"


def test_timeout_handled_distinctly():
    """Verifies that timeout / 504 errors raise a timeout error."""
    service = GeminiService(api_key="valid-mock-key")
    with patch.object(service, "get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("504 Deadline exceeded: request timed out")
        mock_get_client.return_value = mock_client

        with pytest.raises(GeminiServiceError) as exc_info:
            service.generate("Test prompt")
        
        assert "timed out" in str(exc_info.value).lower()
        assert exc_info.value.error_type == "timeout_error"


# -----------------------------------------------------------------------------
# 4. Medical RAG Grounding & Citation Preservation
# -----------------------------------------------------------------------------

def test_medical_rag_preserves_citations(rag_service):
    """Verifies that medical document questions preserve grounded citations."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or "mock" in api_key.lower() or "test" in api_key.lower():
        mock_gemini = MagicMock()
        mock_gemini.generate_answer.return_value = {
            "answer": "Common symptoms of diabetes include increased thirst, frequent urination, unexplained weight loss, and fatigue [Source 1].",
            "model": "gemini-3.5-flash-lite",
            "disclaimer": "Medical disclaimer",
            "generation_time_ms": 120.0,
            "status": "success"
        }
        resp = rag_service.generate_rag_answer("What are the symptoms of diabetes?", user_id=8, gemini_service=mock_gemini)
    else:
        resp = rag_service.generate_rag_answer("What are the symptoms of diabetes?", user_id=8)
    assert resp["retrieval_status"] == "success"
    assert len(resp["sources"]) > 0
    assert "[Source 1]" in resp["answer"] or "[Source" in resp["answer"]


# -----------------------------------------------------------------------------
# 5. Emergency Screening Independence
# -----------------------------------------------------------------------------

def test_emergency_screening_bypasses_gemini_and_rag():
    """Verifies that emergency inquiries intercept immediately even if Gemini is down."""
    assessment = SafetyClassifier.classify_question("Severe chest pain and shortness of breath")
    assert assessment.category == SafetyCategory.EMERGENCY_SYMPTOMS
    assert not assessment.allow_normal_rag
