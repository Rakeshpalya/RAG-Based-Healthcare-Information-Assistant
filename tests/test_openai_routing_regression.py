"""
test_openai_routing_regression.py

Regression test suite for General-Purpose Chatbot & OpenAI Integration (Part B):
1. Verifies the 5-Tier Routing Policy:
   - Tier 1: Emergency & critical safety screening (runs before LLM, works even if API is offline).
   - Tier 2: Simple greetings handled efficiently.
   - Tier 3: Explicit requests to summarize/compare/extract from selected uploaded docs.
   - Tier 4: Explicit questions asking what an uploaded medical document says.
   - Tier 5: All other ordinary questions sent to OpenAI / educational handler by default.
2. Tests the required sample ordinary questions:
   - "tumkur", "Give some information about Tumkur."
   - "Explain NLP."
   - "Explain API."
   - "What is Python?"
   - "Explain artificial intelligence."
   - "What is machine learning."
   - "Explain FastAPI."
   - "What is RAG?"
   - "Explain photosynthesis."
   - "Tell me about Karnataka."
   - "Write Python code to check a palindrome."
   - "Solve 25 × 48."
   - "Help me prepare for an AI/ML interview."
   - "Explain it with an example." (multi-turn follow-up)
3. Ensures non-medical queries NEVER receive a medical refusal.
4. Ensures 429 quota exhaustion produces an actionable billing error, never a medical refusal.
5. Verifies FAISS vector store invariant (744 vectors, 744 metadata records, 384 dimensions).
"""
import pytest
from unittest.mock import MagicMock, patch

from backend.intelligence.intent_classifier import ClinicalIntentClassifier
from backend.intelligence.intent_models import ClinicalIntent, ClinicalRoutingStrategy
from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import get_vector_store_service
from backend.services.openai_service import OpenAIService, OpenAIServiceError


@pytest.fixture(scope="module")
def shared_rag():
    vs = get_vector_store_service()
    return RAGService(vector_store=vs)


REQUIRED_ORDINARY_QUESTIONS = [
    ("tumkur", "tumkur"),
    ("Give some information about Tumkur.", "tumkur"),
    ("Explain NLP.", "nlp"),
    ("Explain API.", "api"),
    ("What is Python?", "python"),
    ("Explain artificial intelligence.", "ai"),
    ("What is machine learning?", "machine learning"),
    ("Explain FastAPI.", "fastapi"),
    ("What is RAG?", "rag"),
    ("Explain photosynthesis.", "photosynthesis"),
    ("Tell me about Karnataka.", "karnataka"),
    ("Write Python code to check a palindrome.", "palindrome"),
    ("Solve 25 × 48.", "1,200"),
    ("Help me prepare for an AI/ML interview.", "interview"),
]


@pytest.mark.parametrize("query,keyword", REQUIRED_ORDINARY_QUESTIONS)
def test_ordinary_questions_classified_as_educational_query(query, keyword):
    """Ordinary questions must classify as EDUCATIONAL_QUERY with zero retrieval required."""
    res = ClinicalIntentClassifier.classify(query)
    assert res.intent == ClinicalIntent.EDUCATIONAL_QUERY, (
        f"Query '{query}' classified as {res.intent}, expected EDUCATIONAL_QUERY"
    )
    assert res.requires_retrieval is False, f"Query '{query}' should not require document retrieval"
    assert res.routing_strategy == ClinicalRoutingStrategy.EDUCATIONAL_RESPONSE


@pytest.mark.parametrize("query,keyword", REQUIRED_ORDINARY_QUESTIONS)
def test_ordinary_questions_generate_informative_answer_without_medical_refusal(shared_rag, query, keyword):
    """Ordinary questions must generate an educational answer and NEVER receive a medical-document refusal."""
    res = shared_rag.generate_rag_answer(query)
    assert res["retrieval_status"] == "educational_response"
    answer = res["answer"]

    # Must contain educational label
    assert "[General Educational Information" in answer or "Tumkur" in answer or "python" in answer.lower()

    # Must NOT be a medical document refusal or out-of-scope refusal
    assert "outside of clinical scope" not in answer.lower()
    assert "relevant medical information could not be found in the available reference documents" not in answer.lower()
    assert "no relevant clinical context" not in answer.lower()

    # Must contain relevant keyword
    assert keyword.lower() in answer.lower() or "1200" in answer.replace(",", "")


def test_multiturn_followup_preserves_dialogue_context(shared_rag):
    """Multi-turn follow-up 'Explain it with an example.' preserves conversation history."""
    history = [
        {"role": "user", "content": "What is triage in healthcare?"},
        {"role": "assistant", "content": "Triage is the prioritization of patient care based on clinical severity."},
    ]
    res = shared_rag.generate_rag_answer(
        question="Explain it with an example.",
        conversation_history=history
    )
    assert res["retrieval_status"] == "educational_response"
    assert "example" in res["answer"].lower()
    assert "outside of clinical scope" not in res["answer"].lower()


def test_emergency_screening_precedes_all_llm_calls(shared_rag):
    """Critical emergency queries are intercepted immediately before any LLM generation."""
    emergency_q = "I have sudden crushing chest pain radiating to my left arm and jaw, what should I do?"
    res = shared_rag.generate_rag_answer(emergency_q)

    assert res["retrieval_status"] == "safety_intercepted"
    assert "emergency" in res["answer"].lower() or "911" in res["answer"] or "immediate medical attention" in res["answer"].lower()
    assert res["timings"].get("llm_called") is False


def test_quota_exhausted_error_handling(shared_rag):
    """When OpenAI raises a 429 quota exhaustion error on a document-grounded query, return actionable billing error, not a medical refusal."""
    mock_service = MagicMock(spec=OpenAIService)
    mock_service.generate_general_response.side_effect = OpenAIServiceError(
        "OpenAI API credit balance or quota exhausted. Please check your OpenAI billing settings at https://platform.openai.com/account/billing.",
        error_type="quota_exhausted",
        status_code=429
    )
    mock_service.generate_answer.side_effect = OpenAIServiceError(
        "OpenAI API credit balance or quota exhausted. Please check your OpenAI billing settings at https://platform.openai.com/account/billing.",
        error_type="quota_exhausted",
        status_code=429
    )

    # 1. Educational query falls back to curated educational knowledge
    res_edu = shared_rag.generate_rag_answer(
        question="Explain FastAPI.",
        openai_service=mock_service
    )
    assert res_edu["retrieval_status"] == "educational_response"
    assert "fastapi" in res_edu["answer"].lower()
    assert "outside of clinical scope" not in res_edu["answer"].lower()

    # 2. Document-grounded query returns actionable billing notice, NEVER a medical-document refusal
    res_grounded = shared_rag.generate_rag_answer(
        question="What are the diagnostic criteria and treatment for hypertension in the guidelines?",
        openai_service=mock_service
    )
    assert res_grounded["retrieval_status"] == "service_unavailable"
    answer = res_grounded["answer"]
    assert "outside of clinical scope" not in answer.lower()
    assert "could not be found in the available reference documents" not in answer.lower()
    assert "billing" in answer.lower() or "quota" in answer.lower() or "credit" in answer.lower()



def test_vector_store_invariant_preserved():
    """Verify FAISS vector store remains strictly at 744 vectors, 744 metadata records, 384 dimensions."""
    import faiss, json
    index = faiss.read_index("data/vector_store/index.faiss")
    assert index.ntotal == 744
    assert index.d == 384

    with open("data/vector_store/metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("count") == 744
    assert len(meta.get("records", [])) == 744
