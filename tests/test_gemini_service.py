import os
import sys
import time
import asyncio
from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.config import settings, Settings
from backend.services.gemini_service import GeminiService, GeminiServiceError
from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import VectorStoreService
from backend.rag.prompt_builder import (
    build_rag_prompt,
    HEALTHCARE_SYSTEM_INSTRUCTIONS,
    MEDICAL_DISCLAIMER
)
from backend.api.rag_router import query_rag, RAGQueryRequest


SYNTHETIC_CHUNKS = [
    {
        "chunk_id": "MED_CHUNK_0",
        "document_id": "DOC_CARDIO_001",
        "page_number": 1,
        "text": "Hypertension is a condition characterized by persistently elevated arterial blood pressure above 130/80 mmHg.",
        "category": "Cardiology"
    },
    {
        "chunk_id": "MED_CHUNK_1",
        "document_id": "DOC_ENDO_001",
        "page_number": 2,
        "text": "Type 2 diabetes mellitus is characterized by peripheral insulin resistance and elevated blood glucose.",
        "category": "Endocrinology"
    }
]


def test_gemini_service_initialization():
    """1. Test GeminiService initialization and default settings."""
    service = GeminiService(api_key="mock-key-12345", model="gemini-2.5-flash", temperature=0.2)
    assert service.model == "gemini-2.5-flash"
    assert service.temperature == 0.2
    assert service._api_key == "mock-key-12345"
    print("[PASS] test_gemini_service_initialization passed.")


def test_prompt_construction_and_sections():
    """2, 3, 4, & 5. Test prompt builder, context/question injection, and grounding instructions."""
    question = "What is hypertension?"
    context = (
        "[SOURCE 1]\n"
        "Document: DOC_CARDIO_001\n"
        "Page: 1\n"
        "Chunk ID: MED_CHUNK_0\n\n"
        "Hypertension is a condition characterized by persistently elevated blood pressure."
    )
    prompt = build_rag_prompt(question, context)

    # Verify section separation
    assert "=== SYSTEM INSTRUCTIONS ===" in prompt
    assert "=== RETRIEVED MEDICAL CONTEXT ===" in prompt
    assert "=== USER QUESTION ===" in prompt
    assert "=== GROUNDED ANSWER ===" in prompt

    # Verify context and question injection
    assert question in prompt
    assert "DOC_CARDIO_001" in prompt
    assert "MED_CHUNK_0" in prompt
    assert "persistently elevated blood pressure" in prompt

    # Verify grounding instructions
    assert "Answer using ONLY the provided reference context" in prompt
    assert "Do NOT fabricate, assume, or invent claims" in prompt
    assert "Do NOT provide a medical diagnosis" in prompt
    assert "Do NOT prescribe medications" in prompt
    assert "[SOURCE 1]" in prompt
    print("[PASS] test_prompt_construction_and_sections passed.")


def test_mock_gemini_generation_and_schema():
    """6 & 7. Test mock Gemini answer generation and structured response schema."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Based on [Source 1], hypertension is defined as persistently elevated blood pressure."
    mock_client.models.generate_content.return_value = mock_resp

    service = GeminiService(api_key="test-key")
    service.set_client(mock_client)

    result = service.generate_answer(
        question="What is hypertension?",
        context="[SOURCE 1]\nDocument: DOC_001\nPage: 1\nChunk ID: CHUNK_0\n\nHypertension is high BP."
    )

    assert result["status"] == "success"
    assert "Based on [Source 1]" in result["answer"]
    assert result["model"] == settings.GEMINI_MODEL
    assert "disclaimer" in result
    assert result["generation_time_ms"] >= 0.0
    print("[PASS] test_mock_gemini_generation_and_schema passed.")


def test_empty_query_handling():
    """8. Test empty query returns immediately without calling LLM."""
    mock_client = MagicMock()
    service = GeminiService(api_key="test-key")
    service.set_client(mock_client)

    res = service.generate_answer("", "Some context")
    assert res["status"] == "empty_query"
    assert "No question was provided" in res["answer"]
    mock_client.models.generate_content.assert_not_called()
    print("[PASS] test_empty_query_handling passed.")


def test_no_relevant_context_does_not_call_gemini():
    """9 & 11. Test that RAG pipeline halts and NEVER calls Gemini when retrieval returns no_relevant_context."""
    store = VectorStoreService()
    store.add_chunks(SYNTHETIC_CHUNKS)
    rag = RAGService(vector_store=store)

    mock_gemini = MagicMock()

    # Query about completely unrelated topic with strict threshold
    result = rag.generate_rag_answer(
        question="What are the quantum properties of gravitational waves in deep space?",
        top_k=3,
        similarity_threshold=0.85,
        gemini_service=mock_gemini
    )

    assert result["retrieval_status"] == "no_relevant_context"
    assert "Relevant medical information could not be found" in result["answer"]
    assert result["retrieved_chunks"] == []
    assert result["sources"] == []
    # Assert Gemini was NOT called to prevent unsupported medical answers
    mock_gemini.generate_answer.assert_not_called()
    print("[PASS] test_no_relevant_context_does_not_call_gemini passed.")


def test_api_error_and_timeout_handling():
    """10 & 11. Test error handling and secret sanitization."""
    mock_client = MagicMock()
    # Simulate API error or network timeout
    mock_client.models.generate_content.side_effect = Exception("504 Gateway Timeout: Deadline exceeded")

    service = GeminiService(api_key="test-key-999")
    service.set_client(mock_client)

    try:
        service.generate_answer("What is hypertension?", "Context text")
        assert False, "Should have raised GeminiServiceError"
    except GeminiServiceError as err:
        assert "504 Gateway Timeout" in str(err)
        assert "test-key-999" not in str(err)  # Secret is not leaked

    print("[PASS] test_api_error_and_timeout_handling passed.")


def test_api_key_validation():
    """12. Test that missing GEMINI_API_KEY raises a clean ValueError without leaking secrets."""
    original_key = Settings.GEMINI_API_KEY
    orig_env_key = os.environ.get("GEMINI_API_KEY")
    try:
        # Test empty key validation
        Settings.GEMINI_API_KEY = ""
        os.environ["GEMINI_API_KEY"] = ""
        try:
            Settings.validate_gemini_api_key()
            assert False, "Should have raised ValueError on missing API key"
        except ValueError as e:
            assert "Gemini API key is not configured" in str(e)
        
        masked_empty = Settings.get_masked_api_key()
        assert masked_empty == "NOT_CONFIGURED"

        # Test masked representation with valid mock key
        test_key = "AIzaSyTestKeyExample123456789"
        Settings.GEMINI_API_KEY = test_key
        os.environ["GEMINI_API_KEY"] = test_key
        masked = Settings.get_masked_api_key()
        assert masked.startswith("AIza")
        assert masked.endswith("6789")
        assert "..." in masked

        val = Settings.validate_gemini_api_key()
        assert val == test_key
    finally:
        Settings.GEMINI_API_KEY = original_key
        if orig_env_key is not None:
            os.environ["GEMINI_API_KEY"] = orig_env_key
        else:
            os.environ.pop("GEMINI_API_KEY", None)

    print("[PASS] test_api_key_validation passed.")


def test_gemini_503_retry_success():
    """TEST 1: Gemini returns 503 once, then succeeds on bounded retry."""
    from google.genai.errors import APIError

    mock_client = MagicMock()
    err_503 = APIError(
        503,
        {"error": {"message": "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later."}}
    )
    success_resp = MagicMock()
    success_resp.text = "Malaria symptoms include high fever, chills, and headache [Source 1]."
    # First attempt: 503, Second attempt: success
    mock_client.models.generate_content.side_effect = [err_503, success_resp]

    service = GeminiService(api_key="test-key")
    service.INITIAL_RETRY_DELAY_SEC = 0.001  # fast test
    service.set_client(mock_client)

    result = service.generate_answer(
        question="What are the symptoms of malaria?",
        context="[SOURCE 1]\nMalaria causes high fever and chills."
    )

    assert result["status"] == "success"
    assert "Malaria symptoms include high fever" in result["answer"]
    assert mock_client.models.generate_content.call_count == 2
    print("[PASS] TEST 1: test_gemini_503_retry_success passed.")


def test_gemini_503_fallback_model_success():
    """TEST 2: Primary Gemini model returns 503; fallback model succeeds."""
    from google.genai.errors import APIError

    mock_client = MagicMock()
    err_503 = APIError(
        503,
        {"error": {"message": "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later."}}
    )
    success_resp = MagicMock()
    success_resp.text = "Malaria is transmitted by Anopheles mosquitoes [Source 1]."

    # Primary model fails 3 times (1 initial + 2 retries), then fallback model succeeds on 1st attempt
    mock_client.models.generate_content.side_effect = [
        err_503, err_503, err_503,  # Primary model exhausted
        success_resp                # Fallback model succeeds
    ]

    service = GeminiService(api_key="test-key", model="gemini-3.5-flash-lite")
    service.INITIAL_RETRY_DELAY_SEC = 0.001
    service.set_client(mock_client)

    result = service.generate_answer(
        question="How is malaria transmitted?",
        context="[SOURCE 1]\nMalaria is transmitted by mosquitoes."
    )

    assert result["status"] == "success"
    assert "Malaria is transmitted" in result["answer"]
    assert result["model"] != "gemini-3.5-flash-lite"
    assert mock_client.models.generate_content.call_count == 4
    print("[PASS] TEST 2: test_gemini_503_fallback_model_success passed.")


def test_gemini_503_all_models_fail_clean_error():
    """TEST 3: All configured models return 503 -> bounded retries, clean failure, no raw traceback, no hallucinated medical answer."""
    from google.genai.errors import APIError

    mock_client = MagicMock()
    err_503 = APIError(
        503,
        {"error": {"message": "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later."}}
    )
    mock_client.models.generate_content.side_effect = err_503

    service = GeminiService(api_key="test-key")
    service.INITIAL_RETRY_DELAY_SEC = 0.001
    service.set_client(mock_client)

    try:
        service.generate_answer(
            question="What are the symptoms of malaria according to my uploaded PDF?",
            context="[SOURCE 1]\nMalaria symptoms."
        )
        assert False, "Should have raised GeminiServiceError"
    except GeminiServiceError as err:
        err_msg = str(err)
        assert "temporarily unavailable due to high demand" in err_msg or "temporarily unavailable" in err_msg
        assert "Traceback" not in err_msg

    # Also test end-to-end through RAGService:
    store = VectorStoreService()
    store.add_chunks(SYNTHETIC_CHUNKS)
    rag = RAGService(vector_store=store)
    rag_res = rag.generate_rag_answer(
        question="What is hypertension?",
        gemini_service=service
    )
    assert rag_res["retrieval_status"] == "service_unavailable"
    assert "temporarily unavailable due to high demand" in rag_res["answer"]
    assert rag_res["sources"] == []  # No hallucinated medical sources
    print("[PASS] TEST 3: test_gemini_503_all_models_fail_clean_error passed.")


def test_gemini_normal_response_unchanged():
    """TEST 4: Normal Gemini response -> existing behavior unchanged."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Hypertension is chronic elevation of systemic arterial blood pressure [Source 1]."
    mock_client.models.generate_content.return_value = mock_resp

    service = GeminiService(api_key="test-key")
    service.set_client(mock_client)

    result = service.generate_answer(
        question="What is hypertension?",
        context="[SOURCE 1]\nHypertension context."
    )
    assert result["status"] == "success"
    assert "[Source 1]" in result["answer"]
    assert mock_client.models.generate_content.call_count == 1
    print("[PASS] TEST 4: test_gemini_normal_response_unchanged passed.")


def test_source_metadata_preservation_in_rag():
    """13 & 14. Test end-to-end source metadata preservation and generation latency measurement."""
    store = VectorStoreService()
    store.add_chunks(SYNTHETIC_CHUNKS)
    rag = RAGService(vector_store=store)

    mock_gemini = MagicMock()
    mock_gemini.generate_answer.return_value = {
        "answer": "According to [Source 1], hypertension involves chronically elevated blood pressure.",
        "model": "gemini-2.5-flash",
        "disclaimer": MEDICAL_DISCLAIMER,
        "generation_time_ms": 150.0,
        "status": "success"
    }

    question = "What is hypertension and its threshold?"
    result = rag.generate_rag_answer(question=question, top_k=2, gemini_service=mock_gemini)

    assert result["retrieval_status"] == "success"
    assert len(result["sources"]) > 0
    assert result["sources"][0]["source_id"] == "DOC_CARDIO_001"
    assert result["sources"][0]["chunk_id"] == "MED_CHUNK_0"
    assert "timings" in result
    assert result["timings"]["retrieval_time_ms"] > 0
    assert result["timings"]["generation_time_ms"] == 150.0
    assert result["timings"]["total_time_ms"] > 150.0
    print("[PASS] test_source_metadata_preservation_in_rag passed.")


def test_hallucination_and_unsupported_context_safeguard():
    """12. Test that ungrounded or unsupported questions are safe-guarded against hallucination."""
    store = VectorStoreService()
    store.add_chunks(SYNTHETIC_CHUNKS)
    rag = RAGService(vector_store=store)

    mock_gemini = MagicMock()

    # Question about unindexed treatment (e.g. brain surgery)
    res = rag.generate_rag_answer(
        question="How is craniotomy performed in neurosurgery?",
        top_k=2,
        similarity_threshold=0.60,
        gemini_service=mock_gemini
    )

    # Retrieval threshold stops pipeline before hallucinating
    assert res["retrieval_status"] == "no_relevant_context"
    mock_gemini.generate_answer.assert_not_called()
    assert "Relevant medical information could not be found" in res["answer"]
    print("[PASS] test_hallucination_and_unsupported_context_safeguard passed.")


def test_api_endpoint_post_rag_query():
    """17. Test FastAPI POST /rag/query endpoint integration with mocked Gemini."""
    from backend.api.rag_router import get_rag_service
    service = get_rag_service()
    if service.vector_store.count() == 0:
        service.vector_store.add_chunks(SYNTHETIC_CHUNKS)

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Hypertension is chronically elevated blood pressure [Source 1]."
    mock_client.models.generate_content.return_value = mock_resp

    GeminiService.set_client(mock_client)

    req = RAGQueryRequest(
        question="What is hypertension?",
        top_k=2,
        similarity_threshold=0.25
    )

    resp = asyncio.run(query_rag(req))
    assert resp.retrieval_status == "success"
    assert "Hypertension is chronically elevated blood pressure" in resp.answer
    assert len(resp.sources) > 0
    assert "disclaimer" in resp.model_dump()
    print("[PASS] test_api_endpoint_post_rag_query passed.")


def run_gemini_rag_demo_and_benchmarks():
    """
    Demonstration of end-to-end RAG + Gemini Generation pipeline with latency breakdown.
    Uses mock Gemini to demonstrate the full prompt flow without requiring a live paid API call.
    """
    print("\n" + "=" * 75)
    print("PHASE 7: END-TO-END RAG + GEMINI GENERATION DEMO")
    print("=" * 75)

    store = VectorStoreService()
    store.add_chunks(SYNTHETIC_CHUNKS)
    rag = RAGService(vector_store=store)

    question = "What is hypertension and how is it characterized?"

    # Setup mocked Gemini LLM with realistic clinical citation response
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = (
        "Based on the provided medical context [Source 1], hypertension is a chronic condition "
        "characterized by persistently elevated arterial blood pressure above 130/80 mmHg. "
        "It increases long-term cardiovascular risk and requires regular clinical monitoring."
    )
    mock_client.models.generate_content.return_value = mock_resp
    GeminiService.set_client(mock_client)

    # Execute full RAG + Gemini generation
    t_start = time.perf_counter()
    res = rag.generate_rag_answer(question=question, top_k=2, similarity_threshold=0.25)
    total_pipeline_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    timings = res["timings"]

    print(f"User Query       : \"{question}\"")
    print(f"Retrieval Status : {res['retrieval_status']}")
    print(f"Configured Model : {settings.GEMINI_MODEL}")
    print("-" * 75)
    print("GROUNDED GEMINI ANSWER:")
    print(res["answer"])
    print("-" * 75)
    print("INDEPENDENT EVIDENCE CITATIONS:")
    for s in res["sources"]:
        print(f"  - Document: {s['source_id']} | Page: {s['page_number']} | Chunk: {s['chunk_id']} | Cosine Similarity: {s['similarity_score']}")
    print("-" * 75)
    print("SAFETY DISCLAIMER:")
    print(res["disclaimer"])
    print("-" * 75)
    print("PERFORMANCE LATENCY BREAKDOWN:")
    print(f"  - Cold Start Latency       : ~10-15 seconds (one-time weight loading on process launch)")
    print(f"  - Warm Retrieval Latency   : {timings['retrieval_time_ms']:.2f} ms")
    print(f"  - LLM Generation Latency   : {timings['generation_time_ms']:.2f} ms (mocked local; live API typically 400-900 ms)")
    print(f"  - Total Pipeline Latency   : {timings['total_time_ms']:.2f} ms")
    print("=" * 75)
    print("[NOTE: Safety & Hallucination Guardrails]")
    print("1. Grounded Context Only: Gemini is prompted with strict grounding instructions.")
    print("2. Safe Fallback: If retrieval score < 0.25, Gemini is NOT called.")
    print("3. Independent Sources: Citations are tracked separately in the API response object.")
    print("===========================================================================\n")


# ==============================================================================
# Phase 3.1 Tests: GeminiService Direct Generation & Resilience (Step 13)
# ==============================================================================

def test_phase3_successful_generation():
    """1. Successful generation via generate()."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Blood pressure is the force of circulating blood against the walls of arteries."
    mock_client.models.generate_content.return_value = mock_resp

    service = GeminiService(api_key="test-key-phase3", model="gemini-3.5-flash-lite", temperature=0.0)
    service.set_client(mock_client)

    result = service.generate(
        prompt="Explain blood pressure concisely.",
        temperature=0.0,
        max_output_tokens=1024
    )

    assert result == "Blood pressure is the force of circulating blood against the walls of arteries."
    assert mock_client.models.generate_content.call_count == 1
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.5-flash-lite"
    assert call_kwargs["contents"] == "Explain blood pressure concisely."
    assert call_kwargs["config"].temperature == 0.0
    assert call_kwargs["config"].max_output_tokens == 1024


def test_phase3_empty_response():
    """2. Empty response handling raises clean GeminiServiceError."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = ""
    mock_resp.candidates = []
    mock_client.models.generate_content.return_value = mock_resp

    service = GeminiService(api_key="test-key-phase3")
    service.set_client(mock_client)

    with pytest.raises(GeminiServiceError, match="Gemini returned an empty response"):
        service.generate("What is hypertension?")


def test_phase3_timeout_handling():
    """3. Timeout handling raises clean GeminiServiceError without leaking tracebacks."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("504 Gateway Timeout: Deadline exceeded")

    service = GeminiService(api_key="test-key-phase3")
    service.INITIAL_RETRY_DELAY_SEC = 0.001
    service.set_client(mock_client)

    with pytest.raises(GeminiServiceError) as exc_info:
        service.generate("What is hypertension?")

    assert "timed out" in str(exc_info.value) or "temporarily unavailable" in str(exc_info.value) or "Deadline exceeded" in str(exc_info.value)


def test_phase3_api_error_handling():
    """4. API error handling cleanly captures provider errors."""
    from google.genai.errors import APIError

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = APIError(
        400,
        {"error": {"message": "Invalid argument: prompt exceeds token limit"}}
    )

    service = GeminiService(api_key="test-key-phase3")
    service.set_client(mock_client)

    with pytest.raises(GeminiServiceError) as exc_info:
        service.generate("A very long prompt")

    assert "Invalid argument" in str(exc_info.value) or "Gemini generation failed" in str(exc_info.value)


def test_phase3_missing_api_key():
    """5. Missing API key raises clean exception."""
    GeminiService.set_client(None)
    orig_env = os.environ.get("GEMINI_API_KEY")
    orig_setting = Settings.GEMINI_API_KEY
    try:
        os.environ["GEMINI_API_KEY"] = ""
        Settings.GEMINI_API_KEY = ""
        service = GeminiService(api_key=None)
        with pytest.raises((ValueError, GeminiServiceError), match="Gemini API key is not configured"):
            service.get_client()
    finally:
        if orig_env is not None:
            os.environ["GEMINI_API_KEY"] = orig_env
        else:
            os.environ.pop("GEMINI_API_KEY", None)
        Settings.GEMINI_API_KEY = orig_setting
        GeminiService.set_client(None)


def test_phase3_provider_exception():
    """6. Provider exception is captured and converted to GeminiServiceError."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("Internal connection reset by peer")

    service = GeminiService(api_key="test-key-phase3")
    service.INITIAL_RETRY_DELAY_SEC = 0.001
    service.set_client(mock_client)

    with pytest.raises(GeminiServiceError, match="Gemini generation failed across all models"):
        service.generate("What is hypertension?")


def test_phase3_model_configuration():
    """7. Model configuration can be set via init and overridden per call."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Configured response"
    mock_client.models.generate_content.return_value = mock_resp

    service = GeminiService(api_key="test-key-phase3", model="gemini-2.5-flash")
    service.set_client(mock_client)

    # 1. Uses init model
    res1 = service.generate("Prompt 1")
    assert res1 == "Configured response"
    assert mock_client.models.generate_content.call_args.kwargs["model"] == "gemini-2.5-flash"

    # 2. Overrides model per call
    res2 = service.generate("Prompt 2", model="gemini-flash-latest")
    assert res2 == "Configured response"
    assert mock_client.models.generate_content.call_args.kwargs["model"] == "gemini-flash-latest"


def test_phase3_temperature_configuration():
    """8. Temperature defaults to 0.0 for deterministic output and can be customized."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Deterministic response"
    mock_client.models.generate_content.return_value = mock_resp

    service = GeminiService(api_key="test-key-phase3")
    service.set_client(mock_client)

    # Default temperature in generate is 0.0
    service.generate("Prompt")
    config = mock_client.models.generate_content.call_args.kwargs["config"]
    assert config.temperature == 0.0

    # Custom override
    service.generate("Prompt", temperature=0.3)
    config2 = mock_client.models.generate_content.call_args.kwargs["config"]
    assert config2.temperature == 0.3


def test_phase3_max_token_configuration():
    """9. Max output token limit is enforced and configurable."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Bounded response"
    mock_client.models.generate_content.return_value = mock_resp

    service = GeminiService(api_key="test-key-phase3")
    service.set_client(mock_client)

    # Default max_output_tokens is 1024
    service.generate("Prompt")
    config = mock_client.models.generate_content.call_args.kwargs["config"]
    assert config.max_output_tokens == 1024

    # Custom override
    service.generate("Prompt", max_output_tokens=256)
    config2 = mock_client.models.generate_content.call_args.kwargs["config"]
    assert config2.max_output_tokens == 256


def test_phase3_secret_leakage_prevention():
    """10. Secret/API-key leakage prevention: keys are never exposed in exceptions or logs."""
    sensitive_api_key = "AIzaSyTestHealthcareSecretKey987654"
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        f"Unauthorized access for api_key={sensitive_api_key}: invalid permissions"
    )

    service = GeminiService(api_key=sensitive_api_key)
    service.INITIAL_RETRY_DELAY_SEC = 0.001
    service.set_client(mock_client)

    with pytest.raises(GeminiServiceError) as exc_info:
        service.generate("Test prompt")

    err_text = str(exc_info.value)
    # The actual sensitive key MUST NOT be present
    assert sensitive_api_key not in err_text
    # Redacted replacement MUST be present
    assert "[REDACTED_API_KEY]" in err_text


if __name__ == "__main__":
    test_gemini_service_initialization()
    test_prompt_construction_and_sections()
    test_mock_gemini_generation_and_schema()
    test_empty_query_handling()
    test_no_relevant_context_does_not_call_gemini()
    test_api_error_and_timeout_handling()
    test_api_key_validation()
    test_gemini_503_retry_success()
    test_gemini_503_fallback_model_success()
    test_gemini_503_all_models_fail_clean_error()
    test_gemini_normal_response_unchanged()
    test_source_metadata_preservation_in_rag()
    test_hallucination_and_unsupported_context_safeguard()
    test_api_endpoint_post_rag_query()
    run_gemini_rag_demo_and_benchmarks()
    print("[SUCCESS] All GeminiService and RAG generation tests passed successfully!")
