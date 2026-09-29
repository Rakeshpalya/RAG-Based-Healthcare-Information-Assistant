import pytest
from backend.evaluation.observability import (
    sanitize_value,
    sanitize_log_dict,
    RAGStructuredLogEvent,
    StructuredRAGLogger
)


def test_sanitize_bearer_and_jwt_tokens():
    """Verify that Bearer tokens and raw JWTs are redacted from log values."""
    raw = "User requested with Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-IDN context."
    sanitized = sanitize_value(raw)
    assert "[REDACTED]" in sanitized or "[REDACTED_JWT]" in sanitized
    assert "eyJhbGciOiJIUzI1Ni" not in sanitized


def test_sanitize_api_keys():
    """Verify that Gemini and OpenAI format API keys are redacted."""
    gemini_key = "AIzaSyD-1234567890abcdefghijklmnopqrstuv"
    openai_key = "sk-1234567890abcdefghijklmnopqrstuv12345"
    
    assert "[REDACTED_API_KEY]" in sanitize_value(f"Key is {gemini_key}")
    assert "[REDACTED_OPENAI_KEY]" in sanitize_value(f"Key is {openai_key}")


def test_sanitize_ssn_and_credit_cards():
    """Verify that Social Security Numbers and credit card numbers are scrubbed."""
    ssn = "Patient SSN is 123-45-6789."
    assert "[REDACTED_SSN]" in sanitize_value(ssn)
    assert "123-45-6789" not in sanitize_value(ssn)

    card = "Paid with card 4111-2222-3333-4444 on file."
    assert "[REDACTED_CARD]" in sanitize_value(card)
    assert "4111-2222-3333-4444" not in sanitize_value(card)


def test_sanitize_database_credentials():
    """Verify that passwords embedded in database connection strings are redacted."""
    db_uri = "postgresql://postgres:SuperSecretPassword123@db.example.com:5432/ai_healthcare"
    sanitized = sanitize_value(db_uri)
    assert "SuperSecretPassword123" not in sanitized
    assert "[REDACTED_PASSWORD]@" in sanitized


def test_sanitize_exception_objects():
    """Verify that raw Exception objects containing sensitive credentials are safely stringified and scrubbed."""
    exc = Exception("Connection failed for postgresql://admin:leaked_pass@localhost:5432 with token eyJhbGciOiJIUzI1Ni.abc")
    sanitized = sanitize_value(exc)
    assert isinstance(sanitized, str)
    assert "leaked_pass" not in sanitized
    assert "eyJhbGciOiJIUzI1Ni" not in sanitized


def test_recursive_nested_structures():
    """Verify deep recursive scrubbing through nested dictionaries, lists, tuples, and sets."""
    payload = {
        "user_id": 42,
        "metadata": {
            "auth_header": "Bearer secret_user_token_12345",
            "api_key": "AIzaSyD-1234567890abcdefghijklmnopqrstuv",
            "nested_list": [
                {"patient_ssn": "987-65-4321"},
                ("sk-abcdefghijklmnopqrstuvwxyz123456",)
            ]
        }
    }

    sanitized = sanitize_log_dict(payload)
    assert sanitized["user_id"] == 42
    assert sanitized["metadata"]["api_key"] == "[REDACTED_CREDENTIAL]"
    assert "[REDACTED]" in sanitized["metadata"]["auth_header"]
    assert "[REDACTED_SSN]" in sanitized["metadata"]["nested_list"][0]["patient_ssn"]
    assert "[REDACTED_OPENAI_KEY]" in sanitized["metadata"]["nested_list"][1][0]


def test_structured_event_json_serialization_safety():
    """Verify that RAGStructuredLogEvent serialization scrubs all sensitive properties."""
    event = RAGStructuredLogEvent(
        request_id="req-test-123",
        query="What is the treatment for patient with SSN 111-22-3333?",
        safety_classification="NORMAL_MEDICAL_INFORMATION",
        risk_level="LOW",
        retrieval_status="success",
        embedding_latency_ms=12.5,
        vector_search_latency_ms=4.2,
        context_construction_latency_ms=1.1,
        llm_latency_ms=250.0,
        total_latency_ms=267.8,
        number_of_retrieved_chunks=3,
        similarity_scores=[0.82, 0.75, 0.68],
        citation_validation_passed=True,
        citation_coverage=1.0,
        hallucination_detected=False,
        hallucination_types=[],
        final_status="success"
    )

    log_dict = event.to_dict()
    assert "111-22-3333" not in log_dict["query"]
    assert "[REDACTED_SSN]" in log_dict["query"]
    json_str = event.to_json()
    assert "111-22-3333" not in json_str
