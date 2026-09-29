import os
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from google import genai
from google.genai import types
from google.genai.errors import APIError

from backend.config import settings
from backend.rag.prompt_builder import (
    build_rag_prompt,
    HEALTHCARE_SYSTEM_INSTRUCTIONS,
    MEDICAL_DISCLAIMER
)

logger = logging.getLogger(__name__)


class GeminiServiceError(Exception):
    """Custom application-level exception for Gemini service failures."""
    pass


class GeminiService:
    """
    Reusable Gemini LLM Service for context-grounded medical answer generation.

    Features:
    - Lazy-loaded singleton client: client is instantiated once and reused across calls.
    - Configurable model and temperature via Settings or per-request overrides.
    - Robust error, fallback, and timeout handling with sanitized, secret-safe exceptions.
    - Controlled fallback chain across supported Google GenAI models (e.g. gemini-3.5-flash, gemini-3.5-flash-lite, gemini-flash-latest).
    - Defensive response parsing for text parts, candidate structures, and safety blocks.
    - Clean integration with prompt construction and grounding instructions.
    """

    _client: Optional[genai.Client] = None

    # Curated, documented fallback chain of verified supported models in Google GenAI SDK
    FALLBACK_MODELS: List[str] = [
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash-lite",
    ]

    MAX_RETRIES_PER_MODEL: int = 2
    INITIAL_RETRY_DELAY_SEC: float = 0.5
    BACKOFF_FACTOR: float = 2.0
    MAX_RETRY_DELAY_SEC: float = 2.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        """
        Initializes GeminiService.

        Args:
            api_key: Optional explicit API key. If omitted, fetched from Settings.
            model: Optional model name override (defaults to settings.GEMINI_MODEL).
            temperature: Optional sampling temperature override (defaults to settings.GEMINI_TEMPERATURE).
        """
        self._api_key = api_key
        self.model = model or settings.GEMINI_MODEL or "gemini-3.5-flash-lite"
        self.temperature = float(temperature if temperature is not None else settings.GEMINI_TEMPERATURE)

    def get_client(self) -> genai.Client:
        """
        Returns or initializes the reusable Google GenAI Client (Singleton pattern).
        Ensures the client is created once and reused across requests.
        """
        if self._client is None:
            # Validate API key without exposing secret
            api_key = self._api_key or settings.validate_gemini_api_key()
            try:
                self._client = genai.Client(api_key=api_key)
            except Exception as e:
                raise GeminiServiceError(f"Failed to initialize Gemini client: {str(e)}")
        return self._client

    @classmethod
    def set_client(cls, client: Optional[genai.Client]) -> None:
        """Allows injecting a mock client for unit testing."""
        cls._client = client

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """
        Safely extracts text content from Google GenAI SDK GenerateContentResponse.
        Defensively checks .text attribute, candidate objects, and content parts.
        """
        if not response:
            return ""

        # 1. Primary: direct response.text
        try:
            txt = getattr(response, "text", None)
            if txt and isinstance(txt, str) and txt.strip():
                return txt.strip()
        except Exception:
            pass

        # 2. Inspect candidates and text parts
        candidates = getattr(response, "candidates", None)
        if candidates and len(candidates) > 0:
            first_candidate = candidates[0]
            content = getattr(first_candidate, "content", None)
            if content:
                parts = getattr(content, "parts", None)
                if parts:
                    extracted = []
                    for part in parts:
                        ptxt = getattr(part, "text", None)
                        if ptxt and isinstance(ptxt, str):
                            extracted.append(ptxt)
                    if extracted:
                        return "".join(extracted).strip()
        return ""

    def generate_answer(
        self,
        question: str,
        context: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        conversation_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a context-grounded medical answer using Gemini with automatic controlled fallback.

        Args:
            question: User's question or medical research query.
            context: Formatted context block with [SOURCE N] headers.
            model: Optional model override.
            temperature: Optional temperature override.
            conversation_context: Optional brief context from prior turn for reference resolution.

        Returns:
            Dictionary containing:
            - answer: The synthesized answer string
            - model: The model name used
            - disclaimer: Standard medical safety disclaimer
            - generation_time_ms: Latency in milliseconds
            - status: "success"
        """
        if not question or not question.strip():
            return {
                "answer": "No question was provided.",
                "model": model or self.model,
                "disclaimer": MEDICAL_DISCLAIMER,
                "generation_time_ms": 0.0,
                "status": "empty_query"
            }

        effective_model = model or self.model or "gemini-3.5-flash-lite"
        effective_temp = float(temperature if temperature is not None else self.temperature)

        prompt = build_rag_prompt(
            question=question,
            context=context,
            conversation_context=conversation_context
        )
        client = self.get_client()

        config = types.GenerateContentConfig(
            temperature=effective_temp,
            system_instruction=HEALTHCARE_SYSTEM_INSTRUCTIONS,
            max_output_tokens=1024,
        )

        # Build ordered candidate model list (primary model first, then supported fallbacks)
        candidate_models: List[str] = [effective_model]
        for fb_model in self.FALLBACK_MODELS:
            if fb_model not in candidate_models:
                candidate_models.append(fb_model)

        last_error: Optional[Exception] = None
        request_start_iso = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()
        calls_count = 0

        for idx, candidate_model in enumerate(candidate_models):
            for retry_attempt in range(self.MAX_RETRIES_PER_MODEL + 1):
                calls_count += 1
                call_start_time = time.perf_counter()
                logger.info(
                    "Gemini API request start: call=%d, model='%s', attempt=%d/%d, max_tokens=1024, temp=%.2f, time=%s",
                    calls_count,
                    candidate_model,
                    retry_attempt + 1,
                    self.MAX_RETRIES_PER_MODEL + 1,
                    effective_temp,
                    request_start_iso,
                )
                try:
                    response = client.models.generate_content(
                        model=candidate_model,
                        contents=prompt,
                        config=config
                    )
                    call_duration_ms = round((time.perf_counter() - call_start_time) * 1000.0, 2)
                    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

                    usage = getattr(response, "usage_metadata", None)
                    input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
                    output_tokens = getattr(usage, "candidates_token_count", None) if usage else None

                    logger.info(
                        "Gemini API call %d complete: model='%s', api_time=%.2fms, total_elapsed=%.2fms, in_tokens=%s, out_tokens=%s",
                        calls_count,
                        candidate_model,
                        call_duration_ms,
                        elapsed_ms,
                        input_tokens,
                        output_tokens,
                    )

                    answer_text = self._extract_response_text(response)
                    if not answer_text:
                        # Check finish reason for safety block
                        candidates = getattr(response, "candidates", None)
                        if candidates and len(candidates) > 0:
                            finish_reason = getattr(candidates[0], "finish_reason", None)
                            if str(finish_reason).upper() in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"):
                                raise GeminiServiceError("Gemini response was blocked by safety filters.")
                        raise GeminiServiceError("Gemini returned an empty response.")

                    from backend.rag.markdown_utils import clean_ai_markdown
                    answer_text = clean_ai_markdown(answer_text)

                    if idx > 0:
                        logger.info(
                            "Primary model '%s' failed, successfully generated answer using fallback model '%s'",
                            effective_model,
                            candidate_model
                        )

                    return {
                        "answer": answer_text,
                        "model": candidate_model,
                        "disclaimer": MEDICAL_DISCLAIMER,
                        "generation_time_ms": elapsed_ms,
                        "api_request_time_ms": call_duration_ms,
                        "request_start_time": request_start_iso,
                        "gemini_calls_count": calls_count,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "time_to_first_token_ms": None,
                        "status": "success"
                    }

                except APIError as api_err:
                    last_error = api_err
                    status_code = getattr(api_err, "code", None)
                    err_msg = getattr(api_err, "message", str(api_err))
                    logger.warning(
                        "Gemini API error (%s) on model '%s' (attempt %d/%d): %s",
                        status_code,
                        candidate_model,
                        retry_attempt + 1,
                        self.MAX_RETRIES_PER_MODEL + 1,
                        err_msg
                    )

                    is_transient = (
                        status_code in (503, 429, 504)
                        or "high demand" in str(err_msg).lower()
                        or "temporarily" in str(err_msg).lower()
                        or "resource exhausted" in str(err_msg).lower()
                        or "deadline exceeded" in str(err_msg).lower()
                    )

                    # Bounded exponential backoff retry on transient errors
                    if is_transient and retry_attempt < self.MAX_RETRIES_PER_MODEL:
                        backoff = min(
                            self.INITIAL_RETRY_DELAY_SEC * (self.BACKOFF_FACTOR ** retry_attempt),
                            self.MAX_RETRY_DELAY_SEC
                        )
                        time.sleep(backoff)
                        continue

                    # Exhausted retries for this model; break to attempt next candidate fallback model
                    break

                except GeminiServiceError:
                    raise

                except Exception as e:
                    last_error = e
                    err_msg = str(e)
                    if self._api_key and self._api_key in err_msg:
                        err_msg = err_msg.replace(self._api_key, "[REDACTED_API_KEY]")
                    logger.warning(
                        "Unexpected error on model '%s' (attempt %d/%d): %s",
                        candidate_model,
                        retry_attempt + 1,
                        self.MAX_RETRIES_PER_MODEL + 1,
                        err_msg
                    )

                    is_transient_exc = any(
                        token in err_msg.lower()
                        for token in ("503", "429", "504", "high demand", "temporarily", "timeout", "deadline")
                    )
                    if is_transient_exc and retry_attempt < self.MAX_RETRIES_PER_MODEL:
                        backoff = min(
                            self.INITIAL_RETRY_DELAY_SEC * (self.BACKOFF_FACTOR ** retry_attempt),
                            self.MAX_RETRY_DELAY_SEC
                        )
                        time.sleep(backoff)
                        continue

                    break

        if last_error:
            err_str = str(last_error).lower()
            is_high_demand = (
                "503" in err_str
                or "high demand" in err_str
                or "temporarily" in err_str
                or "resource exhausted" in err_str
            )
            if is_high_demand:
                raise GeminiServiceError(
                    "The AI generation service is temporarily unavailable due to high demand. "
                    "Please wait a few moments and try your inquiry again."
                )
            raise GeminiServiceError(f"Gemini generation failed across all models: {str(last_error)}")
        raise GeminiServiceError("The AI generation service is temporarily unavailable. Please try again shortly.")
