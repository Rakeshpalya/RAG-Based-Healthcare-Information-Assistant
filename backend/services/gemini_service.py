import os
import re
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
    - Configurable model, temperature, max_output_tokens, and timeout via Settings or per-call overrides.
    - Robust error, fallback, and timeout handling with sanitized, secret-safe exceptions.
    - Controlled fallback chain across supported Google GenAI models (e.g. gemini-3.5-flash-lite, gemini-flash-latest).
    - Defensive response parsing for text parts, candidate structures, and safety blocks.
    - Clean integration with prompt construction, injection defense, and grounding instructions.
    - Dual generation API: direct string generation via `generate()` and structured RAG synthesis via `generate_answer()`.
    """

    _client: Optional[genai.Client] = None
    _client_warm: bool = False
    _warmup_duration_ms: float = 0.0
    _warmup_timestamp: Optional[str] = None

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
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None
    ):
        """
        Initializes GeminiService.

        Args:
            api_key: Optional explicit API key. If omitted, fetched from Settings.
            model: Optional model name override (defaults to settings.GEMINI_MODEL).
            temperature: Optional sampling temperature override (defaults to settings.GEMINI_TEMPERATURE).
            max_output_tokens: Optional max output tokens limit (defaults to settings.GEMINI_MAX_OUTPUT_TOKENS).
            timeout_seconds: Optional timeout in seconds (defaults to settings.GEMINI_TIMEOUT_SECONDS).
        """
        self._api_key = api_key
        self.model = model or settings.GEMINI_MODEL or "gemini-3.5-flash-lite"
        self.temperature = float(temperature if temperature is not None else settings.GEMINI_TEMPERATURE)
        self.max_output_tokens = int(max_output_tokens if max_output_tokens is not None else settings.GEMINI_MAX_OUTPUT_TOKENS)
        self.timeout_seconds = float(timeout_seconds if timeout_seconds is not None else settings.GEMINI_TIMEOUT_SECONDS)

    @classmethod
    def sanitize_secret_static(cls, text: Any, explicit_key: Optional[str] = None) -> str:
        """
        Redacts API keys and sensitive tokens from error messages and logs.
        Never allows raw credentials to leak in logs, tracebacks, or exceptions.
        """
        if text is None:
            return ""
        sanitized = str(text)
        if explicit_key and explicit_key in sanitized:
            sanitized = sanitized.replace(explicit_key, "[REDACTED_API_KEY]")
        if settings.GEMINI_API_KEY and settings.GEMINI_API_KEY in sanitized:
            sanitized = sanitized.replace(settings.GEMINI_API_KEY, "[REDACTED_API_KEY]")
        # Standard Google API key pattern: AIzaSy...
        sanitized = re.sub(r'AIza[0-9A-Za-z\-_]{35}', '[REDACTED_API_KEY]', sanitized)
        return sanitized

    def _sanitize_secret(self, text: Any) -> str:
        """Redacts API keys and sensitive tokens from error messages and logs."""
        return self.sanitize_secret_static(text, self._api_key)

    def get_client(self) -> genai.Client:
        """
        Returns or initializes the reusable Google GenAI Client (Singleton pattern).
        Ensures the client is created once and reused across requests.
        """
        if GeminiService._client is None:
            api_key = self._api_key or settings.validate_gemini_api_key()
            try:
                timeout_ms = int(self.timeout_seconds * 1000) if self.timeout_seconds else 30000
                http_opts = types.HttpOptions(timeout=timeout_ms)
                GeminiService._client = genai.Client(api_key=api_key, http_options=http_opts)
                GeminiService._client_warm = True
            except Exception as e:
                err_msg = self._sanitize_secret(str(e))
                raise GeminiServiceError(f"Failed to initialize Gemini client: {err_msg}")
        return GeminiService._client

    @classmethod
    def prewarm_client(
        cls,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Pre-warms the Google GenAI SDK client at application startup.
        Ensures the client object, connection pooling, and SDK schemas are loaded once.
        Does not execute external generation calls.
        Fails safely if GEMINI_API_KEY is missing or invalid without exposing secrets.
        """
        start = time.perf_counter()
        if cls._client is not None:
            return {
                "status": "already_warm",
                "warm": True,
                "warmup_duration_ms": cls._warmup_duration_ms,
                "timestamp": cls._warmup_timestamp,
            }

        effective_key = api_key
        if not effective_key:
            try:
                effective_key = settings.validate_gemini_api_key()
            except Exception as exc:
                err_msg = cls.sanitize_secret_static(str(exc))
                logger.warning("Gemini pre-warm skipped: %s", err_msg)
                return {
                    "status": "unconfigured",
                    "warm": False,
                    "warmup_duration_ms": 0.0,
                    "error": err_msg,
                }

        try:
            effective_timeout = timeout_seconds if timeout_seconds is not None else settings.GEMINI_TIMEOUT_SECONDS
            timeout_ms = int(effective_timeout * 1000) if effective_timeout else 30000
            http_opts = types.HttpOptions(timeout=timeout_ms)
            cls._client = genai.Client(api_key=effective_key, http_options=http_opts)
            duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
            cls._client_warm = True
            cls._warmup_duration_ms = duration_ms
            cls._warmup_timestamp = datetime.now(timezone.utc).isoformat()
            logger.info("Gemini client successfully pre-warmed in %.2fms (model: %s)", duration_ms, settings.GEMINI_MODEL)
            return {
                "status": "success",
                "warm": True,
                "warmup_duration_ms": duration_ms,
                "timestamp": cls._warmup_timestamp,
            }
        except Exception as e:
            err_msg = cls.sanitize_secret_static(str(e))
            logger.warning("Gemini client pre-warming encountered non-fatal error: %s", err_msg)
            return {
                "status": "failed",
                "warm": False,
                "warmup_duration_ms": round((time.perf_counter() - start) * 1000.0, 2),
                "error": err_msg,
            }

    @classmethod
    def is_client_warm(cls) -> bool:
        """Returns True if the client has been initialized and is ready."""
        return cls._client is not None and cls._client_warm

    @classmethod
    def reset_client(cls) -> None:
        """Resets the singleton client state for testing."""
        cls._client = None
        cls._client_warm = False
        cls._warmup_duration_ms = 0.0
        cls._warmup_timestamp = None

    def prewarm(self) -> Dict[str, Any]:
        """Instance alias for prewarm_client()."""
        return self.prewarm_client(api_key=self._api_key, timeout_seconds=self.timeout_seconds)

    @classmethod
    def set_client(cls, client: Optional[genai.Client]) -> None:
        """Allows injecting a mock client for unit testing."""
        cls._client = client
        cls._client_warm = client is not None

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

    def _execute_generation(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Internal execution engine for Gemini generation with fallback models and bounded retries.

        Returns:
            Dict containing:
            - text: Generated text
            - model: Winning model name
            - elapsed_ms: Total latency across retries/models
            - call_duration_ms: Winning call duration
            - request_start_iso: ISO start timestamp
            - calls_count: Total API attempts
            - input_tokens: Token count if reported
            - output_tokens: Candidate token count if reported
        """
        effective_model = model or self.model or "gemini-3.5-flash-lite"
        effective_temp = float(temperature if temperature is not None else self.temperature)
        effective_max_tokens = int(max_output_tokens if max_output_tokens is not None else self.max_output_tokens)

        client = self.get_client()

        config = types.GenerateContentConfig(
            temperature=effective_temp,
            max_output_tokens=effective_max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

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
                    "Gemini API request start: call=%d, model='%s', attempt=%d/%d, max_tokens=%d, temp=%.2f, time=%s",
                    calls_count,
                    candidate_model,
                    retry_attempt + 1,
                    self.MAX_RETRIES_PER_MODEL + 1,
                    effective_max_tokens,
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

                    if idx > 0:
                        logger.info(
                            "Primary model '%s' failed, successfully generated answer using fallback model '%s'",
                            effective_model,
                            candidate_model
                        )

                    return {
                        "text": answer_text,
                        "model": candidate_model,
                        "elapsed_ms": elapsed_ms,
                        "call_duration_ms": call_duration_ms,
                        "request_start_iso": request_start_iso,
                        "calls_count": calls_count,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }

                except APIError as api_err:
                    last_error = api_err
                    status_code = getattr(api_err, "code", None)
                    raw_msg = getattr(api_err, "message", str(api_err))
                    err_msg = self._sanitize_secret(raw_msg)
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
                        or "timeout" in str(err_msg).lower()
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

                except (KeyboardInterrupt, SystemExit):
                    raise

                except Exception as e:
                    # Fast abort on task cancellation - do not retry cancelled requests
                    if type(e).__name__ in ("CancelledError", "Cancelled"):
                        raise
                    last_error = e
                    err_msg = self._sanitize_secret(str(e))
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
            err_str = self._sanitize_secret(str(last_error)).lower()
            is_high_demand = (
                "503" in err_str
                or "429" in err_str
                or "high demand" in err_str
                or "temporarily" in err_str
                or "resource exhausted" in err_str
            )
            if is_high_demand:
                raise GeminiServiceError(
                    "The AI generation service is temporarily unavailable due to high demand. "
                    "Please wait a few moments and try your inquiry again."
                )
            if "timeout" in err_str or "deadline" in err_str or "504" in err_str:
                raise GeminiServiceError(
                    f"The AI generation service timed out while processing the request: {self._sanitize_secret(str(last_error))}"
                )
            raise GeminiServiceError(f"Gemini generation failed across all models: {self._sanitize_secret(str(last_error))}")

        raise GeminiServiceError("The AI generation service is temporarily unavailable. Please try again shortly.")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Direct production generation method for Phase 3.1.
        Calls the LLM service with conservative clinical parameters and returns raw text.

        Args:
            prompt: Text prompt to generate content for.
            temperature: Sampling temperature (default: 0.0 for deterministic medical generation).
            max_output_tokens: Max output token limit (default: 1024).
            model: Optional model override.
            system_instruction: Optional system instruction override.

        Returns:
            The generated response string.

        Raises:
            GeminiServiceError: If prompt is empty, client call fails, safety filters block, or timeout occurs.
        """
        if not prompt or not prompt.strip():
            raise GeminiServiceError("Prompt cannot be empty.")

        res = self._execute_generation(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            model=model,
            system_instruction=system_instruction,
        )
        return res["text"]

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
        Preserves backward compatibility for RAGService, multi-agent orchestrator, and test suites.

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

        prompt = build_rag_prompt(
            question=question,
            context=context,
            conversation_context=conversation_context
        )

        res = self._execute_generation(
            prompt=prompt,
            temperature=temperature,
            model=model,
            system_instruction=HEALTHCARE_SYSTEM_INSTRUCTIONS,
        )

        from backend.rag.markdown_utils import clean_ai_markdown
        cleaned_text = clean_ai_markdown(res["text"])

        return {
            "answer": cleaned_text,
            "model": res["model"],
            "disclaimer": MEDICAL_DISCLAIMER,
            "generation_time_ms": res["elapsed_ms"],
            "api_request_time_ms": res["call_duration_ms"],
            "request_start_time": res["request_start_iso"],
            "gemini_calls_count": res["calls_count"],
            "input_tokens": res["input_tokens"],
            "output_tokens": res["output_tokens"],
            "time_to_first_token_ms": None,
            "status": "success"
        }

    def generate_answer_from_prompt(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generates a context-grounded answer directly from a pre-built synthesis prompt.
        Preserves automatic model fallback, retry backoff, and error hygiene.
        """
        if not prompt or not prompt.strip():
            return {
                "answer": "No prompt was provided.",
                "model": model or self.model,
                "disclaimer": MEDICAL_DISCLAIMER,
                "generation_time_ms": 0.0,
                "status": "empty_query"
            }

        effective_temp = temperature if temperature is not None else self.temperature
        effective_tokens = max_output_tokens if max_output_tokens is not None else self.max_output_tokens
        effective_model = model or self.model

        res = self._execute_generation(
            prompt=prompt,
            temperature=effective_temp,
            max_output_tokens=effective_tokens,
            model=effective_model,
            system_instruction=HEALTHCARE_SYSTEM_INSTRUCTIONS
        )

        from backend.rag.markdown_utils import clean_ai_markdown
        cleaned_text = clean_ai_markdown(res["text"])

        return {
            "answer": cleaned_text,
            "model": res["model"],
            "disclaimer": MEDICAL_DISCLAIMER,
            "generation_time_ms": res["elapsed_ms"],
            "api_request_time_ms": res["call_duration_ms"],
            "request_start_time": res["request_start_iso"],
            "gemini_calls_count": res["calls_count"],
            "input_tokens": res.get("input_tokens"),
            "output_tokens": res.get("output_tokens"),
            "time_to_first_token_ms": None,
            "status": "success"
        }

    def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Streams generated response chunks from Gemini using Google GenAI SDK.
        Yields text chunks as they arrive.
        Preserves fallback chain and secret redaction.
        """
        if not prompt or not prompt.strip():
            raise GeminiServiceError("Prompt cannot be empty.")

        effective_model = model or self.model or "gemini-3.5-flash-lite"
        effective_temp = float(temperature if temperature is not None else self.temperature)
        effective_max_tokens = int(max_output_tokens if max_output_tokens is not None else self.max_output_tokens)

        client = self.get_client()
        config = types.GenerateContentConfig(
            temperature=effective_temp,
            max_output_tokens=effective_max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        candidate_models = [effective_model]
        for fb_model in self.FALLBACK_MODELS:
            if fb_model not in candidate_models:
                candidate_models.append(fb_model)

        last_error = None
        for candidate_model in candidate_models:
            try:
                logger.info("Attempting Gemini streaming generation with model: %s", candidate_model)
                response_stream = client.models.generate_content_stream(
                    model=candidate_model,
                    contents=prompt,
                    config=config
                )
                yielded_any = False
                for chunk in response_stream:
                    text_chunk = self._extract_response_text(chunk)
                    if text_chunk:
                        yielded_any = True
                        yield text_chunk
                if yielded_any:
                    return
            except Exception as exc:
                last_error = exc
                err_msg = self._sanitize_secret(str(exc))
                logger.warning("Streaming error on model '%s': %s", candidate_model, err_msg)
                continue

        if last_error:
            raise GeminiServiceError(f"Gemini streaming failed across models: {self._sanitize_secret(str(last_error))}")
        raise GeminiServiceError("Gemini streaming returned no content.")
