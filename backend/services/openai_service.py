"""
OpenAI LLM Service for AI-Healthcare-Agent.

Provides robust, context-grounded medical answer generation, document summarization,
and general-purpose conversational responses powered by the official OpenAI Python SDK.
"""

import os
import re
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Generator

try:
    import openai
    from openai import OpenAI, APIError, RateLimitError, AuthenticationError, APITimeoutError, APIConnectionError
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False
    OpenAI = Any  # type: ignore
    APIError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore
    AuthenticationError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    APIConnectionError = Exception  # type: ignore

from backend.config import settings
from backend.rag.prompt_builder import (
    build_rag_prompt,
    HEALTHCARE_SYSTEM_INSTRUCTIONS,
    MEDICAL_DISCLAIMER
)
from backend.rag.markdown_utils import clean_ai_markdown

logger = logging.getLogger(__name__)

GENERAL_SYSTEM_INSTRUCTIONS = (
    "You are HealthAI Assistant, a versatile, helpful, and articulate general-purpose AI assistant. "
    "Provide clear, accurate, well-structured, and helpful answers across education, technology, "
    "science, coding, geography, writing, and general knowledge. "
    "Use clean GitHub-flavored Markdown (headings, lists, bold text, code blocks) where appropriate. "
    "State limitations honestly if a question requires external or real-time data."
)


class OpenAIServiceError(Exception):
    """Custom application-level exception for OpenAI service failures."""
    def __init__(self, message: str, error_type: str = "general", status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code


class OpenAIService:
    """
    Reusable OpenAI LLM Service for context-grounded medical answer generation,
    document summarization, and general-purpose chatbot responses.

    Features:
    - Lazy-loaded singleton client: client is instantiated once and reused across requests.
    - Configurable model, temperature, max_output_tokens, and timeout via Settings or per-call overrides.
    - Robust error handling with sanitized, secret-safe exceptions.
    - Exponential backoff for transient network issues.
    - Distinct handling for quota exhaustion, authentication, and timeout.
    """

    _client: Optional[OpenAI] = None
    _client_warm: bool = False
    _warmup_duration_ms: float = 0.0
    _warmup_timestamp: Optional[str] = None
    _quota_exhausted_until: float = 0.0

    MAX_RETRIES: int = 2
    INITIAL_RETRY_DELAY_SEC: float = 0.5
    BACKOFF_FACTOR: float = 2.0
    MAX_RETRY_DELAY_SEC: float = 2.0

    @classmethod
    def mark_quota_exhausted(cls, cooldown_seconds: float = 60.0) -> None:
        """Marks the OpenAI service as quota-exhausted for a cooldown duration to avoid repeated failing calls."""
        cls._quota_exhausted_until = time.time() + cooldown_seconds

    @classmethod
    def is_quota_exhausted(cls) -> bool:
        """Returns True if the OpenAI API is currently within a quota-exhaustion cooldown window."""
        return time.time() < cls._quota_exhausted_until

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None
    ):
        self._api_key = api_key
        self.model = model or getattr(settings, "OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
        self.temperature = float(
            temperature if temperature is not None
            else getattr(settings, "OPENAI_TEMPERATURE", 0.0)
        )
        self.max_output_tokens = int(
            max_output_tokens if max_output_tokens is not None
            else getattr(settings, "OPENAI_MAX_OUTPUT_TOKENS", 1024)
        )
        self.timeout_seconds = float(
            timeout_seconds if timeout_seconds is not None
            else getattr(settings, "OPENAI_TIMEOUT_SECONDS", 30.0)
        )

    @classmethod
    def sanitize_secret_static(cls, text: Any, explicit_key: Optional[str] = None) -> str:
        """Redacts API keys and sensitive tokens from error messages and logs."""
        if text is None:
            return ""
        sanitized = str(text)
        if explicit_key and explicit_key in sanitized:
            sanitized = sanitized.replace(explicit_key, "[REDACTED_API_KEY]")
        app_key = getattr(settings, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
        if app_key and app_key in sanitized:
            sanitized = sanitized.replace(app_key, "[REDACTED_API_KEY]")
        # Standard OpenAI API key pattern: sk-proj-... or sk-...
        sanitized = re.sub(r'sk-[a-zA-Z0-9_\-]{20,}', '[REDACTED_API_KEY]', sanitized)
        return sanitized

    def _sanitize_secret(self, text: Any) -> str:
        return self.sanitize_secret_static(text, self._api_key)

    def get_client(self) -> OpenAI:
        """Returns or initializes the reusable OpenAI Client singleton."""
        if not OPENAI_SDK_AVAILABLE:
            raise OpenAIServiceError(
                "The 'openai' Python package is not installed. Please install 'openai' to use OpenAI services.",
                error_type="missing_dependency",
                status_code=500
            )

        if OpenAIService._client is None or self._api_key is not None:
            api_key = self._api_key or getattr(settings, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
            if not api_key or not str(api_key).strip():
                raise OpenAIServiceError(
                    "OPENAI_API_KEY is not configured. Please configure a valid OpenAI API key in your .env file.",
                    error_type="missing_config",
                    status_code=401
                )
            try:
                client = OpenAI(
                    api_key=str(api_key).strip(),
                    timeout=self.timeout_seconds,
                    max_retries=0  # Managed by our custom retry loop
                )
                if self._api_key is None:
                    OpenAIService._client = client
                    OpenAIService._client_warm = True
                return client
            except Exception as e:
                err_msg = self._sanitize_secret(str(e))
                raise OpenAIServiceError(
                    f"Failed to initialize OpenAI client: {err_msg}",
                    error_type="initialization_error",
                    status_code=500
                )
        return OpenAIService._client

    @classmethod
    def is_client_warm(cls) -> bool:
        return cls._client_warm and cls._client is not None

    @classmethod
    def prewarm_client(cls) -> Dict[str, Any]:
        """Pre-warms client at startup."""
        t0 = time.perf_counter()
        key = getattr(settings, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY")
        if not key or not str(key).strip():
            return {
                "status": "skipped",
                "warm": False,
                "reason": "OPENAI_API_KEY is not configured.",
                "warmup_duration_ms": 0.0
            }
        try:
            svc = cls()
            svc.get_client()
            duration_ms = (time.perf_counter() - t0) * 1000.0
            cls._warmup_duration_ms = duration_ms
            cls._warmup_timestamp = datetime.now(timezone.utc).isoformat()
            return {
                "status": "ready",
                "warm": True,
                "warmup_duration_ms": round(duration_ms, 2),
                "timestamp": cls._warmup_timestamp
            }
        except Exception as exc:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status": "error",
                "warm": False,
                "error": cls.sanitize_secret_static(str(exc)),
                "warmup_duration_ms": round(duration_ms, 2)
            }

    def _execute_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Executes chat completion with bounded retries and sanitized error handling."""
        effective_model = model or self.model or "gpt-4o-mini"
        effective_temp = float(temperature if temperature is not None else self.temperature)
        effective_tokens = int(max_output_tokens if max_output_tokens is not None else self.max_output_tokens)

        if self.is_quota_exhausted():
            raise OpenAIServiceError(
                "OpenAI API credit balance or quota exhausted. "
                "Please check your OpenAI billing settings at https://platform.openai.com/account/billing.",
                error_type="quota_exhausted",
                status_code=429
            )

        client = self.get_client()
        t0 = time.perf_counter()
        req_start_iso = datetime.now(timezone.utc).isoformat()
        calls_count = 0
        last_error = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                calls_count += 1
                response = client.chat.completions.create(
                    model=effective_model,
                    messages=messages,  # type: ignore
                    temperature=effective_temp,
                    max_tokens=effective_tokens,
                )

                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                choices = getattr(response, "choices", [])
                if not choices:
                    raise OpenAIServiceError("OpenAI returned an empty response choices list.", error_type="empty_response")

                raw_text = choices[0].message.content or ""
                usage = getattr(response, "usage", None)

                return {
                    "text": raw_text,
                    "model": getattr(response, "model", effective_model),
                    "elapsed_ms": round(elapsed_ms, 2),
                    "call_duration_ms": round(elapsed_ms, 2),
                    "request_start_iso": req_start_iso,
                    "calls_count": calls_count,
                    "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                    "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                    "status": "success"
                }

            except AuthenticationError as auth_err:
                sanitized_msg = self._sanitize_secret(str(auth_err))
                logger.error("OpenAI authentication error: %s", sanitized_msg)
                raise OpenAIServiceError(
                    "OpenAI authentication failed: The provided API key is invalid or lacks required permissions. "
                    "Please verify your OPENAI_API_KEY in .env.",
                    error_type="auth_error",
                    status_code=401
                ) from auth_err

            except RateLimitError as rl_err:
                sanitized_msg = self._sanitize_secret(str(rl_err))
                logger.warning("OpenAI rate limit / quota exceeded: %s", sanitized_msg)
                err_lower = sanitized_msg.lower()
                if "quota" in err_lower or "credit" in err_lower or "billing" in err_lower or "balance" in err_lower:
                    self.mark_quota_exhausted(cooldown_seconds=60.0)
                    raise OpenAIServiceError(
                        "OpenAI API credit balance or quota exhausted. "
                        "Please check your OpenAI billing settings at https://platform.openai.com/account/billing.",
                        error_type="quota_exhausted",
                        status_code=429
                    ) from rl_err
                else:
                    raise OpenAIServiceError(
                        "OpenAI API rate limit exceeded. Please wait a moment and try again.",
                        error_type="rate_limit",
                        status_code=429
                    ) from rl_err

            except (APITimeoutError, TimeoutError) as to_err:
                last_error = to_err
                logger.warning("OpenAI API call timed out (attempt %d/%d)", attempt + 1, self.MAX_RETRIES + 1)
                if attempt < self.MAX_RETRIES:
                    delay = min(self.INITIAL_RETRY_DELAY_SEC * (self.BACKOFF_FACTOR ** attempt), self.MAX_RETRY_DELAY_SEC)
                    time.sleep(delay)
                    continue
                raise OpenAIServiceError(
                    "OpenAI API request timed out. Please try again shortly.",
                    error_type="timeout",
                    status_code=504
                ) from to_err

            except (APIConnectionError, ConnectionError) as conn_err:
                last_error = conn_err
                logger.warning("OpenAI network connection failure (attempt %d/%d): %s", attempt + 1, self.MAX_RETRIES + 1, self._sanitize_secret(str(conn_err)))
                if attempt < self.MAX_RETRIES:
                    delay = min(self.INITIAL_RETRY_DELAY_SEC * (self.BACKOFF_FACTOR ** attempt), self.MAX_RETRY_DELAY_SEC)
                    time.sleep(delay)
                    continue
                raise OpenAIServiceError(
                    "Unable to connect to the OpenAI API service. Please verify your internet connection.",
                    error_type="connection_error",
                    status_code=503
                ) from conn_err

            except APIError as api_err:
                last_error = api_err
                sanitized_msg = self._sanitize_secret(str(api_err))
                logger.error("OpenAI API error: %s", sanitized_msg)
                status_code = getattr(api_err, "status_code", 500)
                if status_code and status_code >= 500 and attempt < self.MAX_RETRIES:
                    delay = min(self.INITIAL_RETRY_DELAY_SEC * (self.BACKOFF_FACTOR ** attempt), self.MAX_RETRY_DELAY_SEC)
                    time.sleep(delay)
                    continue
                raise OpenAIServiceError(
                    f"OpenAI service returned an error: {sanitized_msg}",
                    error_type="api_error",
                    status_code=status_code or 500
                ) from api_err

            except OpenAIServiceError:
                raise

            except Exception as exc:
                sanitized_msg = self._sanitize_secret(str(exc))
                logger.error("Unexpected error during OpenAI completion: %s", sanitized_msg)
                raise OpenAIServiceError(
                    f"Unexpected AI service error: {sanitized_msg}",
                    error_type="general",
                    status_code=500
                ) from exc

        raise OpenAIServiceError(
            f"OpenAI completion failed after retries: {self._sanitize_secret(str(last_error))}",
            error_type="retry_exhausted",
            status_code=500
        )

    def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None
    ) -> str:
        """Direct text generation from prompt or message history."""
        if not prompt and not messages:
            raise OpenAIServiceError("Prompt or messages cannot be empty.")

        chat_messages: List[Dict[str, str]] = []
        if system_instruction:
            chat_messages.append({"role": "system", "content": system_instruction})

        if messages:
            chat_messages.extend(messages)
        elif prompt:
            chat_messages.append({"role": "user", "content": prompt})

        res = self._execute_chat_completion(
            messages=chat_messages,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        return clean_ai_markdown(res["text"])

    def generate_answer(
        self,
        question: str,
        context: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        conversation_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates context-grounded medical answer using OpenAI.
        Adheres to strict evidence grounding and citation preservation.
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

        messages = [
            {"role": "system", "content": HEALTHCARE_SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt}
        ]

        res = self._execute_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature
        )

        cleaned_text = clean_ai_markdown(res["text"])

        return {
            "answer": cleaned_text,
            "model": res["model"],
            "disclaimer": MEDICAL_DISCLAIMER,
            "generation_time_ms": res["elapsed_ms"],
            "api_request_time_ms": res["call_duration_ms"],
            "request_start_time": res["request_start_iso"],
            "gemini_calls_count": res["calls_count"],
            "openai_calls_count": res["calls_count"],
            "input_tokens": res.get("input_tokens"),
            "output_tokens": res.get("output_tokens"),
            "status": "success"
        }

    def generate_general_response(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        system_instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates an educational, technical, mathematical, or general response
        to ordinary questions using OpenAI without document-grounded citation constraints.
        """
        if not question or not question.strip():
            return {
                "answer": "Please ask a question.",
                "model": model or self.model,
                "generation_time_ms": 0.0,
                "status": "empty_query"
            }

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_instruction or GENERAL_SYSTEM_INSTRUCTIONS}
        ]

        if conversation_history:
            # Pass recent bounded history (up to last 6 turns)
            bounded_hist = conversation_history[-6:]
            for turn in bounded_hist:
                role = "assistant" if turn.get("role") in ("assistant", "model", "bot") else "user"
                content = str(turn.get("content") or turn.get("text") or "")
                if content.strip():
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": question.strip()})

        res = self._execute_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature if temperature is not None else 0.4
        )

        cleaned_text = clean_ai_markdown(res["text"])

        return {
            "answer": cleaned_text,
            "model": res["model"],
            "generation_time_ms": res["elapsed_ms"],
            "api_request_time_ms": res["call_duration_ms"],
            "request_start_time": res["request_start_iso"],
            "openai_calls_count": res["calls_count"],
            "input_tokens": res.get("input_tokens"),
            "output_tokens": res.get("output_tokens"),
            "status": "success"
        }

    def generate_stream(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Streams generated tokens chunk-by-chunk using OpenAI SDK."""
        if not prompt and not messages:
            raise OpenAIServiceError("Prompt or messages cannot be empty.")

        if self.is_quota_exhausted():
            raise OpenAIServiceError(
                "OpenAI API credit balance or quota exhausted. Please check your OpenAI billing settings at https://platform.openai.com/account/billing.",
                error_type="quota_exhausted",
                status_code=429
            )

        effective_model = model or self.model or "gpt-4o-mini"
        effective_temp = float(temperature if temperature is not None else self.temperature)
        effective_tokens = int(max_output_tokens if max_output_tokens is not None else self.max_output_tokens)

        client = self.get_client()

        chat_messages: List[Dict[str, str]] = []
        if system_instruction:
            chat_messages.append({"role": "system", "content": system_instruction})

        if messages:
            chat_messages.extend(messages)
        elif prompt:
            chat_messages.append({"role": "user", "content": prompt})

        try:
            stream = client.chat.completions.create(
                model=effective_model,
                messages=chat_messages,  # type: ignore
                temperature=effective_temp,
                max_tokens=effective_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except AuthenticationError as auth_err:
            raise OpenAIServiceError(
                "OpenAI authentication failed: The provided API key is invalid or lacks required permissions.",
                error_type="auth_error",
                status_code=401
            ) from auth_err
        except RateLimitError as rl_err:
            sanitized = self._sanitize_secret(str(rl_err)).lower()
            if "quota" in sanitized or "credit" in sanitized or "billing" in sanitized:
                self.mark_quota_exhausted(cooldown_seconds=60.0)
                raise OpenAIServiceError(
                    "OpenAI API credit balance or quota exhausted. Please check your OpenAI billing settings.",
                    error_type="quota_exhausted",
                    status_code=429
                ) from rl_err
            raise OpenAIServiceError(
                "OpenAI rate limit exceeded. Please wait a moment and try again.",
                error_type="rate_limit",
                status_code=429
            ) from rl_err
        except Exception as exc:
            sanitized_msg = self._sanitize_secret(str(exc))
            raise OpenAIServiceError(
                f"OpenAI streaming failure: {sanitized_msg}",
                error_type="stream_error",
                status_code=500
            ) from exc
