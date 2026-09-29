import os
import re
import uuid
import json
import logging
import datetime
import hashlib
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("rag.observability")


def generate_request_id() -> str:
    """Generates a unique traceable request identifier for a RAG invocation."""
    return f"rag-{uuid.uuid4().hex[:12]}"


def mask_clinical_query(query: str, safe_mode: bool = True) -> str:
    """
    Masks or redacts a user's clinical query to protect PHI/PII in logs.
    In development/test mode (safe_mode=False), retains the query.
    In safe/production mode (safe_mode=True), truncates and appends a safe hash.
    """
    if not safe_mode or not query:
        return query
    
    q_len = len(query)
    q_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
    preview = query[:12].strip()
    return f"{preview}... [MASKED_PHI len={q_len} hash={q_hash}]"


def sanitize_value(val: Any) -> Any:
    """
    Sanitizes an individual value, redacting credentials, tokens, and sensitive patterns.
    Safely handles strings, dictionaries, lists, tuples, sets, and Exception objects.
    """
    if isinstance(val, (Exception, BaseException)):
        # Stringify exception message and sanitize to prevent raw credential leakage
        return sanitize_value(f"{type(val).__name__}: {str(val)}")

    if isinstance(val, str):
        # Redact Bearer tokens
        val = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer [REDACTED]", val, flags=re.IGNORECASE)
        # Redact JWT tokens
        val = re.sub(r"eyJ[A-Za-z0-9_\-\.]{15,}", "[REDACTED_JWT]", val)
        # Redact API keys with standard prefixes
        val = re.sub(r"\bAIza[0-9A-Za-z-_]{30,45}\b", "[REDACTED_API_KEY]", val)
        val = re.sub(r"\bsk-[a-zA-Z0-9]{20,}\b", "[REDACTED_OPENAI_KEY]", val)
        # Redact Social Security Numbers
        val = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", val)
        # Redact Credit Card Numbers (16-digit formatted or unformatted)
        val = re.sub(r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[REDACTED_CARD]", val)
        # Redact credentials embedded in database URLs (e.g. postgresql://user:pass@host)
        val = re.sub(r"(:[a-zA-Z0-9_+\-]+@)", ":[REDACTED_PASSWORD]@", val)
        # Redact Email addresses
        val = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]", val)
        # Redact Phone numbers
        val = re.sub(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]", val)
        return val
    elif isinstance(val, dict):
        return sanitize_log_dict(val)
    elif isinstance(val, (list, tuple, set)):
        return [sanitize_value(item) for item in val]
    return val


def sanitize_log_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively scans and sanitizes dictionary fields, stripping or redacting passwords,
    API keys, authentication tokens, connection strings, and sensitive clinical identifiers.
    """
    sensitive_keys = {
        "password", "api_key", "secret", "token", "access_token",
        "authorization", "supabase_key", "gemini_api_key", "openai_api_key",
        "private_key", "client_secret", "db_password", "session_token",
        "refresh_token", "cookie", "database_url"
    }

    sanitized = {}
    for k, v in data.items():
        k_lower = str(k).lower()
        if any(sk in k_lower for sk in sensitive_keys):
            sanitized[k] = "[REDACTED_CREDENTIAL]"
        else:
            sanitized[k] = sanitize_value(v)
    return sanitized


@dataclass
class RAGStructuredLogEvent:
    """Structured, auditable log event for a complete RAG execution cycle."""
    request_id: str
    query: str
    safety_classification: str
    risk_level: str
    retrieval_status: str
    embedding_latency_ms: float
    vector_search_latency_ms: float
    context_construction_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    number_of_retrieved_chunks: int
    similarity_scores: List[float]
    citation_validation_passed: bool
    citation_coverage: float
    hallucination_detected: bool
    hallucination_types: List[str]
    final_status: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        raw_dict = asdict(self)
        # Check safe log mode
        safe_mode = os.getenv("SAFE_LOG_MODE", "").lower() in ("true", "1", "yes")
        if safe_mode:
            raw_dict["query"] = mask_clinical_query(self.query, safe_mode=True)
        return sanitize_log_dict(raw_dict)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class HTTPRequestLogEvent:
    """
    Enterprise-grade structured log event for an incoming HTTP API request.
    Captures request correlation, latency, status code, and clinical metadata.
    """
    request_id: str
    endpoint: str
    method: str
    status_code: int
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    retrieval_status: Optional[str] = None
    safety_status: Optional[str] = None
    error_category: Optional[str] = None
    client_ip: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        raw_dict = asdict(self)
        return sanitize_log_dict(raw_dict)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class StructuredRAGLogger:
    """
    Enterprise-grade structured logger for RAG diagnostics and audit compliance.
    """

    @classmethod
    def emit_rag_log(
        cls,
        event: RAGStructuredLogEvent,
        use_json: bool = True
    ) -> Dict[str, Any]:
        """
        Emits the structured log event to the standard logging stream.
        """
        event_dict = event.to_dict()
        if use_json:
            log_line = json.dumps(event_dict, ensure_ascii=False)
            logger.info("RAG_EVENT_JSON: %s", log_line)
        else:
            logger.info(
                "RAG_EVENT [Req: %s | Status: %s | Safety: %s | Latency: %.2f ms | Citations: %s]",
                event.request_id,
                event.final_status,
                event.safety_classification,
                event.total_latency_ms,
                "VALID" if event.citation_validation_passed else "INVALID"
            )
        return event_dict

    @classmethod
    def emit_http_log(
        cls,
        event: HTTPRequestLogEvent,
        use_json: bool = True
    ) -> Dict[str, Any]:
        """
        Emits the structured HTTP access event to the logging stream.
        """
        event_dict = event.to_dict()
        if use_json:
            log_line = json.dumps(event_dict, ensure_ascii=False)
            logger.info("HTTP_EVENT_JSON: %s", log_line)
        else:
            logger.info(
                "HTTP_EVENT [Req: %s | %s %s -> %d | Latency: %.2f ms | Err: %s]",
                event.request_id,
                event.method,
                event.endpoint,
                event.status_code,
                event.latency_ms,
                event.error_category or "NONE"
            )
        return event_dict
