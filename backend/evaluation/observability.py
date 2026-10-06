import os
import re
import uuid
import json
import math
import logging
import datetime
import hashlib
import threading
from collections import deque
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
    user_id: Optional[Union[str, int]] = None
    query_intent: Optional[str] = None
    llm_called: bool = False
    model_used: Optional[str] = None
    retry_count: int = 0
    fallback_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    # Phase 3.3 Performance, Caching & Streaming Observability
    cache_hit: bool = False
    cache_miss: bool = False
    cache_key_version: Optional[str] = None
    cache_age_ms: Optional[float] = None
    client_warm: bool = False
    streaming: bool = False
    time_to_first_event_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    stream_duration_ms: Optional[float] = None
    tokens_received: Optional[int] = None
    stream_completed: Optional[bool] = None
    stream_error: Optional[str] = None

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
        Emits the structured log event to the standard logging stream and records metrics.
        """
        try:
            get_metrics_collector().record_rag_event(event)
        except Exception:
            pass

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
        Emits the structured HTTP access event to the logging stream and records metrics.
        """
        try:
            get_metrics_collector().record_http_event(event)
        except Exception:
            pass

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


class ProductionMetricsCollector:
    """
    Thread-safe, bounded, production-grade metrics accumulator for RAG, LLM,
    cache, safety, and latency statistics (Phase 3.4).
    """

    def __init__(self, max_samples: int = 1000):
        self._lock = threading.Lock()
        self._max_samples = max_samples

        # Request counters
        self.requests_total: int = 0
        self.requests_successful: int = 0
        self.requests_failed: int = 0
        self.requests_blocked: int = 0

        # Cache counters
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.cache_errors: int = 0

        # LLM counters
        self.llm_calls: int = 0
        self.llm_successes: int = 0
        self.llm_failures: int = 0
        self.llm_retries: int = 0
        self.llm_fallbacks: int = 0

        # Safety counters
        self.safety_blocked: int = 0
        self.safety_passed: int = 0
        self.emergency_blocks: int = 0
        self.prompt_injection_blocks: int = 0
        self.validation_failures: int = 0

        # Retrieval counters
        self.retrieval_count: int = 0
        self.insufficient_context: int = 0

        # Intelligence / Intent counters (Phase 6.1)
        self.intent_classifications: Dict[str, int] = {}
        self.routing_counts: Dict[str, int] = {}
        self.intent_uncertain_total: int = 0

        # Query Planning counters (Phase 6.2)
        self.query_plan_strategies: Dict[str, int] = {}
        self.query_plan_intents: Dict[str, int] = {}
        self.query_plan_expansion_total: int = 0
        self.query_plan_high_evidence_total: int = 0

        # Evidence Fusion counters (Phase 6.3)
        self.evidence_fusion_total: int = 0
        self.evidence_fusion_strategies: Dict[str, int] = {}
        self.evidence_contributing_docs_total: int = 0
        self.evidence_conflicts_total: int = 0
        self.evidence_conflicts_by_type: Dict[str, int] = {}
        self.evidence_insufficient_total: int = 0
        self.evidence_deduped_total: int = 0

        # Answer Synthesis counters (Phase 6.4)
        self.answer_synthesis_total: int = 0
        self.answer_synthesis_intents: Dict[str, int] = {}
        self.answer_synthesis_confidences: Dict[str, int] = {}
        self.answer_synthesis_fallback_total: int = 0
        self.answer_synthesis_fallbacks_by_reason: Dict[str, int] = {}
        self.answer_synthesis_conflict_total: int = 0
        self.answer_synthesis_insufficient_total: int = 0

        # Citation & Attribution counters (Phase 6.5)
        self.citation_validations_total: int = 0
        self.claims_checked_total: int = 0
        self.claims_verified_total: int = 0
        self.claims_unsupported_total: int = 0
        self.citation_spoofing_detected_total: int = 0

        # Clinical Verification counters (Phase 6.6)
        self.clinical_verifications_total: int = 0
        self.verification_grounded_claims_total: int = 0
        self.verification_ungrounded_claims_total: int = 0
        self.verification_contradictions_total: int = 0
        self.verification_hallucinations_total: int = 0
        self.verification_fallbacks_total: int = 0

        # Clinical Decision Support counters (Phase 6.7)
        self.decision_support_total: int = 0
        self.decision_support_uncertainty_levels: Dict[str, int] = {}
        self.decision_support_risk_tiers: Dict[str, int] = {}
        self.decision_support_escalations_total: int = 0
        self.decision_support_red_flags_total: int = 0
        self.decision_support_recommendations_total: int = 0

        # Clinical Orchestration & Audit counters (Phase 6.8)
        self.orchestration_total: int = 0
        self.orchestration_concordant_total: int = 0
        self.orchestration_intercepted_total: int = 0

        # Longitudinal Context & Dialogue State counters (Phase 6.9)
        self.context_resolution_total: int = 0
        self.context_follow_up_total: int = 0
        self.context_entities_tracked_total: int = 0
        self.context_contraindications_flagged_total: int = 0

        # Latency samples (bounded deques in milliseconds)
        self.latencies_retrieval: deque = deque(maxlen=max_samples)
        self.latencies_cache: deque = deque(maxlen=max_samples)
        self.latencies_llm: deque = deque(maxlen=max_samples)
        self.latencies_total: deque = deque(maxlen=max_samples)
        self.latencies_ttft: deque = deque(maxlen=max_samples)
        self.latencies_ttfe: deque = deque(maxlen=max_samples)
        self.latencies_intent: deque = deque(maxlen=max_samples)
        self.latencies_query_plan: deque = deque(maxlen=max_samples)
        self.latencies_evidence_fusion: deque = deque(maxlen=max_samples)
        self.latencies_answer_synthesis: deque = deque(maxlen=max_samples)
        self.latencies_citation_attribution: deque = deque(maxlen=max_samples)
        self.latencies_clinical_verification: deque = deque(maxlen=max_samples)
        self.latencies_decision_support: deque = deque(maxlen=max_samples)
        self.latencies_orchestration: deque = deque(maxlen=max_samples)
        self.latencies_context_resolution: deque = deque(maxlen=max_samples)

    def record_intent_event(
        self,
        intent: str,
        routing_strategy: str,
        latency_ms: float = 0.0,
        is_uncertain: bool = False
    ) -> None:
        """Records clinical intent classification metrics with low cardinality."""
        with self._lock:
            safe_intent = str(intent).strip().upper() if intent else "UNKNOWN"
            self.intent_classifications[safe_intent] = self.intent_classifications.get(safe_intent, 0) + 1

            safe_strat = str(routing_strategy).strip().lower() if routing_strategy else "standard_rag"
            self.routing_counts[safe_strat] = self.routing_counts.get(safe_strat, 0) + 1

            if is_uncertain or safe_intent in ("UNCERTAIN", "UNKNOWN"):
                self.intent_uncertain_total += 1

            if latency_ms > 0:
                self.latencies_intent.append(latency_ms)

    def record_query_plan_event(
        self,
        strategy: str,
        intent: str,
        latency_ms: float = 0.0,
        has_expansion: bool = False,
        high_evidence: bool = False
    ) -> None:
        """Records clinical query planning metrics with strictly low cardinality."""
        with self._lock:
            safe_strat = str(strategy).strip().lower() if strategy else "standard"
            self.query_plan_strategies[safe_strat] = self.query_plan_strategies.get(safe_strat, 0) + 1

            safe_intent = str(intent).strip().upper() if intent else "UNKNOWN"
            self.query_plan_intents[safe_intent] = self.query_plan_intents.get(safe_intent, 0) + 1

            if has_expansion:
                self.query_plan_expansion_total += 1
            if high_evidence:
                self.query_plan_high_evidence_total += 1
            if latency_ms > 0:
                self.latencies_query_plan.append(latency_ms)

    def record_query_plan(
        self,
        strategy: str = "standard",
        intent: str = "UNKNOWN",
        latency_ms: float = 0.0,
        has_expansion: bool = False,
        high_evidence: bool = False,
        **kwargs
    ) -> None:
        """Alias for record_query_plan_event supporting flexible parameters."""
        if "expansions_count" in kwargs:
            has_expansion = bool(kwargs["expansions_count"] > 0)
        if "requires_high_evidence" in kwargs:
            high_evidence = bool(kwargs["requires_high_evidence"])
        self.record_query_plan_event(
            strategy=strategy,
            intent=intent,
            latency_ms=latency_ms,
            has_expansion=has_expansion,
            high_evidence=high_evidence
        )

    def record_evidence_fusion_event(
        self,
        strategy: str = "standard",
        contributing_docs_count: int = 1,
        conflicts_count: int = 0,
        conflict_types: Optional[List[str]] = None,
        deduped_count: int = 0,
        is_sufficient: bool = True,
        latency_ms: float = 0.0
    ) -> None:
        """Records clinical evidence fusion metrics with strictly low cardinality."""
        with self._lock:
            self.evidence_fusion_total += 1
            safe_strat = str(strategy).strip().lower() if strategy else "standard"
            self.evidence_fusion_strategies[safe_strat] = self.evidence_fusion_strategies.get(safe_strat, 0) + 1
            self.evidence_contributing_docs_total += max(0, contributing_docs_count)
            self.evidence_deduped_total += max(0, deduped_count)

            if not is_sufficient:
                self.evidence_insufficient_total += 1

            if conflicts_count > 0:
                self.evidence_conflicts_total += conflicts_count
                if conflict_types:
                    for ct in conflict_types:
                        safe_ct = str(ct).strip().lower()
                        self.evidence_conflicts_by_type[safe_ct] = self.evidence_conflicts_by_type.get(safe_ct, 0) + 1

            if latency_ms > 0:
                self.latencies_evidence_fusion.append(latency_ms)

    def record_evidence_fusion(
        self,
        strategy: str = "standard",
        contributing_docs_count: int = 1,
        conflicts_count: int = 0,
        conflict_types: Optional[List[str]] = None,
        deduped_count: int = 0,
        is_sufficient: bool = True,
        latency_ms: float = 0.0,
        **kwargs
    ) -> None:
        """Alias for record_evidence_fusion_event supporting flexible parameters."""
        self.record_evidence_fusion_event(
            strategy=strategy,
            contributing_docs_count=contributing_docs_count,
            conflicts_count=conflicts_count,
            conflict_types=conflict_types,
            deduped_count=deduped_count,
            is_sufficient=is_sufficient,
            latency_ms=latency_ms
        )

    def record_answer_synthesis_event(
        self,
        intent: str = "GENERAL_HEALTH",
        confidence: str = "HIGH",
        is_fallback: bool = False,
        fallback_reason: Optional[str] = None,
        conflicts_present: bool = False,
        is_insufficient: bool = False,
        latency_ms: float = 0.0
    ) -> None:
        """Records clinical answer synthesis metrics with strictly low cardinality."""
        with self._lock:
            self.answer_synthesis_total += 1
            safe_intent = str(intent).strip().upper() if intent else "GENERAL_HEALTH"
            self.answer_synthesis_intents[safe_intent] = self.answer_synthesis_intents.get(safe_intent, 0) + 1

            safe_conf = str(confidence).strip().upper() if confidence else "HIGH"
            self.answer_synthesis_confidences[safe_conf] = self.answer_synthesis_confidences.get(safe_conf, 0) + 1

            if is_fallback:
                self.answer_synthesis_fallback_total += 1
                if fallback_reason:
                    safe_reason = str(fallback_reason).strip().lower()
                    self.answer_synthesis_fallbacks_by_reason[safe_reason] = self.answer_synthesis_fallbacks_by_reason.get(safe_reason, 0) + 1

            if conflicts_present:
                self.answer_synthesis_conflict_total += 1

            if is_insufficient:
                self.answer_synthesis_insufficient_total += 1

            if latency_ms > 0:
                self.latencies_answer_synthesis.append(latency_ms)

    def record_answer_synthesis(
        self,
        intent: str = "GENERAL_HEALTH",
        confidence: str = "HIGH",
        is_fallback: bool = False,
        fallback_reason: Optional[str] = None,
        conflicts_present: bool = False,
        is_insufficient: bool = False,
        latency_ms: float = 0.0,
        **kwargs
    ) -> None:
        """Alias for record_answer_synthesis_event supporting flexible parameters."""
        self.record_answer_synthesis_event(
            intent=intent,
            confidence=confidence,
            is_fallback=is_fallback,
            fallback_reason=fallback_reason,
            conflicts_present=conflicts_present,
            is_insufficient=is_insufficient,
            latency_ms=latency_ms
        )

    def record_citation_attribution_event(
        self,
        claims_checked: int = 0,
        claims_verified: int = 0,
        claims_unsupported: int = 0,
        spoofing_detected: bool = False,
        latency_ms: float = 0.0
    ) -> None:
        """Records clinical citation attribution metrics with zero PHI and low cardinality."""
        with self._lock:
            self.citation_validations_total += 1
            self.claims_checked_total += max(0, int(claims_checked))
            self.claims_verified_total += max(0, int(claims_verified))
            self.claims_unsupported_total += max(0, int(claims_unsupported))
            if spoofing_detected:
                self.citation_spoofing_detected_total += 1
            if latency_ms > 0:
                self.latencies_citation_attribution.append(latency_ms)

    def record_citation_attribution(
        self,
        claims_checked: int = 0,
        claims_verified: int = 0,
        claims_unsupported: int = 0,
        spoofing_detected: bool = False,
        latency_ms: float = 0.0,
        **kwargs
    ) -> None:
        """Alias for record_citation_attribution_event."""
        self.record_citation_attribution_event(
            claims_checked=claims_checked,
            claims_verified=claims_verified,
            claims_unsupported=claims_unsupported,
            spoofing_detected=spoofing_detected,
            latency_ms=latency_ms
        )

    def record_clinical_verification_event(
        self,
        grounded_claims: int = 0,
        ungrounded_claims: int = 0,
        contradictions: int = 0,
        hallucinations: int = 0,
        fallback_triggered: bool = False,
        latency_ms: float = 0.0
    ) -> None:
        """Records clinical verification metrics with zero PHI and low cardinality."""
        with self._lock:
            self.clinical_verifications_total += 1
            self.verification_grounded_claims_total += max(0, int(grounded_claims))
            self.verification_ungrounded_claims_total += max(0, int(ungrounded_claims))
            self.verification_contradictions_total += max(0, int(contradictions))
            self.verification_hallucinations_total += max(0, int(hallucinations))
            if fallback_triggered:
                self.verification_fallbacks_total += 1
            if latency_ms > 0:
                self.latencies_clinical_verification.append(latency_ms)

    def record_clinical_verification(
        self,
        grounded_claims: int = 0,
        ungrounded_claims: int = 0,
        contradictions: int = 0,
        hallucinations: int = 0,
        fallback_triggered: bool = False,
        latency_ms: float = 0.0,
        **kwargs
    ) -> None:
        """Alias for record_clinical_verification_event."""
        self.record_clinical_verification_event(
            grounded_claims=grounded_claims,
            ungrounded_claims=ungrounded_claims,
            contradictions=contradictions,
            hallucinations=hallucinations,
            fallback_triggered=fallback_triggered,
            latency_ms=latency_ms
        )

    def record_decision_support_event(
        self,
        uncertainty_level: str = "LOW",
        risk_tier: str = "MINIMAL",
        escalation_required: bool = False,
        red_flags_count: int = 0,
        recommendations_count: int = 0,
        latency_ms: float = 0.0
    ) -> None:
        """Records clinical decision support metrics with zero PHI and low cardinality."""
        with self._lock:
            self.decision_support_total += 1
            safe_lvl = str(uncertainty_level).strip().upper() if uncertainty_level else "UNKNOWN"
            self.decision_support_uncertainty_levels[safe_lvl] = self.decision_support_uncertainty_levels.get(safe_lvl, 0) + 1

            safe_tier = str(risk_tier).strip().upper() if risk_tier else "UNKNOWN"
            self.decision_support_risk_tiers[safe_tier] = self.decision_support_risk_tiers.get(safe_tier, 0) + 1

            if escalation_required:
                self.decision_support_escalations_total += 1
            self.decision_support_red_flags_total += max(0, int(red_flags_count))
            self.decision_support_recommendations_total += max(0, int(recommendations_count))

            if latency_ms > 0:
                self.latencies_decision_support.append(latency_ms)

    def record_decision_support(
        self,
        uncertainty_level: str = "LOW",
        risk_tier: str = "MINIMAL",
        escalation_required: bool = False,
        red_flags_count: int = 0,
        recommendations_count: int = 0,
        latency_ms: float = 0.0,
        **kwargs
    ) -> None:
        """Alias for record_decision_support_event."""
        self.record_decision_support_event(
            uncertainty_level=uncertainty_level,
            risk_tier=risk_tier,
            escalation_required=escalation_required,
            red_flags_count=red_flags_count,
            recommendations_count=recommendations_count,
            latency_ms=latency_ms
        )

    def record_orchestration_event(
        self,
        is_concordant: bool = True,
        is_intercepted: bool = False,
        latency_ms: float = 0.0
    ) -> None:
        """Records clinical intelligence orchestration & audit metrics with zero PHI (Phase 6.8)."""
        with self._lock:
            self.orchestration_total += 1
            if is_concordant:
                self.orchestration_concordant_total += 1
            if is_intercepted:
                self.orchestration_intercepted_total += 1
            if latency_ms > 0:
                self.latencies_orchestration.append(latency_ms)

    def record_context_event(
        self,
        is_follow_up: bool = False,
        prior_turns_used: int = 0,
        entities_count: int = 0,
        contraindications_count: int = 0,
        latency_ms: float = 0.0
    ) -> None:
        """Records longitudinal clinical context telemetry with zero PHI (Phase 6.9)."""
        with self._lock:
            self.context_resolution_total += 1
            if is_follow_up:
                self.context_follow_up_total += 1
            self.context_entities_tracked_total += max(0, int(entities_count))
            self.context_contraindications_flagged_total += max(0, int(contraindications_count))
            if latency_ms > 0:
                self.latencies_context_resolution.append(latency_ms)

    def record_rag_event(self, event: RAGStructuredLogEvent) -> None:
        with self._lock:
            self.requests_total += 1
            final_st = str(event.final_status).lower() if event.final_status else ""
            if final_st in ("success", "grounded_boundary"):
                self.requests_successful += 1
            elif final_st in ("safety_intercepted",):
                self.requests_blocked += 1
            else:
                self.requests_failed += 1

            if event.cache_hit:
                self.cache_hits += 1
            elif event.cache_miss:
                self.cache_misses += 1

            if event.llm_called:
                self.llm_calls += 1
                if event.fallback_reason:
                    self.llm_fallbacks += 1
                    self.llm_failures += 1
                else:
                    self.llm_successes += 1
                self.llm_retries += max(0, event.retry_count)

            # Safety breakdown
            is_emergency = event.safety_classification == "EMERGENCY_SYMPTOMS"
            is_injection = (
                event.safety_classification in ("PROMPT_INJECTION", "UNSAFE_OR_UNSUPPORTED_REQUEST")
                or (event.query_intent and "injection" in str(event.query_intent).lower())
            )

            if is_emergency:
                self.emergency_blocks += 1
            if is_injection:
                self.prompt_injection_blocks += 1

            if event.safety_classification in (
                "EMERGENCY_SYMPTOMS",
                "SELF_HARM_OR_SUICIDE",
                "POISONING_OR_OVERDOSE",
                "UNSAFE_OR_UNSUPPORTED_REQUEST"
            ) or event.final_status in ("SAFETY_INTERCEPTED", "safety_intercepted"):
                self.safety_blocked += 1
            else:
                self.safety_passed += 1

            # Retrieval breakdown
            if event.number_of_retrieved_chunks > 0 or event.retrieval_status != "safety_intercepted":
                self.retrieval_count += 1
            if event.retrieval_status in ("no_relevant_context", "insufficient_context"):
                self.insufficient_context += 1

            # Validation failures (citations or hallucinations)
            if not event.citation_validation_passed or event.hallucination_detected:
                self.validation_failures += 1

            if event.vector_search_latency_ms > 0:
                self.latencies_retrieval.append(event.vector_search_latency_ms)
            if event.llm_latency_ms > 0:
                self.latencies_llm.append(event.llm_latency_ms)
            if event.total_latency_ms > 0:
                self.latencies_total.append(event.total_latency_ms)
            if event.time_to_first_token_ms and event.time_to_first_token_ms > 0:
                self.latencies_ttft.append(event.time_to_first_token_ms)
            if event.time_to_first_event_ms and event.time_to_first_event_ms > 0:
                self.latencies_ttfe.append(event.time_to_first_event_ms)
            if event.cache_age_ms and event.cache_age_ms > 0:
                self.latencies_cache.append(event.cache_age_ms)

    def record_http_event(self, event: HTTPRequestLogEvent) -> None:
        with self._lock:
            if event.endpoint not in ("/health", "/health/live", "/health/ready", "/metrics", "/metrics/prometheus"):
                if event.status_code >= 400:
                    if event.status_code in (403, 429):
                        self.requests_blocked += 1
                    else:
                        self.requests_failed += 1
                if event.latency_ms > 0:
                    self.latencies_total.append(event.latency_ms)

    @staticmethod
    def _percentile(data: List[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return round(sorted_data[int(k)], 2)
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        return round(d0 + d1, 2)

    def _calc_stats(self, deq: deque) -> Dict[str, float]:
        data = list(deq)
        if not data:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0}
        return {
            "p50": self._percentile(data, 50),
            "p95": self._percentile(data, 95),
            "p99": self._percentile(data, 99),
            "avg": round(sum(data) / len(data), 2)
        }

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            hits = self.cache_hits
            misses = self.cache_misses
            tot_cache = hits + misses
            hit_rate = round(hits / tot_cache, 4) if tot_cache > 0 else 0.0

            return {
                "requests": {
                    "total": self.requests_total,
                    "successful": self.requests_successful,
                    "failed": self.requests_failed,
                    "blocked": self.requests_blocked
                },
                "cache": {
                    "hits": hits,
                    "misses": misses,
                    "hit_rate": hit_rate
                },
                "llm": {
                    "calls": self.llm_calls,
                    "successes": self.llm_successes,
                    "failures": self.llm_failures,
                    "retries": self.llm_retries,
                    "fallback_usage": self.llm_fallbacks
                },
                "safety": {
                    "blocked": self.safety_blocked,
                    "passed": self.safety_passed
                },
                "intelligence": {
                    "intent_classifications": dict(self.intent_classifications),
                    "routing_counts": dict(self.routing_counts),
                    "uncertain_total": self.intent_uncertain_total,
                    "latency": self._calc_stats(self.latencies_intent),
                    "query_plans": {
                        "strategies": dict(self.query_plan_strategies),
                        "intents": dict(self.query_plan_intents),
                        "expansion_total": self.query_plan_expansion_total,
                        "high_evidence_total": self.query_plan_high_evidence_total,
                        "latency": self._calc_stats(self.latencies_query_plan)
                    },
                    "evidence_fusion": {
                        "total": self.evidence_fusion_total,
                        "strategies": dict(self.evidence_fusion_strategies),
                        "contributing_docs_total": self.evidence_contributing_docs_total,
                        "conflicts_total": self.evidence_conflicts_total,
                        "conflicts_by_type": dict(self.evidence_conflicts_by_type),
                        "insufficient_total": self.evidence_insufficient_total,
                        "deduped_total": self.evidence_deduped_total,
                        "latency": self._calc_stats(self.latencies_evidence_fusion)
                    },
                    "answer_synthesis": {
                        "total": self.answer_synthesis_total,
                        "intents": dict(self.answer_synthesis_intents),
                        "confidences": dict(self.answer_synthesis_confidences),
                        "fallback_total": self.answer_synthesis_fallback_total,
                        "fallbacks_by_reason": dict(self.answer_synthesis_fallbacks_by_reason),
                        "conflict_total": self.answer_synthesis_conflict_total,
                        "insufficient_total": self.answer_synthesis_insufficient_total,
                        "latency": self._calc_stats(self.latencies_answer_synthesis)
                    },
                    "clinical_verification": {
                        "total": self.clinical_verifications_total,
                        "grounded_claims": self.verification_grounded_claims_total,
                        "ungrounded_claims": self.verification_ungrounded_claims_total,
                        "contradictions": self.verification_contradictions_total,
                        "hallucinations": self.verification_hallucinations_total,
                        "fallbacks": self.verification_fallbacks_total,
                        "latency": self._calc_stats(self.latencies_clinical_verification)
                    },
                    "clinical_decision_support": {
                        "total": self.decision_support_total,
                        "uncertainty_levels": dict(self.decision_support_uncertainty_levels),
                        "risk_tiers": dict(self.decision_support_risk_tiers),
                        "escalations_total": self.decision_support_escalations_total,
                        "red_flags_total": self.decision_support_red_flags_total,
                        "recommendations_total": self.decision_support_recommendations_total,
                        "latency": self._calc_stats(self.latencies_decision_support)
                    },
                    "clinical_orchestration": {
                        "total": self.orchestration_total,
                        "concordant_total": self.orchestration_concordant_total,
                        "intercepted_total": self.orchestration_intercepted_total,
                        "latency": self._calc_stats(self.latencies_orchestration)
                    },
                    "longitudinal_context": {
                        "total": self.context_resolution_total,
                        "follow_up_total": self.context_follow_up_total,
                        "entities_tracked_total": self.context_entities_tracked_total,
                        "contraindications_flagged_total": self.context_contraindications_flagged_total,
                        "latency": self._calc_stats(self.latencies_context_resolution)
                    }
                },
                "latency": {
                    "retrieval": self._calc_stats(self.latencies_retrieval),
                    "cache": self._calc_stats(self.latencies_cache),
                    "llm": self._calc_stats(self.latencies_llm),
                    "total": self._calc_stats(self.latencies_total),
                    "ttft": self._calc_stats(self.latencies_ttft),
                    "ttfe": self._calc_stats(self.latencies_ttfe)
                }
            }

    get_summary = get_metrics_snapshot

    def get_prometheus_exposition(self) -> str:
        """
        Builds RFC-compliant Prometheus text exposition (version 0.0.4).
        Strictly adheres to low-cardinality rules:
        - NEVER includes user_id, request_id, queries, or document_id as labels.
        - NEVER exposes secrets or clinical details.
        """
        with self._lock:
            hits = self.cache_hits
            misses = self.cache_misses
            tot_cache = hits + misses
            hit_rate = round(hits / tot_cache, 4) if tot_cache > 0 else 0.0

            # Concurrency saturation
            active_concurrency = 0
            try:
                from backend.security.concurrency import concurrency_controller
                active_concurrency = concurrency_controller.active_count
            except Exception:
                pass

            # Cache error count from cache service
            cache_errs = self.cache_errors
            try:
                from backend.services.llm_cache_service import get_llm_cache_service
                cache_errs = get_llm_cache_service().stats().get("redis_errors", self.cache_errors)
            except Exception:
                pass

            lines: List[str] = [
                "# HELP rag_requests_total Total number of RAG HTTP and service requests processed",
                "# TYPE rag_requests_total counter",
                f'rag_requests_total{{status="successful"}} {self.requests_successful}',
                f'rag_requests_total{{status="failed"}} {self.requests_failed}',
                f'rag_requests_total{{status="blocked"}} {self.requests_blocked}',
                f'rag_requests_total{{status="all"}} {self.requests_total}',
                "",
                "# HELP rag_cache_hits_total Total number of LLM response cache hits",
                "# TYPE rag_cache_hits_total counter",
                f"rag_cache_hits_total {hits}",
                "",
                "# HELP rag_cache_misses_total Total number of LLM response cache misses",
                "# TYPE rag_cache_misses_total counter",
                f"rag_cache_misses_total {misses}",
                "",
                "# HELP rag_cache_errors_total Total number of cache communication or serialization errors",
                "# TYPE rag_cache_errors_total counter",
                f"rag_cache_errors_total {cache_errs}",
                "",
                "# HELP rag_cache_hit_ratio Current ratio of cache hits to total cache lookups",
                "# TYPE rag_cache_hit_ratio gauge",
                f"rag_cache_hit_ratio {hit_rate:.4f}",
                "",
                "# HELP rag_llm_calls_total Total number of LLM answer generation calls",
                "# TYPE rag_llm_calls_total counter",
                f"rag_llm_calls_total {self.llm_calls}",
                "",
                "# HELP rag_llm_successes_total Total number of successful LLM answer generations",
                "# TYPE rag_llm_successes_total counter",
                f"rag_llm_successes_total {self.llm_successes}",
                "",
                "# HELP rag_llm_failures_total Total number of failed LLM generations",
                "# TYPE rag_llm_failures_total counter",
                f"rag_llm_failures_total {self.llm_failures}",
                "",
                "# HELP rag_llm_retries_total Total number of LLM generation retries",
                "# TYPE rag_llm_retries_total counter",
                f"rag_llm_retries_total {self.llm_retries}",
                "",
                "# HELP rag_llm_active_concurrency Number of currently executing concurrent LLM calls",
                "# TYPE rag_llm_active_concurrency gauge",
                f"rag_llm_active_concurrency {active_concurrency}",
                "",
                "# HELP rag_safety_blocked_total Total number of inquiries blocked by medical safety rules",
                "# TYPE rag_safety_blocked_total counter",
                f"rag_safety_blocked_total {self.safety_blocked}",
                "",
                "# HELP rag_safety_emergency_blocks_total Total acute emergency inquiries intercepted",
                "# TYPE rag_safety_emergency_blocks_total counter",
                f"rag_safety_emergency_blocks_total {self.emergency_blocks}",
                "",
                "# HELP rag_safety_prompt_injection_blocks_total Total adversarial or injection attempts blocked",
                "# TYPE rag_safety_prompt_injection_blocks_total counter",
                f"rag_safety_prompt_injection_blocks_total {self.prompt_injection_blocks}",
                "",
                "# HELP rag_validation_failures_total Total citation or grounding verification failures",
                "# TYPE rag_validation_failures_total counter",
                f"rag_validation_failures_total {self.validation_failures}",
                "",
                "# HELP rag_retrieval_total Total number of FAISS vector store retrievals",
                "# TYPE rag_retrieval_total counter",
                f"rag_retrieval_total {self.retrieval_count}",
                "",
                "# HELP rag_insufficient_context_total Total queries yielding no relevant clinical evidence",
                "# TYPE rag_insufficient_context_total counter",
                f"rag_insufficient_context_total {self.insufficient_context}",
                "",
            ]

            # Latency Histograms
            def build_histogram(name: str, help_text: str, samples_ms: deque) -> List[str]:
                h_lines = [
                    f"# HELP {name} {help_text}",
                    f"# TYPE {name} histogram"
                ]
                samples_s = [s / 1000.0 for s in samples_ms]
                buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
                tot = len(samples_s)
                s_sum = round(sum(samples_s), 4)

                for b in buckets:
                    c = sum(1 for v in samples_s if v <= b)
                    h_lines.append(f'{name}_bucket{{le="{b}"}} {c}')
                h_lines.append(f'{name}_bucket{{le="+Inf"}} {tot}')
                h_lines.append(f'{name}_sum {s_sum}')
                h_lines.append(f'{name}_count {tot}')
                h_lines.append("")
                return h_lines

            # Clinical Intent Metrics (Phase 6.1)
            lines.append("# HELP rag_intent_classifications_total Total count of clinical intent classifications by intent")
            lines.append("# TYPE rag_intent_classifications_total counter")
            if self.intent_classifications:
                for intent_name, count in sorted(self.intent_classifications.items()):
                    lines.append(f'rag_intent_classifications_total{{intent="{intent_name}"}} {count}')
            else:
                lines.append('rag_intent_classifications_total{intent="NONE"} 0')

            lines.append("")
            lines.append("# HELP rag_intent_routing_total Total count of clinical query routing by strategy")
            lines.append("# TYPE rag_intent_routing_total counter")
            if self.routing_counts:
                for strat_name, count in sorted(self.routing_counts.items()):
                    lines.append(f'rag_intent_routing_total{{strategy="{strat_name}"}} {count}')
            else:
                lines.append('rag_intent_routing_total{strategy="none"} 0')

            lines.append("")
            lines.append("# HELP rag_intent_uncertain_total Total count of uncertain clinical intent classifications")
            lines.append("# TYPE rag_intent_uncertain_total counter")
            lines.append(f"rag_intent_uncertain_total {self.intent_uncertain_total}")
            lines.append("")

            # Clinical Query Planning Metrics (Phase 6.2)
            lines.append("# HELP rag_query_plan_total Total number of clinical query plans by strategy")
            lines.append("# TYPE rag_query_plan_total counter")
            if self.query_plan_strategies:
                for strat_name, count in sorted(self.query_plan_strategies.items()):
                    lines.append(f'rag_query_plan_total{{strategy="{strat_name}"}} {count}')
            else:
                lines.append('rag_query_plan_total{strategy="none"} 0')
            lines.append("")

            lines.append("# HELP rag_query_plan_intent_total Total number of clinical query plans by intent")
            lines.append("# TYPE rag_query_plan_intent_total counter")
            if self.query_plan_intents:
                for intent_name, count in sorted(self.query_plan_intents.items()):
                    lines.append(f'rag_query_plan_intent_total{{intent="{intent_name}"}} {count}')
            else:
                lines.append('rag_query_plan_intent_total{intent="NONE"} 0')
            lines.append("")

            lines.append("# HELP rag_query_plan_expansion_total Total query plans with query expansion enabled")
            lines.append("# TYPE rag_query_plan_expansion_total counter")
            lines.append(f"rag_query_plan_expansion_total {self.query_plan_expansion_total}")
            lines.append("")

            lines.append("# HELP rag_query_plan_high_evidence_total Total query plans requiring high confidence evidence")
            lines.append("# TYPE rag_query_plan_high_evidence_total counter")
            lines.append(f"rag_query_plan_high_evidence_total {self.query_plan_high_evidence_total}")
            lines.append("")

            # Evidence Fusion Metrics (Phase 6.3)
            lines.append("# HELP rag_evidence_fusion_total Total number of clinical evidence fusion executions")
            lines.append("# TYPE rag_evidence_fusion_total counter")
            lines.append(f"rag_evidence_fusion_total {self.evidence_fusion_total}")
            lines.append("")

            lines.append("# HELP rag_evidence_fusion_strategy_total Total clinical evidence fusions by strategy")
            lines.append("# TYPE rag_evidence_fusion_strategy_total counter")
            if self.evidence_fusion_strategies:
                for strat_name, count in sorted(self.evidence_fusion_strategies.items()):
                    lines.append(f'rag_evidence_fusion_strategy_total{{strategy="{strat_name}"}} {count}')
            else:
                lines.append('rag_evidence_fusion_strategy_total{strategy="none"} 0')
            lines.append("")

            lines.append("# HELP rag_evidence_contributing_docs_total Total number of documents contributing to evidence fusion")
            lines.append("# TYPE rag_evidence_contributing_docs_total counter")
            lines.append(f"rag_evidence_contributing_docs_total {self.evidence_contributing_docs_total}")
            lines.append("")

            lines.append("# HELP rag_evidence_conflicts_total Total clinical evidence conflicts detected across documents")
            lines.append("# TYPE rag_evidence_conflicts_total counter")
            lines.append(f"rag_evidence_conflicts_total {self.evidence_conflicts_total}")
            lines.append("")

            lines.append("# HELP rag_evidence_conflicts_by_type_total Clinical conflicts detected by conflict category")
            lines.append("# TYPE rag_evidence_conflicts_by_type_total counter")
            if self.evidence_conflicts_by_type:
                for ct_name, count in sorted(self.evidence_conflicts_by_type.items()):
                    lines.append(f'rag_evidence_conflicts_by_type_total{{conflict_type="{ct_name}"}} {count}')
            else:
                lines.append('rag_evidence_conflicts_by_type_total{conflict_type="none"} 0')
            lines.append("")

            lines.append("# HELP rag_evidence_insufficient_total Total queries yielding insufficient evidence coverage")
            lines.append("# TYPE rag_evidence_insufficient_total counter")
            lines.append(f"rag_evidence_insufficient_total {self.evidence_insufficient_total}")
            lines.append("")

            lines.append("# HELP rag_evidence_deduped_total Total redundant evidence chunks pruned during fusion")
            lines.append("# TYPE rag_evidence_deduped_total counter")
            lines.append(f"rag_evidence_deduped_total {self.evidence_deduped_total}")
            lines.append("")

            lines.extend(build_histogram("rag_request_duration_seconds", "End-to-end request duration in seconds", self.latencies_total))
            lines.extend(build_histogram("rag_retrieval_latency_seconds", "FAISS similarity retrieval latency in seconds", self.latencies_retrieval))
            lines.extend(build_histogram("rag_cache_latency_seconds", "Cache access latency in seconds", self.latencies_cache))
            lines.extend(build_histogram("rag_llm_latency_seconds", "LLM generation latency in seconds", self.latencies_llm))
            lines.extend(build_histogram("rag_ttft_seconds", "Time to first token in streaming generation in seconds", self.latencies_ttft))
            lines.extend(build_histogram("rag_ttfe_seconds", "Time to first event in streaming generation in seconds", self.latencies_ttfe))
            lines.extend(build_histogram("rag_intent_latency_seconds", "Clinical intent classification latency in seconds", self.latencies_intent))
            lines.extend(build_histogram("rag_query_plan_duration_seconds", "Clinical query planning latency in seconds", self.latencies_query_plan))
            lines.extend(build_histogram("rag_evidence_fusion_duration_seconds", "Clinical evidence fusion latency in seconds", self.latencies_evidence_fusion))

            # Answer Synthesis Metrics (Phase 6.4)
            lines.append("")
            lines.append("# HELP rag_answer_synthesis_total Total number of clinical answer synthesis executions")
            lines.append("# TYPE rag_answer_synthesis_total counter")
            lines.append(f"rag_answer_synthesis_total {self.answer_synthesis_total}")
            lines.append("")

            lines.append("# HELP rag_answer_synthesis_intent_total Total clinical answer syntheses by intent")
            lines.append("# TYPE rag_answer_synthesis_intent_total counter")
            if self.answer_synthesis_intents:
                for int_name, count in sorted(self.answer_synthesis_intents.items()):
                    lines.append(f'rag_answer_synthesis_intent_total{{intent="{int_name}"}} {count}')
            else:
                lines.append('rag_answer_synthesis_intent_total{intent="none"} 0')
            lines.append("")

            lines.append("# HELP rag_answer_synthesis_confidence_total Total clinical answer syntheses by confidence")
            lines.append("# TYPE rag_answer_synthesis_confidence_total counter")
            if self.answer_synthesis_confidences:
                for conf_name, count in sorted(self.answer_synthesis_confidences.items()):
                    lines.append(f'rag_answer_synthesis_confidence_total{{confidence="{conf_name}"}} {count}')
            else:
                lines.append('rag_answer_synthesis_confidence_total{confidence="none"} 0')
            lines.append("")

            lines.append("# HELP rag_answer_synthesis_fallback_total Total clinical answer syntheses requiring conservative fallback")
            lines.append("# TYPE rag_answer_synthesis_fallback_total counter")
            lines.append(f"rag_answer_synthesis_fallback_total {self.answer_synthesis_fallback_total}")
            lines.append("")

            lines.append("# HELP rag_answer_synthesis_conflict_total Total answer syntheses where evidence conflicts were present")
            lines.append("# TYPE rag_answer_synthesis_conflict_total counter")
            lines.append(f"rag_answer_synthesis_conflict_total {self.answer_synthesis_conflict_total}")
            lines.append("")

            lines.append("# HELP rag_answer_synthesis_insufficient_total Total answer syntheses resulting from insufficient evidence")
            lines.append("# TYPE rag_answer_synthesis_insufficient_total counter")
            lines.append(f"rag_answer_synthesis_insufficient_total {self.answer_synthesis_insufficient_total}")
            lines.append("")

            lines.extend(build_histogram("rag_answer_synthesis_duration_seconds", "Clinical answer synthesis latency in seconds", self.latencies_answer_synthesis))

            # Citation & Attribution Metrics (Phase 6.5)
            lines.append("")
            lines.append("# HELP rag_citation_validations_total Total number of clinical citation validations performed")
            lines.append("# TYPE rag_citation_validations_total counter")
            lines.append(f"rag_citation_validations_total {self.citation_validations_total}")
            lines.append("")

            lines.append("# HELP rag_claims_checked_total Total factual claims checked for citation grounding")
            lines.append("# TYPE rag_claims_checked_total counter")
            lines.append(f"rag_claims_checked_total {self.claims_checked_total}")
            lines.append("")

            lines.append("# HELP rag_claims_verified_total Total factual claims verified against evidence")
            lines.append("# TYPE rag_claims_verified_total counter")
            lines.append(f"rag_claims_verified_total {self.claims_verified_total}")
            lines.append("")

            lines.append("# HELP rag_claims_unsupported_total Total factual claims lacking evidence support")
            lines.append("# TYPE rag_claims_unsupported_total counter")
            lines.append(f"rag_claims_unsupported_total {self.claims_unsupported_total}")
            lines.append("")

            lines.append("# HELP rag_citation_spoofing_detected_total Total citation spoofing attempts detected")
            lines.append("# TYPE rag_citation_spoofing_detected_total counter")
            lines.append(f"rag_citation_spoofing_detected_total {self.citation_spoofing_detected_total}")
            lines.append("")

            lines.extend(build_histogram("rag_citation_attribution_duration_seconds", "Clinical citation attribution latency in seconds", self.latencies_citation_attribution))

            # Clinical Verification Metrics (Phase 6.6)
            lines.append("")
            lines.append("# HELP rag_clinical_verifications_total Total number of clinical verification runs")
            lines.append("# TYPE rag_clinical_verifications_total counter")
            lines.append(f"rag_clinical_verifications_total {self.clinical_verifications_total}")
            lines.append("")

            lines.append("# HELP rag_verification_grounded_claims_total Total claims verified as grounded")
            lines.append("# TYPE rag_verification_grounded_claims_total counter")
            lines.append(f"rag_verification_grounded_claims_total {self.verification_grounded_claims_total}")
            lines.append("")

            lines.append("# HELP rag_verification_ungrounded_claims_total Total claims flagged as ungrounded")
            lines.append("# TYPE rag_verification_ungrounded_claims_total counter")
            lines.append(f"rag_verification_ungrounded_claims_total {self.verification_ungrounded_claims_total}")
            lines.append("")

            lines.append("# HELP rag_verification_contradictions_total Total clinical contradictions detected")
            lines.append("# TYPE rag_verification_contradictions_total counter")
            lines.append(f"rag_verification_contradictions_total {self.verification_contradictions_total}")
            lines.append("")

            lines.append("# HELP rag_verification_hallucinations_total Total clinical hallucinations detected")
            lines.append("# TYPE rag_verification_hallucinations_total counter")
            lines.append(f"rag_verification_hallucinations_total {self.verification_hallucinations_total}")
            lines.append("")

            lines.append("# HELP rag_verification_fallbacks_total Total clinical fallbacks triggered by verification")
            lines.append("# TYPE rag_verification_fallbacks_total counter")
            lines.append(f"rag_verification_fallbacks_total {self.verification_fallbacks_total}")
            lines.append("")

            lines.extend(build_histogram("rag_clinical_verification_duration_seconds", "Clinical verification latency in seconds", self.latencies_clinical_verification))

            # Clinical Decision Support Metrics (Phase 6.7)
            lines.append("")
            lines.append("# HELP rag_decision_support_total Total number of clinical decision support runs")
            lines.append("# TYPE rag_decision_support_total counter")
            lines.append(f"rag_decision_support_total {self.decision_support_total}")
            lines.append("")

            lines.append("# HELP rag_uncertainty_level_total Total clinical uncertainty evaluations by level")
            lines.append("# TYPE rag_uncertainty_level_total counter")
            if self.decision_support_uncertainty_levels:
                for lvl_name, count in sorted(self.decision_support_uncertainty_levels.items()):
                    lines.append(f'rag_uncertainty_level_total{{level="{lvl_name}"}} {count}')
            else:
                lines.append('rag_uncertainty_level_total{level="none"} 0')
            lines.append("")

            lines.append("# HELP rag_clinical_risk_tier_total Total clinical risk evaluations by tier")
            lines.append("# TYPE rag_clinical_risk_tier_total counter")
            if self.decision_support_risk_tiers:
                for tier_name, count in sorted(self.decision_support_risk_tiers.items()):
                    lines.append(f'rag_clinical_risk_tier_total{{tier="{tier_name}"}} {count}')
            else:
                lines.append('rag_clinical_risk_tier_total{tier="none"} 0')
            lines.append("")

            lines.append("# HELP rag_decision_support_escalations_total Total clinical escalations triggered")
            lines.append("# TYPE rag_decision_support_escalations_total counter")
            lines.append(f"rag_decision_support_escalations_total {self.decision_support_escalations_total}")
            lines.append("")

            lines.extend(build_histogram("rag_decision_support_duration_seconds", "Clinical decision support latency in seconds", self.latencies_decision_support))

            # Clinical Orchestration Metrics (Phase 6.8)
            lines.append("")
            lines.append("# HELP rag_orchestration_total Total number of clinical intelligence orchestrations")
            lines.append("# TYPE rag_orchestration_total counter")
            lines.append(f"rag_orchestration_total {self.orchestration_total}")
            lines.append("")

            lines.append("# HELP rag_orchestration_concordant_total Total concordant orchestrations")
            lines.append("# TYPE rag_orchestration_concordant_total counter")
            lines.append(f"rag_orchestration_concordant_total {self.orchestration_concordant_total}")
            lines.append("")

            lines.append("# HELP rag_orchestration_intercepted_total Total safety-intercepted orchestrations")
            lines.append("# TYPE rag_orchestration_intercepted_total counter")
            lines.append(f"rag_orchestration_intercepted_total {self.orchestration_intercepted_total}")
            lines.append("")

            lines.extend(build_histogram("rag_orchestration_duration_seconds", "Clinical orchestration latency in seconds", self.latencies_orchestration))

            # Longitudinal Clinical Context Metrics (Phase 6.9)
            lines.append("")
            lines.append("# HELP rag_context_resolution_total Total number of dialogue context resolutions")
            lines.append("# TYPE rag_context_resolution_total counter")
            lines.append(f"rag_context_resolution_total {self.context_resolution_total}")
            lines.append("")

            lines.append("# HELP rag_context_follow_up_total Total number of follow-up queries resolved")
            lines.append("# TYPE rag_context_follow_up_total counter")
            lines.append(f"rag_context_follow_up_total {self.context_follow_up_total}")
            lines.append("")

            lines.append("# HELP rag_context_entities_tracked_total Total clinical entities tracked across dialogue")
            lines.append("# TYPE rag_context_entities_tracked_total counter")
            lines.append(f"rag_context_entities_tracked_total {self.context_entities_tracked_total}")
            lines.append("")

            lines.append("# HELP rag_context_contraindications_flagged_total Total multi-turn contraindications flagged")
            lines.append("# TYPE rag_context_contraindications_flagged_total counter")
            lines.append(f"rag_context_contraindications_flagged_total {self.context_contraindications_flagged_total}")
            lines.append("")

            lines.extend(build_histogram("rag_context_resolution_duration_seconds", "Longitudinal context resolution latency in seconds", self.latencies_context_resolution))

            return "\n".join(lines) + "\n"

    get_prometheus_metrics = get_prometheus_exposition

    def reset(self) -> None:
        with self._lock:
            self.requests_total = 0
            self.requests_successful = 0
            self.requests_failed = 0
            self.requests_blocked = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.cache_errors = 0
            self.llm_calls = 0
            self.llm_successes = 0
            self.llm_failures = 0
            self.llm_retries = 0
            self.llm_fallbacks = 0
            self.safety_blocked = 0
            self.safety_passed = 0
            self.emergency_blocks = 0
            self.prompt_injection_blocks = 0
            self.validation_failures = 0
            self.retrieval_count = 0
            self.insufficient_context = 0
            self.intent_classifications.clear()
            self.routing_counts.clear()
            self.intent_uncertain_total = 0
            self.query_plan_strategies.clear()
            self.query_plan_intents.clear()
            self.query_plan_expansion_total = 0
            self.query_plan_high_evidence_total = 0
            self.evidence_fusion_total = 0
            self.evidence_fusion_strategies.clear()
            self.evidence_contributing_docs_total = 0
            self.evidence_conflicts_total = 0
            self.evidence_conflicts_by_type.clear()
            self.evidence_insufficient_total = 0
            self.evidence_deduped_total = 0
            self.answer_synthesis_total = 0
            self.answer_synthesis_intents.clear()
            self.answer_synthesis_confidences.clear()
            self.answer_synthesis_fallback_total = 0
            self.answer_synthesis_fallbacks_by_reason.clear()
            self.answer_synthesis_conflict_total = 0
            self.answer_synthesis_insufficient_total = 0
            self.latencies_retrieval.clear()
            self.latencies_cache.clear()
            self.latencies_llm.clear()
            self.latencies_total.clear()
            self.latencies_ttft.clear()
            self.latencies_ttfe.clear()
            self.latencies_intent.clear()
            self.latencies_query_plan.clear()
            self.latencies_evidence_fusion.clear()
            self.latencies_answer_synthesis.clear()
            self.citation_validations_total = 0
            self.claims_checked_total = 0
            self.claims_verified_total = 0
            self.claims_unsupported_total = 0
            self.citation_spoofing_detected_total = 0
            self.latencies_citation_attribution.clear()
            self.clinical_verifications_total = 0
            self.verification_grounded_claims_total = 0
            self.verification_ungrounded_claims_total = 0
            self.verification_contradictions_total = 0
            self.verification_hallucinations_total = 0
            self.verification_fallbacks_total = 0
            self.latencies_clinical_verification.clear()
            self.decision_support_total = 0
            self.decision_support_uncertainty_levels.clear()
            self.decision_support_risk_tiers.clear()
            self.decision_support_escalations_total = 0
            self.decision_support_red_flags_total = 0
            self.decision_support_recommendations_total = 0
            self.latencies_decision_support.clear()
            self.orchestration_total = 0
            self.orchestration_concordant_total = 0
            self.orchestration_intercepted_total = 0
            self.latencies_orchestration.clear()
            self.context_resolution_total = 0
            self.context_follow_up_total = 0
            self.context_entities_tracked_total = 0
            self.context_contraindications_flagged_total = 0
            self.latencies_context_resolution.clear()

_metrics_collector = ProductionMetricsCollector()


def get_metrics_collector() -> ProductionMetricsCollector:
    return _metrics_collector


def record_evidence_fusion_event(
    strategy: str = "standard",
    contributing_docs_count: int = 1,
    conflicts_count: int = 0,
    conflict_types: Optional[List[str]] = None,
    deduped_count: int = 0,
    is_sufficient: bool = True,
    latency_ms: float = 0.0
) -> None:
    """Module-level helper to record clinical evidence fusion metrics."""
    get_metrics_collector().record_evidence_fusion_event(
        strategy=strategy,
        contributing_docs_count=contributing_docs_count,
        conflicts_count=conflicts_count,
        conflict_types=conflict_types,
        deduped_count=deduped_count,
        is_sufficient=is_sufficient,
        latency_ms=latency_ms
    )


def record_answer_synthesis_event(
    intent: str = "GENERAL_HEALTH",
    confidence: str = "HIGH",
    is_fallback: bool = False,
    fallback_reason: Optional[str] = None,
    conflicts_present: bool = False,
    is_insufficient: bool = False,
    latency_ms: float = 0.0
) -> None:
    """Module-level helper to record clinical answer synthesis metrics."""
    get_metrics_collector().record_answer_synthesis_event(
        intent=intent,
        confidence=confidence,
        is_fallback=is_fallback,
        fallback_reason=fallback_reason,
        conflicts_present=conflicts_present,
        is_insufficient=is_insufficient,
        latency_ms=latency_ms
    )


def record_citation_attribution_event(
    claims_checked: int = 0,
    claims_verified: int = 0,
    claims_unsupported: int = 0,
    spoofing_detected: bool = False,
    latency_ms: float = 0.0
) -> None:
    """Module-level helper to record clinical citation attribution metrics."""
    get_metrics_collector().record_citation_attribution_event(
        claims_checked=claims_checked,
        claims_verified=claims_verified,
        claims_unsupported=claims_unsupported,
        spoofing_detected=spoofing_detected,
        latency_ms=latency_ms
    )


def record_clinical_verification_event(
    grounded_claims: int = 0,
    ungrounded_claims: int = 0,
    contradictions: int = 0,
    hallucinations: int = 0,
    fallback_triggered: bool = False,
    latency_ms: float = 0.0
) -> None:
    """Module-level helper to record clinical verification telemetry."""
    get_metrics_collector().record_clinical_verification_event(
        grounded_claims=grounded_claims,
        ungrounded_claims=ungrounded_claims,
        contradictions=contradictions,
        hallucinations=hallucinations,
        fallback_triggered=fallback_triggered,
        latency_ms=latency_ms
    )


def record_decision_support_event(
    uncertainty_level: str = "LOW",
    risk_tier: str = "MINIMAL",
    escalation_required: bool = False,
    red_flags_count: int = 0,
    recommendations_count: int = 0,
    latency_ms: float = 0.0
) -> None:
    """Module-level helper to record clinical decision support telemetry."""
    get_metrics_collector().record_decision_support_event(
        uncertainty_level=uncertainty_level,
        risk_tier=risk_tier,
        escalation_required=escalation_required,
        red_flags_count=red_flags_count,
        recommendations_count=recommendations_count,
        latency_ms=latency_ms
    )


def record_orchestration_event(
    is_concordant: bool = True,
    is_intercepted: bool = False,
    latency_ms: float = 0.0
) -> None:
    """Module-level helper to record clinical intelligence orchestration telemetry (Phase 6.8)."""
    get_metrics_collector().record_orchestration_event(
        is_concordant=is_concordant,
        is_intercepted=is_intercepted,
        latency_ms=latency_ms
    )


def record_context_event(
    is_follow_up: bool = False,
    prior_turns_used: int = 0,
    entities_count: int = 0,
    contraindications_count: int = 0,
    latency_ms: float = 0.0
) -> None:
    """Module-level helper to record longitudinal clinical context telemetry (Phase 6.9)."""
    get_metrics_collector().record_context_event(
        is_follow_up=is_follow_up,
        prior_turns_used=prior_turns_used,
        entities_count=entities_count,
        contraindications_count=contraindications_count,
        latency_ms=latency_ms
    )
