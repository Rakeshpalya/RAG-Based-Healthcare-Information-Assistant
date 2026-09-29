"""
Intelligent Query Router for AI-Healthcare-Agent.

Classifies user inquiries and delegates execution to either:
1. Existing Document RAG Assistant (FAISS, RAGService, Gemini grounding, relevance gate)
2. Existing HealthAI Medical Agent (Agno Agent, OpenAI/Gemini, SQLite session persistence)

The router itself does NOT generate medical answers.
It preserves all clinical safeguards, relevance gates, citations, and conversation history isolation.

Routing Priority:
1. Explicit document reference (e.g. 'uploaded PDF', 'in my document', 'according to source') -> RAG
2. Explicit request for general medical information (e.g. 'What is hypertension?', 'What is asthma?') -> AGNO
3. Ambiguous follow-up (e.g. 'What are its symptoms?', 'Which of them can be modified?') -> use previous conversation_mode
4. If no context exists -> default to AGNO for general medical questions
5. Never send a general medical question to RAG merely because documents are indexed.
"""

import re
import time
import logging
import threading
from typing import Dict, Any, Optional, List

from backend.rag.rag_service import RAGService
from backend.agents.medical_agent import generate_medical_chat_response

logger = logging.getLogger(__name__)

# Patterns matching explicit document references (Priority 1)
EXPLICIT_DOC_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:uploaded|provided)\s+(?:pdf|document|paper|file|source|guideline|guidelines|study|trial|report)s?\b", re.IGNORECASE),
    re.compile(r"\b(?:my|the)\s+(?:uploaded\s+|provided\s+)?(?:pdf|document|paper|file|guideline|guidelines|study|trial|source|sources)s?\b", re.IGNORECASE),
    re.compile(r"\b(?:in|from|according to|based on|per)\s+(?:the|my|this|uploaded|provided)?\s*(?:pdf|document|paper|file|source|sources|guideline|guidelines|study|trial|uploaded sources)s?\b", re.IGNORECASE),
    re.compile(r"\bwhat does (?:the|my|this|uploaded|provided)\s+(?:pdf|document|paper|file|guideline|guidelines|study|trial)s?\s+say\b", re.IGNORECASE),
    re.compile(r"\b(?:according to|based on)\s+source\s*(?:#|:)?\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bsource\s*(?:#|:)?\s*\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:page|section|paragraph)\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bclinical trial paper\b", re.IGNORECASE),
    re.compile(r"\baccording to (?:my\s+)?(?:the\s+)?(?:uploaded\s+)?(?:pdf|document|paper|sources?)\b", re.IGNORECASE),
    re.compile(r"\b(?:pdf|pdfs|document|documents)\b", re.IGNORECASE),
]

# Patterns matching ambiguous follow-up questions requiring conversation context (Priority 3)
AMBIGUOUS_FOLLOWUP_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:what are|list|tell me)\s+(?:its|the)\s+(?:symptoms|causes|risk factors|treatments|complications|side effects|signs)\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+(?:of them|one|ones)\s+(?:can be|are|can we)\b", re.IGNORECASE),
    re.compile(r"\b(?:how is it treated|how to treat it|can it be cured|what causes it|why does it happen|can it be prevented)\b", re.IGNORECASE),
    re.compile(r"\b(?:what about|how about)\s+(?:it|them|this|these|those)\b", re.IGNORECASE),
    re.compile(r"\b(?:tell me more|explain more|more details|elaborate)\b", re.IGNORECASE),
    re.compile(r"^(?:which of them|what about them|can they be|are they|is it|does it)\b", re.IGNORECASE),
    re.compile(r"\b(?:its symptoms|its causes|its risks|its risk factors|its treatment|its complications)\b", re.IGNORECASE),
    re.compile(r"\b(?:what are the symptoms|what are the complications|what are the causes|what are the risk factors)\b", re.IGNORECASE),
]

# Common clinical conditions sorted by length descending to match specific entities first
COMMON_CONDITIONS: List[str] = sorted([
    "hypertension",
    "high blood pressure",
    "blood pressure",
    "type 2 diabetes",
    "type 1 diabetes",
    "diabetes",
    "asthma",
    "cardiovascular disease",
    "heart disease",
    "heart failure",
    "coronary artery disease",
    "arrhythmia",
    "pediatric leukemia",
    "leukemia",
    "cancer",
    "pneumonia",
    "cholesterol",
    "high cholesterol",
    "covid-19",
    "covid 19",
    "covid",
    "dengue",
    "aids",
    "hiv",
    "migraine",
    "headache",
    "arthritis",
    "stroke",
    "kidney disease",
    "obesity",
    "sleep apnea",
], key=len, reverse=True)


class ConversationRoutingStateManager:
    """
    Lightweight, thread-safe session state tracking conversation routing context:
    - conversation_mode: 'rag' or 'agno'
    - last_topic: str (e.g., 'hypertension', 'asthma', 'diabetes')
    - last_query: str
    """

    def __init__(self):
        self._states: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_state(self, session_id: Optional[str]) -> Dict[str, Any]:
        if not session_id:
            return {}
        with self._lock:
            return dict(self._states.get(str(session_id), {}))

    def update_state(
        self,
        session_id: Optional[str],
        conversation_mode: str,
        topic: Optional[str] = None,
        last_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not session_id:
            return {"conversation_mode": conversation_mode, "last_topic": topic}
        s_id = str(session_id)
        with self._lock:
            current = self._states.get(s_id, {})
            current["conversation_mode"] = conversation_mode
            if topic:
                current["last_topic"] = topic
            if last_query:
                current["last_query"] = last_query
            current["updated_at"] = time.time()
            self._states[s_id] = current
            return dict(current)

    def clear(self, session_id: Optional[str] = None) -> None:
        with self._lock:
            if session_id:
                self._states.pop(str(session_id), None)
            else:
                self._states.clear()


routing_state_manager = ConversationRoutingStateManager()


def get_routing_state(session_id: Optional[str]) -> Dict[str, Any]:
    """Retrieves routing conversation state for a session."""
    return routing_state_manager.get_state(session_id)


def update_routing_state(
    session_id: Optional[str],
    conversation_mode: str,
    topic: Optional[str] = None,
    last_query: Optional[str] = None,
) -> Dict[str, Any]:
    """Updates routing conversation state for a session."""
    return routing_state_manager.update_state(
        session_id=session_id,
        conversation_mode=conversation_mode,
        topic=topic,
        last_query=last_query,
    )


def reset_routing_state(session_id: Optional[str] = None) -> None:
    """Resets routing conversation state."""
    routing_state_manager.clear(session_id)


def extract_query_topic(text: str) -> Optional[str]:
    """
    Extracts the clinical topic or entity from a user question.
    e.g. 'What are the risk factors for hypertension according to my uploaded PDF?' -> 'hypertension'
    e.g. 'What is asthma?' -> 'asthma'
    e.g. 'What does the uploaded paper say about pediatric leukemia?' -> 'pediatric leukemia'
    """
    if not text:
        return None
    raw = text.strip()

    # 1. Match known clinical conditions (ordered longest to shortest)
    for condition in COMMON_CONDITIONS:
        pattern = r"\b" + re.escape(condition) + r"\b"
        if re.search(pattern, raw, re.IGNORECASE):
            return condition

    # 2. Extract using regex patterns if condition is not in static list
    regex_patterns = [
        r"(?:risk factors|symptoms|causes|complications|treatments?|management)\s+(?:for|of)\s+([a-zA-Z0-9\s\-]+?)(?:\s+(?:according|in|based|from|per)\b|\?|\.|$)",
        r"(?:what is|what are|define|explain)\s+(?:the\s+)?([a-zA-Z0-9\s\-]+?)(?:\s+(?:according|in|based|from|per)\b|\?|\.|$)",
        r"(?:about|regarding|on)\s+([a-zA-Z0-9\s\-]+?)(?:\s+(?:according|in|based|from|per)\b|\?|\.|$)",
    ]
    for pat in regex_patterns:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            extracted = m.group(1).strip()
            # Clean out common document phrases
            extracted = re.sub(
                r"\b(?:my|the|this|uploaded|provided)?\s*(?:pdf|document|paper|file)s?\b",
                "",
                extracted,
                flags=re.IGNORECASE
            ).strip()
            if extracted and len(extracted) > 2:
                return extracted.lower()

    return None


def is_ambiguous_followup(message: str) -> bool:
    """
    Determines if a query is an ambiguous follow-up needing prior conversation context.
    e.g. 'What are its symptoms?' -> True
    e.g. 'Which of them can be modified?' -> True
    e.g. 'What is hypertension?' -> False (explicit standalone condition)
    e.g. 'What are the symptoms of diabetes?' -> False (explicit condition named without pronouns)
    """
    raw = (message or "").strip()
    if not raw:
        return False

    # 1. Pronouns or demonstratives without standalone subject
    if re.search(r"\b(?:its|which of them|what about them|how is it|what causes it|why does it|can it be|can they be)\b", raw, re.IGNORECASE):
        return True

    # 2. Check ambiguous followup patterns
    has_ambiguous_phrase = any(p.search(raw) for p in AMBIGUOUS_FOLLOWUP_PATTERNS)
    if not has_ambiguous_phrase:
        return False

    # If the question explicitly specifies a medical condition without ambiguous pronouns, it is NOT ambiguous
    topic = extract_query_topic(raw)
    if topic:
        # If pronoun exists (e.g. "its"), it remains ambiguous
        if re.search(r"\b(?:its|it|them)\b", raw, re.IGNORECASE):
            return True
        # Explicit topic provided without pronouns (e.g. "What are the symptoms of diabetes?") -> standalone
        return False

    return True


def resolve_ambiguous_message(message: str, topic: Optional[str]) -> str:
    """
    Resolves pronouns and references in an ambiguous query using the prior topic.
    e.g. 'What are its symptoms?' + 'asthma' -> 'What are the symptoms of asthma?'
    e.g. 'Which of them can be modified?' + 'hypertension' -> 'Which risk factors of hypertension can be modified?'
    """
    if not topic:
        return message
    msg = message.strip()
    clean_topic = topic.strip()

    # Exact common follow-up templates
    if re.search(r"\bwhat are its symptoms\b", msg, re.IGNORECASE) or re.search(r"\bwhat are the symptoms\b", msg, re.IGNORECASE):
        return f"What are the symptoms of {clean_topic}?"
    if re.search(r"\bwhat are its complications\b", msg, re.IGNORECASE) or re.search(r"\bwhat are the complications\b", msg, re.IGNORECASE):
        return f"What are the complications of {clean_topic}?"
    if re.search(r"\bwhat are its causes\b", msg, re.IGNORECASE) or re.search(r"\bwhat are the causes\b", msg, re.IGNORECASE):
        return f"What are the causes of {clean_topic}?"
    if re.search(r"\bwhat are its risk factors\b", msg, re.IGNORECASE) or re.search(r"\bwhat are the risk factors\b", msg, re.IGNORECASE):
        return f"What are the risk factors for {clean_topic}?"
    if re.search(r"\bwhich of them can be modified\b", msg, re.IGNORECASE):
        return f"Which risk factors of {clean_topic} can be modified?"
    if re.search(r"\bwhat causes it\b", msg, re.IGNORECASE):
        return f"What causes {clean_topic}?"
    if re.search(r"\bwhy does it happen\b", msg, re.IGNORECASE):
        return f"Why does {clean_topic} happen?"
    if re.search(r"\bhow is it treated\b", msg, re.IGNORECASE):
        return f"How is {clean_topic} treated?"
    if re.search(r"\bcan it be cured\b", msg, re.IGNORECASE):
        return f"Can {clean_topic} be cured?"

    # Pronoun replacements
    subbed = re.sub(r"\bits\b", f"{clean_topic}'s", msg, flags=re.IGNORECASE)
    subbed = re.sub(r"\bit\b", clean_topic, subbed, flags=re.IGNORECASE)
    subbed = re.sub(r"\bthem\b", clean_topic, subbed, flags=re.IGNORECASE)
    return subbed


def classify_query(
    message: str,
    conversation_mode: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Classifies an incoming query following strict priority order:
    1. Explicit document reference -> RAG
    2. Explicit request for general medical information -> AGNO
    3. Ambiguous follow-up -> use previous conversation_mode (rag or agno)
    4. If no context exists -> default to AGNO for general medical questions
    5. Never send a general medical question to RAG merely because documents are indexed.

    Args:
        message: The user's natural language question.
        conversation_mode: Optional indicator of current conversation context ('rag' or 'agno').
        conversation_history: Optional recent messages for contextual resolution.
        session_id: Optional session identifier for retrieving state.

    Returns:
        Dict with keys:
            - 'route': 'rag' or 'agno'
            - 'reason': Human-readable explanation of the routing classification
            - 'topic': Extracted or contextual medical entity
            - 'is_followup': Boolean indicating if this is an ambiguous follow-up
    """
    raw_message = (message or "").strip()
    if not raw_message:
        return {
            "route": "agno",
            "reason": "Empty message routed to general assistant",
            "topic": None,
            "is_followup": False,
        }

    # Retrieve prior session state if available
    prior_state = routing_state_manager.get_state(session_id)
    prior_mode = conversation_mode or prior_state.get("conversation_mode")
    prior_topic = prior_state.get("last_topic")

    # -------------------------------------------------------------------------
    # Priority 1: Explicit Document Reference -> RAG
    # -------------------------------------------------------------------------
    is_explicit_doc = any(p.search(raw_message) for p in EXPLICIT_DOC_PATTERNS)
    if is_explicit_doc:
        topic = extract_query_topic(raw_message) or prior_topic
        return {
            "route": "rag",
            "reason": "User explicitly refers to uploaded document or source",
            "topic": topic,
            "is_followup": False,
        }

    # -------------------------------------------------------------------------
    # Priority 3: Ambiguous Follow-Up -> Use Previous conversation_mode
    # (Checked before general queries to catch queries like "What are its symptoms?")
    # -------------------------------------------------------------------------
    ambiguous = is_ambiguous_followup(raw_message)
    if ambiguous:
        # Determine mode from explicit argument, stored session state, or history
        effective_mode = prior_mode

        if not effective_mode and conversation_history and len(conversation_history) > 0:
            history_text = " ".join(
                str(m.get("content") or m.get("text") or "") for m in conversation_history
            ).lower()
            has_doc_in_history = any(
                p.search(history_text) for p in EXPLICIT_DOC_PATTERNS
            ) or any(k in history_text for k in ["source", "evidence", "document", "pdf", "retrieval"])
            effective_mode = "rag" if has_doc_in_history else "agno"

        # Try extracting topic from history if not in session state
        resolved_topic = prior_topic
        if not resolved_topic and conversation_history:
            for turn in reversed(conversation_history):
                content = str(turn.get("content") or turn.get("text") or "")
                extracted = extract_query_topic(content)
                if extracted:
                    resolved_topic = extracted
                    break

        if effective_mode == "rag":
            return {
                "route": "rag",
                "reason": "Follow-up question within document RAG conversation context",
                "topic": resolved_topic,
                "is_followup": True,
            }
        else:
            # Default ambiguous follow-ups without RAG history to AGNO
            return {
                "route": "agno",
                "reason": "Follow-up question within Medical Agent conversation context",
                "topic": resolved_topic,
                "is_followup": True,
            }

    # -------------------------------------------------------------------------
    # Priority 2: Explicit Request for General Medical Information -> AGNO
    # (Condition named, standalone question, no document reference)
    # -------------------------------------------------------------------------
    query_topic = extract_query_topic(raw_message)
    return {
        "route": "agno",
        "reason": "General medical question without document reference",
        "topic": query_topic,
        "is_followup": False,
    }


def route_and_execute_query(
    message: str,
    mode: str = "auto",
    conversation_mode: Optional[str] = None,
    session_id: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    user_id: Optional[int] = None,
    top_k: int = 5,
    similarity_threshold: float = 0.25,
    rag_service: Optional[RAGService] = None,
) -> Dict[str, Any]:
    """
    Classifies and delegates query execution to the appropriate existing subsystem.

    Preserves:
    - RAG relevance gate (no unsupported hallucination when context is empty)
    - Source citations and evidence metadata
    - Latency and timing breakdowns
    - Agno conversation session isolation
    - Separate conversation states for RAG and Agno

    Args:
        message: Natural language inquiry.
        mode: 'auto' (default), 'rag', or 'agno'.
        conversation_mode: Optional current context ('rag' or 'agno').
        session_id: Session identifier for history tracking.
        conversation_history: List of prior dialogue turns for RAG follow-up resolution.
        user_id: Optional user ID for vector store ownership isolation.
        top_k: Evidence chunk count for RAG retrieval.
        similarity_threshold: Cosine similarity cutoff for RAG evidence.
        rag_service: Optional RAGService instance for dependency injection.

    Returns:
        Structured response dictionary.
    """
    cleaned_message = (message or "").strip()
    effective_session_id = session_id or "default"

    if not cleaned_message:
        return {
            "route": "agno",
            "route_reason": "Empty message",
            "question": cleaned_message,
            "answer": "Please ask a question about your healthcare documents or a general medical topic.",
            "sources": [],
            "retrieval_status": "empty_message",
            "session_id": effective_session_id,
            "status": "empty_message",
        }

    # 1. Classify Route
    normalized_mode = (mode or "auto").strip().lower()
    if normalized_mode in ("rag", "agno"):
        route = normalized_mode
        reason = f"Explicit route override: {normalized_mode.upper()}"
        topic = extract_query_topic(cleaned_message)
        is_followup = False
    else:
        classification = classify_query(
            message=cleaned_message,
            conversation_mode=conversation_mode,
            conversation_history=conversation_history,
            session_id=effective_session_id,
        )
        route = classification["route"]
        reason = classification["reason"]
        topic = classification.get("topic")
        is_followup = classification.get("is_followup", False)

    # 2. Update Lightweight Routing State
    prior_state = routing_state_manager.get_state(effective_session_id)
    active_topic = topic or prior_state.get("last_topic")
    routing_state_manager.update_state(
        session_id=effective_session_id,
        conversation_mode=route,
        topic=active_topic,
        last_query=cleaned_message,
    )

    logger.info(
        "[QueryRouter] Routed query '%s...' to '%s' (Reason: %s, Topic: %s)",
        cleaned_message[:40],
        route,
        reason,
        active_topic,
    )

    # 3. Execute via Document RAG Assistant
    if route == "rag":
        active_rag = rag_service or RAGService()

        # If this is an ambiguous follow-up and caller did not pass conversation_history,
        # construct contextual turns from routing state so RAGService.resolve_followup_query works
        effective_history = conversation_history
        if is_followup and (not effective_history or len(effective_history) == 0):
            last_q = prior_state.get("last_query")
            if last_q:
                effective_history = [
                    {"role": "user", "content": last_q},
                    {"role": "assistant", "content": f"Information regarding {active_topic or 'the medical topic'}."}
                ]

        rag_result = active_rag.generate_rag_answer(
            question=cleaned_message,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            user_id=user_id,
            conversation_history=effective_history,
        )

        return {
            "route": "rag",
            "route_reason": reason,
            "question": rag_result.get("question", cleaned_message),
            "answer": rag_result.get("answer", ""),
            "retrieval_status": rag_result.get("retrieval_status", "success"),
            "sources": rag_result.get("sources", []),
            "context": rag_result.get("context"),
            "disclaimer": rag_result.get("disclaimer", ""),
            "timings": rag_result.get("timings"),
            "session_id": effective_session_id,
            "status": "success",
        }

    # 4. Execute via HealthAI Agno Medical Agent
    # For ambiguous follow-ups, resolve pronouns using active_topic so the agent understands "its" = active_topic
    if is_followup and active_topic:
        agent_message = resolve_ambiguous_message(cleaned_message, active_topic)
    else:
        agent_message = cleaned_message

    agno_result = generate_medical_chat_response(
        message=agent_message,
        session_id=effective_session_id,
    )

    return {
        "route": "agno",
        "route_reason": reason,
        "question": cleaned_message,
        "answer": agno_result.get("answer", ""),
        "session_id": agno_result.get("session_id", effective_session_id),
        "sources": [],
        "retrieval_status": "agno_agent",
        "disclaimer": "General educational information only. Does not diagnose or prescribe.",
        "timings": None,
        "status": agno_result.get("status", "success"),
        "agent": agno_result.get("agent", "HealthAI Medical Assistant"),
    }
