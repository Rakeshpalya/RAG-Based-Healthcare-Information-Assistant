import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

from backend.rag.rag_service import RAGService
from backend.database.models import User
from backend.api.auth_dependencies import get_optional_current_db_user
from backend.security import rate_limiter

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/rag",
    tags=["RAG Retrieval Pipeline"]
)

# Singleton service instance for API router
_rag_service_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Returns a singleton or shared RAGService instance."""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance


def _resolve_active_rag_service(user_id: Optional[int]) -> RAGService:
    """
    Resolves the appropriate RAGService instance.
    For authenticated queries (user_id is not None), prioritizes the shared persistent
    vector store holding uploaded user documents. Falls back to router singleton.
    """
    rag_service = get_rag_service()
    if user_id is not None:
        try:
            from backend.services.vector_store_service import get_vector_store_service
            shared_store = get_vector_store_service()
            if shared_store.count() > 0:
                return RAGService(vector_store=shared_store)
        except Exception as exc:
            logger.warning("Failed to resolve shared vector store: %s", type(exc).__name__)
    return rag_service


class RAGRetrieveRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language question or clinical research query.",
        json_schema_extra={"example": "What are the symptoms and diagnostic criteria for Type 2 Diabetes?"}
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of relevant chunks to retrieve."
    )
    similarity_threshold: Optional[float] = Field(
        default=0.25,
        ge=-1.0,
        le=1.0,
        description="Minimum cosine similarity cutoff threshold."
    )

    model_config = ConfigDict(extra="forbid")


class RAGRetrieveResponse(BaseModel):
    question: str
    retrieved_chunks: List[Dict[str, Any]]
    context: str
    sources: List[Dict[str, Any]]
    retrieval_status: str
    timings: Optional[Dict[str, Any]] = None


@router.post(
    "/retrieve",
    response_model=RAGRetrieveResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve grounded context and sources for a medical query",
    description="Retrieves the most semantically relevant clinical text chunks from FAISS, builds structured context with source attribution, and returns citations without LLM generation."
)
async def retrieve_rag_context(
    request: RAGRetrieveRequest,
    raw_request: Request = None,
    current_user: Optional[User] = Depends(get_optional_current_db_user)
) -> RAGRetrieveResponse:
    """
    RAG Context Retrieval Endpoint:
    1. Validates query string and rate limit.
    2. Embeds query using SentenceTransformers (384 dimensions).
    3. Searches FAISS vector index using cosine similarity with optional user ownership isolation.
    4. Filters results by similarity threshold.
    5. Assembles context string with [SOURCE N] headers and preserves source citations.
    """
    user_id = current_user.id if isinstance(current_user, User) else None
    if raw_request is not None:
        rate_limiter.check_rate_limit(raw_request, endpoint_type="rag_retrieve", user_id=user_id)
    active_service = _resolve_active_rag_service(user_id)
    result = active_service.query(
        question=request.question,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        user_id=user_id
    )

    return RAGRetrieveResponse(
        question=result["question"],
        retrieved_chunks=result["retrieved_chunks"],
        context=result["context"],
        sources=result["sources"],
        retrieval_status=result["retrieval_status"],
        timings=result.get("timings")
    )


class RAGQueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language medical question or clinical research inquiry.",
        json_schema_extra={"example": "What condition is associated with high blood pressure?"}
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of relevant chunks to retrieve for grounding."
    )
    similarity_threshold: Optional[float] = Field(
        default=0.25,
        ge=-1.0,
        le=1.0,
        description="Minimum cosine similarity cutoff threshold."
    )
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional list of recent chat messages for follow-up reference resolution."
    )

    model_config = ConfigDict(extra="forbid")


class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    retrieval_status: str
    sources: List[Dict[str, Any]]
    context: Optional[str] = None
    disclaimer: str
    timings: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    intent: Optional[Dict[str, Any]] = None
    orchestration: Optional[Dict[str, Any]] = None
    clinical_intelligence_orchestration: Optional[Dict[str, Any]] = None


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="End-to-End RAG Query: Retrieve grounded context and generate Gemini answer",
    description="Full RAG pipeline: retrieves semantically relevant clinical chunks from FAISS, verifies relevance threshold, and generates a grounded, cited answer using Google Gemini. Halts generation if context is insufficient."
)
async def query_rag(
    request: RAGQueryRequest,
    raw_request: Request = None,
    current_user: Optional[User] = Depends(get_optional_current_db_user)
) -> RAGQueryResponse:
    """
    End-to-End RAG Query Endpoint:
    1. Embeds question and queries FAISS vector store with optional user ownership isolation.
    2. Applies relevance threshold (≥ 0.25).
    3. If no relevant context is found, returns safe response without invoking Gemini.
    4. If context is found, prompts Gemini with grounding instructions and inline [Source X] citations.
    5. Returns grounded answer with structured source metadata, timings, request_id, and medical disclaimer.
    """
    from backend.services.gemini_service import GeminiServiceError
    from backend.evaluation.observability import generate_request_id

    user_id = current_user.id if isinstance(current_user, User) else None
    if raw_request is not None:
        rate_limiter.check_rate_limit(raw_request, endpoint_type="rag_query", user_id=user_id)
    active_service = _resolve_active_rag_service(user_id)

    req_id = (
        (getattr(raw_request.state, "request_id", None) if raw_request and hasattr(raw_request, "state") else None)
        or (raw_request.headers.get("X-Request-ID") if raw_request else None)
        or generate_request_id()
    )

    try:
        result = active_service.generate_rag_answer(
            question=request.question,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            user_id=user_id,
            conversation_history=request.conversation_history,
            request_id=req_id
        )
        req_id = result.get("request_id") or req_id

        logger.info(
            "RAG query completed for user_id=%s [req_id=%s]: status=%s, sources=%d",
            user_id,
            req_id,
            result.get("retrieval_status"),
            len(result.get("sources", [])),
        )
        return RAGQueryResponse(
            question=result["question"],
            answer=result["answer"],
            retrieval_status=result["retrieval_status"],
            sources=result["sources"],
            context=result.get("context"),
            disclaimer=result.get("disclaimer", ""),
            timings=result.get("timings"),
            request_id=req_id,
            intent=result.get("intent")
        )
    except GeminiServiceError as gse:
        logger.warning("GeminiServiceError handled cleanly in rag_router: %s", str(gse))
        return RAGQueryResponse(
            question=request.question,
            answer=(
                "The AI generation service is temporarily unavailable due to high demand. "
                "Please wait a few moments and try your inquiry again."
            ),
            retrieval_status="service_unavailable",
            sources=[],
            context=None,
            disclaimer="MEDICAL DISCLAIMER: This AI Healthcare Assistant provides educational and research information grounded in reference documents. Always consult a qualified healthcare provider for clinical decisions.",
            timings={"total_time_ms": 0.0}
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Configuration error: {str(ve)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during RAG query processing."
        )


@router.post(
    "/stream",
    summary="Execute Streaming RAG Query (Server-Sent Events)",
    description=(
        "Executes RAG with real-time Server-Sent Events (SSE) streaming. "
        "Strict clinical safety: tokens are buffered and validated before streaming to user."
    ),
    response_class=StreamingResponse
)
async def stream_rag_query(
    request: RAGQueryRequest,
    raw_request: Request = None,
    current_user: Optional[User] = Depends(get_optional_current_db_user)
):
    """
    Server-Sent Events (SSE) Streaming RAG Endpoint:
    Content-Type: text/event-stream
    Events emitted:
    - event: start (trace_id, question, timestamp)
    - event: status (pipeline step updates: retrieval, sufficiency_gate, generation, validation)
    - event: token (streamed text tokens after safety validation)
    - event: complete (final structured response with citations, sources, timings)
    - event: error (safe redacted failure notification if any step fails)
    """
    user_id = current_user.id if isinstance(current_user, User) else None
    if raw_request is not None:
        rate_limiter.check_rate_limit(raw_request, endpoint_type="rag_stream", user_id=user_id)
    active_service = _resolve_active_rag_service(user_id)

    req_id = (
        (getattr(raw_request.state, "request_id", None) if raw_request and hasattr(raw_request, "state") else None)
        or (raw_request.headers.get("X-Request-ID") if raw_request else None)
        or generate_request_id()
    )

    async def event_generator():
        try:
            for event_type, event_payload in active_service.generate_rag_stream(
                question=request.question,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold,
                user_id=user_id,
                conversation_history=request.conversation_history,
                request_id=req_id
            ):
                payload_json = json.dumps(event_payload, ensure_ascii=False)
                yield f"event: {event_type}\ndata: {payload_json}\n\n"
        except Exception as exc:
            logger.error("Unhandled exception during SSE streaming: %s", str(exc))
            err_data = json.dumps({
                "error": "An unexpected error occurred during streaming.",
                "status": "error"
            })
            yield f"event: error\ndata: {err_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
