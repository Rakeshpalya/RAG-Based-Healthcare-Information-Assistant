"""
FastAPI Router for the Intelligent Query Router.

Provides POST /api/query, dynamically dispatching inquiries to either:
- The Document RAG retrieval pipeline (with strict relevance gate & citations)
- The HealthAI Agno Medical Assistant (with SQLite session memory)
"""

import uuid
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field

from backend.database.models import User
from backend.api.auth_dependencies import get_optional_current_db_user
from backend.api.rag_router import _resolve_active_rag_service
from backend.router.query_router import route_and_execute_query

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Intelligent Query Router"]
)


class QueryRequest(BaseModel):
    """Request schema for unified query routing."""
    message: str = Field(
        ...,
        min_length=1,
        description="Natural language question or clinical inquiry.",
        json_schema_extra={"example": "What does the uploaded PDF say about hypertension?"}
    )
    mode: Optional[str] = Field(
        default="auto",
        description="Routing mode: 'auto' (intelligent classifier), 'rag' (force RAG), or 'agno' (force Agno).",
        json_schema_extra={"example": "auto"}
    )
    conversation_mode: Optional[str] = Field(
        default=None,
        description="Optional indicator of caller conversation context ('rag' or 'agno').",
        json_schema_extra={"example": "rag"}
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for Agno history tracking.",
        json_schema_extra={"example": "session-1234"}
    )
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Recent dialogue messages for follow-up reference resolution."
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum evidence chunks to retrieve when RAG is selected."
    )
    similarity_threshold: Optional[float] = Field(
        default=0.25,
        ge=-1.0,
        le=1.0,
        description="Cosine similarity cutoff threshold for RAG retrieval."
    )


class QueryResponse(BaseModel):
    """Response schema returned by the Intelligent Query Router."""
    route: str = Field(..., description="Selected routing pipeline: 'rag' or 'agno'")
    route_reason: str = Field(..., description="Explanation of why this route was selected")
    question: str = Field(..., description="The query processed")
    answer: str = Field(..., description="Synthesized response text")
    sources: List[Dict[str, Any]] = Field(default=[], description="Grounded source citations (empty for Agno)")
    retrieval_status: str = Field(..., description="Status from retrieval or agent pipeline")
    session_id: Optional[str] = Field(default=None, description="Active session ID")
    context: Optional[str] = Field(default=None, description="Grounded document context if RAG")
    disclaimer: Optional[str] = Field(default=None, description="Medical and legal disclaimer")
    timings: Optional[Dict[str, Any]] = Field(default=None, description="Latency breakdown metrics")
    status: str = Field(default="success", description="Overall execution status")
    agent: Optional[str] = Field(default=None, description="Name of the assisting agent")


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Intelligently route and execute inquiry",
    description=(
        "Analyzes the user inquiry and dynamically routes to either the Document RAG pipeline "
        "(for questions referencing uploaded documents/sources) or the Agno Medical Agent "
        "(for general medical inquiries). Preserves RAG relevance gate, citations, and session isolation."
    )
)
async def query_endpoint(
    request: QueryRequest,
    current_user: Optional[User] = Depends(get_optional_current_db_user)
) -> QueryResponse:
    """
    Executes unified intelligent routing turn.
    """
    cleaned_message = request.message.strip()
    if not cleaned_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty or whitespace only."
        )

    user_id = current_user.id if isinstance(current_user, User) else None
    active_rag_service = _resolve_active_rag_service(user_id)

    effective_session_id = request.session_id or f"session-{uuid.uuid4().hex[:10]}"

    try:
        result = route_and_execute_query(
            message=cleaned_message,
            mode=request.mode or "auto",
            conversation_mode=request.conversation_mode,
            session_id=effective_session_id,
            conversation_history=request.conversation_history,
            user_id=user_id,
            top_k=request.top_k or 5,
            similarity_threshold=request.similarity_threshold or 0.25,
            rag_service=active_rag_service,
        )

        return QueryResponse(
            route=result["route"],
            route_reason=result["route_reason"],
            question=result.get("question", cleaned_message),
            answer=result.get("answer", ""),
            sources=result.get("sources", []) or [],
            retrieval_status=result.get("retrieval_status", "success"),
            session_id=result.get("session_id"),
            context=result.get("context"),
            disclaimer=result.get("disclaimer"),
            timings=result.get("timings"),
            status=result.get("status", "success"),
            agent=result.get("agent"),
        )
    except Exception as exc:
        logger.exception("[QueryRouter] Error processing query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query routing error: {str(exc)}"
        )
