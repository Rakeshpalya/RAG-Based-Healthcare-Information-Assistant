import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.agents.agent_state import AgentState, SafetyDecision, OrchestratorDecision
from backend.agents.safety_agent import SafetyAgent
from backend.agents.orchestrator_agent import OrchestratorAgent
from backend.agents.research_agent import ResearchAgent
from backend.agents.explanation_agent import ExplanationAgent
from backend.agents.document_agent import DocumentAgent
from backend.rag.prompt_builder import MEDICAL_DISCLAIMER
from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiService


router = APIRouter(prefix="/agent", tags=["Agentic AI"])


class AgentQueryRequest(BaseModel):
    """Request schema for orchestrated agent queries."""
    question: str = Field(..., description="The user's healthcare, clinical, or research question.")
    explanation_level: Optional[str] = Field("simple", description="Explanation complexity: 'simple', 'intermediate', or 'technical'.")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of evidence chunks to retrieve.")


class AgentQueryResponse(BaseModel):
    """Structured response object containing multi-agent decision metadata and grounded output."""
    question: str
    intent: str
    agent: str
    safety: Dict[str, Any]
    answer: str
    sources: List[Dict[str, Any]]
    citations: Dict[str, Any]
    timings: Dict[str, float]
    disclaimer: str


def execute_agent_workflow(
    question: str,
    explanation_level: str = "simple",
    top_k: int = 5,
    rag_service: Optional[RAGService] = None,
    gemini_service: Optional[GeminiService] = None
) -> AgentQueryResponse:
    """
    Core orchestrator execution function that coordinates the multi-agent pipeline.
    """
    total_start = time.perf_counter()
    timings: Dict[str, float] = {}

    # 1. Step 1: Safety Pre-check
    t_safety_start = time.perf_counter()
    safety_decision = SafetyAgent.evaluate_pre_check(question)
    timings["safety_check_time_ms"] = round((time.perf_counter() - t_safety_start) * 1000.0, 3)

    # If Safety Pre-Check blocks the query (Emergency, Diagnosis, Prescribing)
    # IMMEDIATELY HALT without calling RAG retrieval or Gemini LLM generation.
    if not safety_decision.allowed:
        timings["total_time_ms"] = round((time.perf_counter() - total_start) * 1000.0, 3)
        return AgentQueryResponse(
            question=question or "",
            intent="SAFETY_SENSITIVE",
            agent="safety_agent",
            safety=safety_decision.to_dict(),
            answer=safety_decision.fallback_response,
            sources=[],
            citations={
                "is_valid": True,
                "citations_found": [],
                "valid_citations": [],
                "missing_citations": False,
                "notes": "Safety pre-screening blocked unsafe request."
            },
            timings=timings,
            disclaimer=MEDICAL_DISCLAIMER
        )

    # 2. Step 2: Orchestrator Intent Classification & Routing
    t_orch_start = time.perf_counter()
    orchestrator_decision = OrchestratorAgent.classify_intent(question, safety_decision)
    timings["orchestration_time_ms"] = round((time.perf_counter() - t_orch_start) * 1000.0, 3)

    # 3. Step 3: Initialize State
    state = AgentState(
        user_question=question,
        explanation_level=explanation_level,
        top_k=top_k,
        safety_decision=safety_decision,
        orchestrator_decision=orchestrator_decision,
        timings=timings,
        disclaimer=MEDICAL_DISCLAIMER
    )

    # 4. Step 4: Dispatch to Specialized Agent
    agent_name = orchestrator_decision.agent

    if agent_name == "document_agent":
        agent = DocumentAgent(rag_service=rag_service, gemini_service=gemini_service)
        state = agent.run(state)
    elif agent_name == "explanation_agent":
        agent = ExplanationAgent(rag_service=rag_service, gemini_service=gemini_service)
        state = agent.run(state)
    else:  # Default to research_agent
        agent = ResearchAgent(rag_service=rag_service, gemini_service=gemini_service)
        state = agent.run(state)

    # 5. Step 5: Safety Post-check
    t_post_start = time.perf_counter()
    post_check = SafetyAgent.evaluate_post_check(
        user_question=question,
        generated_answer=state.final_answer,
        sources_present=len(state.sources) > 0
    )
    timings["safety_post_check_time_ms"] = round((time.perf_counter() - t_post_start) * 1000.0, 3)

    if not post_check["post_check_passed"]:
        # If the LLM output contained prohibited diagnostic/prescription phrasing, replace with safe fallback
        state.final_answer = (
            "The generated response was withheld because it contained unauthorized medical advice or clinical conclusions. "
            "Please consult a licensed physician for personalized medical evaluation."
        )

    # 6. Finalize Timings
    timings["total_time_ms"] = round((time.perf_counter() - total_start) * 1000.0, 3)
    state.timings = timings

    return AgentQueryResponse(
        question=state.user_question,
        intent=state.orchestrator_decision.intent,
        agent=state.orchestrator_decision.agent,
        safety=state.safety_decision.to_dict(),
        answer=state.final_answer or "",
        sources=state.sources,
        citations=state.citations or {},
        timings=state.timings,
        disclaimer=state.disclaimer
    )


@router.post(
    "/query",
    response_model=AgentQueryResponse,
    summary="Agentic AI Orchestrated Healthcare Query",
    description="Submits a healthcare inquiry to the multi-agent orchestration pipeline with deterministic safety gatekeeping."
)
async def agent_query_endpoint(request: AgentQueryRequest):
    """
    FastAPI endpoint for multi-agent query routing.
    """
    try:
        response = execute_agent_workflow(
            question=request.question,
            explanation_level=request.explanation_level or "simple",
            top_k=request.top_k or 5
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow error: {str(e)}"
        )
