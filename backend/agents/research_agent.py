import time
from typing import Optional, Dict, Any

from backend.agents.agent_state import AgentState
from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiService
from backend.evaluation.citation_validator import CitationValidator
from backend.rag.prompt_builder import MEDICAL_DISCLAIMER


class ResearchAgent:
    """
    Specialized Agent for clinical research and evidence-based inquiries.

    RESPONSIBILITIES:
    - Queries the authoritative FAISS vector store via RAGService.
    - Strictly prevents LLM hallucination: if context is insufficient, halts safely.
    - Generates grounded answers via GeminiService with inline citations [Source X].
    - Validates citations independently using CitationValidator.
    - Preserves FAISS source metadata alongside the answer.
    """

    def __init__(self, rag_service: Optional[RAGService] = None, gemini_service: Optional[GeminiService] = None):
        self.rag_service = rag_service or RAGService()
        self.gemini_service = gemini_service or GeminiService()

    def run(self, state: AgentState) -> AgentState:
        """
        Executes the research retrieval and grounded generation workflow.
        Updates state in-place and returns it.
        """
        t_start = time.perf_counter()
        question = state.user_question
        top_k = state.top_k

        # 1. Execute RAG Retrieval
        rag_result = self.rag_service.query(question=question, top_k=top_k)
        retrieval_status = rag_result.get("retrieval_status", "no_relevant_context")
        retrieved_chunks = rag_result.get("retrieved_chunks", [])
        context = rag_result.get("context", "")
        sources = rag_result.get("sources", [])

        state.retrieval_status = retrieval_status
        state.retrieved_chunks = retrieved_chunks
        state.context = context
        state.sources = sources

        # Incorporate retrieval timings
        for k, v in rag_result.get("timings", {}).items():
            state.timings[k] = v

        # 2. Hard-Stop Guardrail: If no relevant context found, bypass Gemini completely
        if retrieval_status != "success" or not context.strip():
            state.final_answer = (
                "The available reference medical documents do not contain sufficient evidence to answer this inquiry. "
                "To maintain clinical accuracy and prevent unsupported statements, no generative answer was produced. "
                "Please consult a qualified healthcare provider or verify that relevant documents are indexed."
            )
            state.citations = {
                "is_valid": True,
                "citations_found": [],
                "valid_citations": [],
                "missing_citations": False,
                "notes": "Safe fallback triggered due to insufficient retrieval evidence."
            }
            state.disclaimer = MEDICAL_DISCLAIMER
            state.timings["research_agent_total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
            return state

        # 3. Call Gemini with Grounded Context
        t_gen_start = time.perf_counter()
        gemini_response = self.gemini_service.generate_answer(
            question=question,
            context=context,
            temperature=0.2
        )
        gen_time = round((time.perf_counter() - t_gen_start) * 1000.0, 2)
        state.timings["generation_time_ms"] = gen_time

        raw_answer = gemini_response.get("answer", "")
        state.raw_answer = raw_answer

        # 4. Independent Citation Validation
        citation_res = CitationValidator.validate_citations(
            answer_text=raw_answer,
            retrieved_sources=sources
        )
        state.citations = citation_res.to_dict()

        # 5. Attach Final Answer and Disclaimer
        state.final_answer = raw_answer
        state.disclaimer = MEDICAL_DISCLAIMER
        state.timings["research_agent_total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)

        return state
