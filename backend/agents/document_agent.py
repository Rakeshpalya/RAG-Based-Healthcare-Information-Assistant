import time
from typing import Optional, Dict, Any

from backend.agents.agent_state import AgentState
from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiService
from backend.evaluation.citation_validator import CitationValidator
from backend.rag.prompt_builder import MEDICAL_DISCLAIMER


class DocumentAgent:
    """
    Specialized Agent for clinical document summarization and document-level inquiry workflows.

    RESPONSIBILITIES:
    - Summarizes medical reports and discharge summaries using retrieved chunk context.
    - Reuses the existing RAG pipeline without duplicating extraction or indexing services.
    - Formats summaries into structured sections: Clinical Context, Key Findings, and Action Plan.
    - Validates citations and attaches medical disclaimers.
    """

    def __init__(self, rag_service: Optional[RAGService] = None, gemini_service: Optional[GeminiService] = None):
        self.rag_service = rag_service or RAGService()
        self.gemini_service = gemini_service or GeminiService()

    def _build_summary_prompt(self, question: str, context: str) -> str:
        return (
            "=== ROLE & TASK ===\n"
            "You are the Clinical Document Agent. Summarize the provided document context accurately.\n\n"
            "=== STRUCTURED OUTPUT REQUIREMENTS ===\n"
            "Provide the summary formatted into the following sections:\n"
            "1. CLINICAL BACKGROUND: Patient context and admission/presentation reason.\n"
            "2. KEY FINDINGS: Diagnostic test results, vitals, and significant clinical observations.\n"
            "3. CARE PLAN & INSTRUCTIONS: Medications, follow-up, and monitoring instructions mentioned in the text.\n\n"
            "=== GROUNDING & SAFETY CONSTRAINTS ===\n"
            "- Only summarize information present in the retrieved context below.\n"
            "- Do not invent lab values, dosages, or diagnoses not stated in the text.\n"
            "- Include citations [Source 1], [Source 2] for each major claim.\n"
            "- Do not prescribe new treatments or claim to be the patient's physician.\n\n"
            f"=== RETRIEVED DOCUMENT CONTEXT ===\n"
            f"{context}\n\n"
            f"=== USER QUESTION / INSTRUCTION ===\n"
            f"{question}\n\n"
            "Document Summary:"
        )

    def run(self, state: AgentState) -> AgentState:
        t_start = time.perf_counter()
        question = state.user_question
        top_k = state.top_k

        # 1. Retrieve document context
        rag_result = self.rag_service.query(question=question, top_k=top_k)
        retrieval_status = rag_result.get("retrieval_status", "no_relevant_context")
        retrieved_chunks = rag_result.get("retrieved_chunks", [])
        context = rag_result.get("context", "")
        sources = rag_result.get("sources", [])

        state.retrieval_status = retrieval_status
        state.retrieved_chunks = retrieved_chunks
        state.context = context
        state.sources = sources

        for k, v in rag_result.get("timings", {}).items():
            state.timings[k] = v

        # 2. Guardrail check
        if retrieval_status != "success" or not context.strip():
            state.final_answer = (
                "The requested document or relevant document sections could not be found in the active index. "
                "Please ensure the document has been uploaded and indexed before requesting a summary."
            )
            state.citations = {
                "is_valid": True,
                "citations_found": [],
                "valid_citations": [],
                "missing_citations": False,
                "notes": "Safe fallback triggered due to missing document context."
            }
            state.disclaimer = MEDICAL_DISCLAIMER
            state.timings["document_agent_total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
            return state

        # 3. Generate summary prompt & call Gemini
        summary_prompt = self._build_summary_prompt(question, context)

        t_gen_start = time.perf_counter()
        gemini_response = self.gemini_service.generate_answer(
            question=summary_prompt,
            context=context,
            temperature=0.2
        )
        state.timings["generation_time_ms"] = round((time.perf_counter() - t_gen_start) * 1000.0, 2)

        raw_answer = gemini_response.get("answer", "")
        state.raw_answer = raw_answer

        # 4. Validate citations
        citation_res = CitationValidator.validate_citations(
            answer_text=raw_answer,
            retrieved_sources=sources
        )
        state.citations = citation_res.to_dict()

        # 5. Attach final output
        state.final_answer = raw_answer
        state.disclaimer = MEDICAL_DISCLAIMER
        state.timings["document_agent_total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)

        return state
