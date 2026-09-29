import time
from typing import Optional, Dict, Any

from backend.agents.agent_state import AgentState
from backend.rag.rag_service import RAGService
from backend.services.gemini_service import GeminiService
from backend.evaluation.citation_validator import CitationValidator
from backend.rag.prompt_builder import MEDICAL_DISCLAIMER


class ExplanationAgent:
    """
    Specialized Agent for simplifying and translating complex medical information
    into patient-accessible language while strictly preserving factual grounding.

    SUPPORTED EXPLANATION TIERS:
    - simple: 6th-8th grade reading level; plain language; clear everyday explanations.
    - intermediate: Patient-centered clinical literacy; balanced detail and clarity.
    - technical: Advanced clinical depth with precise pathophysiological nomenclature.
    """

    def __init__(self, rag_service: Optional[RAGService] = None, gemini_service: Optional[GeminiService] = None):
        self.rag_service = rag_service or RAGService()
        self.gemini_service = gemini_service or GeminiService()

    def _build_explanation_prompt(self, question: str, context: str, level: str) -> str:
        level_instructions = {
            "simple": (
                "EXPLANATION TIER: SIMPLE (PATIENT & CAREGIVER FOCUS)\n"
                "- Write at a 6th to 8th grade reading level.\n"
                "- Avoid dense medical jargon; if a clinical term is essential, explain it immediately in plain words.\n"
                "- Use clear, reassuring, and concise sentences.\n"
                "- Preserve all facts and cite sources as [Source 1], [Source 2]."
            ),
            "intermediate": (
                "EXPLANATION TIER: INTERMEDIATE (INFORMED PATIENT FOCUS)\n"
                "- Balance clear language with accurate clinical terminology.\n"
                "- Provide clear context on how the condition or finding affects health.\n"
                "- Preserve all facts and cite sources as [Source 1], [Source 2]."
            ),
            "technical": (
                "EXPLANATION TIER: TECHNICAL (CLINICAL & RESEARCH FOCUS)\n"
                "- Maintain formal medical terminology and pathophysiological precision.\n"
                "- Present details suitable for medical students or clinical researchers.\n"
                "- Preserve all facts and cite sources as [Source 1], [Source 2]."
            )
        }

        tier_instruction = level_instructions.get(level.lower(), level_instructions["simple"])

        return (
            f"=== ROLE & OBJECTIVE ===\n"
            f"You are the Medical Explanation Agent. Your task is to explain the retrieved medical evidence clearly.\n\n"
            f"=== STYLE & READING LEVEL GUIDELINES ===\n"
            f"{tier_instruction}\n\n"
            f"=== GROUNDING & SAFETY CONSTRAINTS ===\n"
            f"1. Base your explanation strictly on the retrieved medical context below.\n"
            f"2. Do not fabricate facts or speculate beyond the provided evidence.\n"
            f"3. Include citations such as [Source 1] whenever referencing evidence.\n"
            f"4. Do NOT diagnose the user and do NOT prescribe or alter medications.\n\n"
            f"=== RETRIEVED MEDICAL CONTEXT ===\n"
            f"{context}\n\n"
            f"=== USER QUESTION ===\n"
            f"{question}\n\n"
            f"Provide a clear, grounded explanation matching the requested level:"
        )

    def run(self, state: AgentState) -> AgentState:
        t_start = time.perf_counter()
        question = state.user_question
        level = state.explanation_level or "simple"
        top_k = state.top_k

        # 1. Retrieve evidence via RAGService
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

        # 2. Insufficient context hard-stop
        if retrieval_status != "success" or not context.strip():
            state.final_answer = (
                "The available reference documents do not contain enough evidence to provide an explanation for this topic. "
                "To maintain medical accuracy, no speculative explanation was generated. Please verify your reference documents."
            )
            state.citations = {
                "is_valid": True,
                "citations_found": [],
                "valid_citations": [],
                "missing_citations": False,
                "notes": "Safe fallback triggered due to insufficient retrieval evidence."
            }
            state.disclaimer = MEDICAL_DISCLAIMER
            state.timings["explanation_agent_total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
            return state

        # 3. Construct Explanation Prompt and Call Gemini
        custom_prompt = self._build_explanation_prompt(question, context, level)

        t_gen_start = time.perf_counter()
        gemini_response = self.gemini_service.generate_answer(
            question=custom_prompt,
            context=context,
            temperature=0.2
        )
        state.timings["generation_time_ms"] = round((time.perf_counter() - t_gen_start) * 1000.0, 2)

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
        state.timings["explanation_agent_total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)

        return state
