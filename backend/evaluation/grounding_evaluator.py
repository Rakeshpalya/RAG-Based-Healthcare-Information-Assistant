import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from backend.evaluation.citation_validator import CitationValidator, CitationValidationResult


@dataclass
class GroundingEvaluationResult:
    """Structured result of deterministic grounding evaluation."""
    is_grounded: bool
    answer_present: bool
    context_present: bool
    citation_validation: CitationValidationResult
    unsupported_fallback_verified: bool
    lexical_overlap_ratio: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_grounded": self.is_grounded,
            "answer_present": self.answer_present,
            "context_present": self.context_present,
            "citation_validation": self.citation_validation.to_dict(),
            "unsupported_fallback_verified": self.unsupported_fallback_verified,
            "lexical_overlap_ratio": self.lexical_overlap_ratio,
            "notes": self.notes
        }


class GroundingEvaluator:
    """
    Lightweight deterministic evaluator for development-time grounding and hallucination checks.

    IMPORTANT CLINICAL & EVALUATION DISCLAIMER:
    Deterministic string matching and citation parsing DO NOT prove factual faithfulness or clinical
    accuracy. True semantic verification requires expert human clinical review and downstream LLM-as-a-judge
    benchmarking (e.g. RAGAS, TruLens). Do NOT claim clinical correctness from this evaluator.
    """

    @classmethod
    def evaluate_grounding(
        cls,
        question: str,
        answer: Optional[str],
        retrieval_status: str,
        retrieved_sources: Optional[List[Dict[str, Any]]]
    ) -> GroundingEvaluationResult:
        """
        Runs deterministic checks on an answer to verify grounding and safety behavior.

        Checks:
        1. Answer exists and is non-empty.
        2. Context exists if retrieval_status is 'success'.
        3. If retrieval_status == 'no_relevant_context', verifies that Gemini was bypassed
           and safe fallback text is returned.
        4. Citation references point only to retrieved source indices.
        5. Computes basic lexical token overlap between answer and retrieved text chunks.
        """
        notes: List[str] = []
        sources = retrieved_sources if isinstance(retrieved_sources, list) else []
        safe_answer = (answer or "").strip()
        answer_present = len(safe_answer) > 0
        context_present = len(sources) > 0

        # Scenario 1: Unsupported context / safe fallback check
        unsupported_fallback_verified = False
        if retrieval_status in ("no_relevant_context", "empty_query"):
            # Check that answer safely indicates lack of information or is empty
            lower_answer = safe_answer.lower()
            safe_phrases = [
                "do not contain",
                "not enough information",
                "insufficient",
                "could not be found",
                "cannot answer",
                "no relevant"
            ]
            if not answer_present or any(p in lower_answer for p in safe_phrases):
                unsupported_fallback_verified = True
                notes.append(f"Safe fallback properly triggered for status '{retrieval_status}'.")
            else:
                notes.append(f"Warning: Answer generated despite retrieval_status '{retrieval_status}'.")

            citation_res = CitationValidator.validate_citations(safe_answer, sources)
            return GroundingEvaluationResult(
                is_grounded=unsupported_fallback_verified,
                answer_present=answer_present,
                context_present=context_present,
                citation_validation=citation_res,
                unsupported_fallback_verified=unsupported_fallback_verified,
                lexical_overlap_ratio=0.0,
                notes=notes
            )

        # Scenario 2: Normal retrieval with expected sources
        citation_res = CitationValidator.validate_citations(safe_answer, sources)
        if not citation_res.is_valid:
            notes.extend(citation_res.errors)

        # Lexical overlap computation (heuristic token matching)
        lexical_ratio = 0.0
        if context_present and answer_present:
            context_text = " ".join([s.get("text", "") for s in sources]).lower()
            context_words = set(re.findall(r'\b[a-z]{4,}\b', context_text))
            answer_words = set(re.findall(r'\b[a-z]{4,}\b', safe_answer.lower()))

            if answer_words and context_words:
                overlap = answer_words.intersection(context_words)
                lexical_ratio = len(overlap) / len(answer_words)

        is_grounded = (
            answer_present and
            context_present and
            citation_res.is_valid and
            len(citation_res.invalid_citations) == 0
        )

        if is_grounded:
            notes.append("Answer passed deterministic grounding and citation checks.")

        return GroundingEvaluationResult(
            is_grounded=is_grounded,
            answer_present=answer_present,
            context_present=context_present,
            citation_validation=citation_res,
            unsupported_fallback_verified=False,
            lexical_overlap_ratio=lexical_ratio,
            notes=notes
        )
