from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from backend.evaluation.citation_validator import CitationValidator, CitationValidationResult
from backend.evaluation.hallucination_guard import HallucinationGuard, HallucinationGuardResult, HallucinationType


@dataclass
class StructuredAnswerEvaluation:
    """Standardized deterministic evaluation result for an LLM clinical answer."""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    claim_support: float
    citation_correctness: float
    citation_completeness: float
    groundedness: float
    unsupported_claims: List[str]
    hallucination_detected: bool
    hallucination_rate: float
    hallucination_types: List[str]
    contradiction_detected: bool
    final_status: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        return self.final_status == "PASS" and not self.hallucination_detected and not self.contradiction_detected

    @property
    def contradictions_detected(self) -> int:
        return 1 if self.contradiction_detected else 0

    @property
    def has_invalid_citations(self) -> bool:
        return bool(self.details.get("invalid_citations"))

    @property
    def invalid_citations(self) -> List[str]:
        raw = self.details.get("invalid_citations", [])
        return [f"[Source {c}]" if isinstance(c, int) else str(c) for c in raw]

    @property
    def unsupported_claims_count(self) -> int:
        return len(self.unsupported_claims)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "claim_support": round(self.claim_support, 4),
            "citation_correctness": round(self.citation_correctness, 4),
            "citation_completeness": round(self.citation_completeness, 4),
            "groundedness": round(self.groundedness, 4),
            "unsupported_claims": self.unsupported_claims,
            "hallucination_detected": self.hallucination_detected,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "hallucination_types": self.hallucination_types,
            "contradiction_detected": self.contradiction_detected,
            "final_status": self.final_status,
            "details": self.details
        }


# Type alias for Phase 5 consistency
AnswerEvaluationResult = StructuredAnswerEvaluation


class AnswerEvaluator:
    """
    Deterministic clinical answer quality and grounding evaluator (Phase 5.4).
    Reuses CitationValidator and HallucinationGuard to measure claim support,
    citation correctness/completeness, groundedness, and contradiction detection.
    """

    @classmethod
    def evaluate_answer(
        cls,
        question: str,
        answer: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        min_groundedness_threshold: float = 0.60,
        retrieved_sources: Optional[List[Dict[str, Any]]] = None,
        answer_text: Optional[str] = None
    ) -> StructuredAnswerEvaluation:
        """
        Evaluates the quality, citation integrity, and factual groundedness of an answer.
        Supports both (answer, sources) and (answer_text, retrieved_sources) argument styles.

        Args:
            question: User clinical query.
            answer: Generated response string.
            sources: List of retrieved evidence source dictionaries.
            min_groundedness_threshold: Minimum groundedness score to achieve PASS status.
            retrieved_sources: Optional alias for sources.
            answer_text: Optional alias for answer.

        Returns:
            StructuredAnswerEvaluation with all computed metrics and final PASS/FAIL status.
        """
        effective_answer = answer if answer is not None else answer_text
        effective_sources = sources if sources is not None else retrieved_sources

        safe_q = question or ""
        safe_ans = (effective_answer or "").strip()
        safe_sources = effective_sources if isinstance(effective_sources, list) else []

        # Handle empty/safe refusal answers
        if not safe_ans:
            return StructuredAnswerEvaluation(
                question=safe_q,
                answer="",
                sources=safe_sources,
                claim_support=0.0,
                citation_correctness=1.0,
                citation_completeness=1.0,
                groundedness=0.0,
                unsupported_claims=[],
                hallucination_detected=False,
                hallucination_rate=0.0,
                hallucination_types=[],
                contradiction_detected=False,
                final_status="FAIL",
                details={"reason": "Empty answer provided."}
            )

        # 1. Run CitationValidator with grounded claim enforcement
        citation_res: CitationValidationResult = CitationValidator.validate_grounded_citations(safe_ans, safe_sources)

        # Citation correctness: valid citations / total citations found (1.0 if no citations)
        if citation_res.citations_found:
            cit_correctness = len(citation_res.valid_citations) / len(citation_res.citations_found)
        else:
            cit_correctness = 1.0

        # Citation completeness: if safe_sources are present and citations are missing -> 0.0
        if safe_sources and not citation_res.has_citations:
            cit_completeness = 0.0
        else:
            cit_completeness = float(citation_res.citation_coverage)

        # Claim support: claims_supported / max(1, claims_checked)
        if citation_res.claims_checked > 0:
            claim_support = float(citation_res.claims_supported) / float(citation_res.claims_checked)
        else:
            claim_support = 1.0 if not citation_res.unsupported_claims else 0.0

        # 2. Run HallucinationGuard
        guard_res: HallucinationGuardResult = HallucinationGuard.guard_answer(safe_ans, safe_sources)

        has_unsupported_citation_claims = bool(citation_res.claims_unsupported > 0 or citation_res.unsupported_claims)
        hallucination_detected = (
            guard_res.hallucinated_claims > 0 or
            not guard_res.is_safe or
            has_unsupported_citation_claims or
            (bool(safe_sources) and not citation_res.has_citations)
        )
        tot_c = max(1, guard_res.total_claims, citation_res.claims_checked)
        unsupported_count = max(guard_res.hallucinated_claims, citation_res.claims_unsupported)
        hallucination_rate = float(unsupported_count) / float(tot_c) if hallucination_detected else 0.0
        hallucination_types = [
            c.hallucination_type.value if hasattr(c.hallucination_type, "value") else str(c.hallucination_type)
            for c in guard_res.claims
            if c.hallucination_type != HallucinationType.NONE
        ]
        if has_unsupported_citation_claims and "UNSUPPORTED_CLAIM" not in hallucination_types:
            hallucination_types.append("UNSUPPORTED_CLAIM")
        if safe_sources and not citation_res.has_citations and "UNCITED_CLAIM" not in hallucination_types:
            hallucination_types.append("UNCITED_CLAIM")

        contradiction_detected = bool(guard_res.contradictions_detected > 0)

        # 3. Handle Safe Fallback / Insufficient Evidence Refusals
        # When reference context is empty or question is unanswerable from sources, a refusal is 100% grounded
        lower_ans = safe_ans.lower()
        refusal_phrases = [
            "do not contain",
            "not enough information",
            "insufficient evidence",
            "could not be found",
            "cannot answer",
            "not mentioned",
            "no relevant"
        ]
        is_safe_refusal = any(p in lower_ans for p in refusal_phrases)
        if is_safe_refusal and not safe_sources:
            return StructuredAnswerEvaluation(
                question=safe_q,
                answer=safe_ans,
                sources=safe_sources,
                claim_support=1.0,
                citation_correctness=1.0,
                citation_completeness=1.0,
                groundedness=1.0,
                unsupported_claims=[],
                hallucination_detected=False,
                hallucination_rate=0.0,
                hallucination_types=[],
                contradiction_detected=False,
                final_status="PASS",
                details={"reason": "Safe refusal on insufficient context."}
            )

        # 4. Compute Composite Groundedness Score
        # Groundedness balances claim support, citation correctness, and absence of hallucinations
        groundedness = (
            (claim_support * 0.40) +
            (cit_correctness * 0.30) +
            (max(0.0, 1.0 - hallucination_rate) * 0.30)
        )
        if contradiction_detected:
            groundedness = max(0.0, groundedness - 0.40)

        # Pass condition:
        # 1. No invalid citations
        # 2. Groundedness meets threshold
        # 3. No critical contradiction detected
        passed = (
            citation_res.is_valid and
            not contradiction_detected and
            groundedness >= min_groundedness_threshold and
            not (hallucination_detected and hallucination_rate >= 0.50)
        )

        return StructuredAnswerEvaluation(
            question=safe_q,
            answer=safe_ans,
            sources=safe_sources,
            claim_support=claim_support,
            citation_correctness=cit_correctness,
            citation_completeness=cit_completeness,
            groundedness=groundedness,
            unsupported_claims=list(citation_res.unsupported_claims),
            hallucination_detected=hallucination_detected,
            hallucination_rate=hallucination_rate,
            hallucination_types=hallucination_types,
            contradiction_detected=contradiction_detected,
            final_status="PASS" if passed else "FAIL",
            details={
                "claims_checked": citation_res.claims_checked,
                "claims_supported": citation_res.claims_supported,
                "claims_unsupported": citation_res.claims_unsupported,
                "citation_errors": citation_res.errors,
                "citations_found": citation_res.citations_found,
                "valid_citations": citation_res.valid_citations,
                "invalid_citations": citation_res.invalid_citations
            }
        )
