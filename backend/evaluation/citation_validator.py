import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import numpy as np

from backend.services.embedding_service import EmbeddingService


@dataclass
class ExtractedClaim:
    """Internal representation of a candidate factual claim segmented from an answer."""
    claim_text: str
    raw_sentence: str
    cited_source_indices: List[int] = field(default_factory=list)
    has_citations: bool = False
    is_supported: bool = False
    support_scores: Dict[int, float] = field(default_factory=dict)
    best_support_score: float = 0.0
    unsupported_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "raw_sentence": self.raw_sentence,
            "cited_source_indices": self.cited_source_indices,
            "has_citations": self.has_citations,
            "is_supported": self.is_supported,
            "support_scores": self.support_scores,
            "best_support_score": round(self.best_support_score, 4),
            "unsupported_reasons": self.unsupported_reasons,
        }


@dataclass
class CitationValidationResult:
    """Structured result of citation validation for an LLM-generated healthcare answer."""
    is_valid: bool
    citations_found: List[int] = field(default_factory=list)
    valid_citations: List[int] = field(default_factory=list)
    invalid_citations: List[int] = field(default_factory=list)
    duplicate_citations: List[int] = field(default_factory=list)
    has_citations: bool = False
    missing_citations: bool = False
    mapped_sources: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Phase 3.3: Grounded citation enforcement & claim validation fields
    claims_checked: int = 0
    claims_supported: int = 0
    claims_unsupported: int = 0
    citation_coverage: float = 1.0
    extracted_claims: List[Dict[str, Any]] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    cleaned_grounded_answer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "citations_found": self.citations_found,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "duplicate_citations": self.duplicate_citations,
            "has_citations": self.has_citations,
            "missing_citations": self.missing_citations,
            "mapped_sources": self.mapped_sources,
            "errors": self.errors,
            "claims_checked": self.claims_checked,
            "claims_supported": self.claims_supported,
            "claims_unsupported": self.claims_unsupported,
            "citation_coverage": round(self.citation_coverage, 4),
            "extracted_claims": self.extracted_claims,
            "unsupported_claims": self.unsupported_claims,
            "cleaned_grounded_answer": self.cleaned_grounded_answer,
        }


class CitationValidator:
    """
    Independently validates citations in generated answers against authoritative retrieved FAISS source metadata.

    Phase 3.3 Hardening:
    - Syntactic validation: Checks citation format [Source N] and source index existence.
    - Semantic support validation: Verifies each factual claim is substantiated by its cited source(s).
    - Hallucination prevention: Detects unsupported medical claims and invalid source references.
    - Deterministic sanitization: Prunes unsupported claims or safely triggers insufficient-evidence fallback.

    SAFETY PRINCIPLE:
    Never blindly trust LLM-generated citations. The underlying FAISS vector metadata
    is the authoritative ground truth for evidence provenance.
    """

    SUPPORT_SIMILARITY_THRESHOLD: float = 0.65

    # Matches full citation bracket, including grouped forms:
    # [Source 1], [1], [Source 1, Source 2], [Source 1, Source 3], [Source 1, Source 2, Source 3], [1, 2, 3]
    CITATION_BRACKET_PATTERN = re.compile(
        r'\[\s*((?:Source\s*(?:#|:)?\s*)?\d+(?:\s*,\s*(?:Source\s*(?:#|:)?\s*)?\d+)*)\s*\]',
        re.IGNORECASE,
    )
    # Legacy alias: first number immediately before ']' (single-citation brackets only)
    CITATION_PATTERN = re.compile(r'\[(?:Source\s*(?:#|:)?\s*)?(\d+)\]', re.IGNORECASE)

    @classmethod
    def extract_citations(cls, text: Optional[str]) -> List[int]:
        """
        Safely extracts all integer citation references from text.
        Supports single and grouped Source tags. Handles None, empty strings,
        and malformed text without raising exceptions.
        """
        if not text or not isinstance(text, str):
            return []

        citations = []
        for match in cls.CITATION_BRACKET_PATTERN.finditer(text):
            for num_str in re.findall(r'\d+', match.group(1)):
                try:
                    citations.append(int(num_str))
                except (ValueError, TypeError):
                    continue
        return citations

    @classmethod
    def strip_invalid_citations(cls, text: Optional[str], invalid_citations: Optional[List[int]]) -> str:
        """
        Removes hallucinated source numbers from the answer while preserving valid citations.

        Grouped tags keep remaining valid numbers, e.g. [Source 1, Source 4] -> [Source 1]
        when 4 is invalid. Brackets with no remaining valid numbers are removed entirely.
        Valid grouped citations are left unchanged when none of their numbers are invalid.
        """
        if not text or not isinstance(text, str):
            return ""
        if not invalid_citations:
            return text

        invalid = set()
        for cit in invalid_citations:
            try:
                invalid.add(int(cit))
            except (ValueError, TypeError):
                continue
        if not invalid:
            return text

        def _replace(match: re.Match) -> str:
            numbers = []
            for num_str in re.findall(r'\d+', match.group(1)):
                try:
                    numbers.append(int(num_str))
                except (ValueError, TypeError):
                    continue
            kept = [n for n in numbers if n not in invalid]
            if not kept:
                return ""
            if kept == numbers:
                return match.group(0)
            if len(kept) == 1:
                return f"[Source {kept[0]}]"
            inner = ", ".join(f"Source {n}" for n in kept)
            return f"[{inner}]"

        return cls.CITATION_BRACKET_PATTERN.sub(_replace, text)

    @classmethod
    def is_structural_or_meta(cls, text: str) -> bool:
        """Identifies non-factual structural elements, headings, disclaimers, or refusals."""
        s = text.strip()
        if not s:
            return True
        # Markdown headings (# Heading, ## Subheading)
        if re.match(r'^#{1,6}\s+', s):
            return True
        # Short headers ending in colon without citations
        if s.endswith(':') and '[' not in s and len(s.split()) <= 8:
            return True
        # Pure bold headers (**Recommendations**)
        if re.match(r'^\*\*[^*]+\*\*$', s) and '[' not in s:
            return True
        # Delimiters
        if re.match(r'^(?:[-=_*~]{3,}|===.*===)$', s):
            return True
        # Disclaimers & standard refusals
        lower = s.lower()
        if any(p in lower for p in [
            "medical disclaimer",
            "always consult a qualified",
            "always consult your doctor",
            "experiencing a medical emergency",
            "relevant medical information could not be found",
            "to prevent unsupported healthcare answers",
            "the available documents do not contain",
            "not enough information",
            "could not be found in the available reference documents",
            "generation was halted"
        ]):
            return True
        return False

    @classmethod
    def extract_claims(cls, answer_text: Optional[str]) -> List[ExtractedClaim]:
        """
        Segments generated answer into discrete candidate factual claims.
        Handles normal sentences, bullet points, numbered lists, and inline citations.
        Excludes structural headings, empty lines, and pure disclaimers.
        """
        if not answer_text or not isinstance(answer_text, str):
            return []

        lines = answer_text.strip().split('\n')
        claims: List[ExtractedClaim] = []

        for line in lines:
            line_str = line.strip()
            if not line_str or cls.is_structural_or_meta(line_str):
                continue

            # Extract paragraph-level citations (e.g. trailing citation for a multi-sentence paragraph)
            p_cits = cls.extract_citations(line_str)

            # Strip list bullet or numbering prefix if present
            bullet_match = re.match(r'^(?:[-*•]|\d+\.)\s+(.*)$', line_str)
            content_to_split = bullet_match.group(1).strip() if bullet_match else line_str

            # Normalize citation position: if citation is after punctuation (e.g. '. [Source 1]'),
            # move citation before punctuation (' [Source 1].') so sentence splitting keeps them together
            normalized_content = re.sub(r'([.!?])\s*(\[[^\]]+\])', r' \2\1', content_to_split)

            # Split sentences by standard sentence punctuation followed by space and capital letter or number
            raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', normalized_content)
            for s in raw_sentences:
                s_clean = s.strip()
                if not s_clean or len(s_clean) < 6 or cls.is_structural_or_meta(s_clean):
                    continue

                # Extract inline citations for this specific sentence
                s_cits = cls.extract_citations(s_clean)
                effective_cits = s_cits if s_cits else p_cits

                # Clean claim text for semantic comparison (strip citation markers & normalize whitespace)
                clean_text = cls.CITATION_BRACKET_PATTERN.sub("", s_clean)
                clean_text = re.sub(r'\s+([,.:;])', r'\1', clean_text)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip(' .:,;')

                if len(clean_text) < 6:
                    continue

                claims.append(ExtractedClaim(
                    claim_text=clean_text,
                    raw_sentence=s_clean,
                    cited_source_indices=effective_cits,
                    has_citations=len(effective_cits) > 0
                ))

        return claims

    @classmethod
    def compute_support_score(cls, claim_text: str, source_text: str) -> float:
        """
        Computes maximum semantic similarity between a claim and a source passage.
        Checks both the overall source passage and individual lines/sentences to prevent
        dilution in long chunks.
        """
        if not claim_text or not source_text:
            return 0.0

        claim_vec = EmbeddingService.embed_query(claim_text)
        if not claim_vec:
            return 0.0

        # Overall source passage similarity
        src_vec = EmbeddingService.embed_query(source_text)
        if not src_vec:
            return 0.0

        scores = [float(np.dot(claim_vec, src_vec))]

        # If source has multiple lines or sentences, also check best matching segment
        segments = [
            seg.strip()
            for seg in re.split(r'(?:\n+|\.\s+)', source_text)
            if len(seg.strip()) > 10
        ]
        if len(segments) > 1:
            for seg in segments[:15]:
                seg_vec = EmbeddingService.embed_query(seg)
                if seg_vec:
                    scores.append(float(np.dot(claim_vec, seg_vec)))

        return max(scores)

    @classmethod
    def check_claim_support(
        cls,
        claim: ExtractedClaim,
        source_map: Dict[int, Dict[str, Any]]
    ) -> bool:
        """
        Validates whether a factual claim is supported by its cited sources.
        Checks cited source existence, medical entity presence, negation contradictions,
        and semantic similarity against cited text.
        """
        # Phase 3.4: Deep medical hallucination & contradiction guard
        from backend.evaluation.hallucination_guard import HallucinationGuard
        guard_detail = HallucinationGuard.inspect_claim(claim, source_map)
        claim.best_support_score = guard_detail.best_support_score
        claim.support_scores = guard_detail.support_scores

        if not guard_detail.is_supported:
            claim.is_supported = False
            for reason in guard_detail.reasons:
                if reason not in claim.unsupported_reasons:
                    claim.unsupported_reasons.append(reason)
            return False

        claim.is_supported = True
        return True

    @classmethod
    def prune_unsupported_claims(
        cls,
        answer_text: Optional[str],
        validation_result: CitationValidationResult,
        safe_fallback_text: Optional[str] = None
    ) -> str:
        """
        Safely strips unsupported claims from answer_text.
        If no supported claims remain, returns safe_fallback_text.
        """
        default_fallback = (
            "Relevant medical information could not be found in the available reference documents. "
            "To prevent unsupported healthcare answers, generation was halted. "
            "Please refine your query or consult authorized clinical guidelines."
        )
        fallback = safe_fallback_text or default_fallback

        if not answer_text or not answer_text.strip():
            return fallback

        # If all checked claims are unsupported, return fallback immediately
        if validation_result.claims_checked > 0 and validation_result.claims_supported == 0:
            return fallback

        # If there are no unsupported claims, return text with invalid citations stripped
        if validation_result.claims_unsupported == 0:
            return cls.strip_invalid_citations(answer_text, validation_result.invalid_citations)

        # Otherwise, remove the raw sentences of unsupported claims
        cleaned = answer_text
        for clm in validation_result.extracted_claims:
            if not clm.get("is_supported", False):
                raw_s = clm.get("raw_sentence", "")
                if raw_s:
                    cleaned = cleaned.replace(raw_s, "")

        # Strip invalid citations from what remains
        cleaned = cls.strip_invalid_citations(cleaned, validation_result.invalid_citations)

        # Clean excess blank lines / spaces
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        # If remaining text has no substantive content, return fallback
        substantive = re.sub(r'\[.*?\]', '', cleaned).strip(' \n\r\t.,;:')
        if len(substantive) < 15:
            return fallback

        return cleaned

    @classmethod
    def validate_citations(
        cls,
        answer_text: Optional[str],
        retrieved_sources: Optional[List[Dict[str, Any]]],
        check_claim_support: bool = False
    ) -> CitationValidationResult:
        """
        Validates citation numbers in `answer_text` against `retrieved_sources`.
        When check_claim_support=True, also performs semantic claim-to-source grounding checks.

        Args:
            answer_text: The generated text containing inline citations (e.g., [Source 1]).
            retrieved_sources: The authoritative list of retrieved sources from FAISS,
                               each containing at least 'source_index' (or 1-indexed by position).
            check_claim_support: Whether to execute semantic claim-to-source grounding validation.

        Returns:
            CitationValidationResult with detailed validation status.
        """
        errors: List[str] = []
        safe_sources = retrieved_sources if isinstance(retrieved_sources, list) else []

        # Build authoritative source index mapping
        source_map: Dict[int, Dict[str, Any]] = {}
        for idx, src in enumerate(safe_sources, start=1):
            if isinstance(src, dict):
                src_num = src.get("source_index", idx)
                try:
                    src_num = int(src_num)
                except (ValueError, TypeError):
                    src_num = idx
                source_map[src_num] = src

        available_source_indices = set(source_map.keys())

        # Extract citations from answer safely
        extracted_citations = cls.extract_citations(answer_text)
        has_citations = len(extracted_citations) > 0

        # Detect duplicates while preserving order
        seen: Set[int] = set()
        duplicates: Set[int] = set()
        for cit in extracted_citations:
            if cit in seen:
                duplicates.add(cit)
            seen.add(cit)
        duplicate_citations = sorted(list(duplicates))

        valid_citations: List[int] = []
        invalid_citations: List[int] = []
        mapped_sources: List[Dict[str, Any]] = []

        for cit in extracted_citations:
            if cit in available_source_indices:
                if cit not in valid_citations:
                    valid_citations.append(cit)
                    mapped_sources.append({
                        "citation_index": cit,
                        "chunk_id": source_map[cit].get("chunk_id"),
                        "document_id": source_map[cit].get("document_id"),
                        "similarity_score": source_map[cit].get("similarity_score")
                    })
            else:
                if cit not in invalid_citations:
                    invalid_citations.append(cit)
                    errors.append(f"Citation [Source {cit}] does not correspond to any retrieved source.")

        # Determine missing citations condition
        missing_citations = False
        safe_text = (answer_text or "").lower()
        is_refusal = any(
            phrase in safe_text for phrase in [
                "do not contain",
                "not enough information",
                "insufficient",
                "could not be found",
                "cannot answer",
                "no relevant",
                "generation was halted"
            ]
        )

        if len(available_source_indices) > 0 and not has_citations:
            if not is_refusal:
                missing_citations = True
                errors.append("Retrieved sources were available but no citations were included in the answer.")

        # Claim-level grounding & semantic support check (Phase 3.3)
        claims_checked = 0
        claims_supported = 0
        claims_unsupported = 0
        citation_coverage = 1.0
        extracted_claim_dicts: List[Dict[str, Any]] = []
        unsupported_claim_texts: List[str] = []

        if check_claim_support and answer_text and not is_refusal:
            claims = cls.extract_claims(answer_text)
            claims_checked = len(claims)

            for clm in claims:
                supported = cls.check_claim_support(clm, source_map)
                extracted_claim_dicts.append(clm.to_dict())

                if supported:
                    claims_supported += 1
                else:
                    claims_unsupported += 1
                    unsupported_claim_texts.append(clm.claim_text)
                    errors.extend(clm.unsupported_reasons)

            if claims_checked > 0:
                citation_coverage = round(claims_supported / claims_checked, 4)
            elif not has_citations and len(available_source_indices) > 0:
                citation_coverage = 0.0

        # Overall validity check:
        # If check_claim_support is True: requires both syntactic and semantic validity
        # If check_claim_support is False: preserves purely syntactic validity for legacy callers
        if check_claim_support:
            is_valid = (
                (len(invalid_citations) == 0) and
                (not missing_citations) and
                (claims_unsupported == 0) and
                (answer_text is not None and bool(answer_text.strip()))
            )
        else:
            is_valid = (
                (len(invalid_citations) == 0) and
                (not missing_citations) and
                (answer_text is not None and bool(answer_text.strip()))
            )

        temp_res = CitationValidationResult(
            is_valid=is_valid,
            citations_found=extracted_citations,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            duplicate_citations=duplicate_citations,
            has_citations=has_citations,
            missing_citations=missing_citations,
            mapped_sources=mapped_sources,
            errors=errors,
            claims_checked=claims_checked,
            claims_supported=claims_supported,
            claims_unsupported=claims_unsupported,
            citation_coverage=citation_coverage,
            extracted_claims=extracted_claim_dicts,
            unsupported_claims=unsupported_claim_texts,
            cleaned_grounded_answer=None
        )

        temp_res.cleaned_grounded_answer = cls.prune_unsupported_claims(answer_text, temp_res)
        return temp_res

    @classmethod
    def validate_grounded_citations(
        cls,
        answer_text: Optional[str],
        retrieved_sources: Optional[List[Dict[str, Any]]]
    ) -> CitationValidationResult:
        """
        Enforces both syntactic citation correctness and deterministic semantic claim grounding.
        Validates that every factual statement is substantiated by its cited source passage.
        """
        return cls.validate_citations(answer_text, retrieved_sources, check_claim_support=True)
