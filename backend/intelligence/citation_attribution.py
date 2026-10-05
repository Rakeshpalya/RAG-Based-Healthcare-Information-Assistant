"""
Clinical Citation & Attribution Engine for AI-Healthcare-Agent (Phase 6.5).

Implements claim-level evidence attribution, citation bounds validation,
spoofing and prompt injection defense, medical contradiction checking,
and deterministic answer sanitization.
"""

import re
import time
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

from backend.intelligence.citation_models import (
    CitationVerificationStatus,
    ClinicalClaimType,
    AttributedEvidenceSpan,
    ClinicalClaimAttribution,
    CitationAttributionReport,
)

logger = logging.getLogger(__name__)


class ClinicalCitationAttributionEngine:
    """
    Production-grade Clinical Citation & Attribution Engine.

    Guarantees:
    1. Claim-level attribution: Maps each factual claim to its exact contributing evidence passages.
    2. Citation verification: Verifies syntactic existence, bounds, and semantic support.
    3. Anti-spoofing defense: Neutralizes out-of-bounds numbers, fabricated names, and prompt injection brackets.
    4. Conservative sanitization: Prunes unsupported claims or safely falls back if evidence is compromised.
    5. Pure Python speed: Sub-millisecond local execution without secondary LLM round-trips.
    """

    # Matches full citation bracket, including grouped forms:
    # [Source 1], [1], [Source 1, Source 2], [Source 1, Source 3], [1, 2]
    CITATION_BRACKET_PATTERN = re.compile(
        r'\[\s*((?:Source\s*(?:#|:)?\s*)?\d+(?:\s*,\s*(?:Source\s*(?:#|:)?\s*)?\d+)*)\s*\]',
        re.IGNORECASE,
    )

    # Legacy / single citation tag pattern
    SINGLE_CITATION_PATTERN = re.compile(
        r'\[(?:Source\s*(?:#|:)?\s*)?(\d+)\]',
        re.IGNORECASE
    )

    # Pattern detecting potentially spoofed / non-integer citation tags:
    # e.g. [Source OVERRIDE], [Source SYSTEM_OVERRIDE: ...], [Source CDC: 2024], [Source NONE]
    SPOOFED_CITATION_PATTERN = re.compile(
        r'\[\s*Source\b([^\]]*)\]',
        re.IGNORECASE
    )

    # Medical assertion patterns for claim classification
    DOSAGE_PATTERN = re.compile(
        r'\b(?:\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?|tablets?|capsules?)|'
        r'once\s+daily|twice\s+daily|bid|tid|qid|daily|titrat(?:e|ion)|starting\s+dose|'
        r'initial\s+dose|maintenance\s+dose|maximum\s+dose)\b',
        re.IGNORECASE
    )

    CONTRAINDICATION_PATTERN = re.compile(
        r'\b(?:contraindicat(?:ed|ion)|adverse\s+(?:effects?|reactions?)|side\s+effects?|'
        r'black\s*box\s*warning|do\s+not\s+(?:use|administer|take)|toxicity|'
        r'risk\s+of|precaution|warning|allergic\s+reaction)\b',
        re.IGNORECASE
    )

    RECOMMENDATION_PATTERN = re.compile(
        r'\b(?:first-line|second-line|recommended|recommendation|should\s+be\s+(?:prescribed|used|initiated)|'
        r'indicated\s+for|preferred\s+agent|guidelines?\s+recommend|therapy\s+includes)\b',
        re.IGNORECASE
    )

    COMPARISON_PATTERN = re.compile(
        r'\b(?:compared\s+to|in\s+contrast\s+to|versus|vs\.?|superior\s+to|inferior\s+to|'
        r'more\s+effective\s+than|less\s+effective\s+than|higher\s+risk\s+than|lower\s+risk\s+than)\b',
        re.IGNORECASE
    )

    LIMITATION_PATTERN = re.compile(
        r'\b(?:does\s+not\s+(?:state|mention|specify|explicitly)|not\s+enough\s+information|'
        r'insufficient\s+evidence|consult\s+(?:your|a)\s+(?:doctor|physician|healthcare)|'
        r'disclaimer|evidence\s+is\s+inconclusive|cannot\s+be\s+determined)\b',
        re.IGNORECASE
    )

    @classmethod
    def extract_citations(cls, text: Optional[str]) -> List[int]:
        """
        Safely extracts all integer citation indices from text.
        Supports single and grouped Source tags.
        """
        if not text or not isinstance(text, str):
            return []

        citations: List[int] = []
        for match in cls.CITATION_BRACKET_PATTERN.finditer(text):
            for num_str in re.findall(r'\d+', match.group(1)):
                try:
                    citations.append(int(num_str))
                except (ValueError, TypeError):
                    continue
        return citations

    @classmethod
    def detect_spoofed_citations(cls, text: Optional[str]) -> List[str]:
        """
        Identifies non-integer, fabricated, or adversarial source citations (e.g. [Source OVERRIDE],
        [Source CDC: 2024], [Source SYSTEM_OVERRIDE: ...]).
        """
        if not text or not isinstance(text, str):
            return []

        spoofed = []
        for match in cls.SPOOFED_CITATION_PATTERN.finditer(text):
            inner = match.group(1).strip()
            tokens = re.findall(r'[A-Za-z0-9_]+', inner)
            # A valid citation bracket only contains digits (or the word "Source" in grouped tags)
            is_spoofed = False
            if not tokens:
                is_spoofed = True
            else:
                for tok in tokens:
                    if tok.lower() != "source" and not tok.isdigit():
                        is_spoofed = True
                        break
            if is_spoofed:
                spoofed.append(match.group(0))
        return spoofed

    @classmethod
    def strip_spoofed_and_invalid_citations(
        cls,
        text: Optional[str],
        invalid_citations: Optional[List[int]] = None,
        spoofed_tags: Optional[List[str]] = None
    ) -> str:
        """
        Removes hallucinated and spoofed source brackets from text while preserving valid citations.
        Grouped tags keep remaining valid numbers (e.g. [Source 1, Source 99] -> [Source 1]).
        """
        if not text or not isinstance(text, str):
            return ""

        cleaned = text

        # Strip explicit spoofed tags (e.g. [Source OVERRIDE] -> "")
        if spoofed_tags:
            for s_tag in spoofed_tags:
                cleaned = cleaned.replace(s_tag, "")

        # Also strip any detected spoofed brackets
        for s_match in cls.detect_spoofed_citations(cleaned):
            cleaned = cleaned.replace(s_match, "")

        # Strip numeric out-of-bounds citations
        invalid_set = set(int(c) for c in (invalid_citations or []))
        if invalid_set:
            def _replace_grouped(match: re.Match) -> str:
                numbers = []
                for num_str in re.findall(r'\d+', match.group(1)):
                    try:
                        numbers.append(int(num_str))
                    except (ValueError, TypeError):
                        continue
                kept = [n for n in numbers if n not in invalid_set]
                if not kept:
                    return ""
                if kept == numbers:
                    return match.group(0)
                if len(kept) == 1:
                    return f"[Source {kept[0]}]"
                inner = ", ".join(f"Source {n}" for n in kept)
                return f"[{inner}]"

            cleaned = cls.CITATION_BRACKET_PATTERN.sub(_replace_grouped, cleaned)

        # Normalize whitespace after bracket removals
        cleaned = re.sub(r'[ \t]+([.,;:!?])', r'\1', cleaned)
        cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
        return cleaned.strip()

    @classmethod
    def is_structural_or_meta(cls, text: str) -> bool:
        """Determines if a segmented text is structural, header, or meta disclaimer."""
        s = text.strip()
        if not s:
            return True
        # Markdown headings (# Heading, ## Subheading)
        if re.match(r'^#{1,6}\s+', s):
            return True
        # Short headers ending in colon without citations
        if s.endswith(':') and '[' not in s and len(s.split()) <= 8:
            return True
        # Pure bold headers (**Section Title**)
        if re.match(r'^\*\*[^*]+\*\*$', s) and '[' not in s:
            return True
        # Markdown horizontal delimiters
        if re.match(r'^(?:[-=_*~]{3,}|===.*===)$', s):
            return True
        # Disclaimer or standard refusal phrases
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
            "generation was halted",
            "please consult clinical guidelines"
        ]):
            return True
        return False

    @classmethod
    def classify_claim_type(cls, text: str) -> ClinicalClaimType:
        """Classifies a claim into a semantic clinical assertion type."""
        if cls.is_structural_or_meta(text):
            if any(p in text.lower() for p in ("disclaimer", "not enough information", "could not be found")):
                return ClinicalClaimType.LIMITATION_OR_DISCLAIMER
            return ClinicalClaimType.STRUCTURAL

        if cls.LIMITATION_PATTERN.search(text):
            return ClinicalClaimType.LIMITATION_OR_DISCLAIMER

        if cls.DOSAGE_PATTERN.search(text):
            return ClinicalClaimType.DOSAGE_INSTRUCTION

        if cls.CONTRAINDICATION_PATTERN.search(text):
            return ClinicalClaimType.CONTRAINDICATION_OR_WARNING

        if cls.COMPARISON_PATTERN.search(text):
            return ClinicalClaimType.COMPARATIVE_CLAIM

        if cls.RECOMMENDATION_PATTERN.search(text):
            return ClinicalClaimType.TREATMENT_RECOMMENDATION

        return ClinicalClaimType.FACTUAL_MEDICAL

    @classmethod
    def segment_claims(cls, answer_text: Optional[str]) -> List[ClinicalClaimAttribution]:
        """
        Segments a synthesized clinical answer into individual candidate assertions.
        Preserves inline citations with their corresponding sentence.
        """
        if not answer_text or not isinstance(answer_text, str):
            return []

        lines = answer_text.strip().split('\n')
        claims: List[ClinicalClaimAttribution] = []
        claim_counter = 1

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if entire line is structural
            if cls.is_structural_or_meta(line_str) and not re.search(r'\[Source', line_str, re.I):
                claims.append(ClinicalClaimAttribution(
                    claim_id=f"CLM_{claim_counter:03d}",
                    claim_text=line_str,
                    raw_sentence=line_str,
                    claim_type=ClinicalClaimType.STRUCTURAL,
                    cited_source_indices=[],
                    verification_status=CitationVerificationStatus.STRUCTURAL_OR_DISCLAIMER,
                    best_support_score=1.0,
                    is_supported=True
                ))
                claim_counter += 1
                continue

            # Paragraph-level citations
            p_cits = cls.extract_citations(line_str)

            # Strip list bullet or numbering prefix if present
            bullet_match = re.match(r'^(?:[-*•]|\d+\.)\s+(.*)$', line_str)
            content_to_split = bullet_match.group(1).strip() if bullet_match else line_str

            # Normalize citation position: if citation is after punctuation (e.g. '. [Source 1]'),
            # move citation before punctuation (' [Source 1].')
            normalized_content = re.sub(r'([.!?])\s*(\[[^\]]+\])', r' \2\1', content_to_split)

            # Split sentences by standard sentence punctuation followed by space and capital letter or number,
            # or after a citation bracket followed by a conjunction (e.g. [Source 1] but ...)
            raw_sentences = re.split(
                r'(?:(?<=[.!?])\s+(?=[A-Z0-9])|(?<=\])\s*(?:,|;)?\s+(?=(?:but|however|whereas|while)\b))',
                normalized_content,
                flags=re.IGNORECASE
            )
            for s in raw_sentences:
                s_clean = s.strip()
                if not s_clean or len(s_clean) < 6:
                    continue

                s_cits = cls.extract_citations(s_clean)
                effective_cits = s_cits if s_cits else (p_cits if len(raw_sentences) == 1 else [])

                # Clean claim text (strip citation brackets)
                clean_text = cls.CITATION_BRACKET_PATTERN.sub("", s_clean)
                clean_text = cls.SPOOFED_CITATION_PATTERN.sub("", clean_text)
                clean_text = re.sub(r'\s+([,.:;])', r'\1', clean_text)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip(' .:,;')

                if len(clean_text) < 5:
                    continue

                claim_type = cls.classify_claim_type(s_clean)

                claims.append(ClinicalClaimAttribution(
                    claim_id=f"CLM_{claim_counter:03d}",
                    claim_text=clean_text,
                    raw_sentence=s_clean,
                    claim_type=claim_type,
                    cited_source_indices=effective_cits,
                    verification_status=CitationVerificationStatus.UNSPECIFIED_CITATION,
                    best_support_score=0.0,
                    is_supported=False
                ))
                claim_counter += 1

        return claims

    @classmethod
    def compute_lexical_similarity(cls, text_a: str, text_b: str) -> float:
        """
        Fast token Jaccard similarity between claim words and source passage.
        """
        tokens_a = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', text_a.lower()))
        tokens_b = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', text_b.lower()))

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        return round(intersection / union if union > 0 else 0.0, 4)

    @classmethod
    def check_dosage_contradiction(cls, claim_text: str, source_text: str) -> Tuple[bool, Optional[str]]:
        """
        Validates that specific numerical dosage quantities in the claim exist in the source passage.
        """
        claim_doses = re.findall(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)\b', claim_text, re.I)
        if not claim_doses:
            return False, None

        source_lower = source_text.lower()
        for num, unit in claim_doses:
            exact_dose = f"{num} {unit}".lower()
            joined_dose = f"{num}{unit}".lower()
            if exact_dose not in source_lower and joined_dose not in source_lower:
                return True, f"Dosage quantity '{num} {unit}' is not substantiated by cited evidence."

        return False, None

    @classmethod
    def check_negation_contradiction(cls, claim_text: str, source_text: str) -> Tuple[bool, Optional[str]]:
        """
        Detects direct medical negation contradictions (e.g. claim asserts contraindicated when source says indicated).
        """
        c_lower = claim_text.lower()
        s_lower = source_text.lower()

        # Contraindication contradiction
        claim_contra = bool(re.search(r'\b(?:contraindicated|must\s+not\s+be\s+used|strictly\s+prohibited)\b', c_lower))
        source_contra = bool(re.search(r'\b(?:contraindicated|must\s+not\s+be\s+used|strictly\s+prohibited)\b', s_lower))

        if claim_contra and not source_contra and re.search(r'\b(?:indicated|recommended|prescribed)\b', s_lower):
            return True, "Claim asserts contraindication contrary to cited source indication."

        if not claim_contra and source_contra and re.search(r'\b(?:recommended|first-line|indicated)\b', c_lower):
            return True, "Claim recommends therapy which is contraindicated in cited source."

        return False, None

    @classmethod
    def verify_claim(
        cls,
        claim: ClinicalClaimAttribution,
        source_map: Dict[int, Dict[str, Any]],
        use_embeddings: bool = False
    ) -> None:
        """
        Verifies an individual claim against its cited sources in source_map.
        """
        # If structural or limitation, exempt from citation verification
        if claim.claim_type in (ClinicalClaimType.STRUCTURAL, ClinicalClaimType.LIMITATION_OR_DISCLAIMER):
            claim.verification_status = CitationVerificationStatus.STRUCTURAL_OR_DISCLAIMER
            claim.is_supported = True
            claim.best_support_score = 1.0
            return

        # If factual claim has no citations
        if not claim.cited_source_indices:
            claim.verification_status = CitationVerificationStatus.UNSPECIFIED_CITATION
            claim.is_supported = False
            claim.best_support_score = 0.0
            claim.unsupported_reasons.append("Clinical assertion lacks required inline citation [Source N].")
            return

        # Check each cited source
        spans: List[AttributedEvidenceSpan] = []
        scores: List[float] = []
        has_invalid_idx = False

        for idx in claim.cited_source_indices:
            if idx not in source_map:
                has_invalid_idx = True
                claim.unsupported_reasons.append(f"Citation [Source {idx}] does not correspond to any retrieved source.")
                continue

            src = source_map[idx]
            src_text = src.get("text", "")
            src_doc = src.get("document_name", src.get("filename", "Unknown Document"))
            src_doc_id = src.get("document_id", "")
            src_chunk_id = src.get("chunk_id", "")
            src_page = src.get("page_number")
            sim_score = float(src.get("similarity_score", 0.0))

            # Dosage contradiction check
            dosage_conflict, dosage_reason = cls.check_dosage_contradiction(claim.claim_text, src_text)
            if dosage_conflict and dosage_reason:
                claim.unsupported_reasons.append(dosage_reason)

            # Negation contradiction check
            neg_conflict, neg_reason = cls.check_negation_contradiction(claim.claim_text, src_text)
            if neg_conflict and neg_reason:
                claim.unsupported_reasons.append(neg_reason)

            # Compute support score
            support_score = cls.compute_lexical_similarity(claim.claim_text, src_text)

            # Semantic embedding score if enabled
            if use_embeddings:
                try:
                    from backend.services.embedding_service import EmbeddingService
                    import numpy as np
                    clm_vec = EmbeddingService.embed_query(claim.claim_text)
                    src_vec = EmbeddingService.embed_query(src_text[:500])
                    if clm_vec and src_vec:
                        sem_score = float(np.dot(clm_vec, src_vec))
                        support_score = max(support_score, sem_score)
                except Exception:
                    pass

            scores.append(support_score)

            # Extract snippet
            passage_snippet = src_text[:200] + ("..." if len(src_text) > 200 else "")

            # Matched medical entities
            claim_tokens = set(re.findall(r'\b[a-zA-Z]{4,}\b', claim.claim_text.lower()))
            source_tokens = set(re.findall(r'\b[a-zA-Z]{4,}\b', src_text.lower()))
            matched_entities = sorted(list(claim_tokens & source_tokens))[:8]

            spans.append(AttributedEvidenceSpan(
                source_index=idx,
                chunk_id=src_chunk_id,
                document_id=src_doc_id,
                document_name=src_doc,
                page_number=src_page,
                passage_snippet=passage_snippet,
                similarity_score=sim_score,
                support_score=support_score,
                matched_entities=matched_entities
            ))

        claim.attributed_sources = spans
        best_score = max(scores) if scores else 0.0
        claim.best_support_score = best_score

        if has_invalid_idx:
            clinical_contradictions = [
                r for r in claim.unsupported_reasons
                if not r.startswith("Citation [Source ")
            ]
            has_substantive_support = (best_score >= 0.05 or any(len(s.matched_entities) > 0 for s in spans))
            if spans and not clinical_contradictions and has_substantive_support:
                claim.verification_status = CitationVerificationStatus.PARTIALLY_VERIFIED
                claim.is_supported = True
            else:
                claim.verification_status = CitationVerificationStatus.INVALID_SOURCE
                claim.is_supported = False
        elif claim.unsupported_reasons:
            claim.verification_status = CitationVerificationStatus.UNSUPPORTED
            claim.is_supported = False
        elif best_score >= 0.05 or any(len(s.matched_entities) > 0 for s in spans):
            # When matched to valid retrieved source with support or overlapping entities
            claim.verification_status = CitationVerificationStatus.VERIFIED
            claim.is_supported = True
        else:
            claim.verification_status = CitationVerificationStatus.UNSUPPORTED
            claim.is_supported = False
            claim.unsupported_reasons.append("Low semantic/lexical alignment with cited source passage.")

    @classmethod
    def prune_unsupported_claims(
        cls,
        answer_text: Optional[str],
        report: CitationAttributionReport,
        fallback_text: Optional[str] = None
    ) -> str:
        """
        Deterministically sanitizes the answer text:
        1. If all factual claims are unsupported, returns safe fallback.
        2. If partial claims are supported, removes sentences corresponding to unsupported claims.
        3. Strips invalid and spoofed citation brackets.
        """
        default_fallback = (
            "Relevant medical information could not be found in the available reference documents. "
            "To prevent unsupported healthcare answers, generation was halted. "
            "Please refine your query or consult authorized clinical guidelines."
        )
        safe_fallback = fallback_text or default_fallback

        if not answer_text or not answer_text.strip():
            return safe_fallback

        # If there are factual claims and 0 of them are verified -> halt immediately
        if report.factual_claims_count > 0 and report.verified_claims_count == 0:
            return safe_fallback

        cleaned = answer_text

        # Remove raw sentences for unsupported claims
        for clm in report.claims:
            if not clm.is_supported and clm.raw_sentence:
                cleaned = cleaned.replace(clm.raw_sentence, "")

        # Strip invalid and spoofed brackets
        cleaned = cls.strip_spoofed_and_invalid_citations(
            text=cleaned,
            invalid_citations=report.invalid_citations,
            spoofed_tags=report.spoofed_citation_tags
        )

        # Normalize remaining whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

        # If remaining text has no substantive medical content, return fallback
        substantive = re.sub(r'\[.*?\]', '', cleaned).strip(' \n\r\t.,;:')
        if len(substantive) < 15:
            return safe_fallback

        return cleaned

    @classmethod
    def attribute_and_validate(
        cls,
        answer_text: Optional[str],
        retrieved_sources: Optional[List[Dict[str, Any]]],
        use_embeddings: bool = False
    ) -> CitationAttributionReport:
        """
        End-to-end citation verification and evidence attribution audit.

        Args:
            answer_text: Synthesized clinical answer text.
            retrieved_sources: Authoritative source metadata list.
            use_embeddings: Whether to compute deep semantic embedding cosine scores.

        Returns:
            CitationAttributionReport with complete claim attribution breakdown.
        """
        t0 = time.perf_counter()

        safe_sources = retrieved_sources if isinstance(retrieved_sources, list) else []

        # Build authoritative source mapping
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

        # Extract citations & detect spoofing
        extracted_citations = cls.extract_citations(answer_text)
        spoofed_tags = cls.detect_spoofed_citations(answer_text)
        spoofed_detected = len(spoofed_tags) > 0

        # Categorize citation numbers
        valid_citations: List[int] = []
        invalid_citations: List[int] = []
        seen: Set[int] = set()
        duplicates: Set[int] = set()

        for cit in extracted_citations:
            if cit in seen:
                duplicates.add(cit)
            seen.add(cit)

            if cit in available_source_indices:
                if cit not in valid_citations:
                    valid_citations.append(cit)
            else:
                if cit not in invalid_citations:
                    invalid_citations.append(cit)

        # Segment claims
        claims = cls.segment_claims(answer_text)

        total_claims_count = len(claims)
        factual_claims_count = 0
        verified_claims_count = 0
        unsupported_claims_count = 0
        unsupported_claim_texts: List[str] = []

        for clm in claims:
            cls.verify_claim(clm, source_map, use_embeddings=use_embeddings)

            if clm.claim_type not in (ClinicalClaimType.STRUCTURAL, ClinicalClaimType.LIMITATION_OR_DISCLAIMER):
                factual_claims_count += 1
                if clm.is_supported:
                    verified_claims_count += 1
                else:
                    unsupported_claims_count += 1
                    unsupported_claim_texts.append(clm.claim_text)

        # Precision & Coverage calculations
        total_found = len(extracted_citations)
        citation_precision = round(len(valid_citations) / total_found, 4) if total_found > 0 else 1.0

        if factual_claims_count > 0:
            claim_attribution_coverage = round(verified_claims_count / factual_claims_count, 4)
        else:
            claim_attribution_coverage = 1.0

        is_valid = (
            len(invalid_citations) == 0 and
            not spoofed_detected and
            unsupported_claims_count == 0 and
            bool(answer_text and answer_text.strip())
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        report = CitationAttributionReport(
            is_valid=is_valid,
            claims=claims,
            total_claims_count=total_claims_count,
            factual_claims_count=factual_claims_count,
            verified_claims_count=verified_claims_count,
            unsupported_claims_count=unsupported_claims_count,
            citations_found=extracted_citations,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            duplicate_citations=sorted(list(duplicates)),
            citation_precision=citation_precision,
            claim_attribution_coverage=claim_attribution_coverage,
            spoofed_citations_detected=spoofed_detected,
            spoofed_citation_tags=spoofed_tags,
            unsupported_claims=unsupported_claim_texts,
            cleaned_attributed_answer=None,
            latency_ms=elapsed_ms,
            metadata={
                "available_sources_count": len(available_source_indices),
                "spoofed_tags_count": len(spoofed_tags)
            }
        )

        # Produce pruned & sanitized answer text
        report.cleaned_attributed_answer = cls.prune_unsupported_claims(answer_text, report)

        return report
