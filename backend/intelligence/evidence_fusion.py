"""
Clinical Evidence Fusion & Multi-Document Reasoning Engine (Phase 6.3).

Combines evidence retrieved across multiple documents, detects clinical contradictions,
evaluates inquiry aspect coverage, prunes near-duplicates, balances document diversity,
and constructs provenance-tracked clinical context for generation.
"""

import re
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

from backend.intelligence.query_plan import (
    QueryPlan,
    ChunkWeightingStrategy,
    MultiDocumentStrategy,
    GenerationStrategy
)
from backend.intelligence.evidence_models import (
    ConflictType,
    ConflictSeverity,
    CoverageStatus,
    EvidenceConflict,
    EvidenceCoverage,
    FusedEvidenceChunk,
    FusedContextResult
)

logger = logging.getLogger(__name__)


class ClinicalEvidenceFusionEngine:
    """
    Deterministic clinical evidence fusion, deduplication, conflict detection,
    and multi-document reasoning engine.
    """

    # -------------------------------------------------------------------------
    # CLINICAL ENTITIES & PATTERNS FOR CONFLICT DETECTION
    # -------------------------------------------------------------------------

    KNOWN_DRUGS: Set[str] = {
        "metformin", "lisinopril", "atorvastatin", "amlodipine", "losartan",
        "albuterol", "levothyroxine", "omeprazole", "amoxicillin",
        "hydrochlorothiazide", "ibuprofen", "acetaminophen", "aspirin",
        "warfarin", "apixaban", "clopidogrel", "gabapentin", "sertraline",
        "metoprolol", "furosemide", "prednisone", "simvastatin", "insulin"
    }

    KNOWN_CONDITIONS: Set[str] = {
        "hypertension", "diabetes", "type 2 diabetes", "type 1 diabetes",
        "asthma", "heart failure", "cardiovascular disease", "stroke",
        "chronic kidney disease", "copd", "pneumonia", "myocardial infarction",
        "atrial fibrillation", "rheumatoid arthritis", "hypothyroidism"
    }

    @staticmethod
    def _extract_doc_name(chunk: Dict[str, Any]) -> str:
        """Extracts human-readable document filename or identifier."""
        meta = chunk.get("metadata", {})
        fname = meta.get("filename") or chunk.get("filename")
        if fname and str(fname).strip():
            return str(fname).strip()
        doc_id = chunk.get("document_id") or meta.get("document_id")
        if doc_id and str(doc_id).strip():
            return str(doc_id).strip()
        return "UNKNOWN_DOCUMENT"

    @staticmethod
    def _extract_doc_id(chunk: Dict[str, Any]) -> str:
        """Extracts stable document unique ID."""
        doc_id = chunk.get("document_id") or chunk.get("metadata", {}).get("document_id")
        if doc_id and str(doc_id).strip():
            return str(doc_id).strip()
        fname = chunk.get("metadata", {}).get("filename")
        if fname and str(fname).strip():
            return str(fname).strip()
        return "UNKNOWN_DOCUMENT"

    # -------------------------------------------------------------------------
    # 1. DEDUPLICATION & NEAR-DUPLICATE PRUNING
    # -------------------------------------------------------------------------

    @classmethod
    def deduplicate_evidence(
        cls,
        chunks: List[Dict[str, Any]],
        jaccard_threshold: float = 0.85
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Removes exact duplicates (same doc_id + chunk_id or content hash)
        and near-duplicate passages across documents/pages using token set overlap.

        Returns:
            (deduped_chunks, num_pruned)
        """
        if not chunks:
            return [], 0

        # Sort descending by similarity score so highest-quality chunk is retained
        sorted_chunks = sorted(
            chunks,
            key=lambda c: float(c.get("similarity_score", 0.0)),
            reverse=True
        )

        survived: List[Dict[str, Any]] = []
        seen_exact_hashes: Set[str] = set()
        seen_token_sets: List[Tuple[Set[str], Dict[str, Any]]] = []
        pruned_count = 0

        for chunk in sorted_chunks:
            text = str(chunk.get("text", "")).strip()
            if not text:
                pruned_count += 1
                continue

            # Exact normalized content hash
            norm_text = " ".join(text.lower().split())
            c_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
            doc_id = cls._extract_doc_id(chunk)
            chunk_id = str(chunk.get("chunk_id", ""))
            exact_key = f"{doc_id}:{chunk_id}:{c_hash}"

            if exact_key in seen_exact_hashes or c_hash in seen_exact_hashes:
                pruned_count += 1
                continue

            # Token-level Jaccard similarity for near-duplicates
            tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", norm_text))
            is_near_dup = False
            if tokens:
                for existing_tokens, _ in seen_token_sets:
                    intersection = len(tokens & existing_tokens)
                    union = len(tokens | existing_tokens)
                    if union > 0 and (intersection / union) >= jaccard_threshold:
                        is_near_dup = True
                        break

            if is_near_dup:
                pruned_count += 1
                continue

            seen_exact_hashes.add(exact_key)
            seen_exact_hashes.add(c_hash)
            if tokens:
                seen_token_sets.append((tokens, chunk))
            survived.append(chunk)

        return survived, pruned_count

    # -------------------------------------------------------------------------
    # 2. EVIDENCE RANKING & DIVERSITY
    # -------------------------------------------------------------------------

    @classmethod
    def rank_and_balance_evidence(
        cls,
        chunks: List[Dict[str, Any]],
        query_plan: Optional[QueryPlan] = None,
        top_k: int = 5
    ) -> List[FusedEvidenceChunk]:
        """
        Ranks retrieved chunks using similarity score + clinical weighting boost,
        applying multi-document diversity constraints to prevent single-source monopolization.
        """
        if not chunks:
            return []

        effective_top_k = top_k
        weighting_strat = ChunkWeightingStrategy.STANDARD
        multi_doc_strat = MultiDocumentStrategy.MULTI_DOCUMENT

        if query_plan is not None:
            effective_top_k = min(int(query_plan.top_k), int(query_plan.max_chunks))
            weighting_strat = query_plan.chunk_weighting_strategy
            multi_doc_strat = query_plan.multi_document_strategy

        # Compute initial fused scores with weighting boost
        scored_candidates: List[Dict[str, Any]] = []
        for c in chunks:
            cand = dict(c)
            sim = float(cand.get("similarity_score", 0.0))
            boost = float(cand.get("weighting_boost", 0.0))
            cand["fused_score"] = round(min(1.0, sim + boost), 4)
            scored_candidates.append(cand)

        # Sort descending by fused score
        scored_candidates.sort(key=lambda x: x["fused_score"], reverse=True)

        selected: List[Dict[str, Any]] = []

        if multi_doc_strat == MultiDocumentStrategy.BALANCED_DOCUMENT_RETRIEVAL:
            # Round-robin selection across distinct documents
            docs: Dict[str, List[Dict[str, Any]]] = {}
            for c in scored_candidates:
                d_id = cls._extract_doc_id(c)
                docs.setdefault(d_id, []).append(c)

            doc_keys = list(docs.keys())
            idx = 0
            while len(selected) < effective_top_k and any(docs.values()):
                curr_doc = doc_keys[idx % len(doc_keys)]
                if docs[curr_doc]:
                    selected.append(docs[curr_doc].pop(0))
                idx += 1
                if not any(docs.values()):
                    break

        elif multi_doc_strat == MultiDocumentStrategy.MAP_REDUCE:
            # Broad coverage: max 2 chunks per doc until each doc is represented
            doc_counts: Dict[str, int] = {}
            remaining: List[Dict[str, Any]] = []
            for c in scored_candidates:
                d_id = cls._extract_doc_id(c)
                if doc_counts.get(d_id, 0) < 2 and len(selected) < effective_top_k:
                    selected.append(c)
                    doc_counts[d_id] = doc_counts.get(d_id, 0) + 1
                else:
                    remaining.append(c)
            # Backfill if space remains
            if len(selected) < effective_top_k:
                for c in remaining:
                    selected.append(c)
                    if len(selected) >= effective_top_k:
                        break
        else:
            # Standard / Single document
            selected = scored_candidates[:effective_top_k]

        # Final sort by fused_score descending and conversion to strongly typed FusedEvidenceChunk
        selected.sort(key=lambda x: x["fused_score"], reverse=True)
        fused_chunks: List[FusedEvidenceChunk] = []

        for idx, c in enumerate(selected, start=1):
            doc_id = cls._extract_doc_id(c)
            doc_name = cls._extract_doc_name(c)
            sim_score = float(c.get("similarity_score", 0.0))
            w_boost = float(c.get("weighting_boost", 0.0))
            f_score = float(c.get("fused_score", sim_score))

            aspects = cls._identify_chunk_aspects(c.get("text", ""))

            fused_chunks.append(
                FusedEvidenceChunk(
                    chunk_id=str(c.get("chunk_id", f"chunk_{idx}")),
                    document_id=doc_id,
                    document_name=doc_name,
                    page_number=c.get("page_number"),
                    text=str(c.get("text", "")).strip(),
                    similarity_score=sim_score,
                    weighting_boost=w_boost,
                    fused_score=f_score,
                    rank=idx,
                    source_index=idx,
                    aspects=aspects,
                    metadata=dict(c.get("metadata", {}))
                )
            )

        return fused_chunks

    # -------------------------------------------------------------------------
    # 3. CLINICAL CONFLICT DETECTION
    # -------------------------------------------------------------------------

    @classmethod
    def detect_conflicts(
        cls,
        chunks: List[FusedEvidenceChunk],
        query: str = ""
    ) -> List[EvidenceConflict]:
        """
        Identifies material clinical contradictions between retrieved evidence documents:
        1. Dosage discrepancies (e.g. 10 mg daily vs 25 mg daily starting dose).
        2. Contraindication conflicts (e.g. contraindicated in pregnancy vs safe).
        3. Treatment recommendation conflicts (e.g. first-line vs contraindicated/second-line).
        4. Diagnostic criteria conflicts (e.g. cutoff BP > 140/90 vs > 130/80).

        Explicitly distinguishes genuine contradictions from complementary information.
        """
        if len(chunks) < 2:
            return []

        conflicts: List[EvidenceConflict] = []
        seen_pairs: Set[Tuple[str, str, str]] = set()

        # Group chunks by document ID
        doc_chunks: Dict[str, List[FusedEvidenceChunk]] = {}
        for c in chunks:
            doc_chunks.setdefault(c.document_id, []).append(c)

        # Cross-document pairwise comparison
        doc_ids = list(doc_chunks.keys())
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                doc_a_id = doc_ids[i]
                doc_b_id = doc_ids[j]

                chunks_a = doc_chunks[doc_a_id]
                chunks_b = doc_chunks[doc_b_id]

                for ca in chunks_a:
                    for cb in chunks_b:
                        # 1. Dosage Discrepancy Check
                        dosage_conf = cls._check_dosage_conflict(ca, cb)
                        if dosage_conf:
                            pair_key = (doc_a_id, doc_b_id, dosage_conf.topic)
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
                                conflicts.append(dosage_conf)

                        # 2. Contraindication Conflict Check
                        contra_conf = cls._check_contraindication_conflict(ca, cb)
                        if contra_conf:
                            pair_key = (doc_a_id, doc_b_id, contra_conf.topic)
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
                                conflicts.append(contra_conf)

                        # 3. Treatment Recommendation Conflict Check
                        treatment_conf = cls._check_treatment_conflict(ca, cb)
                        if treatment_conf:
                            pair_key = (doc_a_id, doc_b_id, treatment_conf.topic)
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
                                conflicts.append(treatment_conf)

                        # 4. Diagnostic Criteria Conflict Check
                        diag_conf = cls._check_diagnostic_conflict(ca, cb)
                        if diag_conf:
                            pair_key = (doc_a_id, doc_b_id, diag_conf.topic)
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
                                conflicts.append(diag_conf)

        return conflicts

    @classmethod
    def _extract_drugs(cls, text: str) -> Set[str]:
        """Finds known drug mentions in text."""
        t_low = text.lower()
        return {d for d in cls.KNOWN_DRUGS if re.search(rf"\b{d}\b", t_low)}

    @classmethod
    def _check_dosage_conflict(
        cls,
        ca: FusedEvidenceChunk,
        cb: FusedEvidenceChunk
    ) -> Optional[EvidenceConflict]:
        """Detects conflicting numerical dosages for the same drug."""
        common_drugs = cls._extract_drugs(ca.text) & cls._extract_drugs(cb.text)
        if not common_drugs:
            return None

        # Dosage pattern: number + unit
        dose_pat = re.compile(
            r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|units?|g)\b",
            re.IGNORECASE
        )

        for drug in common_drugs:
            # Check if both chunks describe starting or daily doses for this drug
            is_start_a = bool(re.search(rf"\b(?:starting|initial|recommended|daily)\s+(?:dose\s+of\s+)?{drug}\b", ca.text, re.I) or
                             re.search(rf"\b{drug}\s+(?:starting|initial|recommended|daily)\s+dose\b", ca.text, re.I))
            is_start_b = bool(re.search(rf"\b(?:starting|initial|recommended|daily)\s+(?:dose\s+of\s+)?{drug}\b", cb.text, re.I) or
                             re.search(rf"\b{drug}\s+(?:starting|initial|recommended|daily)\s+dose\b", cb.text, re.I))

            if is_start_a and is_start_b:
                doses_a = dose_pat.findall(ca.text)
                doses_b = dose_pat.findall(cb.text)
                if doses_a and doses_b:
                    val_a = float(doses_a[0][0])
                    val_b = float(doses_b[0][0])
                    unit_a = doses_a[0][1].lower()
                    unit_b = doses_b[0][1].lower()

                    if unit_a == unit_b and abs(val_a - val_b) > 0.01:
                        # Genuine conflict in quantified dose
                        stmt_a = ca.text[:140].strip() + "..."
                        stmt_b = cb.text[:140].strip() + "..."
                        return EvidenceConflict(
                            topic=f"{drug.capitalize()} Dosage",
                            conflict_type=ConflictType.DOSAGE_DISCREPANCY,
                            severity=ConflictSeverity.MODERATE,
                            doc_a_id=ca.document_id,
                            doc_b_id=cb.document_id,
                            doc_a_name=ca.document_name,
                            doc_b_name=cb.document_name,
                            statement_a=stmt_a,
                            statement_b=stmt_b,
                            resolution_guidance=(
                                f"Discrepancy in recommended {drug} dosing ({val_a} {unit_a} vs {val_b} {unit_b}). "
                                "Present both guideline figures with their respective sources and advise physician consultation."
                            )
                        )
        return None

    @classmethod
    def _check_contraindication_conflict(
        cls,
        ca: FusedEvidenceChunk,
        cb: FusedEvidenceChunk
    ) -> Optional[EvidenceConflict]:
        """Detects conflicting contraindication/safety statements for the same entity."""
        common_drugs = cls._extract_drugs(ca.text) & cls._extract_drugs(cb.text)
        if not common_drugs:
            return None

        # Populations of interest
        populations = [
            ("pregnancy", [r"\bpregnancy\b", r"\bpregnant\b"]),
            ("renal impairment", [r"\brenal\b", r"\bkidney\s+disease\b", r"\begfr\b"]),
            ("pediatric", [r"\bpediatric\b", r"\bchildren\b", r"\binfants?\b"]),
            ("elderly", [r"\belderly\b", r"\bgeriatric\b", r"\bolder\s+adults\b"])
        ]

        pos_pat = re.compile(r"\b(?:safe|recommended|indicated|can\s+be\s+used|well-tolerated|acceptable)\b", re.I)
        neg_pat = re.compile(r"\b(?:contraindicated|strictly\s+contraindicated|do\s+not\s+use|avoid|teratogenic|hazardous|toxic)\b", re.I)

        for drug in common_drugs:
            for pop_name, pop_pats in populations:
                has_pop_a = any(re.search(p, ca.text, re.I) for p in pop_pats)
                has_pop_b = any(re.search(p, cb.text, re.I) for p in pop_pats)

                if has_pop_a and has_pop_b:
                    is_pos_a = bool(pos_pat.search(ca.text))
                    is_neg_a = bool(neg_pat.search(ca.text))
                    is_pos_b = bool(pos_pat.search(cb.text))
                    is_neg_b = bool(neg_pat.search(cb.text))

                    if (is_pos_a and not is_neg_a and is_neg_b) or (is_neg_a and is_pos_b and not is_neg_b):
                        stmt_a = ca.text[:140].strip() + "..."
                        stmt_b = cb.text[:140].strip() + "..."
                        return EvidenceConflict(
                            topic=f"{drug.capitalize()} in {pop_name}",
                            conflict_type=ConflictType.CONTRAINDICATION_CONFLICT,
                            severity=ConflictSeverity.HIGH,
                            doc_a_id=ca.document_id,
                            doc_b_id=cb.document_id,
                            doc_a_name=ca.document_name,
                            doc_b_name=cb.document_name,
                            statement_a=stmt_a,
                            statement_b=stmt_b,
                            resolution_guidance=(
                                f"Direct safety conflict: contradictory evidence regarding {drug} use in {pop_name}. "
                                "Prioritize caution, highlight the strict contraindication warning, and emphasize doctor consultation."
                            )
                        )
        return None

    @classmethod
    def _check_treatment_conflict(
        cls,
        ca: FusedEvidenceChunk,
        cb: FusedEvidenceChunk
    ) -> Optional[EvidenceConflict]:
        """Detects conflicting first-line treatment recommendations."""
        common_conditions = {c for c in cls.KNOWN_CONDITIONS if re.search(rf"\b{c}\b", ca.text.lower()) and re.search(rf"\b{c}\b", cb.text.lower())}
        if not common_conditions:
            return None

        first_line_pat = re.compile(r"\b(?:first-line|first\s+line|initial\s+therapy|drug\s+of\s+choice|preferred\s+treatment)\b", re.I)
        opposing_pat = re.compile(r"\b(?:not\s+recommended\s+as\s+first-line|second-line|second\s+line|avoid\s+as\s+initial|no\s+longer\s+recommended)\b", re.I)

        for cond in common_conditions:
            drugs_a = cls._extract_drugs(ca.text)
            drugs_b = cls._extract_drugs(cb.text)
            overlap_drugs = drugs_a & drugs_b

            for drug in overlap_drugs:
                is_fl_a = bool(first_line_pat.search(ca.text))
                is_opp_b = bool(opposing_pat.search(cb.text))

                is_fl_b = bool(first_line_pat.search(cb.text))
                is_opp_a = bool(opposing_pat.search(ca.text))

                if (is_fl_a and is_opp_b) or (is_fl_b and is_opp_a):
                    return EvidenceConflict(
                        topic=f"{cond.capitalize()} First-Line Therapy ({drug.capitalize()})",
                        conflict_type=ConflictType.TREATMENT_RECOMMENDATION_CONFLICT,
                        severity=ConflictSeverity.MODERATE,
                        doc_a_id=ca.document_id,
                        doc_b_id=cb.document_id,
                        doc_a_name=ca.document_name,
                        doc_b_name=cb.document_name,
                        statement_a=ca.text[:140].strip() + "...",
                        statement_b=cb.text[:140].strip() + "...",
                        resolution_guidance=(
                            f"Contradictory clinical guidelines regarding {drug} as first-line therapy for {cond}. "
                            "Report the divergence across guidelines explicitly."
                        )
                    )
        return None

    @classmethod
    def _check_diagnostic_conflict(
        cls,
        ca: FusedEvidenceChunk,
        cb: FusedEvidenceChunk
    ) -> Optional[EvidenceConflict]:
        """Detects conflicting diagnostic criteria cutoffs (e.g. BP or HbA1c threshold)."""
        bp_pat = re.compile(r"\b(?:blood\s+pressure|bp)\s+(?:threshold|cutoff|criteria|diagnosis)?\s*(?:of|>=|>)?\s*(\d{2,3}/\d{2,3})\b", re.I)
        hba1c_pat = re.compile(r"\b(?:hba1c|a1c)\s*(?:>=|>|cutoff|of)?\s*(\d+(?:\.\d+)?)\s*%", re.I)

        # Check BP cutoff conflict
        bp_a = bp_pat.findall(ca.text)
        bp_b = bp_pat.findall(cb.text)
        if bp_a and bp_b and bp_a[0] != bp_b[0]:
            return EvidenceConflict(
                topic="Hypertension Diagnostic Threshold",
                conflict_type=ConflictType.DIAGNOSTIC_CRITERIA_CONFLICT,
                severity=ConflictSeverity.MODERATE,
                doc_a_id=ca.document_id,
                doc_b_id=cb.document_id,
                doc_a_name=ca.document_name,
                doc_b_name=cb.document_name,
                statement_a=ca.text[:140].strip() + "...",
                statement_b=cb.text[:140].strip() + "...",
                resolution_guidance=(
                    f"Differing diagnostic thresholds ({bp_a[0]} vs {bp_b[0]}). "
                    "Highlight differences between historical or organizational guidelines (e.g. ACC/AHA vs JNC)."
                )
            )

        # Check HbA1c cutoff conflict
        a1c_a = hba1c_pat.findall(ca.text)
        a1c_b = hba1c_pat.findall(cb.text)
        if a1c_a and a1c_b and abs(float(a1c_a[0]) - float(a1c_b[0])) >= 0.3:
            return EvidenceConflict(
                topic="Diabetes Diagnostic HbA1c Cutoff",
                conflict_type=ConflictType.DIAGNOSTIC_CRITERIA_CONFLICT,
                severity=ConflictSeverity.MODERATE,
                doc_a_id=ca.document_id,
                doc_b_id=cb.document_id,
                doc_a_name=ca.document_name,
                doc_b_name=cb.document_name,
                statement_a=ca.text[:140].strip() + "...",
                statement_b=cb.text[:140].strip() + "...",
                resolution_guidance=(
                    f"Differing HbA1c diagnostic cutoffs ({a1c_a[0]}% vs {a1c_b[0]}%). "
                    "Report organizational criteria variance transparently."
                )
            )

        # Check Fasting Blood Glucose (FBG) cutoff conflict
        fbg_pat = re.compile(r"\b(?:fasting\s+blood\s+glucose|fasting\s+glucose|fbg)\s*(?:is|of|>=|>|cutoff)?\s*(\d{2,3})\s*mg/dl\b", re.I)
        fbg_a = fbg_pat.findall(ca.text)
        fbg_b = fbg_pat.findall(cb.text)
        if fbg_a and fbg_b and abs(int(fbg_a[0]) - int(fbg_b[0])) >= 10:
            return EvidenceConflict(
                topic="Fasting Blood Glucose Diagnostic Cutoff",
                conflict_type=ConflictType.DIAGNOSTIC_CRITERIA_CONFLICT,
                severity=ConflictSeverity.MODERATE,
                doc_a_id=ca.document_id,
                doc_b_id=cb.document_id,
                doc_a_name=ca.document_name,
                doc_b_name=cb.document_name,
                statement_a=ca.text[:140].strip() + "...",
                statement_b=cb.text[:140].strip() + "...",
                resolution_guidance=(
                    f"Differing fasting blood glucose diagnostic cutoffs ({fbg_a[0]} mg/dL vs {fbg_b[0]} mg/dL). "
                    "Report organizational criteria variance transparently."
                )
            )

        return None

    # -------------------------------------------------------------------------
    # 4. EVIDENCE COVERAGE EVALUATION
    # -------------------------------------------------------------------------

    CLINICAL_ASPECT_PATTERNS: Dict[str, re.Pattern] = {
        "dosage": re.compile(r"\b(dose|dosage|how\s+much|daily\s+amount|mg\b|mcg\b|units?\b|titrat)\b", re.I),
        "side_effects": re.compile(r"\b(side\s+effects?|adverse\s+effects?|safety|risks?|complications?)\b", re.I),
        "contraindications": re.compile(r"\b(contraindicated|contraindications?|who\s+should\s+not|avoid|interactions?)\b", re.I),
        "mechanism": re.compile(r"\b(mechanism|how\s+does\s+it\s+work|pharmacology|action\b)\b", re.I),
        "diagnosis": re.compile(r"\b(diagnosis|diagnostic|symptoms?|signs?|criteria|how\s+to\s+test)\b", re.I),
        "treatment": re.compile(r"\b(treatment|therapy|management|cure|medications?|how\s+to\s+treat)\b", re.I),
        "prevention": re.compile(r"\b(prevent|prevention|lifestyle|diet|exercise|prophylaxis)\b", re.I),
    }

    @classmethod
    def _identify_chunk_aspects(cls, text: str) -> List[str]:
        """Identifies which clinical aspects are discussed in an evidence chunk."""
        aspects = []
        for aspect_name, pat in cls.CLINICAL_ASPECT_PATTERNS.items():
            if pat.search(text):
                aspects.append(aspect_name)
        return aspects

    @classmethod
    def assess_coverage(
        cls,
        query: str,
        fused_chunks: List[FusedEvidenceChunk],
        query_plan: Optional[QueryPlan] = None
    ) -> EvidenceCoverage:
        """
        Determines whether the retrieved evidence covers all required facets of the user question.
        """
        q_lower = query.lower()
        requested_aspects = []

        for aspect_name, pat in cls.CLINICAL_ASPECT_PATTERNS.items():
            if pat.search(q_lower):
                requested_aspects.append(aspect_name)

        if not requested_aspects:
            # Fallback: if query doesn't specify multi-aspect terms, assume general clinical inquiry
            requested_aspects = ["treatment" if "treat" in q_lower else "general_clinical_info"]

        # Collect all aspects covered by retrieved chunks
        covered_set: Set[str] = set()
        combined_text = " ".join(c.text for c in fused_chunks).lower()

        for aspect_name in requested_aspects:
            if aspect_name == "general_clinical_info":
                if len(fused_chunks) > 0:
                    covered_set.add("general_clinical_info")
            elif aspect_name in cls.CLINICAL_ASPECT_PATTERNS:
                if cls.CLINICAL_ASPECT_PATTERNS[aspect_name].search(combined_text):
                    covered_set.add(aspect_name)

        covered_aspects = [a for a in requested_aspects if a in covered_set]
        missing_aspects = [a for a in requested_aspects if a not in covered_set]

        coverage_score = len(covered_aspects) / max(1, len(requested_aspects))

        if not fused_chunks or len(covered_aspects) == 0:
            status = CoverageStatus.INSUFFICIENT
        elif coverage_score >= 0.90:
            status = CoverageStatus.FULL
        else:
            status = CoverageStatus.PARTIAL

        return EvidenceCoverage(
            coverage_score=round(coverage_score, 3),
            status=status,
            query_aspects=requested_aspects,
            covered_aspects=covered_aspects,
            missing_aspects=missing_aspects
        )

    # -------------------------------------------------------------------------
    # 5. MULTI-DOCUMENT CONTEXT FORMATTING & REASONING
    # -------------------------------------------------------------------------

    @classmethod
    def format_multi_document_context(
        cls,
        chunks: List[FusedEvidenceChunk],
        conflicts: Optional[List[EvidenceConflict]] = None,
        coverage: Optional[EvidenceCoverage] = None
    ) -> str:
        """
        Formats a clean, attribution-preserving multi-document context block
        clearly identifying documents, chunk metadata, and conflict advisories.
        """
        if not chunks:
            return ""

        context_lines: List[str] = []

        # 1. Conflict Warning Advisory Header (if clinical contradictions exist)
        if conflicts:
            context_lines.append("=== CLINICAL EVIDENCE NOTICE: CONFLICTING RECOMMENDATIONS IDENTIFIED ===")
            for c in conflicts:
                context_lines.append(
                    f"Notice regarding {c.topic} (Severity: {c.severity.value.upper()}):\n"
                    f"  - Source '{c.doc_a_name}': {c.statement_a}\n"
                    f"  - Source '{c.doc_b_name}': {c.statement_b}\n"
                    f"  - Clinical Guidance: {c.resolution_guidance}"
                )
            context_lines.append("========================================================================\n")

        # 2. Multi-Document Summary Header
        contributing_docs = sorted(list({c.document_name for c in chunks}))
        if len(contributing_docs) > 1:
            context_lines.append(
                f"Multi-Document Context: {len(chunks)} evidence sources synthesized from "
                f"{len(contributing_docs)} distinct references: [{', '.join(contributing_docs)}]\n"
            )

        # 3. Individual Provenance-Preserved Source Blocks
        for chunk in chunks:
            pg_str = str(chunk.page_number) if chunk.page_number is not None else "N/A"
            doc_label = f"{chunk.document_name} (ID: {chunk.document_id})" if (chunk.document_name and chunk.document_id and chunk.document_name != chunk.document_id) else (chunk.document_name or chunk.document_id)
            block = (
                f"[SOURCE {chunk.source_index}]\n"
                f"Document: {doc_label}\n"
                f"Page: {pg_str}\n"
                f"Chunk ID: {chunk.chunk_id}\n\n"
                f"{chunk.text}"
            )
            context_lines.append(block)

        return "\n\n".join(context_lines)

    # -------------------------------------------------------------------------
    # 6. MASTER FUSION ORCHESTRATOR
    # -------------------------------------------------------------------------

    @classmethod
    def fuse_evidence(
        cls,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        query_plan: Optional[QueryPlan] = None,
        user_id: Optional[int] = None
    ) -> FusedContextResult:
        """
        Master orchestrator: deduplicates, ranks, detects conflicts, assesses coverage,
        and constructs attributed context.
        """
        t0 = time.perf_counter()

        # Step 0: User / tenant isolation filtering
        effective_user_id = user_id
        if effective_user_id is None and query_plan is not None:
            effective_user_id = getattr(query_plan, "user_id", None) or (
                query_plan.metadata.get("user_id") if hasattr(query_plan, "metadata") and query_plan.metadata else None
            )

        if effective_user_id is not None:
            filtered_chunks = []
            for c in retrieved_chunks:
                c_uid = c.get("user_id") if "user_id" in c else c.get("metadata", {}).get("user_id")
                if c_uid is None or str(c_uid) == str(effective_user_id):
                    filtered_chunks.append(c)
            retrieved_chunks = filtered_chunks

        # Step 1: Deduplicate and prune near-duplicate chunks
        deduped_raw, pruned_count = cls.deduplicate_evidence(retrieved_chunks)

        # Step 2: Rank and balance evidence across documents
        fused_chunks = cls.rank_and_balance_evidence(
            chunks=deduped_raw,
            query_plan=query_plan,
            top_k=query_plan.top_k if query_plan else 5
        )

        # Step 3: Extract contributing documents
        contributing_docs = sorted(list({c.document_name for c in fused_chunks}))

        # Step 4: Clinical conflict detection
        conflicts = cls.detect_conflicts(fused_chunks, query=query)
        has_conflicts = len(conflicts) > 0

        # Step 5: Evidence coverage assessment
        coverage = cls.assess_coverage(query=query, fused_chunks=fused_chunks, query_plan=query_plan)

        # Step 6: Multi-document context formatting
        formatted_context = cls.format_multi_document_context(
            chunks=fused_chunks,
            conflicts=conflicts,
            coverage=coverage
        )

        # Step 7: Build structured citation sources mapping
        sources = []
        for c in fused_chunks:
            sources.append({
                "source_index": c.source_index,
                "source_label": f"[Source {c.source_index}]",
                "source_id": c.document_id,
                "document_id": c.document_id,
                "filename": c.document_name,
                "page_number": c.page_number,
                "chunk_id": c.chunk_id,
                "similarity_score": round(float(c.similarity_score), 4),
                "fused_score": round(float(c.fused_score), 4),
                "preview_text": c.text[:250] if c.text else "",
                "text": c.text,
                "metadata": c.metadata
            })

        # Step 8: Determine overall evidence sufficiency
        is_sufficient = bool(fused_chunks) and coverage.status != CoverageStatus.INSUFFICIENT
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return FusedContextResult(
            fused_chunks=fused_chunks,
            contributing_documents=contributing_docs,
            contributing_documents_count=len(contributing_docs),
            conflicts=conflicts,
            has_conflicts=has_conflicts,
            coverage=coverage,
            formatted_context=formatted_context,
            sources=sources,
            deduped_count=pruned_count,
            is_sufficient=is_sufficient,
            latency_ms=elapsed_ms,
            metadata={
                "weighting_strategy": query_plan.chunk_weighting_strategy.value if query_plan else "standard",
                "multi_document_strategy": query_plan.multi_document_strategy.value if query_plan else "multi_document"
            }
        )
