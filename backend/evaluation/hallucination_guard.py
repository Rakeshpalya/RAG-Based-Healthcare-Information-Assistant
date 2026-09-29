import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np

from backend.services.embedding_service import EmbeddingService
from backend.evaluation.citation_validator import (
    ExtractedClaim,
    CitationValidator,
    CitationValidationResult
)


class HallucinationType:
    NONE = "NONE"
    MEDICATION_HALLUCINATION = "MEDICATION_HALLUCINATION"
    DOSAGE_HALLUCINATION = "DOSAGE_HALLUCINATION"
    NEGATION_CONTRADICTION = "NEGATION_CONTRADICTION"
    DIRECTIONAL_CONTRADICTION = "DIRECTIONAL_CONTRADICTION"
    UNSUPPORTED_DIAGNOSIS = "UNSUPPORTED_DIAGNOSIS"
    UNSUPPORTED_RECOMMENDATION = "UNSUPPORTED_RECOMMENDATION"
    WRONG_SOURCE = "WRONG_SOURCE"
    SEMANTIC_MISMATCH = "SEMANTIC_MISMATCH"
    UNCITED_CLAIM = "UNCITED_CLAIM"
    INVALID_SOURCE = "INVALID_SOURCE"


@dataclass
class ExtractedEntities:
    """Entities extracted from a clinical claim for hallucination analysis."""
    medications: List[str] = field(default_factory=list)
    dosages: List[str] = field(default_factory=list)
    diagnoses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    has_negation: bool = False
    negation_terms: List[str] = field(default_factory=list)
    direction: Optional[str] = None  # 'UP', 'DOWN', or None
    direction_terms: List[str] = field(default_factory=list)


@dataclass
class ClaimGuardDetail:
    """Detailed hallucination audit for a single segmented claim."""
    claim_text: str
    raw_sentence: str
    cited_sources: List[int]
    is_supported: bool
    hallucination_type: str = HallucinationType.NONE
    reasons: List[str] = field(default_factory=list)
    support_scores: Dict[int, float] = field(default_factory=dict)
    best_support_score: float = 0.0
    best_source_id: Optional[int] = None
    entities: Optional[ExtractedEntities] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_text": self.claim_text,
            "raw_sentence": self.raw_sentence,
            "cited_sources": self.cited_sources,
            "is_supported": self.is_supported,
            "hallucination_type": self.hallucination_type,
            "reasons": self.reasons,
            "support_scores": self.support_scores,
            "best_support_score": round(self.best_support_score, 4),
            "best_source_id": self.best_source_id,
            "entities": {
                "medications": self.entities.medications if self.entities else [],
                "dosages": self.entities.dosages if self.entities else [],
                "diagnoses": self.entities.diagnoses if self.entities else [],
                "recommendations": self.entities.recommendations if self.entities else [],
                "has_negation": self.entities.has_negation if self.entities else False,
                "direction": self.entities.direction if self.entities else None,
            } if self.entities else None
        }


@dataclass
class HallucinationGuardResult:
    """Structured result of complete hallucination inspection on an answer."""
    is_safe: bool
    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    hallucinated_claims: int = 0
    contradictions_detected: int = 0
    medication_hallucinations: int = 0
    dosage_hallucinations: int = 0
    claims: List[ClaimGuardDetail] = field(default_factory=list)
    sanitized_answer: str = ""
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "hallucinated_claims": self.hallucinated_claims,
            "contradictions_detected": self.contradictions_detected,
            "medication_hallucinations": self.medication_hallucinations,
            "dosage_hallucinations": self.dosage_hallucinations,
            "claims": [c.to_dict() for c in self.claims],
            "sanitized_answer": self.sanitized_answer,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason,
        }


class HallucinationGuard:
    """
    Phase 3.4 Hallucination Protection Engine.
    
    Provides deterministic, multi-level clinical grounding validation:
    1. Medication entity validation (checks against cited sources).
    2. Numerical dosage and unit validation (prevents dosage fabrications).
    3. Negation mismatch & polarity contradiction detection.
    4. Directional clinical contradiction detection (e.g. increases vs decreases).
    5. Diagnostic and prescriptive recommendation ground checking.
    6. Wrong-source attribution detection.
    7. Partial/mixed support filtering and safe fallback.
    """

    SUPPORT_SIMILARITY_THRESHOLD: float = 0.65

    SAFE_FALLBACK_TEXT = (
        "Based on the provided medical documents, there is insufficient evidence to answer your question."
    )

    # Curated medical pharmacological entity dictionary (generic & brand names)
    KNOWN_MEDICATIONS: Set[str] = {
        # Antihypertensives & Cardiovascular
        "amlodipine", "lisinopril", "enalapril", "losartan", "valsartan", "candesartan",
        "hydrochlorothiazide", "chlorthalidone", "atenolol", "metoprolol", "propranolol",
        "carvedilol", "labetalol", "diltiazem", "verapamil", "nifedipine", "clonidine",
        "hydralazine", "spironolactone", "ramipril", "benazepril", "furosemide", "torsemide",
        "bumetanide", "digoxin", "nitroglycerin", "isosorbide",
        # Antidiabetics
        "metformin", "glipizide", "glyburide", "glimepiride", "pioglitazone", "rosiglitazone",
        "sitagliptin", "saxagliptin", "linagliptin", "empagliflozin", "dapagliflozin",
        "canagliflozin", "liraglutide", "semaglutide", "ozempic", "wegovy", "mounjaro",
        "tirzepatide", "dulaglutide", "insulin", "glargine", "lispro", "aspart", "detemir",
        # Statins & Lipid Lowering
        "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin",
        "ezetimibe", "fenofibrate", "gemfibrozil",
        # Analgesics & Anti-inflammatories
        "aspirin", "ibuprofen", "naproxen", "acetaminophen", "paracetamol", "celecoxib",
        "meloxicam", "diclofenac", "indomethacin", "ketorolac", "tramadol", "codeine",
        "morphine", "oxycodone", "hydrocodone", "fentanyl", "buprenorphine",
        # Antibiotics & Antivirals
        "amoxicillin", "ampicillin", "penicillin", "cephalexin", "cefuroxime", "ceftriaxone",
        "azithromycin", "clarithromycin", "ciprofloxacin", "levofloxacin", "doxycycline",
        "trimethoprim", "sulfamethoxazole", "bactrim", "metronidazole", "vancomycin",
        "acyclovir", "valacyclovir", "oseltamivir", "tamiflu", "paxlovid", "remdesivir",
        # Anticoagulants & Antiplatelets
        "warfarin", "coumadin", "heparin", "enoxaparin", "lovenox", "apixaban", "eliquis",
        "rivaroxaban", "xarelto", "dabigatran", "pradaxa", "clopidogrel", "plavix", "ticagrelor",
        # Gastrointestinal & Steroids
        "omeprazole", "pantoprazole", "esomeprazole", "lansoprazole", "famotidine",
        "prednisone", "prednisolone", "dexamethasone", "hydrocortisone", "budesonide",
        # Respiratory & Psychiatric
        "albuterol", "ipratropium", "tiotropium", "montelukast", "fluticasone",
        "levothyroxine", "synthroid", "gabapentin", "pregabalin", "sertraline", "fluoxetine",
        "escitalopram", "citalopram", "duloxetine", "venlafaxine", "bupropion"
    }

    # Common pharmacological suffix patterns
    MEDICATION_SUFFIX_REGEX = re.compile(
        r'\b[A-Za-z]+(?:olol|pril|sartan|statin|gliptin|gliflozin|tide|prazole|cillin|mycin|micin|cycline|floxacin|mab|nib|vir|afil|zosin|triptan|dipine)\b',
        re.IGNORECASE
    )

    # Numerical dosage patterns: e.g., 500 mg, 10 mg, 0.5 mcg, 100 units, 5 ml
    DOSAGE_REGEX = re.compile(
        r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|units?|iu)\b',
        re.IGNORECASE
    )

    # Numerical blood pressure measurements: e.g. 158/96 mmHg or 120/80
    BP_MEASUREMENT_REGEX = re.compile(
        r'\b\d{2,3}/\d{2,3}(?:\s*mm\s*hg)?\b',
        re.IGNORECASE
    )

    # Directional clinical terms
    DIRECTION_UP_TERMS: Set[str] = {
        "increase", "increases", "increased", "increasing",
        "elevate", "elevates", "elevated", "elevating",
        "raise", "raises", "raised", "raising",
        "worsen", "worsens", "worsened", "worsening",
        "aggravate", "aggravates", "aggravated", "aggravating",
        "heighten", "heightens", "heightened", "higher"
    }

    DIRECTION_DOWN_TERMS: Set[str] = {
        "decrease", "decreases", "decreased", "decreasing",
        "lower", "lowers", "lowered", "lowering",
        "reduce", "reduces", "reduced", "reducing",
        "attenuate", "attenuates", "attenuated", "attenuating",
        "alleviate", "alleviates", "alleviated", "alleviating",
        "ameliorate", "ameliorates", "ameliorated", "ameliorating",
        "lessen", "lessens", "lessened", "drop", "drops", "dropped"
    }

    # Explicit clinical negation markers and phrases
    NEGATION_PHRASES: List[str] = [
        "not recommended", "not advised", "not indicated", "no evidence",
        "should not", "do not", "does not", "is not", "are not",
        "cannot", "contraindicated", "contraindication", "contraindications",
        "avoid", "unrecommended", "prohibited", "prohibit", "never",
        "neither", "nor", "without", "unlikely", "ineffective",
        "discontinue", "stop taking", "unsafe", "harmful"
    ]

    # Explicit diagnostic assertion patterns
    DIAGNOSIS_PATTERNS = [
        re.compile(r'\b(?:diagnosed\s+with|diagnosis\s+of|confirmed\s+case\s+of|suffers\s+from|patient\s+has)\s+([a-z0-9\s\-]{3,35})(?=[.,;]|\[|\band\b|$)', re.IGNORECASE),
        re.compile(r'\b(?:stage\s+[1-4]|type\s+[12]|grade\s+[1-4]|class\s+[i-iv]+)\s+([a-z0-9\s\-]{3,35})', re.IGNORECASE)
    ]

    # Explicit prescriptive recommendation patterns
    RECOMMENDATION_PATTERNS = [
        re.compile(r'\b(?:prescribe|prescribes|prescribed|prescribing|prescription)\s+([a-z0-9\s\-]{3,30})', re.IGNORECASE),
        re.compile(r'\b(?:recommend|recommends|recommended|recommending)\s+(?:that\s+patient\s+take|taking|the\s+use\s+of)?\s*([a-z0-9\s\-]{3,30})', re.IGNORECASE),
        re.compile(r'\b(?:should\s+(?:take|receive|start|administer|initiate))\s+([a-z0-9\s\-]{3,30})', re.IGNORECASE),
        re.compile(r'\b(?:cure|cures|cured|curing|proven\s+to\s+cure|permanently\s+cure|treat|treats|treated|treating)\s+([a-z0-9\s\-]{3,35})(?=[.,;]|\[|\band\b|$)', re.IGNORECASE)
    ]

    @classmethod
    def extract_entities(cls, text: str) -> ExtractedEntities:
        """Extracts clinical medications, dosages, directions, and negations from text."""
        if not text:
            return ExtractedEntities()

        text_lower = text.lower()
        words = re.findall(r'\b[a-z0-9\-]+\b', text_lower)
        word_set = set(words)

        # 1. Medications
        known_meds = [w for w in words if w in cls.KNOWN_MEDICATIONS]
        suffix_meds = [m.lower() for m in cls.MEDICATION_SUFFIX_REGEX.findall(text)]
        medications = sorted(set(known_meds + suffix_meds))

        # 2. Dosages & Measurements
        dosages = [d.strip().lower() for d in cls.DOSAGE_REGEX.findall(text)]
        bps = [bp.strip().lower() for bp in cls.BP_MEASUREMENT_REGEX.findall(text)]
        all_dosages = sorted(set(dosages + bps))

        # 3. Negations
        found_negations = []
        for phrase in cls.NEGATION_PHRASES:
            if phrase in text_lower:
                found_negations.append(phrase)
        has_negation = len(found_negations) > 0

        # 4. Directional terms
        ups = sorted(list(word_set & cls.DIRECTION_UP_TERMS))
        downs = sorted(list(word_set & cls.DIRECTION_DOWN_TERMS))
        direction = None
        direction_terms = []
        if ups and not downs:
            direction = "UP"
            direction_terms = ups
        elif downs and not ups:
            direction = "DOWN"
            direction_terms = downs
        elif ups and downs:
            # If both are present, pick dominant or mark mixed
            direction = "MIXED"
            direction_terms = ups + downs

        # 5. Diagnoses
        diagnoses = []
        for pat in cls.DIAGNOSIS_PATTERNS:
            for m in pat.finditer(text):
                diag_term = m.group(1).strip().lower()
                if len(diag_term) > 3 and diag_term not in ("the", "this", "that"):
                    diagnoses.append(diag_term)

        # 6. Recommendations
        recommendations = []
        for pat in cls.RECOMMENDATION_PATTERNS:
            for m in pat.finditer(text):
                rec_term = m.group(1).strip().lower()
                if len(rec_term) > 3:
                    recommendations.append(rec_term)

        return ExtractedEntities(
            medications=medications,
            dosages=all_dosages,
            diagnoses=sorted(set(diagnoses)),
            recommendations=sorted(set(recommendations)),
            has_negation=has_negation,
            negation_terms=found_negations,
            direction=direction,
            direction_terms=direction_terms
        )

    @classmethod
    def find_best_source_segment(cls, claim_text: str, source_text: str) -> Tuple[float, str]:
        """Finds the source sentence/segment with the highest semantic similarity to the claim."""
        if not claim_text or not source_text:
            return 0.0, ""

        claim_vec = EmbeddingService.embed_query(claim_text)
        if not claim_vec:
            return 0.0, ""

        candidates = [source_text.strip()]
        segments = [
            seg.strip()
            for seg in re.split(r'(?:\n+|\.\s+)', source_text)
            if len(seg.strip()) > 8
        ]
        candidates.extend(segments)

        best_score = -1.0
        best_segment = candidates[0]

        for cand in candidates[:25]:
            s_vec = EmbeddingService.embed_query(cand)
            if s_vec:
                score = float(np.dot(claim_vec, s_vec))
                if score > best_score:
                    best_score = score
                    best_segment = cand

        return max(0.0, best_score), best_segment

    @classmethod
    def check_negation_contradiction(
        cls,
        claim_text: str,
        claim_entities: ExtractedEntities,
        best_source_segment: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Detects negation polarity contradictions between a claim and its best matching source segment.
        Returns (is_contradiction, explanation).
        """
        if not best_source_segment:
            return False, None

        source_entities = cls.extract_entities(best_source_segment)

        # Polarity mismatch detection
        # Case A: Source explicitly negates/contraindicates, but claim asserts affirmatively
        if source_entities.has_negation and not claim_entities.has_negation:
            # Verify clinical entity overlap between claim and source segment
            shared_words = set(re.findall(r'\b[a-z]{4,}\b', claim_text.lower())) & \
                           set(re.findall(r'\b[a-z]{4,}\b', best_source_segment.lower()))
            if len(shared_words) >= 2:
                neg_reasons = ", ".join(f"'{p}'" for p in source_entities.negation_terms)
                return True, (
                    f"Negation contradiction: Source negates assertion ({neg_reasons}), "
                    f"but claim asserts it affirmatively."
                )

        # Case B: Source asserts affirmatively, but claim asserts negatively when source does not support negation
        if claim_entities.has_negation and not source_entities.has_negation:
            # If claim claims something is not recommended / contraindicated, but source recommends or describes it
            shared_words = set(re.findall(r'\b[a-z]{4,}\b', claim_text.lower())) & \
                           set(re.findall(r'\b[a-z]{4,}\b', best_source_segment.lower()))
            if len(shared_words) >= 2 and any(p in claim_text.lower() for p in ("not recommend", "contraindicated", "avoid", "should not")):
                return True, (
                    f"Negation contradiction: Claim introduces unsupported negation "
                    f"('{', '.join(claim_entities.negation_terms)}') not present in source."
                )

        return False, None

    @classmethod
    def check_directional_contradiction(
        cls,
        claim_entities: ExtractedEntities,
        best_source_segment: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Detects opposing clinical directions (e.g. claim asserts increase, source asserts decrease).
        """
        if not best_source_segment or not claim_entities.direction:
            return False, None

        source_entities = cls.extract_entities(best_source_segment)
        if not source_entities.direction or source_entities.direction == "MIXED":
            return False, None

        if claim_entities.direction != "MIXED" and claim_entities.direction != source_entities.direction:
            # Check for inverted direction
            c_dir = "increase/elevate" if claim_entities.direction == "UP" else "decrease/reduce"
            s_dir = "increase/elevate" if source_entities.direction == "UP" else "decrease/reduce"
            return True, (
                f"Directional contradiction: Claim asserts {c_dir} "
                f"({', '.join(claim_entities.direction_terms)}), but source indicates {s_dir} "
                f"({', '.join(source_entities.direction_terms)})."
            )

        return False, None

    @classmethod
    def check_medication_hallucinations(
        cls,
        claim_medications: List[str],
        source_text: str
    ) -> List[str]:
        """
        Verifies that every medication mentioned in the claim exists in the cited source.
        Returns list of hallucinated medications.
        """
        if not claim_medications or not source_text:
            return []

        src_lower = source_text.lower()
        hallucinated = []
        for med in claim_medications:
            # Check for exact word boundary match or common variation
            if not re.search(r'\b' + re.escape(med) + r'\b', src_lower):
                hallucinated.append(med)

        return hallucinated

    @classmethod
    def check_dosage_hallucinations(
        cls,
        claim_dosages: List[str],
        source_text: str
    ) -> List[str]:
        """
        Verifies that numerical dosages mentioned in the claim exist in the cited source.
        Returns list of hallucinated dosages.
        """
        if not claim_dosages or not source_text:
            return []

        src_lower = source_text.lower()
        hallucinated = []
        for d in claim_dosages:
            parts = d.split()
            pattern = r'\s*'.join(re.escape(p) for p in parts)
            if not re.search(r'\b' + pattern + r'\b', src_lower):
                hallucinated.append(d)

        return hallucinated

    @classmethod
    def inspect_claim(
        cls,
        claim: ExtractedClaim,
        source_map: Dict[int, Dict[str, Any]]
    ) -> ClaimGuardDetail:
        """
        Performs comprehensive multi-layer hallucination audit on a single claim.
        """
        entities = cls.extract_entities(claim.claim_text)
        reasons: List[str] = []

        # 1. Uncited factual claim check
        if not claim.has_citations or not claim.cited_source_indices:
            return ClaimGuardDetail(
                claim_text=claim.claim_text,
                raw_sentence=claim.raw_sentence,
                cited_sources=[],
                is_supported=False,
                hallucination_type=HallucinationType.UNCITED_CLAIM,
                reasons=["Factual medical claim contains no source citation."],
                entities=entities
            )

        # 2. Source index existence check
        valid_cits = [c for c in claim.cited_source_indices if c in source_map]
        invalid_cits = [c for c in claim.cited_source_indices if c not in source_map]
        if not valid_cits:
            for cit in invalid_cits:
                reasons.append(f"Cited [Source {cit}] does not exist in retrieved sources.")
            return ClaimGuardDetail(
                claim_text=claim.claim_text,
                raw_sentence=claim.raw_sentence,
                cited_sources=claim.cited_source_indices,
                is_supported=False,
                hallucination_type=HallucinationType.INVALID_SOURCE,
                reasons=reasons,
                entities=entities
            )

        # Gather source texts
        available_texts: List[Tuple[int, str]] = []
        for cit in valid_cits:
            txt = source_map[cit].get("text") or source_map[cit].get("preview_text") or ""
            if txt:
                available_texts.append((cit, txt))

        if not available_texts:
            # No text available (metadata only) -> allow fallback
            return ClaimGuardDetail(
                claim_text=claim.claim_text,
                raw_sentence=claim.raw_sentence,
                cited_sources=valid_cits,
                is_supported=True,
                hallucination_type=HallucinationType.NONE,
                best_support_score=1.0,
                entities=entities
            )

        # Evaluate support score and find best segment per cited source
        scores: Dict[int, float] = {}
        best_seg_per_src: Dict[int, Tuple[float, str]] = {}
        for cit, txt in available_texts:
            s_sim, best_seg = cls.find_best_source_segment(claim.claim_text, txt)
            scores[cit] = round(s_sim, 4)
            best_seg_per_src[cit] = (s_sim, best_seg)

        best_source_id = max(scores, key=scores.get)
        best_support_score = scores[best_source_id]
        _, best_matching_segment = best_seg_per_src[best_source_id]

        combined_cited_text = " ".join(t for _, t in available_texts)
        if len(available_texts) > 1:
            comb_sim, comb_seg = cls.find_best_source_segment(claim.claim_text, combined_cited_text)
            if comb_sim > best_support_score:
                best_support_score = comb_sim
                best_matching_segment = comb_seg

        # 3. Medication Hallucination Check
        hallucinated_meds = cls.check_medication_hallucinations(entities.medications, combined_cited_text)
        if hallucinated_meds:
            med_str = ", ".join(f"'{m}'" for m in hallucinated_meds)
            reasons.append(f"Medication hallucination: {med_str} not mentioned in cited source(s).")
            return ClaimGuardDetail(
                claim_text=claim.claim_text,
                raw_sentence=claim.raw_sentence,
                cited_sources=valid_cits,
                is_supported=False,
                hallucination_type=HallucinationType.MEDICATION_HALLUCINATION,
                reasons=reasons,
                support_scores=scores,
                best_support_score=best_support_score,
                best_source_id=best_source_id,
                entities=entities
            )

        # 4. Dosage Hallucination Check
        hallucinated_dosages = cls.check_dosage_hallucinations(entities.dosages, combined_cited_text)
        if hallucinated_dosages:
            dose_str = ", ".join(f"'{d}'" for d in hallucinated_dosages)
            reasons.append(f"Dosage hallucination: {dose_str} not substantiated by cited source(s).")
            return ClaimGuardDetail(
                claim_text=claim.claim_text,
                raw_sentence=claim.raw_sentence,
                cited_sources=valid_cits,
                is_supported=False,
                hallucination_type=HallucinationType.DOSAGE_HALLUCINATION,
                reasons=reasons,
                support_scores=scores,
                best_support_score=best_support_score,
                best_source_id=best_source_id,
                entities=entities
            )

        # 5. Negation Contradiction Check
        is_neg_contra, neg_expl = cls.check_negation_contradiction(
            claim.claim_text, entities, best_matching_segment
        )
        if is_neg_contra and neg_expl:
            reasons.append(neg_expl)
            return ClaimGuardDetail(
                claim_text=claim.claim_text,
                raw_sentence=claim.raw_sentence,
                cited_sources=valid_cits,
                is_supported=False,
                hallucination_type=HallucinationType.NEGATION_CONTRADICTION,
                reasons=reasons,
                support_scores=scores,
                best_support_score=best_support_score,
                best_source_id=best_source_id,
                entities=entities
            )

        # 6. Directional Contradiction Check
        is_dir_contra, dir_expl = cls.check_directional_contradiction(entities, best_matching_segment)
        if is_dir_contra and dir_expl:
            reasons.append(dir_expl)
            return ClaimGuardDetail(
                claim_text=claim.claim_text,
                raw_sentence=claim.raw_sentence,
                cited_sources=valid_cits,
                is_supported=False,
                hallucination_type=HallucinationType.DIRECTIONAL_CONTRADICTION,
                reasons=reasons,
                support_scores=scores,
                best_support_score=best_support_score,
                best_source_id=best_source_id,
                entities=entities
            )

        # 6B. Check if another uncited retrieved source actually has the evidence (Wrong Source detection)
        wrong_source_match: Optional[int] = None
        for other_id, other_src in source_map.items():
            if other_id not in valid_cits:
                other_txt = other_src.get("text") or other_src.get("preview_text") or ""
                if other_txt:
                    other_sim, _ = cls.find_best_source_segment(claim.claim_text, other_txt)
                    if other_sim >= cls.SUPPORT_SIMILARITY_THRESHOLD:
                        wrong_source_match = other_id
                        break

        # 6C. Clinical Target (Diagnosis & Recommendation) Grounding Check
        targets_to_check = entities.diagnoses + entities.recommendations
        if targets_to_check:
            src_lower = combined_cited_text.lower()
            unsupported_targets = []
            for target in targets_to_check:
                clean_target = target.strip().lower()
                tokens = [w for w in re.findall(r'\b[a-z]{3,}\b', clean_target) if w not in {"the", "for", "and", "with", "patient", "has", "been", "immediate", "proven", "permanently"}]
                if tokens and not any(t in src_lower for t in tokens):
                    unsupported_targets.append(clean_target)

            if unsupported_targets:
                if wrong_source_match is not None:
                    reasons.append(
                        f"Claim is unsupported by cited source(s) (wrong source attribution): Claim cites [Source {valid_cits[0]}], "
                        f"but evidence appears in [Source {wrong_source_match}]."
                    )
                    return ClaimGuardDetail(
                        claim_text=claim.claim_text,
                        raw_sentence=claim.raw_sentence,
                        cited_sources=valid_cits,
                        is_supported=False,
                        hallucination_type=HallucinationType.WRONG_SOURCE,
                        reasons=reasons,
                        support_scores=scores,
                        best_support_score=best_support_score,
                        best_source_id=best_source_id,
                        entities=entities
                    )

                h_type = HallucinationType.UNSUPPORTED_DIAGNOSIS if any(t in entities.diagnoses for t in unsupported_targets) else HallucinationType.UNSUPPORTED_RECOMMENDATION
                reasons.append(
                    f"Unsupported clinical target: {', '.join(unsupported_targets)} not substantiated by cited source(s)."
                )
                return ClaimGuardDetail(
                    claim_text=claim.claim_text,
                    raw_sentence=claim.raw_sentence,
                    cited_sources=valid_cits,
                    is_supported=False,
                    hallucination_type=h_type,
                    reasons=reasons,
                    support_scores=scores,
                    best_support_score=best_support_score,
                    best_source_id=best_source_id,
                    entities=entities
                )

        # 7. Semantic Grounding Threshold Check (0.65)
        if best_support_score < cls.SUPPORT_SIMILARITY_THRESHOLD:
            if wrong_source_match is not None:
                reasons.append(
                    f"Claim is unsupported by cited source(s) (wrong source attribution): Claim cites [Source {valid_cits[0]}], "
                    f"but evidence appears in [Source {wrong_source_match}]."
                )
                h_type = HallucinationType.WRONG_SOURCE
            else:
                reasons.append(
                    f"Claim is unsupported by cited source(s) (similarity: {best_support_score:.4f} < {cls.SUPPORT_SIMILARITY_THRESHOLD})."
                )
                h_type = HallucinationType.SEMANTIC_MISMATCH

            return ClaimGuardDetail(
                claim_text=claim.claim_text,
                raw_sentence=claim.raw_sentence,
                cited_sources=valid_cits,
                is_supported=False,
                hallucination_type=h_type,
                reasons=reasons,
                support_scores=scores,
                best_support_score=best_support_score,
                best_source_id=best_source_id,
                entities=entities
            )

        # Claim passed all verification checks!
        return ClaimGuardDetail(
            claim_text=claim.claim_text,
            raw_sentence=claim.raw_sentence,
            cited_sources=valid_cits,
            is_supported=True,
            hallucination_type=HallucinationType.NONE,
            support_scores=scores,
            best_support_score=best_support_score,
            best_source_id=best_source_id,
            entities=entities
        )

    @classmethod
    def guard_answer(
        cls,
        answer_text: Optional[str],
        retrieved_sources: Optional[List[Dict[str, Any]]]
    ) -> HallucinationGuardResult:
        """
        Executes the full hallucination protection pipeline on an answer text:
        1. Segments candidate claims.
        2. Validates evidence against cited sources.
        3. Identifies medication/dosage fabrications and contradictions.
        4. Filters unsupported claims or safely falls back.
        """
        if not answer_text or not isinstance(answer_text, str) or not answer_text.strip():
            return HallucinationGuardResult(
                is_safe=False,
                sanitized_answer=cls.SAFE_FALLBACK_TEXT,
                fallback_triggered=True,
                fallback_reason="Empty or non-string answer received."
            )

        sources = retrieved_sources if isinstance(retrieved_sources, list) else []
        source_map = {
            src.get("source_number", idx + 1): src
            for idx, src in enumerate(sources)
        }

        # Check refusal or non-contextual statements
        lower_ans = answer_text.lower()
        if any(p in lower_ans for p in [
            "could not be found in the available reference documents",
            "the available documents do not contain",
            "relevant medical information could not be found",
            "insufficient evidence to answer your question",
            "to prevent unsupported healthcare answers"
        ]):
            return HallucinationGuardResult(
                is_safe=True,
                sanitized_answer=answer_text,
                fallback_triggered=False
            )

        claims = CitationValidator.extract_claims(answer_text)
        total_claims = len(claims)

        # If no factual claims extracted (e.g. pure disclaimers or structural text)
        if total_claims == 0:
            return HallucinationGuardResult(
                is_safe=True,
                total_claims=0,
                sanitized_answer=answer_text
            )

        claim_details: List[ClaimGuardDetail] = []
        supported_count = 0
        unsupported_count = 0
        medication_halls = 0
        dosage_halls = 0
        contradictions = 0

        for clm in claims:
            detail = cls.inspect_claim(clm, source_map)
            claim_details.append(detail)

            if detail.is_supported:
                supported_count += 1
            else:
                unsupported_count += 1
                if detail.hallucination_type == HallucinationType.MEDICATION_HALLUCINATION:
                    medication_halls += 1
                elif detail.hallucination_type == HallucinationType.DOSAGE_HALLUCINATION:
                    dosage_halls += 1
                elif detail.hallucination_type in (
                    HallucinationType.NEGATION_CONTRADICTION,
                    HallucinationType.DIRECTIONAL_CONTRADICTION
                ):
                    contradictions += 1

        is_safe = (unsupported_count == 0)

        # Sanitization: prune unsupported/hallucinated sentences
        if unsupported_count == 0:
            sanitized = answer_text
            fallback_triggered = False
            fallback_reason = None
        elif supported_count == 0:
            # Complete hallucination / 0 supported claims -> trigger safe fallback
            sanitized = cls.SAFE_FALLBACK_TEXT
            fallback_triggered = True
            fallback_reason = "All factual claims were unsupported or contained hallucinations."
        else:
            # Partial support: remove unsupported claims
            cleaned = answer_text
            for d in claim_details:
                if not d.is_supported and d.raw_sentence:
                    cleaned = cleaned.replace(d.raw_sentence, "")

            # Clean excess whitespace
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
            substantive = re.sub(r'\[.*?\]', '', cleaned).strip(' \n\r\t.,;:')

            if len(substantive) < 25:
                sanitized = cls.SAFE_FALLBACK_TEXT
                fallback_triggered = True
                fallback_reason = "Pruning unsupported claims left insufficient grounded content."
            else:
                sanitized = cleaned
                fallback_triggered = False
                fallback_reason = None

        return HallucinationGuardResult(
            is_safe=is_safe,
            total_claims=total_claims,
            supported_claims=supported_count,
            unsupported_claims=unsupported_count,
            hallucinated_claims=unsupported_count,
            contradictions_detected=contradictions,
            medication_hallucinations=medication_halls,
            dosage_hallucinations=dosage_halls,
            claims=claim_details,
            sanitized_answer=sanitized,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason
        )
