"""
Clinical Grounding Verification & Hallucination Guardrails Engine (Phase 6.6).
Provides deterministic, sub-millisecond clinical grounding validation, directional
contradiction analysis, ungrounded entity detection, post-generation clinical safety
sanitization, and explicit negative scope boundary enforcement.
"""

import re
import time
from typing import List, Dict, Any, Optional, Tuple, Set

from backend.intelligence.verification_models import (
    GroundingVerificationStatus,
    ClinicalHallucinationType,
    SafetyPostScreenAction,
    ExtractedClinicalEntities,
    ClinicalVerificationClaim,
    ClinicalVerificationResult
)
from backend.intelligence.citation_models import (
    CitationAttributionReport,
    ClinicalClaimAttribution,
    CitationVerificationStatus,
    ClinicalClaimType
)
from backend.rag.prompt_builder import MEDICAL_DISCLAIMER


class ClinicalVerificationEngine:
    """
    Phase 6.6 Clinical Grounding Verification & Hallucination Guardrail Engine.

    Elevates clinical RAG evaluation from lexical citation checks to deep assertion-level
    grounding verification:
    1. Directional Contradiction Detection (trajectory inversion: UP vs DOWN).
    2. Polarity & Negation Conflict Detection (indicated vs contraindicated).
    3. Numerical Dosage & Measurement Verification (detects dose hallucinations).
    4. Pharmacological & Clinical Entity Grounding (unsubstantiated drugs/diagnoses).
    5. Post-Generation Prescriptive & Diagnostic Scrubbing (replaces absolute pronouncements).
    6. Explicit Negative Scope Boundary Enforcement (respects out-of-scope testing boundaries).
    7. Conservative Claim Pruning & Authoritative Fallback Halting.
    """

    SUPPORT_SIMILARITY_THRESHOLD: float = 0.65

    SAFE_FALLBACK_TEXT: str = (
        "Relevant medical information could not be found in the available reference documents. "
        "To prevent unsupported healthcare answers, generation was halted. "
        "Please refine your query or consult authorized clinical guidelines."
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

    # Pharmacological suffix patterns
    MEDICATION_SUFFIX_REGEX = re.compile(
        r'\b[A-Za-z]+(?:olol|pril|sartan|statin|gliptin|gliflozin|tide|prazole|cillin|mycin|micin|cycline|floxacin|mab|nib|vir|afil|zosin|triptan|dipine)\b',
        re.IGNORECASE
    )

    # Numerical dosage patterns: e.g., 500 mg, 10 mg, 0.5 mcg, 100 units, 5 ml
    DOSAGE_REGEX = re.compile(
        r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|units?|iu)(?:/(?:day|dose|kg))?\b',
        re.IGNORECASE
    )

    # Blood pressure patterns: e.g., 140/90 mmHg or 120/80
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
        re.compile(r'\b(?:cure|cures|cured|curing|proven\s+to\s+cure|permanently\s+cure)\s+([a-z0-9\s\-]{3,35})(?=[.,;]|\[|\band\b|$)', re.IGNORECASE)
    ]

    # Prohibited post-generation prescriptive phrases to sanitize
    PROHIBITED_PRESCRIBING_PATTERNS = [
        re.compile(r'\b(?:you\s+must\s+take|you\s+should\s+take|i\s+prescribe|take\s+\d+\s*mg|i\s+recommend\s+you\s+take)\b', re.IGNORECASE),
        re.compile(r'\b(?:you\s+need\s+to\s+start\s+taking|you\s+must\s+administer)\b', re.IGNORECASE)
    ]

    # Prohibited post-generation definitive diagnostic statements to sanitize
    PROHIBITED_DIAGNOSTIC_PATTERNS = [
        re.compile(r'\b(?:you\s+have\s+been\s+diagnosed\s+with|my\s+diagnosis\s+is\s+that\s+you\s+have|i\s+diagnose\s+you\s+with|based\s+on\s+what\s+you\s+said,\s+you\s+have)\b', re.IGNORECASE),
        re.compile(r'\byou\s+definitely\s+have\b', re.IGNORECASE),
        re.compile(r'\byou\s+have\s+(?=stage\s+[1-4]|type\s+[12]|severe|acute|chronic)', re.IGNORECASE),
        re.compile(r'\b(?:you\s+are\s+diagnosed\s+with|my\s+diagnosis\s+is|you\s+suffer\s+from)\b', re.IGNORECASE)
    ]

    # Document boundary patterns
    DOCUMENT_BOUNDARY_REGEX = re.compile(
        r'\b(?:testing\s+boundary|scope\s+boundary|testing\s+boundaries|boundary\b|boundaries\b|out\s+of\s+scope|outside\s+(?:the\s+)?scope|unsupported\s+by\s+this\s+document|treated\s+as\s+unsupported|not\s+contain\s+information\s+about|does\s+not\s+contain\s+information\s+about|not\s+covered\s+(?:in|by)\s+this\s+document|exclusion\s+criteria)\b',
        re.IGNORECASE
    )

    @classmethod
    def extract_clinical_entities(cls, text: str) -> ExtractedClinicalEntities:
        """Extracts pharmacological and clinical entities from text."""
        if not text or not isinstance(text, str):
            return ExtractedClinicalEntities()

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
        found_negations = [p for p in cls.NEGATION_PHRASES if p in text_lower]
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
            direction = "MIXED"
            direction_terms = ups + downs

        # 5. Diagnoses
        diagnoses = []
        for pat in cls.DIAGNOSIS_PATTERNS:
            for m in pat.finditer(text):
                diag_term = m.group(1).strip().lower()
                if len(diag_term) > 3 and diag_term not in ("the", "this", "that", "what", "which"):
                    diagnoses.append(diag_term)

        # 6. Recommendations
        recommendations = []
        for pat in cls.RECOMMENDATION_PATTERNS:
            for m in pat.finditer(text):
                rec_term = m.group(1).strip().lower()
                if len(rec_term) > 3:
                    recommendations.append(rec_term)

        return ExtractedClinicalEntities(
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
    def check_directional_contradiction(
        cls,
        claim_entities: ExtractedClinicalEntities,
        source_text: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Detects if clinical trajectory in claim inverts the trajectory in the reference text.
        e.g., Claim says 'increases blood pressure' while Source says 'decreases blood pressure'.
        """
        if not claim_entities.direction or claim_entities.direction == "MIXED":
            return False, None

        src_lower = source_text.lower()
        src_words = set(re.findall(r'\b[a-z0-9\-]+\b', src_lower))

        src_ups = bool(src_words & cls.DIRECTION_UP_TERMS)
        src_downs = bool(src_words & cls.DIRECTION_DOWN_TERMS)

        # If claim says UP but source only mentions DOWN
        if claim_entities.direction == "UP" and src_downs and not src_ups:
            return True, (
                f"Directional Contradiction: Claim asserts elevation/increase "
                f"({', '.join(claim_entities.direction_terms)}), but evidence indicates reduction/decrease."
            )

        # If claim says DOWN but source only mentions UP
        if claim_entities.direction == "DOWN" and src_ups and not src_downs:
            return True, (
                f"Directional Contradiction: Claim asserts reduction/decrease "
                f"({', '.join(claim_entities.direction_terms)}), but evidence indicates elevation/increase."
            )

        return False, None

    @classmethod
    def check_negation_conflict(
        cls,
        claim_entities: ExtractedClinicalEntities,
        source_text: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Detects polarity inversion between claim and reference passage.
        e.g., Claim asserts drug is indicated when source marks it contraindicated or to be avoided.
        """
        src_lower = source_text.lower()
        src_has_contraindication = any(
            p in src_lower for p in [
                "contraindicated", "contraindication", "contraindications",
                "not recommended", "avoid", "should not be used", "harmful"
            ]
        )

        if not claim_entities.has_negation and src_has_contraindication:
            # Claim recommends or asserts safe use, but source contraindicates
            return True, "Negation Contradiction: Evidence explicitly contraindicates or warns against the intervention."

        if claim_entities.has_negation and not src_has_contraindication:
            # Check if claim falsely asserts contraindication
            claim_contra = any(
                p in claim_entities.negation_terms
                for p in ["contraindicated", "contraindication", "should not", "avoid"]
            )
            if claim_contra and any(p in src_lower for p in ["indicated", "recommended", "first-line", "effective"]):
                return True, "Negation Contradiction: Claim asserts contraindication contrary to supporting guideline."

        return False, None

    @classmethod
    def check_dosage_discrepancy(
        cls,
        claim_entities: ExtractedClinicalEntities,
        source_text: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Checks if numerical dosages in the claim match quantities in the reference source.
        """
        if not claim_entities.dosages:
            return False, None

        src_lower = source_text.lower()
        unmatched_dosages = []

        for dose in claim_entities.dosages:
            dose_clean = dose.strip().lower()
            # Normalize whitespace: "10 mg" -> "10\s*mg"
            dose_num = re.search(r'\d+(?:\.\d+)?', dose_clean)
            if dose_num:
                num_str = dose_num.group(0)
                # Check if number appears anywhere in source
                if num_str not in src_lower:
                    unmatched_dosages.append(dose)

        if unmatched_dosages:
            return True, (
                f"Dosage Discrepancy: Numerical dosage/measurement {', '.join(unmatched_dosages)} "
                f"is not substantiated by the reference evidence."
            )

        return False, None

    @classmethod
    def check_entity_grounding(
        cls,
        claim_entities: ExtractedClinicalEntities,
        combined_evidence: str
    ) -> Tuple[bool, List[str]]:
        """
        Verifies that specific medications and diagnoses in the claim exist in the evidence.
        """
        if not combined_evidence:
            return False, ["Evidence context is empty."]

        ev_lower = combined_evidence.lower()
        unsupported = []

        # Check medications
        for med in claim_entities.medications:
            if med not in ev_lower:
                unsupported.append(f"Medication '{med}' is absent from retrieved evidence")

        # Check diagnostic assertions
        for diag in claim_entities.diagnoses:
            diag_words = [w for w in re.findall(r'\b[a-z]{3,}\b', diag) if w not in {"the", "and", "stage", "type"}]
            if diag_words and not any(w in ev_lower for w in diag_words):
                unsupported.append(f"Diagnostic term '{diag}' is unsupported by evidence")

        return len(unsupported) > 0, unsupported

    @classmethod
    def sanitize_prescriptions_and_diagnoses(cls, answer_text: str) -> Tuple[str, List[str], bool]:
        """
        Scans answer text for prohibited prescriptive imperatives or definitive diagnostic
        pronouncements, rewriting them to safe, objective clinical informational framing.
        """
        sanitized = answer_text
        warnings = []
        action_needed = False

        # 1. Sanitize prescriptive statements
        for pat in cls.PROHIBITED_PRESCRIBING_PATTERNS:
            if pat.search(sanitized):
                action_needed = True
                warnings.append("Unauthorized prescriptive phrasing detected and sanitized.")
                sanitized = pat.sub("clinical guidelines note that physicians may prescribe", sanitized)

        # 2. Sanitize diagnostic statements
        for pat in cls.PROHIBITED_DIAGNOSTIC_PATTERNS:
            if pat.search(sanitized):
                action_needed = True
                warnings.append("Unauthorized diagnostic pronouncement detected and sanitized.")
                sanitized = pat.sub("clinical evidence discusses", sanitized)

        return sanitized, warnings, action_needed

    @classmethod
    def check_negative_boundary(
        cls,
        query: str,
        answer_text: str,
        retrieved_sources: List[Dict[str, Any]]
    ) -> Tuple[bool, bool]:
        """
        Checks whether the query touches an explicit document exclusion/testing boundary
        and whether the answer appropriately reflects this negative boundary without hallucination.

        Returns:
            (has_boundary_in_doc, answer_reflects_boundary)
        """
        if not retrieved_sources:
            return False, False

        combined_src = " ".join(s.get("text", "") or s.get("preview_text", "") for s in retrieved_sources).lower()
        has_boundary = bool(cls.DOCUMENT_BOUNDARY_REGEX.search(combined_src))
        if not has_boundary:
            return False, False

        ans_lower = answer_text.lower()
        ans_reflects = bool(re.search(
            r'\b(?:not\s+contain|do\s+not\s+contain|does\s+not\s+contain|not\s+covered|unsupported|not\s+supported|testing\s+boundary|outside\s+(?:the\s+)?scope)\b',
            ans_lower
        ))

        return has_boundary, ans_reflects

    @classmethod
    def prune_ungrounded_content(
        cls,
        answer_text: str,
        verification_claims: List[ClinicalVerificationClaim],
        fallback_text: Optional[str] = None
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Prunes sentences containing ungrounded entities or contradictions.
        If all factual claims are ungrounded or if substantive content falls below
        acceptable clinical length, triggers safe fallback.
        """
        fallback = fallback_text or cls.SAFE_FALLBACK_TEXT

        factual_claims = [c for c in verification_claims if c.verification_status != GroundingVerificationStatus.EXEMPT_STRUCTURAL]
        if not factual_claims:
            return answer_text, False, None

        grounded_claims = [c for c in factual_claims if c.is_grounded]
        ungrounded_claims = [c for c in factual_claims if not c.is_grounded]

        # Case 1: 100% ungrounded
        if len(grounded_claims) == 0:
            return fallback, True, "All factual claims were ungrounded or contradictory."

        # Case 2: 100% grounded
        if len(ungrounded_claims) == 0:
            return answer_text, False, None

        # Case 3: Mixed support - prune ungrounded sentences
        cleaned = answer_text
        for claim in ungrounded_claims:
            if claim.raw_sentence and claim.raw_sentence in cleaned:
                cleaned = cleaned.replace(claim.raw_sentence, "")

        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        substantive = re.sub(r'\[.*?\]', '', cleaned).strip(' \n\r\t.,;:')

        if len(substantive) < 25:
            return fallback, True, "Pruning ungrounded claims left insufficient clinical content."

        return cleaned, False, None

    @classmethod
    def verify_and_guard(
        cls,
        answer_text: Optional[str],
        retrieved_sources: Optional[List[Dict[str, Any]]],
        attribution_report: Optional[CitationAttributionReport] = None,
        query: Optional[str] = None,
        intent: Optional[str] = None,
        safety_assessment: Optional[Any] = None,
        cumulative_profile: Optional[Any] = None
    ) -> ClinicalVerificationResult:
        """
        Main Phase 6.6 verification entrypoint.

        Executes comprehensive post-synthesis verification:
        - Evaluates claim-level attribution report or extracts claims
        - Detects directional contradictions and negation conflicts
        - Identifies dosage discrepancies and ungrounded entities
        - Sanitizes prohibited prescriptive and diagnostic assertions
        - Enforces explicit negative scope boundaries
        - Enforces medical disclaimer
        - Prunes ungrounded content or triggers safe fallback
        """
        t0 = time.perf_counter()

        # Handle empty/invalid input
        if not answer_text or not isinstance(answer_text, str) or not answer_text.strip():
            latency = (time.perf_counter() - t0) * 1000.0
            return ClinicalVerificationResult(
                is_verified_safe=False,
                overall_grounding_score=0.0,
                total_claims_analyzed=0,
                grounded_claims_count=0,
                ungrounded_claims_count=0,
                contradictions_count=0,
                hallucinations_detected=0,
                claim_verifications=[],
                action_taken=SafetyPostScreenAction.TRIGGER_FALLBACK,
                sanitized_answer=cls.SAFE_FALLBACK_TEXT,
                fallback_triggered=True,
                fallback_reason="Empty or non-string answer provided.",
                latency_ms=latency,
                metadata={"reason": "empty_input"}
            )

        sources = retrieved_sources or []
        source_map = {
            src.get("source_number", src.get("source_index", idx + 1)): src
            for idx, src in enumerate(sources)
        }
        combined_evidence = " ".join(s.get("text", "") or s.get("preview_text", "") for s in sources)

        # 1. Check if the answer is already a standard safe fallback or refusal
        lower_ans = answer_text.lower()
        if any(p in lower_ans for p in [
            "could not be found in the available reference documents",
            "the available documents do not contain",
            "relevant medical information could not be found",
            "insufficient evidence to answer your question",
            "to prevent unsupported healthcare answers",
            "emergency advisory",
            "support notice",
            "urgent toxicology"
        ]):
            latency = (time.perf_counter() - t0) * 1000.0
            return ClinicalVerificationResult(
                is_verified_safe=True,
                overall_grounding_score=1.0,
                total_claims_analyzed=0,
                grounded_claims_count=0,
                ungrounded_claims_count=0,
                contradictions_count=0,
                hallucinations_detected=0,
                claim_verifications=[],
                action_taken=SafetyPostScreenAction.ALLOW,
                sanitized_answer=answer_text,
                fallback_triggered=False,
                latency_ms=latency,
                metadata={"type": "standard_safe_advisory_or_fallback"}
            )

        # 2. Check Negative Document Boundary
        query_str = query or ""
        has_doc_boundary, ans_reflects_boundary = cls.check_negative_boundary(
            query=query_str,
            answer_text=answer_text,
            retrieved_sources=sources
        )
        negative_boundary_enforced = has_doc_boundary and ans_reflects_boundary

        # 3. Post-Generation Prescriptive and Diagnostic Sanitization
        sanitized_ans, sanitization_warnings, sanitization_performed = cls.sanitize_prescriptions_and_diagnoses(answer_text)
        action_taken = SafetyPostScreenAction.ALLOW
        if sanitization_performed:
            if any("prescriptive" in w for w in sanitization_warnings):
                action_taken = SafetyPostScreenAction.SANITIZE_PRESCRIPTION
            elif any("diagnostic" in w for w in sanitization_warnings):
                action_taken = SafetyPostScreenAction.SANITIZE_DIAGNOSIS

        # 4. Claim-Level Verification Loop
        verification_claims: List[ClinicalVerificationClaim] = []
        contradictions_count = 0
        hallucinations_count = 0

        # Leverage Phase 6.5 attribution report if available
        if attribution_report and attribution_report.claims:
            claims_to_verify = attribution_report.claims
        else:
            # Segment claims internally
            from backend.intelligence.citation_attribution import ClinicalCitationAttributionEngine
            attr_rep = ClinicalCitationAttributionEngine.attribute_and_validate(
                answer_text=sanitized_ans,
                retrieved_sources=sources
            )
            claims_to_verify = attr_rep.claims

        for clm in claims_to_verify:
            # Extract entities
            entities = cls.extract_clinical_entities(clm.claim_text)

            # Skip structural and disclaimer claims
            if clm.claim_type in (ClinicalClaimType.STRUCTURAL, ClinicalClaimType.LIMITATION_OR_DISCLAIMER):
                verification_claims.append(ClinicalVerificationClaim(
                    claim_id=clm.claim_id,
                    claim_text=clm.claim_text,
                    raw_sentence=clm.raw_sentence,
                    verification_status=GroundingVerificationStatus.EXEMPT_STRUCTURAL,
                    hallucination_type=ClinicalHallucinationType.NONE,
                    is_grounded=True,
                    confidence_score=1.0,
                    supporting_sources=clm.cited_source_indices,
                    entities=entities
                ))
                continue

            # Gather cited text
            cited_texts = []
            for src_idx in clm.cited_source_indices:
                if src_idx in source_map:
                    s_txt = source_map[src_idx].get("text", "") or source_map[src_idx].get("preview_text", "")
                    if s_txt:
                        cited_texts.append(s_txt)
            target_evidence = " ".join(cited_texts) if cited_texts else combined_evidence

            # A. Directional contradiction check
            is_dir_contra, dir_reason = cls.check_directional_contradiction(entities, target_evidence)
            if is_dir_contra:
                contradictions_count += 1
                verification_claims.append(ClinicalVerificationClaim(
                    claim_id=clm.claim_id,
                    claim_text=clm.claim_text,
                    raw_sentence=clm.raw_sentence,
                    verification_status=GroundingVerificationStatus.DIRECTIONAL_CONTRADICTION,
                    hallucination_type=ClinicalHallucinationType.DIRECTIONAL_INVERSION,
                    is_grounded=False,
                    confidence_score=0.0,
                    discrepancy_details=[dir_reason],
                    supporting_sources=[],
                    entities=entities
                ))
                continue

            # B. Negation conflict check
            is_neg_contra, neg_reason = cls.check_negation_conflict(entities, target_evidence)
            if is_neg_contra:
                contradictions_count += 1
                verification_claims.append(ClinicalVerificationClaim(
                    claim_id=clm.claim_id,
                    claim_text=clm.claim_text,
                    raw_sentence=clm.raw_sentence,
                    verification_status=GroundingVerificationStatus.DIRECTIONAL_CONTRADICTION,
                    hallucination_type=ClinicalHallucinationType.NEGATION_CONFLICT,
                    is_grounded=False,
                    confidence_score=0.0,
                    discrepancy_details=[neg_reason],
                    supporting_sources=[],
                    entities=entities
                ))
                continue

            # C. Dosage discrepancy check
            has_dose_err, dose_reason = cls.check_dosage_discrepancy(entities, target_evidence)
            if has_dose_err:
                hallucinations_count += 1
                verification_claims.append(ClinicalVerificationClaim(
                    claim_id=clm.claim_id,
                    claim_text=clm.claim_text,
                    raw_sentence=clm.raw_sentence,
                    verification_status=GroundingVerificationStatus.NUMERICAL_DISCREPANCY,
                    hallucination_type=ClinicalHallucinationType.DOSAGE_DISCREPANCY,
                    is_grounded=False,
                    confidence_score=0.0,
                    discrepancy_details=[dose_reason],
                    supporting_sources=[],
                    entities=entities
                ))
                continue

            # D. Entity presence check (medications & diagnoses)
            has_unsub_entity, entity_reasons = cls.check_entity_grounding(entities, combined_evidence)
            if has_unsub_entity:
                hallucinations_count += 1
                h_type = (
                    ClinicalHallucinationType.MEDICATION_FABRICATION
                    if any("Medication" in r for r in entity_reasons)
                    else ClinicalHallucinationType.UNGROUNDED_DIAGNOSTIC_CLAIM
                )
                verification_claims.append(ClinicalVerificationClaim(
                    claim_id=clm.claim_id,
                    claim_text=clm.claim_text,
                    raw_sentence=clm.raw_sentence,
                    verification_status=GroundingVerificationStatus.UNSUBSTANTIATED_ENTITY,
                    hallucination_type=h_type,
                    is_grounded=False,
                    confidence_score=0.0,
                    discrepancy_details=entity_reasons,
                    supporting_sources=[],
                    entities=entities
                ))
                continue

            # E. Baseline attribution status from Phase 6.5
            if not clm.is_supported or clm.verification_status in (
                CitationVerificationStatus.UNSUPPORTED,
                CitationVerificationStatus.INVALID_SOURCE
            ):
                hallucinations_count += 1
                verification_claims.append(ClinicalVerificationClaim(
                    claim_id=clm.claim_id,
                    claim_text=clm.claim_text,
                    raw_sentence=clm.raw_sentence,
                    verification_status=GroundingVerificationStatus.UNGROUNDED,
                    hallucination_type=ClinicalHallucinationType.UNSUBSTANTIATED_ASSERTION,
                    is_grounded=False,
                    confidence_score=clm.best_support_score,
                    discrepancy_details=clm.unsupported_reasons or ["Claim lacks evidence support"],
                    supporting_sources=[],
                    entities=entities
                ))
                continue

            # F. Verified claim
            status = (
                GroundingVerificationStatus.PARTIALLY_GROUNDED
                if clm.verification_status == CitationVerificationStatus.PARTIALLY_VERIFIED
                else GroundingVerificationStatus.GROUNDED
            )
            verification_claims.append(ClinicalVerificationClaim(
                claim_id=clm.claim_id,
                claim_text=clm.claim_text,
                raw_sentence=clm.raw_sentence,
                verification_status=status,
                hallucination_type=ClinicalHallucinationType.NONE,
                is_grounded=True,
                confidence_score=clm.best_support_score,
                discrepancy_details=[],
                supporting_sources=clm.cited_source_indices,
                entities=entities
            ))

        # 4.5. Check Cumulative Profile Multi-Turn Contraindications (Phase 6.9)
        if cumulative_profile:
            active_conds = getattr(cumulative_profile, "active_conditions", [])
            allergies = getattr(cumulative_profile, "confirmed_allergies", [])
            conds_lower = {str(c).lower() for c in active_conds}
            allergies_lower = {str(a).lower() for a in allergies}
            ans_lower = sanitized_ans.lower()

            # A. Renal impairment (CKD / AKI) vs NSAIDs
            renal_keys = {"chronic kidney disease", "ckd", "renal failure", "renal impairment", "acute kidney injury", "aki"}
            if bool(conds_lower & renal_keys):
                nsaid_terms = ["ibuprofen", "naproxen", "celecoxib", "diclofenac", "indomethacin", "ketorolac", "meloxicam"]
                found_nsaids = [n for n in nsaid_terms if re.search(rf'\b{re.escape(n)}\b', ans_lower)]
                if found_nsaids and not any(p in ans_lower for p in ["contraindicated", "avoid", "caution", "not recommended", "renal risk", "kidney risk"]):
                    contradictions_count += 1
                    verification_claims.append(ClinicalVerificationClaim(
                        claim_id=f"cumul_contra_{len(verification_claims)+1}",
                        claim_text=f"Recommends {', '.join(found_nsaids)} despite patient having chronic kidney disease/renal impairment.",
                        raw_sentence=f"Cross-turn contraindication: {', '.join(found_nsaids)} in renal disease",
                        verification_status=GroundingVerificationStatus.DIRECTIONAL_CONTRADICTION,
                        hallucination_type=ClinicalHallucinationType.NEGATION_CONFLICT,
                        is_grounded=False,
                        confidence_score=0.0,
                        discrepancy_details=[f"Cross-Turn Renal Contraindication: NSAIDs ({', '.join(found_nsaids)}) contraindicated in patient with renal impairment."],
                        supporting_sources=[],
                        entities=ExtractedClinicalEntities(medications=found_nsaids)
                    ))

            # B. Penicillin allergy vs Beta-lactam antibiotics
            if "penicillin" in allergies_lower or "amoxicillin" in allergies_lower:
                pen_terms = ["penicillin", "amoxicillin", "ampicillin", "augmentin"]
                found_pen = [p for p in pen_terms if re.search(rf'\b{re.escape(p)}\b', ans_lower)]
                if found_pen and not any(p in ans_lower for p in ["allergic", "allergy", "contraindicated", "avoid", "do not take"]):
                    contradictions_count += 1
                    verification_claims.append(ClinicalVerificationClaim(
                        claim_id=f"cumul_allergy_{len(verification_claims)+1}",
                        claim_text=f"Recommends {', '.join(found_pen)} despite confirmed penicillin allergy.",
                        raw_sentence=f"Cross-turn allergy contraindication: {', '.join(found_pen)} in penicillin allergy",
                        verification_status=GroundingVerificationStatus.DIRECTIONAL_CONTRADICTION,
                        hallucination_type=ClinicalHallucinationType.NEGATION_CONFLICT,
                        is_grounded=False,
                        confidence_score=0.0,
                        discrepancy_details=[f"Cross-Turn Allergy Contraindication: Penicillin-class antibiotics ({', '.join(found_pen)}) contraindicated in penicillin-allergic patient."],
                        supporting_sources=[],
                        entities=ExtractedClinicalEntities(medications=found_pen)
                    ))

        # 5. Calculate Metrics
        total_claims = len(verification_claims)
        factual_claims = [c for c in verification_claims if c.verification_status != GroundingVerificationStatus.EXEMPT_STRUCTURAL]
        grounded_claims = [c for c in factual_claims if c.is_grounded]
        ungrounded_claims = [c for c in factual_claims if not c.is_grounded]

        grounded_count = len(grounded_claims)
        ungrounded_count = len(ungrounded_claims)

        overall_grounding_score = (
            grounded_count / len(factual_claims)
            if factual_claims else 1.0
        )

        # 6. Content Pruning & Safe Fallback Determination
        final_answer, fallback_triggered, fallback_reason = cls.prune_ungrounded_content(
            answer_text=sanitized_ans,
            verification_claims=verification_claims
        )

        contra_claims = [c for c in verification_claims if c.claim_id.startswith("cumul_contra") or c.claim_id.startswith("cumul_allergy")]
        if contra_claims:
            contra_details = [c.discrepancy_details[0] for c in contra_claims if c.discrepancy_details]
            advisory = f"CLINICAL CAUTION & CONTRAINDICATION: {'; '.join(contra_details)}"
            final_answer = f"{advisory}\n\n{final_answer}"

        if fallback_triggered:
            action_taken = SafetyPostScreenAction.TRIGGER_FALLBACK
            is_safe = False
        elif ungrounded_count > 0:
            action_taken = SafetyPostScreenAction.PRUNE_UNSUPPORTED
            is_safe = True
        else:
            is_safe = True

        # 7. Enforce Medical Disclaimer
        disclaimer_enforced = False
        if not fallback_triggered:
            if MEDICAL_DISCLAIMER not in final_answer and not any(k in final_answer for k in ("EMERGENCY ADVISORY", "SUPPORT NOTICE", "URGENT TOXICOLOGY")):
                final_answer = f"{final_answer}\n\n{MEDICAL_DISCLAIMER}"
                disclaimer_enforced = True
            else:
                disclaimer_enforced = True

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ClinicalVerificationResult(
            is_verified_safe=is_safe,
            overall_grounding_score=overall_grounding_score,
            total_claims_analyzed=total_claims,
            grounded_claims_count=grounded_count,
            ungrounded_claims_count=ungrounded_count,
            contradictions_count=contradictions_count,
            hallucinations_detected=hallucinations_count,
            claim_verifications=verification_claims,
            action_taken=action_taken,
            sanitized_answer=final_answer,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
            negative_boundary_enforced=negative_boundary_enforced,
            disclaimer_enforced=disclaimer_enforced,
            latency_ms=latency_ms,
            metadata={
                "sanitization_warnings": sanitization_warnings,
                "sanitization_performed": sanitization_performed,
                "negative_boundary_detected": has_doc_boundary,
                "intent": intent,
            }
        )
