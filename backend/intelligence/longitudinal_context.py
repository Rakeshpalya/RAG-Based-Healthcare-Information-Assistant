"""
Longitudinal Clinical Context & Multi-Turn Interaction Memory Engine (Phase 6.9).

Provides deterministic, sub-millisecond multi-turn dialogue state tracking,
clinical entity extraction, cumulative patient profile assembly, contextual
query reformulation, and cross-turn contraindication propagation.
"""

import re
import time
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple

from backend.intelligence.context_models import (
    ClinicalEntityType,
    EntityTemporalState,
    ClinicalEntity,
    CumulativeClinicalProfile,
    TurnContextResolution,
    DialogueStateAuditRecord,
    ContraindicationAlert
)


class ClinicalContextEngine:
    """
    Phase 6.9 Longitudinal Clinical Context Engine.
    Executes in pure Python with deterministic rule-based algorithms, zero LLM dependency,
    and sub-millisecond execution overhead.
    """

    MAX_WINDOW_TURNS: int = 6  # Bounded window: last 6 turns (3 dialogue cycles)

    # --------------------------------------------------------------------------
    # Curated Clinical Entity Dictionaries & Patterns
    # --------------------------------------------------------------------------
    KNOWN_CONDITIONS: Dict[str, str] = {
        # Renal & Urinary
        "chronic kidney disease": "Chronic Kidney Disease",
        "ckd": "Chronic Kidney Disease",
        "renal failure": "Renal Failure",
        "renal impairment": "Renal Impairment",
        "kidney disease": "Kidney Disease",
        "acute kidney injury": "Acute Kidney Injury",
        "aki": "Acute Kidney Injury",
        "nephropathy": "Nephropathy",
        "end stage renal disease": "End Stage Renal Disease",
        "esrd": "End Stage Renal Disease",
        # Cardiovascular
        "hypertension": "Hypertension",
        "high blood pressure": "Hypertension",
        "htn": "Hypertension",
        "heart failure": "Heart Failure",
        "congestive heart failure": "Congestive Heart Failure",
        "chf": "Congestive Heart Failure",
        "coronary artery disease": "Coronary Artery Disease",
        "cad": "Coronary Artery Disease",
        "atrial fibrillation": "Atrial Fibrillation",
        "afib": "Atrial Fibrillation",
        "myocardial infarction": "Myocardial Infarction",
        "heart attack": "Myocardial Infarction",
        "stroke": "Stroke",
        "transient ischemic attack": "Transient Ischemic Attack",
        "tia": "Transient Ischemic Attack",
        # Endocrine & Metabolic
        "diabetes": "Diabetes Mellitus",
        "type 2 diabetes": "Type 2 Diabetes",
        "t2d": "Type 2 Diabetes",
        "t2dm": "Type 2 Diabetes",
        "type 1 diabetes": "Type 1 Diabetes",
        "t1d": "Type 1 Diabetes",
        "hyperlipidemia": "Hyperlipidemia",
        "dyslipidemia": "Dyslipidemia",
        "high cholesterol": "Hyperlipidemia",
        "hypothyroidism": "Hypothyroidism",
        "hyperthyroidism": "Hyperthyroidism",
        # Respiratory
        "asthma": "Asthma",
        "copd": "Chronic Obstructive Pulmonary Disease",
        "chronic bronchitis": "Chronic Bronchitis",
        "emphysema": "Emphysema",
        "pneumonia": "Pneumonia",
        "pulmonary embolism": "Pulmonary Embolism",
        # Gastrointestinal & Hepatic
        "peptic ulcer": "Peptic Ulcer Disease",
        "peptic ulcer disease": "Peptic Ulcer Disease",
        "pud": "Peptic Ulcer Disease",
        "gerd": "Gastroesophageal Reflux Disease",
        "acid reflux": "Gastroesophageal Reflux Disease",
        "cirrhosis": "Cirrhosis",
        "liver cirrhosis": "Cirrhosis",
        "liver failure": "Liver Failure",
        "chronic liver failure": "Chronic Liver Failure",
        "liver disease": "Liver Disease",
        "hepatic impairment": "Hepatic Impairment",
        "hepatitis": "Hepatitis",
        "inflammatory bowel disease": "Inflammatory Bowel Disease",
        "ibd": "Inflammatory Bowel Disease",
        "crohn's": "Crohn's Disease",
        "ulcerative colitis": "Ulcerative Colitis",
        # Neurological & Psychiatric
        "epilepsy": "Epilepsy",
        "seizure disorder": "Seizure Disorder",
        "migraine": "Migraine",
        "dementia": "Dementia",
        "alzheimer's": "Alzheimer's Disease",
        "parkinson's": "Parkinson's Disease",
        "depression": "Major Depressive Disorder",
        "anxiety": "Anxiety Disorder",
        # Autoimmune & Others
        "rheumatoid arthritis": "Rheumatoid Arthritis",
        "lupus": "Systemic Lupus Erythematosus",
        "sle": "Systemic Lupus Erythematosus",
        "osteoporosis": "Osteoporosis",
        "gout": "Gout"
    }

    KNOWN_MEDICATIONS: Set[str] = {
        # Antihypertensives & Cardiovascular
        "amlodipine", "lisinopril", "enalapril", "losartan", "valsartan", "candesartan",
        "hydrochlorothiazide", "hctz", "chlorthalidone", "atenolol", "metoprolol", "propranolol",
        "carvedilol", "labetalol", "diltiazem", "verapamil", "nifedipine", "clonidine",
        "hydralazine", "spironolactone", "ramipril", "benazepril", "furosemide", "lasix",
        "torsemide", "bumetanide", "digoxin", "nitroglycerin", "isosorbide",
        # Antidiabetics
        "metformin", "glipizide", "glyburide", "glimepiride", "pioglitazone", "rosiglitazone",
        "sitagliptin", "saxagliptin", "linagliptin", "empagliflozin", "jardiance", "dapagliflozin",
        "farxiga", "canagliflozin", "invokana", "liraglutide", "victoza", "semaglutide", "ozempic",
        "wegovy", "rybelsus", "mounjaro", "tirzepatide", "dulaglutide", "trulicity", "insulin",
        "glargine", "lantus", "lispro", "humalog", "aspart", "novolog", "detemir", "levemir",
        # Statins & Lipid Lowering
        "atorvastatin", "lipitor", "simvastatin", "zocor", "rosuvastatin", "crestor",
        "pravastatin", "lovastatin", "ezetimibe", "fenofibrate", "gemfibrozil",
        # Analgesics & Anti-inflammatories
        "aspirin", "ibuprofen", "advil", "motrin", "naproxen", "aleve", "acetaminophen",
        "tylenol", "paracetamol", "celecoxib", "celebrex", "meloxicam", "mobic", "diclofenac",
        "voltaren", "indomethacin", "ketorolac", "toradol", "tramadol", "codeine", "morphine",
        "oxycodone", "hydrocodone", "fentanyl", "buprenorphine",
        # Antibiotics & Antivirals
        "amoxicillin", "ampicillin", "penicillin", "augmentin", "cephalexin", "keflex",
        "cefuroxime", "ceftriaxone", "azithromycin", "zithromax", "clarithromycin", "ciprofloxacin",
        "cipro", "levofloxacin", "levaquin", "doxycycline", "trimethoprim", "sulfamethoxazole",
        "bactrim", "septra", "metronidazole", "flagyl", "vancomycin", "acyclovir", "valacyclovir",
        "valtrex", "oseltamivir", "tamiflu", "paxlovid", "remdesivir",
        # Anticoagulants & Antiplatelets
        "warfarin", "coumadin", "heparin", "enoxaparin", "lovenox", "apixaban", "eliquis",
        "rivaroxaban", "xarelto", "dabigatran", "pradaxa", "clopidogrel", "plavix", "ticagrelor",
        # Gastrointestinal & Steroids
        "omeprazole", "prilosec", "pantoprazole", "protonix", "esomeprazole", "nexium",
        "lansoprazole", "prevacid", "famotidine", "pepcid", "prednisone", "prednisolone",
        "dexamethasone", "hydrocortisone", "budesonide",
        # Respiratory & Psychiatric
        "albuterol", "ventolin", "proair", "ipratropium", "atrovent", "tiotropium", "spiriva",
        "montelukast", "singulair", "fluticasone", "flovent", "flonase", "levothyroxine",
        "synthroid", "gabapentin", "neurontin", "pregabalin", "lyrica", "sertraline", "zoloft",
        "fluoxetine", "prozac", "escitalopram", "lexapro", "citalopram", "celexa", "duloxetine",
        "cymbalta", "venlafaxine", "effexor", "bupropion", "wellbutrin"
    }

    KNOWN_ALLERGIES: Set[str] = {
        "penicillin", "amoxicillin", "ampicillin", "cephalosporin", "cephalexin",
        "sulfa", "sulfonamides", "bactrim", "aspirin", "nsaids", "ibuprofen",
        "codeine", "morphine", "opioids", "iodine", "contrast", "radiocontrast",
        "latex", "peanuts", "tree nuts", "shellfish", "eggs"
    }

    KNOWN_SYMPTOMS: Set[str] = {
        "cough", "dry cough", "productive cough", "fever", "chills", "dyspnea",
        "shortness of breath", "chest pain", "angina", "palpitations", "fatigue",
        "malaise", "headache", "dizziness", "lightheadedness", "nausea", "vomiting",
        "diarrhea", "constipation", "abdominal pain", "joint pain", "arthralgia",
        "back pain", "edema", "swelling", "weight gain", "weight loss", "rash",
        "pruritus", "itching", "sore throat", "wheezing", "hemoptysis", "hematuria"
    }

    KNOWN_RISK_FACTORS: Dict[str, str] = {
        "pregnant": "Pregnancy",
        "pregnancy": "Pregnancy",
        "weeks pregnant": "Pregnancy",
        "week pregnant": "Pregnancy",
        "trimester": "Pregnancy",
        "breastfeeding": "Lactation",
        "nursing": "Lactation",
        "elderly": "Geriatric",
        "older adult": "Geriatric",
        "smoker": "Tobacco Use",
        "smoking": "Tobacco Use",
        "alcohol": "Alcohol Consumption",
        "obese": "Obesity",
        "obesity": "Obesity",
        "immunocompromised": "Immunocompromised",
        "immunosuppressed": "Immunocompromised",
        "chemotherapy": "Active Chemotherapy",
        "transplant": "Organ Transplant Recipient"
    }

    # Negation trigger patterns
    NEGATION_REGEX = re.compile(
        r'\b(?:no\s+history\s+of|denies|denied|denying|negative\s+for|without|does\s+not\s+have|'
        r'doesn\'t\s+have|do\s+not\s+have|don\'t\s+have|do\s+not\s+take|don\'t\s+take|not\s+taking|'
        r'has\s+no|not\s+diagnosed\s+with|never\s+had|free\s+of|'
        r'not\s+allergic|no\s+allergy|no\s+known\s+allergies|nka|nkda)\b',
        re.IGNORECASE
    )

    # Allergy explicit affirmation patterns
    ALLERGY_REGEX = re.compile(
        r'\b(?:allergic\s+to|allergy\s+to|has\s+an\s+allergy\s+to|anaphylaxis\s+to|'
        r'reaction\s+to|intolerant\s+to)\s+([a-z0-9\s\-]+?)(?=[.,;]|\[|\band\b|$)',
        re.IGNORECASE
    )

    # Pronoun / follow-up demonstrative patterns
    FOLLOW_UP_INDICATORS = re.compile(
        r'\b(it|this|that|these|those|them|they|the\s+medication|the\s+drug|the\s+treatment|'
        r'the\s+condition|the\s+disease|the\s+illness|such\s+condition|same|dosage|dose|'
        r'side\s+effects?|adverse\s+effects?|alternatives?|contraindications?|interactions?|'
        r'first-line|how\s+long|how\s+often|frequency|monitoring|its|their|diet|eat|food|'
        r'lifestyle|symptoms?)\b',
        re.IGNORECASE
    )

    # Elliptical inquiry patterns (short follow-ups lacking subject)
    ELLIPTICAL_PATTERNS = [
        re.compile(r'^(?:what\s+about\s+the\s+)?side\s+effects\??$', re.IGNORECASE),
        re.compile(r'^(?:what\s+is\s+the\s+)?(?:dose|dosage)\??$', re.IGNORECASE),
        re.compile(r'^(?:what\s+are\s+the\s+)?alternatives\??$', re.IGNORECASE),
        re.compile(r'^(?:how\s+often\s+to\s+)?monitor\??$', re.IGNORECASE),
        re.compile(r'^(?:is\s+it\s+)?safe\??$', re.IGNORECASE),
        re.compile(r'^(?:can\s+we\s+)?switch\??$', re.IGNORECASE),
        re.compile(r'^(?:what\s+are\s+)?contraindications\??$', re.IGNORECASE)
    ]

    @classmethod
    def resolve_context(
        cls,
        query: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[Any] = None,
        current_turn: Optional[str] = None,
    ) -> TurnContextResolution:
        """
        Main Phase 6.9 entrypoint. Resolves multi-turn context, builds cumulative profile,
        reformulates query if follow-up, and exports inherited safety contraindications.

        Args:
            query: Current user query string (or current_turn).
            conversation_history: Optional prior messages list [{role: str, content: str}].
            tenant_id: Optional multi-tenant organizational scope.
            session_id: Optional session identifier.
            user_id: Optional user identifier.
            current_turn: Alias for query.

        Returns:
            TurnContextResolution with effective_query, cumulative_profile, and safety flags.
        """
        t0 = time.perf_counter()
        raw_text = query if query is not None else (current_turn or "")
        raw_query = raw_text.strip()
        history = conversation_history or []

        # Graceful handling for single-turn requests
        if not history or not isinstance(history, list) or len(history) == 0:
            profile = cls._extract_single_turn_profile(raw_query)
            contraindications: List[str] = []
            alerts: List[ContraindicationAlert] = []
            latency = (time.perf_counter() - t0) * 1000.0
            return TurnContextResolution(
                original_query=raw_query,
                effective_query=raw_query,
                is_follow_up=False,
                resolved_topic=None,
                prior_turn_count=0,
                cumulative_profile=profile,
                inherited_contraindications=contraindications,
                contraindication_alerts=alerts,
                resolution_confidence=1.0,
                latency_ms=latency
            )

        # 1. Bounded Windowing: Restrict to last MAX_WINDOW_TURNS
        bounded_history = cls._bound_history(history)
        prior_turn_count = len(bounded_history)

        # 2. Extract Entities Across All Bounded Turns + Current Query
        extracted_entities: List[ClinicalEntity] = []
        for idx, turn in enumerate(bounded_history):
            content = turn.get("content") or turn.get("text") or ""
            role = turn.get("role") or turn.get("sender") or "user"
            # We extract from user disclosures and assistant prior topics
            turn_entities = cls._extract_turn_entities(str(content), turn_index=idx, role=role)
            extracted_entities.extend(turn_entities)

        # Extract from current query as current turn index
        current_turn_idx = prior_turn_count
        current_query_entities = cls._extract_turn_entities(raw_query, turn_index=current_turn_idx, role="user")
        extracted_entities.extend(current_query_entities)

        # 3. Assemble Cumulative Profile
        profile = cls._assemble_cumulative_profile(extracted_entities)

        # 4. Contextual Query Disambiguation & Follow-Up Reformulation
        is_follow_up, resolved_topic, effective_query, conf = cls._reformulate_query(
            raw_query=raw_query,
            bounded_history=bounded_history,
            profile=profile
        )

        # 5. Determine Inherited Contraindications from Cumulative Profile
        contraindications, alerts = cls._evaluate_inherited_contraindications(profile)

        latency = (time.perf_counter() - t0) * 1000.0

        return TurnContextResolution(
            original_query=raw_query,
            effective_query=effective_query,
            is_follow_up=is_follow_up,
            resolved_topic=resolved_topic,
            prior_turn_count=prior_turn_count,
            cumulative_profile=profile,
            inherited_contraindications=contraindications,
            contraindication_alerts=alerts,
            resolution_confidence=conf,
            latency_ms=latency
        )

    @classmethod
    def _bound_history(cls, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Restricts history to the most recent bounded turns (default: 6 turns)."""
        valid_turns = []
        for msg in history:
            if isinstance(msg, dict):
                content = msg.get("content") or msg.get("text") or ""
                if isinstance(content, str) and content.strip():
                    valid_turns.append(msg)
        return valid_turns[-cls.MAX_WINDOW_TURNS:]

    @classmethod
    def _extract_turn_entities(cls, text: str, turn_index: int, role: str) -> List[ClinicalEntity]:
        """Extracts conditions, medications, symptoms, allergies, and risk factors from a single turn."""
        entities: List[ClinicalEntity] = []
        if not text:
            return entities

        text_lower = text.lower()

        # Check for global negation in sentence fragments
        sentences = re.split(r'[.!?;\n]+', text)

        for sentence in sentences:
            s_clean = sentence.strip().lower()
            if not s_clean:
                continue

            is_sentence_negated = bool(cls.NEGATION_REGEX.search(s_clean))

            # 1. Allergies (explicit patterns)
            allergy_match = cls.ALLERGY_REGEX.findall(s_clean)
            for raw_allergen in allergy_match:
                allergen_clean = raw_allergen.strip()
                for known_a in cls.KNOWN_ALLERGIES:
                    if known_a in allergen_clean:
                        entities.append(ClinicalEntity(
                            name=known_a.capitalize(),
                            entity_type=ClinicalEntityType.ALLERGY,
                            temporal_state=EntityTemporalState.ACTIVE,
                            turn_index=turn_index,
                            negated=is_sentence_negated,
                            source_text=sentence.strip()
                        ))

            # Direct allergy word check
            if "allerg" in s_clean:
                for known_a in cls.KNOWN_ALLERGIES:
                    if re.search(rf'\b{re.escape(known_a)}\b', s_clean):
                        entities.append(ClinicalEntity(
                            name=known_a.capitalize(),
                            entity_type=ClinicalEntityType.ALLERGY,
                            temporal_state=EntityTemporalState.ACTIVE,
                            turn_index=turn_index,
                            negated=is_sentence_negated,
                            source_text=sentence.strip()
                        ))

            # 2. Conditions
            for cond_key, canonical_name in cls.KNOWN_CONDITIONS.items():
                if cond_key not in s_clean:
                    continue
                pattern = rf'\b{re.escape(cond_key)}\b'
                if re.search(pattern, s_clean):
                    entities.append(ClinicalEntity(
                        name=canonical_name,
                        entity_type=ClinicalEntityType.CONDITION,
                        temporal_state=EntityTemporalState.NEGATED if is_sentence_negated else EntityTemporalState.ACTIVE,
                        turn_index=turn_index,
                        negated=is_sentence_negated,
                        source_text=sentence.strip()
                    ))

            # 3. Medications
            for med in cls.KNOWN_MEDICATIONS:
                if med not in s_clean:
                    continue
                pattern = rf'\b{re.escape(med)}\b'
                if re.search(pattern, s_clean):
                    entities.append(ClinicalEntity(
                        name=med.capitalize(),
                        entity_type=ClinicalEntityType.MEDICATION,
                        temporal_state=EntityTemporalState.NEGATED if is_sentence_negated else EntityTemporalState.ACTIVE,
                        turn_index=turn_index,
                        negated=is_sentence_negated,
                        source_text=sentence.strip()
                    ))

            # 4. Symptoms
            for sym in cls.KNOWN_SYMPTOMS:
                if sym not in s_clean:
                    continue
                pattern = rf'\b{re.escape(sym)}\b'
                if re.search(pattern, s_clean):
                    entities.append(ClinicalEntity(
                        name=sym.capitalize(),
                        entity_type=ClinicalEntityType.SYMPTOM,
                        temporal_state=EntityTemporalState.NEGATED if is_sentence_negated else EntityTemporalState.ACTIVE,
                        turn_index=turn_index,
                        negated=is_sentence_negated,
                        source_text=sentence.strip()
                    ))

            # 5. Risk Factors / Demographics
            for rf_key, canonical_rf in cls.KNOWN_RISK_FACTORS.items():
                if rf_key not in s_clean:
                    continue
                pattern = rf'\b{re.escape(rf_key)}\b'
                if re.search(pattern, s_clean):
                    entities.append(ClinicalEntity(
                        name=canonical_rf,
                        entity_type=ClinicalEntityType.RISK_FACTOR,
                        temporal_state=EntityTemporalState.NEGATED if is_sentence_negated else EntityTemporalState.ACTIVE,
                        turn_index=turn_index,
                        negated=is_sentence_negated,
                        source_text=sentence.strip()
                    ))

        return entities

    @classmethod
    def _assemble_cumulative_profile(cls, entities: List[ClinicalEntity]) -> CumulativeClinicalProfile:
        """Assembles a clean, deduplicated, unnegated cumulative patient profile."""
        active_conds: Set[str] = set()
        negated_conds: Set[str] = set()

        active_meds: Set[str] = set()
        negated_meds: Set[str] = set()

        active_syms: Set[str] = set()
        negated_syms: Set[str] = set()

        allergies: Set[str] = set()
        negated_allergies: Set[str] = set()

        risk_factors: Set[str] = set()
        negated_risk: Set[str] = set()

        for ent in entities:
            if ent.entity_type == ClinicalEntityType.CONDITION:
                if ent.negated or ent.temporal_state == EntityTemporalState.NEGATED:
                    negated_conds.add(ent.name)
                else:
                    active_conds.add(ent.name)
            elif ent.entity_type == ClinicalEntityType.MEDICATION:
                if ent.negated or ent.temporal_state == EntityTemporalState.NEGATED:
                    negated_meds.add(ent.name)
                else:
                    active_meds.add(ent.name)
            elif ent.entity_type == ClinicalEntityType.SYMPTOM:
                if ent.negated or ent.temporal_state == EntityTemporalState.NEGATED:
                    negated_syms.add(ent.name)
                else:
                    active_syms.add(ent.name)
            elif ent.entity_type == ClinicalEntityType.ALLERGY:
                if ent.negated or ent.temporal_state == EntityTemporalState.NEGATED:
                    negated_allergies.add(ent.name)
                else:
                    allergies.add(ent.name)
            elif ent.entity_type == ClinicalEntityType.RISK_FACTOR:
                if ent.negated or ent.temporal_state == EntityTemporalState.NEGATED:
                    negated_risk.add(ent.name)
                else:
                    risk_factors.add(ent.name)

        # Apply negation resolution (remove negated items)
        final_conds = sorted(list(active_conds - negated_conds))
        final_meds = sorted(list(active_meds - negated_meds))
        final_syms = sorted(list(active_syms - negated_syms))
        final_allergies = sorted(list(allergies - negated_allergies))
        final_risk = sorted(list(risk_factors - negated_risk))

        # Clinical Invariant: If a drug is in confirmed allergies, it is NOT an active medication
        final_allergies_lower = {a.lower() for a in final_allergies}
        final_meds = [m for m in final_meds if m.lower() not in final_allergies_lower]

        total_extracted = len(final_conds) + len(final_meds) + len(final_syms) + len(final_allergies) + len(final_risk)

        # Generate deterministic profile hash
        hash_seed = f"C:{'|'.join(final_conds)};M:{'|'.join(final_meds)};A:{'|'.join(final_allergies)};R:{'|'.join(final_risk)}"
        profile_hash = hashlib.sha256(hash_seed.encode("utf-8")).hexdigest()[:16]

        return CumulativeClinicalProfile(
            active_conditions=final_conds,
            active_symptoms=final_syms,
            active_medications=final_meds,
            confirmed_allergies=final_allergies,
            risk_factors=final_risk,
            demographics={},
            entities=entities,
            total_entities_extracted=total_extracted,
            profile_hash=profile_hash
        )

    @classmethod
    def _extract_single_turn_profile(cls, query: str) -> CumulativeClinicalProfile:
        """Extracts profile for a single standalone query."""
        entities = cls._extract_turn_entities(query, turn_index=0, role="user")
        return cls._assemble_cumulative_profile(entities)

    @classmethod
    def _reformulate_query(
        cls,
        raw_query: str,
        bounded_history: List[Dict[str, str]],
        profile: CumulativeClinicalProfile
    ) -> Tuple[bool, Optional[str], str, float]:
        """
        Identifies follow-up inquiries and reformulates raw_query into an effective search query.
        Returns:
            (is_follow_up, resolved_topic, effective_query, confidence)
        """
        q_strip = raw_query.strip()
        q_lower = q_strip.lower()

        # Check if query is an elliptical or demonstrative follow-up
        is_elliptical = any(p.match(q_strip) for p in cls.ELLIPTICAL_PATTERNS)
        has_follow_up_tokens = bool(cls.FOLLOW_UP_INDICATORS.search(q_lower))

        # Check if query already has a rich standalone medical subject
        has_own_med = any(med.lower() in q_lower for med in profile.active_medications)
        has_own_cond = any(cond.lower() in q_lower for cond in profile.active_conditions)

        # Find the most recent focal topic from prior turns
        prior_focal_topic = cls._extract_prior_focal_topic(bounded_history)

        # If it's explicitly elliptical or contains pronouns and lacks its own subject
        is_follow_up = False
        resolved_topic = None
        effective_query = q_strip
        confidence = 1.0

        if prior_focal_topic:
            if is_elliptical or (has_follow_up_tokens and not (has_own_med or has_own_cond)):
                is_follow_up = True
                resolved_topic = prior_focal_topic

                # Possessive pronoun substitution: "its", "their" -> "{topic}'s"
                possessive_sub = re.sub(
                    r'\b(its|their)\b',
                    f"{prior_focal_topic}'s",
                    q_strip,
                    flags=re.IGNORECASE
                )

                # Direct pronoun / noun phrase substitution
                pronoun_sub = re.sub(
                    r'\b(it|this|that|these|those|them|the medication|the drug|the treatment|the condition|the disease|such condition)\b',
                    prior_focal_topic,
                    possessive_sub,
                    flags=re.IGNORECASE
                )

                if pronoun_sub != q_strip:
                    effective_query = pronoun_sub
                else:
                    # Prepend or append topic for follow-up inquiries
                    if re.match(r'^(?:what|how|can|is|are|should|which)\b', q_strip, re.IGNORECASE):
                        effective_query = f"{q_strip.rstrip('?. ')} for {prior_focal_topic}?"
                    else:
                        effective_query = f"{prior_focal_topic} {q_strip}"

                confidence = 0.95
            elif has_follow_up_tokens and (has_own_med or has_own_cond):
                # Query has its own subject but refers to context (e.g., "Compared to that, what about metformin?")
                is_follow_up = True
                resolved_topic = prior_focal_topic
                confidence = 0.90

        return is_follow_up, resolved_topic, effective_query, confidence

    @classmethod
    def _extract_prior_focal_topic(cls, history: List[Dict[str, str]]) -> Optional[str]:
        """Extracts the primary focal clinical entity from the most recent user or assistant turns."""
        for turn in reversed(history):
            content = (turn.get("content") or turn.get("text") or "").strip()
            if not content:
                continue

            content_lower = content.lower()

            # Check for medication mentions first (most common clinical follow-up target)
            for med in cls.KNOWN_MEDICATIONS:
                if med in content_lower and re.search(rf'\b{re.escape(med)}\b', content_lower):
                    return med.capitalize()

            # Check for condition mentions
            for cond_key, canonical in cls.KNOWN_CONDITIONS.items():
                if cond_key in content_lower and re.search(rf'\b{re.escape(cond_key)}\b', content_lower):
                    return canonical

        return None

    @classmethod
    def _evaluate_inherited_contraindications(cls, profile: CumulativeClinicalProfile) -> Tuple[List[str], List[ContraindicationAlert]]:
        """
        Evaluates cumulative patient conditions/allergies and exports strict contraindication rules
        and structured ContraindicationAlert objects.
        """
        contraindications: List[str] = []
        alerts: List[ContraindicationAlert] = []
        conds_lower = {c.lower() for c in profile.active_conditions}
        allergies_lower = {a.lower() for a in profile.confirmed_allergies}
        meds_lower = {m.lower() for m in profile.active_medications}
        risks_lower = {r.lower() for r in profile.risk_factors}

        # 1. Renal Contraindications (CKD / Renal Failure / AKI / Nephropathy / Kidney Disease)
        renal_keys = {
            "chronic kidney disease", "ckd", "renal failure", "renal impairment",
            "acute kidney injury", "aki", "kidney disease", "nephropathy", "end stage renal disease", "esrd"
        }
        if bool(conds_lower & renal_keys):
            msg = (
                "Renal Contraindication: Patient has diagnosed renal impairment/CKD. "
                "NSAIDs (ibuprofen, naproxen, celecoxib) are contraindicated or high-risk for acute decompensation."
            )
            contraindications.append(msg)
            alerts.append(ContraindicationAlert(
                contraindication_id="RENAL_IMPAIRMENT_NSAID",
                reason=msg,
                severity="CRITICAL"
            ))

        # 2. Allergy Contraindications (Penicillin / Sulfa / etc.)
        if "penicillin" in allergies_lower or "amoxicillin" in allergies_lower:
            msg = (
                "Allergy Contraindication: Confirmed penicillin allergy. "
                "Beta-lactam antibiotics (penicillins, ampicillin, amoxicillin, augmentin) are strictly contraindicated."
            )
            contraindications.append(msg)
            alerts.append(ContraindicationAlert(
                contraindication_id="PENICILLIN_ALLERGY_BETA_LACTAM",
                reason=msg,
                severity="CRITICAL"
            ))
        if "sulfa" in allergies_lower or "sulfonamides" in allergies_lower:
            msg = (
                "Allergy Contraindication: Confirmed sulfonamide allergy. "
                "Sulfamethoxazole/trimethoprim (Bactrim) is strictly contraindicated."
            )
            contraindications.append(msg)
            alerts.append(ContraindicationAlert(
                contraindication_id="SULFONAMIDE_ALLERGY",
                reason=msg,
                severity="CRITICAL"
            ))

        # 3. Hepatic Contraindications (Cirrhosis / Liver Failure / Liver Disease / Hepatic Impairment)
        hepatic_keys = {
            "cirrhosis", "liver cirrhosis", "liver failure", "chronic liver failure",
            "hepatic impairment", "liver disease", "hepatitis"
        }
        if bool(conds_lower & hepatic_keys):
            msg = (
                "Hepatic Contraindication: Patient has diagnosed liver disease/cirrhosis. "
                "High-dose acetaminophen (Tylenol) carries severe risk of acute-on-chronic hepatotoxicity."
            )
            contraindications.append(msg)
            alerts.append(ContraindicationAlert(
                contraindication_id="HEPATIC_IMPAIRMENT_ACETAMINOPHEN",
                reason=msg,
                severity="HIGH"
            ))

        # 4. Pregnancy / Teratogen Contraindications
        if "pregnancy" in risks_lower or any("pregnant" in r for r in risks_lower):
            msg = (
                "Pregnancy Contraindication: Patient is pregnant. "
                "ACE inhibitors, ARBs, and statins are teratogenic and strictly contraindicated."
            )
            contraindications.append(msg)
            alerts.append(ContraindicationAlert(
                contraindication_id="PREGNANCY_TERATOGEN_ACE_INHIBITOR",
                reason=msg,
                severity="CRITICAL"
            ))

        # 5. Anticoagulant Bleeding Risk (Warfarin / DOAC + Aspirin / NSAIDs)
        if "warfarin" in meds_lower or "coumadin" in meds_lower or "apixaban" in meds_lower or "rivaroxaban" in meds_lower:
            msg = (
                "Anticoagulant Warning: Patient is on anticoagulant therapy (Warfarin/DOAC). "
                "Concomitant aspirin or NSAIDs significantly elevates gastrointestinal hemorrhage risk."
            )
            contraindications.append(msg)
            alerts.append(ContraindicationAlert(
                contraindication_id="ANTICOAGULANT_BLEEDING_RISK",
                reason=msg,
                severity="HIGH"
            ))

        # 6. Asthma Contraindications
        if "asthma" in conds_lower:
            msg = (
                "Asthma Contraindication: Patient has asthma. "
                "Non-selective beta blockers (propranolol, labetalol, carvedilol) may provoke bronchospasm."
            )
            contraindications.append(msg)
            alerts.append(ContraindicationAlert(
                contraindication_id="ASTHMA_BETA_BLOCKER",
                reason=msg,
                severity="HIGH"
            ))

        # 7. Peptic Ulcer Disease / GI Bleeding
        if "peptic ulcer disease" in conds_lower or "peptic ulcer" in conds_lower:
            msg = (
                "Gastrointestinal Contraindication: Patient has peptic ulcer disease. "
                "Systemic NSAIDs and high-dose aspirin carry high risk of gastrointestinal hemorrhage."
            )
            contraindications.append(msg)
            alerts.append(ContraindicationAlert(
                contraindication_id="PEPTIC_ULCER_NSAID_BLEEDING",
                reason=msg,
                severity="HIGH"
            ))

        return contraindications, alerts
