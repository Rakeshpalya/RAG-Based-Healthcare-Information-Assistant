"""
Medical Query Expander and Multi-Aspect Query Analyzer for Phase 2E and Phase 2F.

Provides deterministic, lightweight medical synonym expansion and multi-aspect
query decomposition to improve retrieval recall while preserving Phase 2C precision.
Includes strict clinical subject anchoring to prevent cross-condition contamination.
"""

import re
from typing import List, Dict, Any, Optional, Set


class MedicalQueryExpander:
    """
    Deterministic medical synonym expander and multi-aspect query detector.

    Enhances query recall by:
    1. Identifying clinical synonyms (e.g., 'elevated arterial blood pressure' -> 'hypertension')
    2. Appending normalized medical terminology to the retrieval query
    3. Detecting multi-aspect clinical queries and decomposing them into focused sub-queries
    4. Guarding against cross-condition pollution (never defaults to hypertension)
    """

    # Comprehensive medical synonym mapping: pattern -> list of canonical expansion terms
    SYNONYM_MAP: List[Dict[str, Any]] = [
        # Blood pressure / Hypertension
        {
            "pattern": re.compile(r'\bhypertension\b', re.IGNORECASE),
            "canonical": "blood pressure",
            "expansions": ["blood pressure", "high blood pressure"]
        },
        {
            "pattern": re.compile(r'\belevated\s+arterial\s+blood\s+pressure\b', re.IGNORECASE),
            "canonical": "hypertension",
            "expansions": ["hypertension", "high blood pressure"]
        },
        {
            "pattern": re.compile(r'\barterial\s+blood\s+pressure\b', re.IGNORECASE),
            "canonical": "blood pressure",
            "expansions": ["blood pressure", "hypertension"]
        },
        {
            "pattern": re.compile(r'\bhigh\s+blood\s+pressure\b', re.IGNORECASE),
            "canonical": "hypertension",
            "expansions": ["hypertension"]
        },
        {
            "pattern": re.compile(r'\b(?:uncontrolled\s+)?vascular\s+pressure\b', re.IGNORECASE),
            "canonical": "blood pressure",
            "expansions": ["blood pressure", "hypertension", "high blood pressure"]
        },
        {
            "pattern": re.compile(r'\bsystemic\s+arterial\s+tension\b', re.IGNORECASE),
            "canonical": "blood pressure",
            "expansions": ["blood pressure", "hypertension", "arterial pressure"]
        },
        # Glycemic / Diabetes (condition-level only; medications separated)
        {
            "pattern": re.compile(r'\b(?:elevated\s+|high\s+)?blood\s+glucose\b', re.IGNORECASE),
            "canonical": "blood sugar",
            "expansions": ["blood sugar", "diabetes", "hyperglycemia"]
        },
        {
            "pattern": re.compile(r'\bglycemic\s+disorders?\b', re.IGNORECASE),
            "canonical": "diabetes",
            "expansions": ["diabetes", "type 2 diabetes", "blood sugar", "glucose"]
        },
        {
            "pattern": re.compile(r'\bglycemic\s+control\b', re.IGNORECASE),
            "canonical": "blood sugar",
            "expansions": ["blood sugar", "diabetes", "glucose"]
        },
        {
            "pattern": re.compile(r'\bblood\s+sugar\b', re.IGNORECASE),
            "canonical": "glucose",
            "expansions": ["glucose", "diabetes"]
        },
        {
            "pattern": re.compile(r'\bhigh\s+blood\s+sugar\b', re.IGNORECASE),
            "canonical": "hyperglycemia",
            "expansions": ["hyperglycemia", "diabetes"]
        },
        # Complications / Sequelae / Organ Damage
        {
            "pattern": re.compile(r'\bsequelae\b', re.IGNORECASE),
            "canonical": "complications",
            "expansions": ["complications", "long-term complications", "organ damage"]
        },
        {
            "pattern": re.compile(r'\borgan\s+damage\b', re.IGNORECASE),
            "canonical": "complications",
            "expansions": ["complications", "target organ damage"]
        },
        {
            "pattern": re.compile(r'\bsecondary\s+complications?\b', re.IGNORECASE),
            "canonical": "complications",
            "expansions": ["complications", "target organ damage"]
        },
        # Cardiac & Coronary
        {
            "pattern": re.compile(r'\b(?:acute\s+)?coronary\s+(?:occlusion|syndrome|artery\s+disease|events?)\b', re.IGNORECASE),
            "canonical": "cardiology",
            "expansions": ["cardiology", "myocardial infarction", "heart attack", "cardiovascular", "coronary artery disease"]
        },
        {
            "pattern": re.compile(r'\bheart\s+attack\b', re.IGNORECASE),
            "canonical": "myocardial infarction",
            "expansions": ["myocardial infarction", "cardiac event", "cardiology"]
        },
        {
            "pattern": re.compile(r'\bmyocardial\s+infarction\b', re.IGNORECASE),
            "canonical": "heart attack",
            "expansions": ["heart attack", "cardiology", "cardiac event"]
        },
        {
            "pattern": re.compile(r'\bcardiac\s+failure\b', re.IGNORECASE),
            "canonical": "heart failure",
            "expansions": ["heart failure", "congestive heart failure", "cardiology"]
        },
        # Respiratory / Asthma
        {
            "pattern": re.compile(r'\bbronchial\s+spasm\b', re.IGNORECASE),
            "canonical": "asthma",
            "expansions": ["asthma", "wheezing", "bronchospasm"]
        },
        # Renal / Kidney
        {
            "pattern": re.compile(r'\brenal\s+(?:disease|failure|impairment)\b', re.IGNORECASE),
            "canonical": "kidney disease",
            "expansions": ["kidney disease", "kidney problems", "renal failure"]
        },
        {
            "pattern": re.compile(r'\bkidney\s+(?:problems?|failure|damage)\b', re.IGNORECASE),
            "canonical": "renal disease",
            "expansions": ["renal disease", "kidney problems"]
        }
    ]

    # Clinical aspect categories for multi-aspect query detection
    ASPECT_PATTERNS = [
        ("definition", re.compile(r'\b(?:definition|define|what is|meaning|overview)\b', re.IGNORECASE)),
        ("symptoms", re.compile(r'\b(?:symptoms?|clinical\s+signs?|manifestations?|presentation)\b', re.IGNORECASE)),
        ("diagnosis", re.compile(r'\b(?:diagnosis|diagnostic|criteria|thresholds?|evaluation|testing)\b', re.IGNORECASE)),
        ("risk_factors", re.compile(r'\b(?:risk\s+factors?|causes?|predisposing|etiology|pathophysiology|triggers?)\b', re.IGNORECASE)),
        ("lifestyle_measures", re.compile(r'\b(?:lifestyle(?:\s+measures?|\s+interventions?|\s+modifications?)?|diet|exercise|physical\s+activity|sodium)\b', re.IGNORECASE)),
        ("complications", re.compile(r'\b(?:complications?|sequelae|organ\s+damage|long-term\s+(?:consequences?|effects?)|exacerbation\s+risks?)\b', re.IGNORECASE)),
        ("medications", re.compile(r'\b(?:medications?|pharmacotherapy|drug\s+therapy|prescribed|medicines?|management|efficacy|inhaler|dosage|pharmacology|therapy|treatment|treatments)\b', re.IGNORECASE)),
        ("clinical_guidelines", re.compile(r'\b(?:clinical\s+guidelines?|guidelines?|protocols?|standards?)\b', re.IGNORECASE)),
        ("cardiology_records", re.compile(r'\b(?:cardiology(?:\s+records?)?|patient\s+records?|clinical\s+records?)\b', re.IGNORECASE)),
    ]

    # Known primary clinical subjects
    PRIMARY_SUBJECTS = [
        ("hypertension", re.compile(r'\b(?:hypertension|blood\s+pressure|arterial\s+pressure|arterial\s+tension|vascular\s+pressure)\b', re.IGNORECASE)),
        ("diabetes", re.compile(r'\b(?:diabetes|diabetic|glycemic|blood\s+sugar|glucose|insulin|metformin)\b', re.IGNORECASE)),
        ("asthma", re.compile(r'\b(?:asthma|bronch\w*|wheez\w*|pulmonology|respiratory)\b', re.IGNORECASE)),
        ("cardiovascular", re.compile(r'\b(?:cardiology|cardiovascular|coronary|myocardi\w*|heart\s+attack|heart\s+failure|cardiac)\b', re.IGNORECASE)),
        ("cancer", re.compile(r'\b(?:cancer|tumor|carcinoma|oncology|pathology|biopsy|malignan\w*)\b', re.IGNORECASE)),
        ("renal", re.compile(r'\b(?:renal|kidney|nephro\w*|ckd)\b', re.IGNORECASE)),
        ("pheochromocytoma", re.compile(r'\b(?:pheochromocytoma|adrenal)\b', re.IGNORECASE)),
        ("stroke", re.compile(r'\b(?:stroke|cerebrovascular|ischemi\w*)\b', re.IGNORECASE)),
        ("pneumonia", re.compile(r'\b(?:pneumoni\w*|lung\s+infiltrat\w*)\b', re.IGNORECASE)),
        ("tuberculosis", re.compile(r'\b(?:tuberculosis|tb)\b', re.IGNORECASE)),
        ("malaria", re.compile(r'\b(?:malaria|plasmodi\w*)\b', re.IGNORECASE)),
        ("arthritis", re.compile(r'\b(?:arthrit\w*|joint\s+inflamm\w*)\b', re.IGNORECASE)),
        ("hepatitis", re.compile(r'\b(?:hepatit\w*|liver\s+inflamm\w*)\b', re.IGNORECASE)),
        ("cholesterol", re.compile(r'\b(?:cholesterol|lipid|statin|ldl|hdl)\b', re.IGNORECASE)),
        ("dermatology", re.compile(r'\b(?:dermatol\w*|skin|rash|lesion)\b', re.IGNORECASE)),
    ]

    MEDICATION_INTENT_PATTERN = re.compile(
        r'\b(?:medications?|drugs?|prescribed|prescription|pills?|pharmacotherapy|pharmacology|dosage|dose|tablets?|medicines?|inhaler)\b',
        re.IGNORECASE
    )

    @classmethod
    def has_medication_intent(cls, query: str) -> bool:
        """Returns True if the query explicitly asks about medications, drugs, or dosages."""
        if not query:
            return False
        return bool(cls.MEDICATION_INTENT_PATTERN.search(query))

    @classmethod
    def get_expanded_terms(cls, query: str) -> List[str]:
        """
        Extracts a deduplicated list of clinical expansion terms matching the query.
        Separates condition-level synonyms from medication-level terms.
        Medication terms are ONLY added when explicit medication intent is detected.
        """
        if not query or not query.strip():
            return []

        q_lower = query.lower()
        expanded: List[str] = []
        seen: Set[str] = set()

        for entry in cls.SYNONYM_MAP:
            if entry["pattern"].search(q_lower):
                for term in entry["expansions"]:
                    term_norm = term.strip().lower()
                    if term_norm not in seen and term_norm not in q_lower:
                        seen.add(term_norm)
                        expanded.append(term.strip())

        # Intent-driven expansion: add specific medications ONLY when medication intent is present
        if cls.has_medication_intent(query):
            if any(term in q_lower for term in ["glycemic", "diabetes", "blood sugar", "glucose"]):
                for med in ["metformin", "insulin"]:
                    if med not in seen and med not in q_lower:
                        seen.add(med)
                        expanded.append(med)
            if any(term in q_lower for term in ["hypertension", "blood pressure"]):
                for med in ["amlodipine", "lisinopril"]:
                    if med not in seen and med not in q_lower:
                        seen.add(med)
                        expanded.append(med)
            if any(term in q_lower for term in ["asthma", "bronchial", "wheezing"]):
                for med in ["inhaler", "albuterol"]:
                    if med not in seen and med not in q_lower:
                        seen.add(med)
                        expanded.append(med)

        # Context-driven expansion for organ damage in cardiovascular/hypertension queries
        if any(term in q_lower for term in ["organ damage", "secondary complications", "sequelae"]):
            if any(term in q_lower for term in ["blood pressure", "hypertension", "arterial", "vascular pressure"]):
                for cv_term in ["stroke", "heart disease"]:
                    if cv_term not in seen and cv_term not in q_lower:
                        seen.add(cv_term)
                        expanded.append(cv_term)

        return expanded

    @classmethod
    def normalize_query(cls, query: str) -> str:
        """
        Normalizes medical synonym phrasing in a query into canonical clinical phrasing.
        e.g., 'elevated arterial blood pressure' -> 'hypertension'
        """
        if not query or not query.strip():
            return query or ""

        normalized = query
        for entry in cls.SYNONYM_MAP:
            normalized = entry["pattern"].sub(entry["canonical"], normalized)

        return normalized

    @classmethod
    def expand_query(cls, query: str) -> str:
        """
        Expands the query by appending non-redundant medical synonym terms.
        Preserves the original query at the front to maintain embedding focus.

        Example:
            'What are the sequelae of elevated arterial blood pressure?'
            -> 'What are the sequelae of elevated arterial blood pressure? hypertension high blood pressure complications'
        """
        if not query or not query.strip():
            return query or ""

        terms = cls.get_expanded_terms(query)
        if not terms:
            return query.strip()

        expanded_str = " ".join(terms)
        return f"{query.strip()} {expanded_str}"

    @classmethod
    def is_multi_aspect_query(cls, query: str) -> bool:
        """
        Returns True if the query contains 2 or more distinct medical aspects
        (e.g., definition + risk factors + complications).
        """
        if not query or not query.strip():
            return False

        q_lower = query.lower()
        matched_aspects = sum(1 for _, pat in cls.ASPECT_PATTERNS if pat.search(q_lower))
        return matched_aspects >= 2

    @classmethod
    def extract_primary_subject(cls, query: str) -> str:
        """
        Extracts the main clinical condition/subject from the query.
        Falls back to empty string if no known subject is identified.
        """
        if not query:
            return ""

        q_lower = query.lower()
        for subj_name, pat in cls.PRIMARY_SUBJECTS:
            if pat.search(q_lower):
                return subj_name.replace("_", " ")

        return ""

    @classmethod
    def _extract_fallback_subject(cls, query: str) -> str:
        """
        Extracts a candidate medical condition/noun phrase from the query when not in PRIMARY_SUBJECTS.
        Never invents an unrelated disease like hypertension.
        Returns empty string if no valid clinical subject can be established.
        """
        if not query:
            return ""
        q_lower = query.lower()
        # Remove common aspect phrases and question framing
        clean = re.sub(
            r'\b(?:what\s+is|what\s+are|provide\s+an\s+overview\s+of|overview\s+of|explain|detail|describe|'
            r'definition|define|risk\s+factors?|causes?|lifestyle(?:\s+measures?)?|diet|exercise|'
            r'complications?|sequelae|medications?|treatment|management|guidelines?|protocols?|'
            r'patient\s+records?|clinical|report|summary|chart|notes?|findings?)\b',
            '',
            q_lower,
            flags=re.IGNORECASE
        )
        words = [w for w in re.findall(r'\b[a-z]{3,}\b', clean) if w not in {
            "and", "the", "for", "with", "from", "about", "which", "how", "does", "can", "into", "across",
            "all", "any", "are", "but", "not", "our", "per", "that", "this", "these", "those"
        }]
        if words:
            return " ".join(words[:2])
        return ""

    @classmethod
    def decompose_multi_aspect_query(cls, query: str) -> List[str]:
        """
        Decomposes a multi-aspect clinical query into targeted retrieval sub-queries.

        Guarantees:
        - Never defaults to 'hypertension' when querying other diseases.
        - Identifies the actual clinical subject (known or extracted).
        - If no subject can be established, returns empty list (no manufactured sub-queries).
        - Uses generic, facet-appropriate clinical sub-query templates.

        Returns:
            List of generated sub-queries. Returns an empty list if not multi-aspect.
        """
        if not query or not query.strip():
            return []

        if not cls.is_multi_aspect_query(query):
            return []

        q_lower = query.lower()
        primary_subject = cls.extract_primary_subject(query)
        if not primary_subject:
            primary_subject = cls._extract_fallback_subject(query)

        # If no identifiable subject exists, do NOT manufacture one
        if not primary_subject:
            return []

        sub_queries: List[str] = []
        for aspect_name, pat in cls.ASPECT_PATTERNS:
            if pat.search(q_lower):
                aspect_clean = aspect_name.replace("_", " ")
                # Format focused sub-query
                if aspect_name == "cardiology_records":
                    sub_q = f"{primary_subject} cardiology patient records"
                elif aspect_name == "medications":
                    sub_q = f"{primary_subject} medication management treatment plan"
                elif aspect_name == "clinical_guidelines":
                    sub_q = f"{primary_subject} clinical guidelines recommendations"
                elif aspect_name == "complications":
                    sub_q = f"{primary_subject} complications sequelae target organ damage"
                elif aspect_name == "lifestyle_measures":
                    sub_q = f"{primary_subject} lifestyle measures diet physical activity"
                elif aspect_name == "risk_factors":
                    sub_q = f"{primary_subject} risk factors causes pathophysiology"
                elif aspect_name == "definition":
                    sub_q = f"{primary_subject} definition clinical criteria overview"
                elif aspect_name == "symptoms":
                    sub_q = f"{primary_subject} symptoms clinical signs manifestations"
                elif aspect_name == "diagnosis":
                    sub_q = f"{primary_subject} diagnostic criteria clinical evaluation"
                else:
                    sub_q = f"{primary_subject} {aspect_clean}"

                sub_queries.append(sub_q)

        # Cap at maximum 4 sub-queries for optimal latency
        return sub_queries[:4]
