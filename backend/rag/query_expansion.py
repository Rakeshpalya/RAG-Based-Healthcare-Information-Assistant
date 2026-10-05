"""
Medical Query Intent Normalization and Expansion Engine for Phase 2E.2.

Provides deterministic, rule-based medical intent classification, negative
pharmaceutical phrase handling, and non-pharmacological/lifestyle query expansion.

Guarantees:
1. Strict Negation Priority: Phrases like "non-medication", "non-pharmacological",
   "non-drug", "without medication" NEVER activate pharmaceutical sufficiency requirements.
2. Distinct Intent Resolution: Accurately classifies LIFESTYLE, MEDICATION, MIXED, and GENERAL.
3. Deterministic Query Expansion: Appends high-signal lifestyle terms for retrieval while
   preserving the exact original user question for downstream generation and citation validation.
4. Zero LLM Dependencies: All classifications and expansions execute deterministically via regex.
"""

import re
from functools import lru_cache
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set


class QueryIntent(str, Enum):
    """Enumeration of recognized medical query intents."""
    LIFESTYLE = "LIFESTYLE"
    MEDICATION = "MEDICATION"
    MIXED = "MIXED"
    GENERAL = "GENERAL"


@dataclass
class QueryIntentResult:
    """Structured result of medical query intent classification and expansion."""
    raw_query: str
    normalized_query: str
    intent: QueryIntent
    is_lifestyle: bool
    is_medication: bool
    is_mixed: bool
    has_negative_pharmaceutical: bool
    negative_phrases_detected: List[str] = field(default_factory=list)
    lifestyle_phrases_detected: List[str] = field(default_factory=list)
    medication_phrases_detected: List[str] = field(default_factory=list)
    expansion_terms: List[str] = field(default_factory=list)
    expanded_retrieval_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "intent": self.intent.value if isinstance(self.intent, QueryIntent) else str(self.intent),
            "is_lifestyle": self.is_lifestyle,
            "is_medication": self.is_medication,
            "is_mixed": self.is_mixed,
            "has_negative_pharmaceutical": self.has_negative_pharmaceutical,
            "negative_phrases_detected": self.negative_phrases_detected,
            "lifestyle_phrases_detected": self.lifestyle_phrases_detected,
            "medication_phrases_detected": self.medication_phrases_detected,
            "expansion_terms": self.expansion_terms,
            "expanded_retrieval_query": self.expanded_retrieval_query,
        }


# ==============================================================================
# Deterministic Patterns for Intent Classification
# ==============================================================================

# Base medication noun group and compound conjunctions (e.g. "drugs and pharmaceuticals")
_MED_NOUN = r'(?:medications?|drugs?|medicines?|pharmaceuticals?|prescriptions?)'
_MED_NOUN_COMPOUND = rf'(?:{_MED_NOUN}(?:\s+(?:and|or)\s+{_MED_NOUN})?)'

# Explicit negative pharmaceutical patterns (negating pharmaceutical intent)
NEGATIVE_PHARMACEUTICAL_PATTERNS = [
    re.compile(r'\bnon[- ]medications?(?:\s+(?:measures?|approaches?|interventions?|strategies?|options?|management))?\b', re.IGNORECASE),
    re.compile(r'\bnon[- ]pharmacologic(?:al)?(?:\s+(?:measures?|approaches?|interventions?|strategies?|therap(?:y|ies)|management))?\b', re.IGNORECASE),
    re.compile(r'\bnon[- ]drugs?(?:\s+(?:measures?|approaches?|interventions?|strategies?|options?|therap(?:y|ies)|management))?\b', re.IGNORECASE),
    re.compile(rf'\bwithout\s+(?:any\s+|the\s+use\s+of\s+)?{_MED_NOUN_COMPOUND}\b', re.IGNORECASE),
    re.compile(rf'\b(?:instead\s+of|rather\s+than)\s+{_MED_NOUN_COMPOUND}\b', re.IGNORECASE),
    re.compile(rf'\bavoiding\s+{_MED_NOUN_COMPOUND}\b', re.IGNORECASE),
    re.compile(rf'\bno\s+{_MED_NOUN_COMPOUND}\b', re.IGNORECASE),
    re.compile(rf'\bnot\s+(?:using\s+|taking\s+)?{_MED_NOUN_COMPOUND}\b', re.IGNORECASE),
    re.compile(rf'\boff\s+{_MED_NOUN_COMPOUND}\b', re.IGNORECASE),
    re.compile(r'\bdrug[- ]free\b', re.IGNORECASE),
    re.compile(r'\bunmedicated\b', re.IGNORECASE),
    re.compile(r'\bnon[- ]invasive\b', re.IGNORECASE),
]

# Lifestyle & Non-pharmacological intent patterns
LIFESTYLE_PATTERNS = [
    re.compile(r'\blifestyle(?:\s+(?:changes?|measures?|interventions?|approaches?|modifications?|habits?|management))?\b', re.IGNORECASE),
    re.compile(r'\b(?:diet|dietary|nutrition|nutritional|eating\s+habits?)\b', re.IGNORECASE),
    re.compile(r'\b(?:exercise|physical\s+activity|aerobic|workout|fitness|walking)\b', re.IGNORECASE),
    re.compile(r'\b(?:weight\s+(?:loss|reduction|management|control)|healthy\s+weight|bmi)\b', re.IGNORECASE),
    re.compile(r'\b(?:sodium|salt\s+intake|low[- ]sodium|salt\s+restriction)\b', re.IGNORECASE),
    re.compile(r'\b(?:tobacco|smoking(?:\s+cessation)?|cigarettes?)\b', re.IGNORECASE),
    re.compile(r'\b(?:alcohol(?:\s+intake|\s+moderation|\s+reduction|\s+consumption)?|drinking)\b', re.IGNORECASE),
    re.compile(r'\b(?:sleep(?:\s+hygiene|\s+duration|\s+apnea)?|adequate\s+sleep|rest)\b', re.IGNORECASE),
    re.compile(r'\b(?:stress\s+(?:reduction|management|relief)|relaxation|meditation)\b', re.IGNORECASE),
    re.compile(r'\bnon[- ]pharmacologic(?:al)?\b', re.IGNORECASE),
    re.compile(r'\bnon[- ]medication\b', re.IGNORECASE),
    re.compile(r'\bnon[- ]drug\b', re.IGNORECASE),
    re.compile(r'\blifestyle\s+approaches\b', re.IGNORECASE),
]

# Affirmative medication / pharmaceutical intent patterns (evaluated after masking negative phrases)
AFFIRMATIVE_MEDICATION_PATTERNS = [
    re.compile(r'\b(?:what|which|any)\s+(?:medications?|drugs?|medicines?|pharmaceuticals?|pharmacotherapy)\b', re.IGNORECASE),
    re.compile(r'\b(?:what|which)\s+(?:treatment|therapy|prescription)\s+(?:was\s+)?prescribed\b', re.IGNORECASE),
    re.compile(r'\b(?:treatment|therapy)\s+(?:was\s+)?prescribed\b', re.IGNORECASE),
    re.compile(r'\bprescribed\s+(?:treatment|therapy)\b', re.IGNORECASE),
    re.compile(r'\b(?:medications?|drugs?|medicines?)\s+(?:are|is)?\s*recommended\b', re.IGNORECASE),
    re.compile(r'\b(?:recommended|prescribed)\s+(?:medications?|drugs?|medicines?|pharmaceuticals?|pharmacotherapy)\b', re.IGNORECASE),
    re.compile(r'\b(?:medications?|drugs?|medicines?)\s+(?:for|to)\s+treat\b', re.IGNORECASE),
    re.compile(r'\b(?:medications?|drugs?|medicines?)\s+prescribed\b', re.IGNORECASE),
    re.compile(r'\b(?:side\s+effects?|adverse\s+effects?|contraindications?)\s+of\s+.*?(?:medications?|drugs?|pharmaceuticals?)\b', re.IGNORECASE),
    re.compile(r'\b(?:medications?|drugs?|pharmaceuticals?)\s+side\s+effects?\b', re.IGNORECASE),
    re.compile(r'\b(?:side\s+effects?|adverse\s+reactions?)\b.*?\b(?:medication|drug|pill|tablet)s?\b', re.IGNORECASE),
    re.compile(r'\bpharmaceutical\s+(?:treatments?|therap(?:y|ies)|interventions?|agents?)\b', re.IGNORECASE),
    re.compile(r'\b(?:pharmacotherapy|pharmacology|drug\s+therapy|drug\s+treatment)\b', re.IGNORECASE),
    re.compile(r'\b(?:dosage|dosing|dose|prescriptions?|tablets?|pills?|inhalers?)\b', re.IGNORECASE),
    re.compile(r'\b(?:first[- ]line\s+(?:medications?|agents?|drugs?)|antihypertensive\s+(?:medications?|drugs?|agents?))\b', re.IGNORECASE),
    re.compile(r'\b(?:ace\s+inhibitors?|arbs?|beta\s+blockers?|calcium\s+channel\s+blockers?|diuretics?|thiazides?|statins?)\b', re.IGNORECASE),
    re.compile(r'\b(?:antibiotics?|antivirals?|antimicrobials?|antibacterial)\b', re.IGNORECASE),
    re.compile(r'\b(?:lisinopril|amlodipine|losartan|metoprolol|hydrochlorothiazide|furosemide|metformin|insulin|atorvastatin|albuterol|fluticasone)\b', re.IGNORECASE),
    # Standalone medication/drug tokens (only matched when affirmative)
    re.compile(r'\b(?:medications?|drugs?|medicines?|pharmaceuticals?)\b', re.IGNORECASE),
]

# Canonical lifestyle expansion terms for retrieval embedding
CANONICAL_LIFESTYLE_EXPANSION_TERMS = [
    "lifestyle changes",
    "lifestyle measures",
    "non-pharmacological approaches",
    "non-drug interventions",
    "diet",
    "exercise",
    "physical activity",
    "healthy weight",
    "sodium",
    "tobacco",
    "alcohol",
    "sleep"
]


@lru_cache(maxsize=256)
def normalize_medical_query(query: str) -> QueryIntentResult:
    """
    Deterministically normalizes and classifies a clinical user query.

    Resolves:
    - Negative pharmaceutical phrasing (e.g. 'non-medication', 'without medication')
    - Lifestyle / non-pharmacological intent
    - Affirmative pharmaceutical intent
    - Mixed intent (e.g. 'medication and lifestyle')
    - Deterministic retrieval query expansion

    Args:
        query: Raw user query string.

    Returns:
        QueryIntentResult containing intent, flags, detected phrases, and expanded retrieval query.
    """
    if not query or not query.strip():
        return QueryIntentResult(
            raw_query=query or "",
            normalized_query="",
            intent=QueryIntent.GENERAL,
            is_lifestyle=False,
            is_medication=False,
            is_mixed=False,
            has_negative_pharmaceutical=False,
            expansion_terms=[],
            expanded_retrieval_query=""
        )

    raw_clean = query.strip()
    q_lower = raw_clean.lower()

    # Step 1: Detect and record negative pharmaceutical phrases
    neg_phrases: List[str] = []
    masked_query = q_lower
    for pattern in NEGATIVE_PHARMACEUTICAL_PATTERNS:
        for match in pattern.finditer(q_lower):
            matched_phrase = match.group().strip()
            if matched_phrase not in neg_phrases:
                neg_phrases.append(matched_phrase)
        # Mask out negative phrases with spaces to prevent false triggers on residual medication checks
        masked_query = pattern.sub(" ", masked_query)

    has_neg_pharm = len(neg_phrases) > 0

    # Step 2: Detect lifestyle phrases (checked on both original query and negative phrases)
    lifestyle_phrases: List[str] = []
    for pattern in LIFESTYLE_PATTERNS:
        for match in pattern.finditer(q_lower):
            matched_phrase = match.group().strip()
            if matched_phrase not in lifestyle_phrases:
                lifestyle_phrases.append(matched_phrase)

    # Any negative pharmaceutical phrase inherently signifies non-pharmacological / lifestyle intent
    if has_neg_pharm:
        for p in neg_phrases:
            if p not in lifestyle_phrases:
                lifestyle_phrases.append(p)

    has_lifestyle = len(lifestyle_phrases) > 0

    # Step 3: Detect affirmative medication phrases on the MASKED query
    # (where negative phrases like "non-medication", "without medication" have been stripped)
    medication_phrases: List[str] = []
    for pattern in AFFIRMATIVE_MEDICATION_PATTERNS:
        for match in pattern.finditer(masked_query):
            matched_phrase = match.group().strip()
            if matched_phrase not in medication_phrases:
                medication_phrases.append(matched_phrase)

    has_medication = len(medication_phrases) > 0

    # Step 4: Resolve overall QueryIntent
    if has_lifestyle and has_medication:
        intent = QueryIntent.MIXED
        is_lifestyle = True
        is_medication = True
        is_mixed = True
    elif has_lifestyle:
        intent = QueryIntent.LIFESTYLE
        is_lifestyle = True
        is_medication = False
        is_mixed = False
    elif has_medication:
        intent = QueryIntent.MEDICATION
        is_lifestyle = False
        is_medication = True
        is_mixed = False
    else:
        intent = QueryIntent.GENERAL
        is_lifestyle = False
        is_medication = False
        is_mixed = False

    # Step 5: Deterministic Query Expansion for Retrieval
    # For LIFESTYLE or MIXED queries, append high-signal non-pharmacological terms not already present
    expansion_terms: List[str] = []
    if is_lifestyle or is_mixed:
        seen_terms: Set[str] = set()
        for term in CANONICAL_LIFESTYLE_EXPANSION_TERMS:
            term_clean = term.strip().lower()
            if term_clean not in seen_terms and term_clean not in q_lower:
                seen_terms.add(term_clean)
                expansion_terms.append(term.strip())

    expanded_retrieval_query = raw_clean
    if expansion_terms:
        expanded_retrieval_query = f"{raw_clean} {' '.join(expansion_terms)}"

    return QueryIntentResult(
        raw_query=raw_clean,
        normalized_query=raw_clean,
        intent=intent,
        is_lifestyle=is_lifestyle,
        is_medication=is_medication,
        is_mixed=is_mixed,
        has_negative_pharmaceutical=has_neg_pharm,
        negative_phrases_detected=neg_phrases,
        lifestyle_phrases_detected=lifestyle_phrases,
        medication_phrases_detected=medication_phrases,
        expansion_terms=expansion_terms,
        expanded_retrieval_query=expanded_retrieval_query
    )
