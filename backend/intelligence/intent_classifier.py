"""
Deterministic Clinical Intent Classifier for AI-Healthcare-Agent (Phase 6.1).

Provides high-speed, predictable, rule-based clinical intent detection and query routing
before RAG retrieval and generation. Adheres to strict medical safety precedence.
"""

import re
import time
import html
import logging
from typing import Dict, Any, Optional, List, Tuple

from backend.intelligence.intent_models import (
    ClinicalIntent,
    SafetyPriority,
    ClinicalRoutingStrategy,
    IntentClassificationResult
)
from backend.safety.safety_types import SafetyCategory, SafetyAssessment
from backend.safety.safety_classifier import SafetyClassifier

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# COMPILED REGEX PATTERNS FOR CLINICAL INTENT DETECTION
# -------------------------------------------------------------------------

# Emergency Patterns (Aligned with Critical Pre-Screening)
EMERGENCY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:severe|crushing|sharp|radiating|intense|acute)\s+(?:\w+\s+)?chest\s+pain\b", re.I), "CHEST_PAIN_SEVERE"),
    (re.compile(r"\bchest\s+pain\s+(?:radiating\s+[\w\s]{0,25}|with|and)\s+(?:sweating|nausea|shortness\s+of\s+breath|difficulty\s+breathing|jaw|left\s+arm)\b", re.I), "CHEST_PAIN_RADIATION"),
    (re.compile(r"\b(?:difficulty|trouble|can't|cannot|unable\s+to|struggling\s+to)\s+breath(?:e|ing)?\b", re.I), "ACUTE_DYSPNEA"),
    (re.compile(r"\b(?:unconscious|unresponsive|passed\s+out|fainted|loss\s+of\s+consciousness|collapsed\s+and\s+unresponsive)\b", re.I), "UNCONSCIOUS"),
    (re.compile(r"\b(?:facial\s+droop\w*|face\s+is\s+droop\w*|slurred\s+speech|sudden\s+numbness|(?:arm|leg|face)\s+is\s+numb|stroke\s+symptoms)\b", re.I), "STROKE_SYMPTOMS"),
    (re.compile(r"\b(?:anaphylaxis|throat\s+closing|throat\s+swelling|severe\s+allergic\s+reaction)\b", re.I), "ANAPHYLAXIS"),
    (re.compile(r"\b(?:active\s+cardiac\s+arrest|heart\s+attack|choking\s+on)\b", re.I), "CARDIAC_ARREST"),
    (re.compile(r"\bblood\s+pressure\s+(?:is\s+)?(?:1[89]\d|2\d\d)\s*/\s*(?:1[2-9]\d|2\d\d)\b", re.I), "HYPERTENSIVE_CRISIS_VALUES"),
]

# Self-Harm & Suicide Patterns
SELF_HARM_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:suicid\w*|kill\s+myself|end\s+my\s+life|want\s+to\s+die|hang\s+myself|cut\s+my\s+wrists?|self-harm|take\s+my\s+own\s+life)\b", re.I), "SELF_HARM_DIRECT"),
    (re.compile(r"\bhow\s+to\s+(?:commit\s+suicide|kill\s+oneself|die\s+painlessly)\b", re.I), "SUICIDE_METHOD"),
    (re.compile(r"\b(?:never\s+wake\s+up|sleep\s+forever|to\s+never\s+wake\s+up)\b", re.I), "SUICIDE_IDEATION"),
]

# Poisoning & Overdose Patterns
POISONING_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:swallowed|drank|ingested)\s+(?:an?\s+)?(?:entire|whole)?\s*(?:bottle|glass|cup|amount)?\s*(?:of\s+)?(?:bleach|poison|antifreeze|cleaner|detergent|toxic|chemicals?)\b", re.I), "TOXIC_INGESTION"),
    (re.compile(r"\b(?:overdos\w*|took\s+(?:an?\s+)?(?:entire|whole)?\s*bottle\s+of|took\s+too\s+many\s+pills|swallowed\s+(?:a\s+handful|too\s+many)\s+pills)\b", re.I), "ACUTE_OVERDOSE"),
]

# Dosage Patterns (High Priority Clinical Query)
DOSAGE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:what\s+is\s+the\s+)?(?:dose|dosage|dosing)(?:\s+of|\s+for|\s+recommendations?|\s+guidelines?|\s+schedule|\s+regimen)?\b", re.I), "DOSAGE_GENERAL"),
    (re.compile(r"\b(?:what\s+dose|how\s+much|how\s+many\s+mg|how\s+many\s+tablets|how\s+many\s+pills)(?:\s+of\s+[\w\-]+)?\s+(?:should|can|to|is|do)\b", re.I), "DOSAGE_AMOUNT"),
    (re.compile(r"\b(?:starting\s+dose|maximum\s+dose|daily\s+dose|pediatric\s+dose|maintenance\s+dose|titration\s+dose)\b", re.I), "DOSAGE_SPECIFICATION"),
    (re.compile(r"\b(?:take\s+\d+\s*(?:mg|mcg|g|ml)|prescribed\s+\d+\s*(?:mg|mcg|g|ml)|\d+\s*mg\s+(?:daily|twice|once))\b", re.I), "DOSAGE_UNITS"),
    (re.compile(r"\b(?:dosing\s+frequency|how\s+often\s+to\s+take|times\s+per\s+day)\b", re.I), "DOSAGE_FREQUENCY"),
]

# Medication & Pharmacological Patterns
MEDICATION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:side\s+effects?|adverse\s+effects?|adverse\s+reactions?)\b", re.I), "MED_SIDE_EFFECTS"),
    (re.compile(r"\b(?:drug\s+interactions?|interact\s+with|contraindications?|contraindicated)\b", re.I), "MED_INTERACTIONS"),
    (re.compile(r"\b(?:mechanism\s+of\s+action|pharmacokinetics|pharmacodynamics|drug\s+class)\b", re.I), "MED_PHARMACOLOGY"),
    (re.compile(r"\b(?:medication|medicine|drug|pharmaceutical|prescription|generic\s+name|brand\s+name)\b", re.I), "MED_GENERAL"),
    (re.compile(r"\b(?:metformin|lisinopril|atorvastatin|amlodipine|losartan|albuterol|levothyroxine|omeprazole|amoxicillin|hydrochlorothiazide|ibuprofen|acetaminophen|aspirin|warfarin|apixaban|clopidogrel|gabapentin|sertraline|metoprolol)\b", re.I), "COMMON_DRUG_NAMES"),
]

# Lab Result & Diagnostic Testing Patterns
LAB_RESULT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:lab\s+result|lab\s+work|blood\s+test|blood\s+work|test\s+results?|panel\s+results?)\b", re.I), "LAB_GENERAL"),
    (re.compile(r"\b(?:cbc|complete\s+blood\s+count|lipid\s+panel|metabolic\s+panel|cmp|bmp|urinalysis)\b", re.I), "LAB_PANELS"),
    (re.compile(r"\b(?:hba1c|a1c|creatinine|eGFR|bun|troponin|alt|ast|bilirubin|tsh|inr|wbc|rbc|platelets?|hemoglobin|hematocrit|potassium|sodium|calcium)\b", re.I), "LAB_BIOMARKERS"),
    (re.compile(r"\b(?:reference\s+range|normal\s+range|elevated|abnormal|high\s+levels?|low\s+levels?|positive\s+result|negative\s+result)\b", re.I), "LAB_INTERPRETATION"),
    (re.compile(r"\bwhat\s+does\s+(?:my|this)?\s*(?:cbc|lab|report|test|level|result)\s+mean\b", re.I), "LAB_MEANING"),
]

# Document Summary Patterns
DOCUMENT_SUMMARY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:summarize|give\s+me\s+a\s+summary|summary\s+of|clinical\s+summary|executive\s+summary)\b", re.I), "DOC_SUMMARY_VERB"),
    (re.compile(r"\b(?:this\s+report|this\s+document|the\s+attached|uploaded\s+document|the\s+paper|the\s+study|the\s+records?)\b", re.I), "DOC_REFERENCE"),
    (re.compile(r"\b(?:key\s+findings|main\s+conclusions|overview\s+of\s+this\s+document|tl;?dr\s+of)\b", re.I), "DOC_KEY_FINDINGS"),
]

# Document Comparison Patterns
DOCUMENT_COMPARISON_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:compare|comparison|contrast|difference\s+between)\s+(?:these|the\s+two|both|the\s+uploaded|multiple)\s*(?:documents?|reports?|studies?|files?|guidelines?)\b", re.I), "DOC_COMPARE_DIRECT"),
    (re.compile(r"\bhow\s+do\s+(?:these\s+two|both\s+documents?|the\s+reports?)\s+(?:differ|compare)\b", re.I), "DOC_COMPARE_HOW"),
    (re.compile(r"\bcross-document\s+analysis\b", re.I), "DOC_COMPARE_CROSS"),
]

# Diagnosis Patterns
DIAGNOSIS_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:do\s+i\s+have|could\s+i\s+have|am\s+i\s+suffering\s+from|diagnose\s+me)\b", re.I), "DIAGNOSIS_INQUIRY"),
    (re.compile(r"\b(?:diagnostic\s+criteria|how\s+is\s+[\w\s]{2,25}\s+diagnosed|differential\s+diagnosis|confirmed\s+diagnosis)\b", re.I), "DIAGNOSIS_CRITERIA"),
    (re.compile(r"\bwhat\s+(?:disease|illness|condition|disorder)\s+(?:do\s+i\s+have|causes\s+these\s+symptoms)\b", re.I), "DIAGNOSIS_WHAT_DISEASE"),
]

# Symptom Query Patterns
SYMPTOM_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:symptoms?|signs?|clinical\s+manifestations?|presentation)\s+of\b", re.I), "SYMPTOM_OF"),
    (re.compile(r"\b(?:warning\s+signs?|early\s+signs?|common\s+symptoms?)\b", re.I), "SYMPTOM_SIGNS"),
    (re.compile(r"\b(?:i\s+have|experiencing|suffering\s+from)\s+(?:a\s+fever|headache|cough|fatigue|nausea|dizziness|chills|joint\s+pain|rash)\b", re.I), "SYMPTOM_REPORT"),
]

# Treatment Patterns
TREATMENT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:treatment\s+for|how\s+to\s+treat|how\s+is\s+[\w\s]{2,25}\s+treated|management\s+of)\b", re.I), "TREATMENT_HOW_TO"),
    (re.compile(r"\b(?:first-line\s+therapy|therapeutic\s+options?|standard\s+of\s+care|interventions?)\b", re.I), "TREATMENT_OPTIONS"),
    (re.compile(r"\b(?:cure\s+for|can\s+[\w\s]{2,25}\s+be\s+cured|remedies?|rehabilitation)\b", re.I), "TREATMENT_CURE"),
]

# Prevention Patterns
PREVENTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:prevent|prevention\s+of|how\s+to\s+prevent|can\s+[\w\s]{2,25}\s+be\s+prevented)\b", re.I), "PREVENTION_DIRECT"),
    (re.compile(r"\b(?:lifestyle\s+modifications?|preventative\s+measures?|prophylaxis|risk\s+reduction|avoid\s+getting)\b", re.I), "PREVENTION_LIFESTYLE"),
    (re.compile(r"\b(?:vaccin\w*|immuniz\w*|screenings?\s+to\s+prevent)\b", re.I), "PREVENTION_VACCINES"),
]

# General Health & Pathophysiology Patterns
GENERAL_HEALTH_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:what\s+causes|cause\s+of|etiology\s+of|pathophysiology\s+of)\b", re.I), "GH_PATHOPHYSIOLOGY"),
    (re.compile(r"\b(?:what\s+is|what\s+are|explain|overview\s+of|definition\s+of|how\s+does\s+the\s+[\w\s]{2,20}\s+work)\b", re.I), "GH_EXPLANATION"),
    (re.compile(r"\b(?:anatomy\s+of|physiology\s+of|role\s+of\s+[\w\s]{2,20}\s+in\s+the\s+body)\b", re.I), "GH_ANATOMY"),
]

# Explicit Non-Medical Out-of-Scope Patterns
OUT_OF_SCOPE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:bake|recipe|cooking|chocolate\s+cake|roast|dinner\s+ideas)\b", re.I), "NON_MED_COOKING"),
    (re.compile(r"\b(?:capital\s+of|who\s+won|president\s+of|weather\s+in|stock\s+price|crypto|bitcoin|football|soccer|nba|world\s+cup)\b", re.I), "NON_MED_GENERAL_TRIVIA"),
    (re.compile(r"\b(?:python\s+code|javascript|write\s+a\s+program|sql\s+query|debug\s+this\s+code|html|css)\b", re.I), "NON_MED_PROGRAMMING"),
    (re.compile(r"\b(?:fix\s+my\s+car|engine\s+oil|tire\s+pressure|brake\s+pads)\b", re.I), "NON_MED_AUTOMOTIVE"),
    (re.compile(r"\b(?:ignore\s+(?:all\s+)?previous\s+instructions|bypass\s+safety|jailbreak|pretend\s+you\s+are)\b", re.I), "PROMPT_INJECTION_OVERRIDE"),
]


class ClinicalIntentClassifier:
    """
    Deterministic clinical intent classification engine for AI-Healthcare-Agent.

    Evaluates natural language inquiries across the full Phase 6.1 clinical intent taxonomy,
    assigns confidence scores, enforces medical safety precedence, and generates structured
    routing recommendations.
    """

    @staticmethod
    def sanitize_input(query: Optional[str]) -> str:
        """
        Sanitizes raw input text by stripping null bytes, HTML tags, unprintable characters,
        and collapsing redundant whitespace. Bounded at 4000 characters to prevent ReDoS.
        """
        if not query:
            return ""
        text = str(query)
        # Strip null bytes
        text = text.replace("\x00", "")
        # Bounded length (prevents ReDoS and guarantees sub-10ms classification)
        if len(text) > 2000:
            text = text[:2000]
        # Strip HTML/script tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Unescape HTML entities
        text = html.unescape(text)
        # Collapse whitespace
        return " ".join(text.split())

    @classmethod
    def classify(
        cls,
        query: Optional[str],
        safety_assessment: Optional[SafetyAssessment] = None
    ) -> IntentClassificationResult:
        """
        Classifies user query intent using deterministic rules and signals.
        Enforces strict precedence for medical emergencies and safety violations.

        Args:
            query: User's natural language input string.
            safety_assessment: Optional pre-computed SafetyAssessment from MedicalSafetyGuard.

        Returns:
            IntentClassificationResult with intent, confidence, signals, and routing strategy.
        """
        start_time = time.perf_counter()
        clean_text = cls.sanitize_input(query)

        # -------------------------------------------------------------
        # STEP 1: EMPTY / GIBBERISH / UNCERTAIN INPUT HANDLING
        # -------------------------------------------------------------
        if not clean_text or len(clean_text.strip()) == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return IntentClassificationResult(
                intent=ClinicalIntent.UNCERTAIN,
                confidence=0.0,
                matched_signals=["EMPTY_INPUT"],
                requires_retrieval=False,
                requires_document_context=False,
                safety_priority=SafetyPriority.NONE,
                routing_strategy=ClinicalRoutingStrategy.STANDARD_RAG,
                latency_ms=elapsed_ms,
                metadata={"reason": "Query string is empty or contains only whitespace/null characters."}
            )

        # Purely punctuation / non-alphanumeric inputs
        if not any(c.isalnum() for c in clean_text):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return IntentClassificationResult(
                intent=ClinicalIntent.UNCERTAIN,
                confidence=0.1,
                matched_signals=["PUNCTUATION_ONLY"],
                requires_retrieval=False,
                requires_document_context=False,
                safety_priority=SafetyPriority.NONE,
                routing_strategy=ClinicalRoutingStrategy.STANDARD_RAG,
                latency_ms=elapsed_ms,
                metadata={"reason": "Query contains no alphanumeric tokens."}
            )

        # -------------------------------------------------------------
        # STEP 2: SAFETY PRECEDENCE (ABSOLUTE PRIORITY)
        # -------------------------------------------------------------
        # If pre-screen safety assessment is provided or evaluated:
        effective_safety = safety_assessment or SafetyClassifier.classify_question(clean_text)

        # Acute Emergency Interception
        if effective_safety.category == SafetyCategory.EMERGENCY_SYMPTOMS or not effective_safety.allow_normal_rag and "emergency" in effective_safety.reason.lower():
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return IntentClassificationResult(
                intent=ClinicalIntent.EMERGENCY,
                confidence=1.0,
                matched_signals=effective_safety.matched_rules or ["SAFETY_PRESCREEN_EMERGENCY"],
                requires_retrieval=False,
                requires_document_context=False,
                safety_priority=SafetyPriority.CRITICAL,
                routing_strategy=ClinicalRoutingStrategy.EMERGENCY_SAFETY,
                latency_ms=elapsed_ms,
                recommended_top_k=0,
                metadata={"safety_reason": effective_safety.reason, "category": effective_safety.category.value}
            )

        # Self-Harm & Suicide Interception
        if effective_safety.category == SafetyCategory.SELF_HARM_OR_SUICIDE or any("SELF_HARM" in r for r in effective_safety.matched_rules):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return IntentClassificationResult(
                intent=ClinicalIntent.SELF_HARM,
                confidence=1.0,
                matched_signals=effective_safety.matched_rules or ["SAFETY_PRESCREEN_SELF_HARM"],
                requires_retrieval=False,
                requires_document_context=False,
                safety_priority=SafetyPriority.CRITICAL,
                routing_strategy=ClinicalRoutingStrategy.SELF_HARM_SAFETY,
                latency_ms=elapsed_ms,
                recommended_top_k=0,
                metadata={"safety_reason": effective_safety.reason, "category": effective_safety.category.value}
            )

        # Poisoning & Toxic Overdose Interception
        if effective_safety.category == SafetyCategory.POISONING_OR_OVERDOSE or any("POISONING" in r or "OVERDOSE" in r for r in effective_safety.matched_rules):
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return IntentClassificationResult(
                intent=ClinicalIntent.POISONING,
                confidence=1.0,
                matched_signals=effective_safety.matched_rules or ["SAFETY_PRESCREEN_POISONING"],
                requires_retrieval=False,
                requires_document_context=False,
                safety_priority=SafetyPriority.CRITICAL,
                routing_strategy=ClinicalRoutingStrategy.POISONING_SAFETY,
                latency_ms=elapsed_ms,
                recommended_top_k=0,
                metadata={"safety_reason": effective_safety.reason, "category": effective_safety.category.value}
            )

        # Direct pattern match for emergencies if safety assessment was benign
        for pattern, sig in EMERGENCY_PATTERNS:
            if pattern.search(clean_text):
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return IntentClassificationResult(
                    intent=ClinicalIntent.EMERGENCY,
                    confidence=0.98,
                    matched_signals=[sig],
                    requires_retrieval=False,
                    requires_document_context=False,
                    safety_priority=SafetyPriority.CRITICAL,
                    routing_strategy=ClinicalRoutingStrategy.EMERGENCY_SAFETY,
                    latency_ms=elapsed_ms,
                    recommended_top_k=0,
                    metadata={"reason": "Matched critical emergency pattern."}
                )

        for pattern, sig in SELF_HARM_PATTERNS:
            if pattern.search(clean_text):
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return IntentClassificationResult(
                    intent=ClinicalIntent.SELF_HARM,
                    confidence=0.99,
                    matched_signals=[sig],
                    requires_retrieval=False,
                    requires_document_context=False,
                    safety_priority=SafetyPriority.CRITICAL,
                    routing_strategy=ClinicalRoutingStrategy.SELF_HARM_SAFETY,
                    latency_ms=elapsed_ms,
                    recommended_top_k=0,
                    metadata={"reason": "Matched self-harm pattern."}
                )

        for pattern, sig in POISONING_PATTERNS:
            if pattern.search(clean_text):
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return IntentClassificationResult(
                    intent=ClinicalIntent.POISONING,
                    confidence=0.99,
                    matched_signals=[sig],
                    requires_retrieval=False,
                    requires_document_context=False,
                    safety_priority=SafetyPriority.CRITICAL,
                    routing_strategy=ClinicalRoutingStrategy.POISONING_SAFETY,
                    latency_ms=elapsed_ms,
                    recommended_top_k=0,
                    metadata={"reason": "Matched toxic poisoning pattern."}
                )

        # -------------------------------------------------------------
        # STEP 3: OUT-OF-SCOPE & INJECTION OVERRIDE DETECTION
        # -------------------------------------------------------------
        out_of_scope_signals = []
        for pattern, sig in OUT_OF_SCOPE_PATTERNS:
            if pattern.search(clean_text):
                out_of_scope_signals.append(sig)

        if out_of_scope_signals:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return IntentClassificationResult(
                intent=ClinicalIntent.OUT_OF_SCOPE,
                confidence=0.95,
                matched_signals=out_of_scope_signals,
                requires_retrieval=False,
                requires_document_context=False,
                safety_priority=SafetyPriority.NONE,
                routing_strategy=ClinicalRoutingStrategy.OUT_OF_SCOPE_RESPONSE,
                latency_ms=elapsed_ms,
                recommended_top_k=0,
                metadata={"reason": "Query is non-medical or an injection override attempt."}
            )

        # -------------------------------------------------------------
        # STEP 4: MULTI-INTENT SIGNAL SCORING & DISCRIMINATION
        # -------------------------------------------------------------
        intent_scores: Dict[ClinicalIntent, float] = {
            ClinicalIntent.DOSAGE_QUERY: 0.0,
            ClinicalIntent.MEDICATION_QUERY: 0.0,
            ClinicalIntent.LAB_RESULT_QUERY: 0.0,
            ClinicalIntent.DOCUMENT_COMPARISON: 0.0,
            ClinicalIntent.DOCUMENT_SUMMARY: 0.0,
            ClinicalIntent.DIAGNOSIS_QUERY: 0.0,
            ClinicalIntent.SYMPTOM_QUERY: 0.0,
            ClinicalIntent.TREATMENT_QUERY: 0.0,
            ClinicalIntent.PREVENTION_QUERY: 0.0,
            ClinicalIntent.GENERAL_HEALTH: 0.0,
        }
        matched_signals_map: Dict[ClinicalIntent, List[str]] = {k: [] for k in intent_scores}

        # 4.1 Document Comparison (Strong specific intent)
        for pattern, sig in DOCUMENT_COMPARISON_PATTERNS:
            if pattern.search(clean_text):
                intent_scores[ClinicalIntent.DOCUMENT_COMPARISON] += 3.0
                matched_signals_map[ClinicalIntent.DOCUMENT_COMPARISON].append(sig)

        # 4.2 Document Summary (Strong specific intent)
        has_summary_verb = False
        has_doc_ref = False
        for pattern, sig in DOCUMENT_SUMMARY_PATTERNS:
            if pattern.search(clean_text):
                if sig == "DOC_SUMMARY_VERB":
                    has_summary_verb = True
                if sig == "DOC_REFERENCE":
                    has_doc_ref = True
                intent_scores[ClinicalIntent.DOCUMENT_SUMMARY] += 1.5
                matched_signals_map[ClinicalIntent.DOCUMENT_SUMMARY].append(sig)
        if has_summary_verb and has_doc_ref:
            intent_scores[ClinicalIntent.DOCUMENT_SUMMARY] += 2.0

        # 4.3 Dosage Query (Takes precedence over general medication query)
        for pattern, sig in DOSAGE_PATTERNS:
            if pattern.search(clean_text):
                intent_scores[ClinicalIntent.DOSAGE_QUERY] += 2.5
                matched_signals_map[ClinicalIntent.DOSAGE_QUERY].append(sig)

        # 4.4 Medication Query
        for pattern, sig in MEDICATION_PATTERNS:
            if pattern.search(clean_text):
                intent_scores[ClinicalIntent.MEDICATION_QUERY] += 1.8
                matched_signals_map[ClinicalIntent.MEDICATION_QUERY].append(sig)

        # 4.5 Lab Result Query
        for pattern, sig in LAB_RESULT_PATTERNS:
            if pattern.search(clean_text):
                intent_scores[ClinicalIntent.LAB_RESULT_QUERY] += 2.0
                matched_signals_map[ClinicalIntent.LAB_RESULT_QUERY].append(sig)

        # 4.6 Prevention Query
        for pattern, sig in PREVENTION_PATTERNS:
            if pattern.search(clean_text):
                intent_scores[ClinicalIntent.PREVENTION_QUERY] += 2.0
                matched_signals_map[ClinicalIntent.PREVENTION_QUERY].append(sig)

        # 4.7 Diagnosis Query
        for pattern, sig in DIAGNOSIS_PATTERNS:
            if pattern.search(clean_text):
                intent_scores[ClinicalIntent.DIAGNOSIS_QUERY] += 2.0
                matched_signals_map[ClinicalIntent.DIAGNOSIS_QUERY].append(sig)

        # 4.8 Treatment Query
        for pattern, sig in TREATMENT_PATTERNS:
            if pattern.search(clean_text):
                intent_scores[ClinicalIntent.TREATMENT_QUERY] += 1.8
                matched_signals_map[ClinicalIntent.TREATMENT_QUERY].append(sig)

        # 4.9 Symptom Query
        for pattern, sig in SYMPTOM_PATTERNS:
            if pattern.search(clean_text):
                intent_scores[ClinicalIntent.SYMPTOM_QUERY] += 1.6
                matched_signals_map[ClinicalIntent.SYMPTOM_QUERY].append(sig)

        # 4.10 General Health Query
        for pattern, sig in GENERAL_HEALTH_PATTERNS:
            if pattern.search(clean_text):
                intent_scores[ClinicalIntent.GENERAL_HEALTH] += 1.2
                matched_signals_map[ClinicalIntent.GENERAL_HEALTH].append(sig)

        # -------------------------------------------------------------
        # STEP 5: RANKING, DISAMBIGUATION & STRATEGY ASSIGNMENT
        # -------------------------------------------------------------
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_score = sorted_intents[0]
        second_intent, second_score = sorted_intents[1]

        # Clinical Disambiguation Heuristic:
        # If DOSAGE has matched signals and MEDICATION also matched signals, DOSAGE wins.
        if intent_scores[ClinicalIntent.DOSAGE_QUERY] > 0 and intent_scores[ClinicalIntent.MEDICATION_QUERY] > 0:
            top_intent = ClinicalIntent.DOSAGE_QUERY
            top_score = max(intent_scores[ClinicalIntent.DOSAGE_QUERY], intent_scores[ClinicalIntent.MEDICATION_QUERY])

        # If highest score is 0.0, evaluate whether this is a general inquiry or uncertain
        if top_score <= 0.0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            # If question has a medical condition word or question mark
            has_question_word = bool(re.search(r"\b(?:what|how|why|when|where|which|can|should|is|are|does)\b", clean_text, re.I))
            if has_question_word and len(clean_text.split()) >= 3:
                return IntentClassificationResult(
                    intent=ClinicalIntent.GENERAL_HEALTH,
                    confidence=0.55,
                    matched_signals=["QUESTION_FALLBACK"],
                    requires_retrieval=True,
                    requires_document_context=False,
                    safety_priority=SafetyPriority.NORMAL,
                    routing_strategy=ClinicalRoutingStrategy.GENERAL_HEALTH_RAG,
                    latency_ms=elapsed_ms,
                    recommended_top_k=5,
                    metadata={"fallback": True}
                )
            else:
                return IntentClassificationResult(
                    intent=ClinicalIntent.UNCERTAIN,
                    confidence=0.30,
                    matched_signals=["NO_SIGNALS_MATCHED"],
                    requires_retrieval=True,
                    requires_document_context=False,
                    safety_priority=SafetyPriority.NONE,
                    routing_strategy=ClinicalRoutingStrategy.STANDARD_RAG,
                    latency_ms=elapsed_ms,
                    recommended_top_k=5,
                    metadata={"fallback": True}
                )

        # Calculate confidence based on margin and raw match strength
        confidence = min(0.96, 0.70 + (top_score * 0.08))
        if second_score > 0 and (top_score - second_score) < 0.5:
            confidence = max(0.55, confidence - 0.15)

        # Map Intent to Routing Strategy and Safety Priority
        strategy_map = {
            ClinicalIntent.DOSAGE_QUERY: (ClinicalRoutingStrategy.DOSAGE_RAG, SafetyPriority.HIGH, 4, 0.30, False),
            ClinicalIntent.MEDICATION_QUERY: (ClinicalRoutingStrategy.MEDICATION_RAG, SafetyPriority.NORMAL, 5, 0.25, False),
            ClinicalIntent.LAB_RESULT_QUERY: (ClinicalRoutingStrategy.LAB_RAG, SafetyPriority.NORMAL, 5, 0.25, False),
            ClinicalIntent.DOCUMENT_COMPARISON: (ClinicalRoutingStrategy.DOCUMENT_COMPARISON_RAG, SafetyPriority.NORMAL, 8, 0.20, True),
            ClinicalIntent.DOCUMENT_SUMMARY: (ClinicalRoutingStrategy.DOCUMENT_SUMMARY_RAG, SafetyPriority.NORMAL, 6, 0.20, True),
            ClinicalIntent.DIAGNOSIS_QUERY: (ClinicalRoutingStrategy.DIAGNOSIS_RAG, SafetyPriority.HIGH, 5, 0.25, False),
            ClinicalIntent.SYMPTOM_QUERY: (ClinicalRoutingStrategy.SYMPTOM_RAG, SafetyPriority.NORMAL, 5, 0.25, False),
            ClinicalIntent.TREATMENT_QUERY: (ClinicalRoutingStrategy.TREATMENT_RAG, SafetyPriority.NORMAL, 5, 0.25, False),
            ClinicalIntent.PREVENTION_QUERY: (ClinicalRoutingStrategy.PREVENTION_RAG, SafetyPriority.NORMAL, 5, 0.25, False),
            ClinicalIntent.GENERAL_HEALTH: (ClinicalRoutingStrategy.GENERAL_HEALTH_RAG, SafetyPriority.NORMAL, 5, 0.25, False),
        }

        routing_strat, safety_prio, top_k, thresh, req_doc = strategy_map.get(
            top_intent,
            (ClinicalRoutingStrategy.STANDARD_RAG, SafetyPriority.NORMAL, 5, 0.25, False)
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return IntentClassificationResult(
            intent=top_intent,
            confidence=round(confidence, 3),
            matched_signals=matched_signals_map[top_intent],
            requires_retrieval=True,
            requires_document_context=req_doc,
            safety_priority=safety_prio,
            routing_strategy=routing_strat,
            latency_ms=elapsed_ms,
            recommended_top_k=top_k,
            recommended_similarity_threshold=thresh,
            metadata={
                "top_score": round(top_score, 2),
                "second_intent": second_intent.value if second_score > 0 else None,
                "second_score": round(second_score, 2) if second_score > 0 else 0.0
            }
        )
