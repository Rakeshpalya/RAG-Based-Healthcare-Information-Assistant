import re
from typing import Optional, Dict, Any

from backend.agents.agent_state import SafetyDecision
from backend.rag.prompt_builder import MEDICAL_DISCLAIMER


class SafetyAgent:
    """
    Dual-layer clinical safety gatekeeper.

    SAFETY MANDATE:
    This AI Assistant is strictly an informational and educational research tool.
    It does NOT provide clinical diagnoses, medical treatment plans, emergency triage,
    or medication prescribing/dosing instructions.

    Deterministic rules execute FIRST. High-risk patterns immediately trigger
    safe clinical fallback guidance without calling RAG retrieval or LLM generation.
    """

    # 1. Acute Emergency Patterns
    EMERGENCY_PATTERNS = [
        r"\b(?:severe|crushing|sharp|radiating)\s+chest\s+pain\b",
        r"\b(?:difficulty|trouble|can't|cannot|unable\s+to)\s+breath(?:e|ing)?\b",
        r"\b(?:shortness\s+of\s+breath|dyspnea)\b",
        r"\b(?:unconscious|unresponsive|passed\s+out|fainted|loss\s+of\s+consciousness)\b",
        r"\b(?:severe|uncontrolled|heavy)\s+bleeding\b",
        r"\b(?:facial\s+droop\w*|face\s+is\s+droop\w*|slurred\s+speech|sudden\s+numbness|(?:arm|leg|face)\s+is\s+numb|stroke\s+symptoms)\b",
        r"\b(?:suicide|kill\s+myself|end\s+my\s+life|self-harm)\b",
        r"\b(?:heart\s+attack|anaphylaxis|choking)\b"
    ]

    # 2. Direct Diagnostic Requests
    DIAGNOSIS_PATTERNS = [
        r"\bdo\s+i\s+have\b",
        r"\bwhat\s+disease\s+do\s+i\s+have\b",
        r"\bwhat\s+illness\s+do\s+i\s+have\b",
        r"\bdiagnose\s+me\b",
        r"\bam\s+i\s+suffering\s+from\b",
        r"\bcould\s+i\s+have\b",
        r"\bwhat\s+is\s+my\s+diagnosis\b",
        r"\btell\s+me\s+if\s+i\s+have\b"
    ]

    # 3. Medication & Dosage Prescribing Inquiries
    MEDICATION_PATTERNS = [
        r"\bwhat\s+(?:medicine|medication|drug|pill)\s+should\s+i\s+take\b",
        r"\bwhat\s+(?:dose|dosage)(?:\s+of\s+\w+)?\s+should\s+i\s+take\b",
        r"\bhow\s+much\s+(?:mg|milligrams|dose)\s+should\s+i\s+take\b",
        r"\bprescribe\s+(?:me|for\s+me)\b",
        r"\bcan\s+you\s+give\s+me\s+a\s+prescription\b"
    ]

    # 4. Treatment & Medication Alteration Inquiries
    TREATMENT_ALTERATION_PATTERNS = [
        r"\bshould\s+i\s+stop\s+taking\b",
        r"\bcan\s+i\s+stop\s+(?:my\s+)?medication\b",
        r"\bcan\s+i\s+change\s+(?:my\s+)?dosage\b",
        r"\bcan\s+i\s+increase\s+(?:my\s+)?dose\b",
        r"\bshould\s+i\s+skip\s+(?:my\s+)?dose\b",
        r"\bdiscontinue\s+(?:my\s+)?treatment\b"
    ]

    # Compile regex patterns for performance
    _re_emergency = [re.compile(p, re.IGNORECASE) for p in EMERGENCY_PATTERNS]
    _re_diagnosis = [re.compile(p, re.IGNORECASE) for p in DIAGNOSIS_PATTERNS]
    _re_medication = [re.compile(p, re.IGNORECASE) for p in MEDICATION_PATTERNS]
    _re_treatment = [re.compile(p, re.IGNORECASE) for p in TREATMENT_ALTERATION_PATTERNS]

    @classmethod
    def evaluate_pre_check(cls, user_question: Optional[str]) -> SafetyDecision:
        """
        Executes deterministic clinical safety pre-screening on the incoming user question.
        Delegates to the Phase 4 SafetyClassifier while maintaining full backwards compatibility.

        Returns:
            SafetyDecision object indicating whether the question is permitted to proceed,
            and providing immediate clinical guidance if blocked.
        """
        if not user_question or not user_question.strip():
            return SafetyDecision(
                category="EMPTY_QUERY",
                allowed=False,
                requires_clinician=False,
                requires_emergency_guidance=False,
                reason="Inquiry cannot be empty or whitespace.",
                fallback_response="Please enter a valid medical or health research question."
            )

        from backend.safety.safety_classifier import SafetyClassifier
        from backend.safety.safety_types import SafetyCategory
        assessment = SafetyClassifier.classify_question(user_question)

        # 1. Emergency Symptoms / Immediate Danger / Self-Harm / Overdose
        if assessment.category in (
            SafetyCategory.EMERGENCY_SYMPTOMS,
            SafetyCategory.IMMEDIATE_DANGER,
            SafetyCategory.SELF_HARM_OR_SUICIDE,
            SafetyCategory.POISONING_OR_OVERDOSE,
            SafetyCategory.UNSAFE_OR_UNSUPPORTED_REQUEST
        ):
            return SafetyDecision(
                category="EMERGENCY_SYMPTOM",
                allowed=False,
                requires_clinician=True,
                requires_emergency_guidance=True,
                reason=assessment.reason,
                fallback_response=assessment.emergency_message or SafetyClassifier.EMERGENCY_ADVISORY
            )

        # 2. Diagnostic Request
        if assessment.category == SafetyCategory.DIAGNOSIS_REQUEST:
            return SafetyDecision(
                category="DIAGNOSIS_REQUEST",
                allowed=False,
                requires_clinician=True,
                requires_emergency_guidance=False,
                reason=assessment.reason,
                fallback_response=assessment.guidance_message or SafetyClassifier.DIAGNOSIS_GUIDANCE
            )

        # 3. Treatment Alteration / Stopping Medication
        if assessment.category == SafetyCategory.TREATMENT_REQUEST:
            return SafetyDecision(
                category="TREATMENT_REQUEST",
                allowed=False,
                requires_clinician=True,
                requires_emergency_guidance=False,
                reason=assessment.reason,
                fallback_response=assessment.guidance_message or SafetyClassifier.TREATMENT_GUIDANCE
            )

        # 4. Medication Prescribing / Dosage Request
        if assessment.category in (SafetyCategory.MEDICATION_REQUEST, SafetyCategory.DOSAGE_REQUEST):
            return SafetyDecision(
                category="MEDICATION_REQUEST",
                allowed=False,
                requires_clinician=True,
                requires_emergency_guidance=False,
                reason=assessment.reason,
                fallback_response=SafetyClassifier.MEDICATION_GUIDANCE
            )

        # Default: Permitted informational / educational inquiry
        return SafetyDecision(
            category="SAFE_INFORMATIONAL",
            allowed=True,
            requires_clinician=False,
            requires_emergency_guidance=False,
            reason=assessment.reason,
            fallback_response=""
        )

    @classmethod
    def evaluate_post_check(
        cls,
        user_question: str,
        generated_answer: Optional[str],
        sources_present: bool = True
    ) -> Dict[str, Any]:
        """
        Executes post-generation safety screening on the generated answer.
        Ensures the model did not generate unauthorized diagnostic or prescribing claims.
        """
        from backend.safety.medical_safety_guard import MedicalSafetyGuard
        res = MedicalSafetyGuard.post_screen_answer(
            user_question=user_question,
            generated_answer=generated_answer,
            sources_present=sources_present
        )
        return {
            "post_check_passed": res["post_check_passed"],
            "warnings": res["warnings"],
            "disclaimer_appended": res["disclaimer_appended"],
            "sanitized_answer": res["sanitized_answer"]
        }
