import re
from typing import Dict, Any, Optional, Tuple, List

from backend.safety.safety_types import SafetyCategory, RiskLevel, SafetyAssessment
from backend.safety.safety_classifier import SafetyClassifier
from backend.rag.prompt_builder import MEDICAL_DISCLAIMER


class MedicalSafetyGuard:
    """
    Phase 4 Medical Safety Guard & Response Gatekeeper.
    
    Orchestrates pre-generation safety interception and post-generation compliance checks:
    - Pre-generation: Intercepts acute emergencies, self-harm, and poisonings BEFORE calling LLM/RAG.
    - Post-generation: Verifies that answers do not contain unauthorized diagnostic or prescriptive assertions,
      attaches mandatory clinical advisories, and ensures regulatory disclaimer compliance.
    """

    PROHIBITED_DIAGNOSTIC_PATTERNS = [
        re.compile(r'\b(?:you\s+have\s+been\s+diagnosed\s+with|my\s+diagnosis\s+is\s+that\s+you\s+have|i\s+diagnose\s+you\s+with|based\s+on\s+what\s+you\s+said,\s+you\s+have)\b', re.IGNORECASE),
        re.compile(r'\byou\s+definitely\s+have\s+(?:cancer|diabetes|hypertension|a\s+tumor)\b', re.IGNORECASE)
    ]

    PROHIBITED_PRESCRIBING_PATTERNS = [
        re.compile(r'\b(?:i\s+prescribe|i\s+am\s+prescribing|you\s+must\s+take\s+this\s+medication|you\s+need\s+to\s+take\s+\d+\s*mg)\b', re.IGNORECASE),
        re.compile(r'\b(?:stop\s+taking\s+your\s+medication\s+immediately|increase\s+your\s+dose\s+to\s+\d+)\b', re.IGNORECASE)
    ]

    @classmethod
    def pre_screen_inquiry(cls, user_question: Optional[str]) -> Tuple[bool, SafetyAssessment, Optional[str]]:
        """
        Executes pre-generation safety check on the user's question.
        
        Returns:
            (allow_generation, assessment, immediate_response)
            If allow_generation is False, immediate_response contains the emergency/safety advisory,
            and downstream retrieval and LLM generation MUST be bypassed.
        """
        assessment = SafetyClassifier.classify_question(user_question)

        if not assessment.allow_normal_rag:
            immediate_msg = assessment.emergency_message or assessment.guidance_message or (
                "This inquiry cannot be processed due to clinical safety boundaries."
            )
            return False, assessment, immediate_msg

        return True, assessment, None

    @classmethod
    def post_screen_answer(
        cls,
        user_question: str,
        generated_answer: Optional[str],
        assessment: Optional[SafetyAssessment] = None,
        sources_present: bool = True
    ) -> Dict[str, Any]:
        """
        Executes post-generation clinical boundary screening on the generated answer.
        Enforces diagnosis boundaries, prescribing boundaries, and category advisories.
        """
        ans = (generated_answer or "").strip()
        warnings: List[str] = []
        is_safe = True

        safety_meta = assessment or SafetyClassifier.classify_question(user_question)

        # 1. Prohibited Diagnostic Assertions
        for pat in cls.PROHIBITED_DIAGNOSTIC_PATTERNS:
            if pat.search(ans):
                is_safe = False
                warnings.append("Generated answer contained unauthorized diagnostic statement.")
                # Replace with clinical boundary statement
                ans = pat.sub("clinical evidence discusses", ans)

        # 2. Prohibited Prescribing Assertions
        for pat in cls.PROHIBITED_PRESCRIBING_PATTERNS:
            if pat.search(ans):
                is_safe = False
                warnings.append("Generated answer contained unauthorized prescriptive statement.")
                ans = pat.sub("guidelines note that physicians may prescribe", ans)

        # 3. Append category-specific guidance if needed
        guidance = safety_meta.guidance_message
        if guidance and guidance not in ans:
            # Check if answer is an emergency / refusal already
            if not any(k in ans for k in ("EMERGENCY ADVISORY", "SUPPORT NOTICE", "URGENT TOXICOLOGY")):
                ans = f"{ans}\n\n{guidance}"

        # 4. Enforce Medical Disclaimer
        disclaimer_appended = False
        if MEDICAL_DISCLAIMER not in ans and not any(k in ans for k in ("EMERGENCY ADVISORY", "SUPPORT NOTICE", "URGENT TOXICOLOGY")):
            ans = f"{ans}\n\n{MEDICAL_DISCLAIMER}"
            disclaimer_appended = True

        return {
            "post_check_passed": is_safe,
            "sanitized_answer": ans,
            "warnings": warnings,
            "safety_assessment": safety_meta.to_dict(),
            "disclaimer_appended": disclaimer_appended or (MEDICAL_DISCLAIMER in ans)
        }
