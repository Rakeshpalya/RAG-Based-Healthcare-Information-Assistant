import re
from typing import List, Dict, Any, Optional, Tuple

from backend.safety.safety_types import SafetyCategory, RiskLevel, SafetyAssessment


class SafetyClassifier:
    """
    Deterministic rule-based clinical safety classification engine (Phase 4.1).
    Evaluates incoming inquiries across 15 medical safety categories, assigns risk levels,
    and defines strict permissions for downstream RAG retrieval and generation.
    """

    # 1. Critical Emergency Patterns
    EMERGENCY_PATTERNS = [
        (r"\b(?:severe|crushing|sharp|radiating|intense|acute)\s+(?:\w+\s+)?chest\s+pain\b", "EMERGENCY_CHEST_PAIN"),
        (r"\bchest\s+pain\s+(?:radiating\s+[\w\s]{0,25}|with|and)\s+(?:sweating|nausea|shortness\s+of\s+breath|difficulty\s+breathing|jaw|left\s+arm)\b", "EMERGENCY_CARDIAC_SYMPTOMS"),
        (r"\b(?:difficulty|trouble|can't|cannot|unable\s+to|struggling\s+to)\s+breath(?:e|ing)?\b", "EMERGENCY_DYSPNEA"),
        (r"\b(?:shortness\s+of\s+breath|severe\s+dyspnea|gasping\s+for\s+air|suffocating)\b", "EMERGENCY_RESPIRATORY"),
        (r"\b(?:unconscious|unresponsive|passed\s+out|fainted|loss\s+of\s+consciousness|collapsed\s+and\s+unresponsive)\b", "EMERGENCY_UNCONSCIOUS"),
        (r"\b(?:severe|uncontrolled|heavy|arterial|spurting)\s+bleeding\b", "EMERGENCY_HEMORRHAGE"),
        (r"\b(?:facial\s+droop\w*|face\s+is\s+droop\w*|slurred\s+speech|sudden\s+numbness|(?:arm|leg|face)\s+is\s+numb|stroke\s+symptoms|fast\s+stroke)\b", "EMERGENCY_STROKE"),
        (r"\b(?:anaphylaxis|throat\s+closing|throat\s+swelling|severe\s+allergic\s+reaction)\b", "EMERGENCY_ANAPHYLAXIS"),
        (r"\b(?:having\s+a\s+seizure|active\s+seizure|continuous\s+convulsions|grand\s+mal)\b", "EMERGENCY_SEIZURE"),
        (r"\b(?:worst\s+headache\s+of\s+my\s+life|thunderclap\s+headache|sudden\s+vision\s+loss)\b", "EMERGENCY_NEUROLOGICAL"),
        (r"\b(?:heart\s+attack|active\s+cardiac\s+arrest|choking\s+on)\b", "EMERGENCY_CARDIORESPIRATORY")
    ]

    # Educational / Non-acute context filters for chest pain, stroke, etc.
    ACUTE_PERSONAL_INDICATORS = [
        r"\bi\s+(?:have|am\s+having|feel|am\s+experiencing)\b",
        r"\bmy\s+(?:chest|arm|face|head|throat|heart|breathing)\b",
        r"\bhelp\s+(?:me|my)\b",
        r"\bpatient\s+is\s+(?:unresponsive|bleeding|having|unconscious)\b",
        r"\bright\s+now\b",
        r"\bsuddenly\b"
    ]

    EDUCATIONAL_PATTERNS = [
        r"\b(?:what\s+(?:is|are|causes)|explain|describe|pathophysiology|mechanism|etiology|difference\s+between|overview\s+of)\b",
        r"\b(?:can\s+(?:gastroesophageal\s+reflux|gerd|anxiety|muscle\s+strain)\s+cause)\b",
        r"\b(?:in\s+a\s+textbook|academic|historical|definition\s+of)\b"
    ]

    # 2. Self-Harm & Suicide Patterns
    SELF_HARM_PATTERNS = [
        (r"\b(?:suicid\w*|kill\s+myself|end\s+my\s+life|want\s+to\s+die|hang\s+myself|cut\s+my\s+wrists?|self-harm|take\s+my\s+own\s+life)\b", "SELF_HARM_DIRECT"),
        (r"\bhow\s+to\s+(?:commit\s+suicide|kill\s+oneself|die\s+painlessly)\b", "SELF_HARM_METHODS"),
        (r"\b(?:never\s+wake\s+up|sleep\s+forever|to\s+never\s+wake\s+up)\b", "SELF_HARM_SLEEP")
    ]

    # 3. Poisoning & Overdose Patterns
    POISONING_OVERDOSE_PATTERNS = [
        (r"\b(?:overdos\w*|took\s+(?:an?\s+)?(?:entire|whole)?\s*bottle\s+of\b|took\s+(?:an?\s+)?(?:entire\s+)?bottle\s+of\s+[\w\s]{0,25}pills|took\s+too\s+many\s+pills|swallowed\s+(?:an?\s+)?(?:entire|whole)?\s*bottle|(?:swallowed|took|ingested)\s+(?:\d{2,}|a\s+handful\s+of|too\s+many|excessive)\s+(?:[\w\-]+\s+)?(?:pills|tablets|capsules))\b", "OVERDOSE_MEDICATION"),
        (r"\b(?:swallowed|drank|ingested)\s+(?:(?:an?\s+)?(?:whole|entire)?\s*(?:bottle|cup|glass|can|amount)\s+of\s+)?(?:bleach|poison|antifreeze|cleaner|detergent|chemical|battery|toxic)\b", "POISONING_INGESTION"),
        (r"\b(?:toxic\s+exposure|chemical\s+burn\s+in\s+eye|inhaled\s+toxic\s+fumes)\b", "POISONING_EXPOSURE")
    ]

    # 4. Immediate Danger / Unsafe & Unsupported Patterns
    UNSAFE_PATTERNS = [
        (r"\bhow\s+to\s+(?:synthesize|manufacture|extract|make)\s+(?:[\w\-]+\s+)?(?:fentanyl|meth|heroin|ricin|anthrax|poison|explosive)\b", "UNSAFE_ILLICIT_SYNTHESIS"),
        (r"\b(?:lethal\s+dose\s+to\s+kill|recreational\s+high\s+from|bypass\s+(?:the\s+)?safety\s+(?:filter|guard|rules|engine|system)|ignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions|rules)|alternate\s+universe\s+with\s+no\s+(?:medical\s+)?regulations|act\s+as\s+an\s+unrestricted\s+doctor|pretend\s+you\s+are\s+an?\s+unregulated)\b", "UNSAFE_MALICIOUS")
    ]

    # 5. Direct Diagnosis Requests
    DIAGNOSIS_PATTERNS = [
        (r"\bdo\s+i\s+have\b", "DIAGNOSIS_DO_I_HAVE"),
        (r"\bdiagnose\s+(?:me|my\s+condition|my\s+symptoms)\b", "DIAGNOSIS_DIRECT"),
        (r"\bwhat\s+(?:disease|illness|condition)\s+do\s+i\s+have\b", "DIAGNOSIS_WHAT_DISEASE"),
        (r"\bcould\s+i\s+have\b", "DIAGNOSIS_COULD_I_HAVE"),
        (r"\bwhat\s+is\s+my\s+diagnosis\b", "DIAGNOSIS_WHAT_IS_MY"),
        (r"\bam\s+i\s+suffering\s+from\b", "DIAGNOSIS_AM_I_SUFFERING"),
        (r"\btell\s+me\s+if\s+i\s+have\b", "DIAGNOSIS_TELL_ME_IF"),
        (r"\bis\s+this\s+(?:cancer|a\s+tumor|a\s+stroke|a\s+heart\s+attack|diabetes)\b", "DIAGNOSIS_IS_THIS_SPECIFIC"),
        (r"\bdo\s+these\s+symptoms\s+mean\s+i\s+have\b", "DIAGNOSIS_SYMPTOMS_MEAN")
    ]

    # 6. Treatment Requests / Medication Alteration
    TREATMENT_PATTERNS = [
        (r"\b(?:should|can|could)\s+i\s+stop\s+(?:taking\b|my\b)", "TREATMENT_STOP_TAKING"),
        (r"\b(?:should|can|could)\s+i\s+(?:stop|discontinue|pause|quit)\s+(?:taking\s+)?(?:my\s+)?(?:prescribed\s+)?(?:\w+\s+)*(?:medication|treatment|pills?|drugs?|medicine)\b", "TREATMENT_CAN_I_STOP"),
        (r"\bcan\s+i\s+change\s+(?:my\s+)?dosage\b", "TREATMENT_CHANGE_DOSAGE"),
        (r"\bshould\s+i\s+skip\s+(?:my\s+)?dose\b", "TREATMENT_SKIP_DOSE"),
        (r"\bshould\s+i\s+discontinue\s+(?:my\s+)?(?:treatment|medication)\b", "TREATMENT_DISCONTINUE"),
        (r"\bhow\s+should\s+i\s+treat\s+my\b", "TREATMENT_HOW_TREAT"),
        (r"\bcure\s+my\s+[a-z0-9\-]+\b", "TREATMENT_CURE_MY")
    ]

    # 7. Medication Prescribing Requests
    MEDICATION_PATTERNS = [
        (r"\bwhat\s+(?:medicine|medication|drug|pill)\s+should\s+i\s+take\b", "MEDICATION_WHAT_SHOULD_I_TAKE"),
        (r"\bprescribe\s+(?:me|for\s+me)\b", "MEDICATION_PRESCRIBE_ME"),
        (r"\b(?:can\s+you\s+)?(?:write|give|get)\s+(?:me\s+)?(?:a\s+)?prescription\b", "MEDICATION_GIVE_PRESCRIPTION"),
        (r"\bprescription\s+(?:for|refill)\b", "MEDICATION_PRESCRIPTION_REQUEST"),
        (r"\brecommend\s+a\s+(?:medicine|drug|pill)\s+for\s+me\b", "MEDICATION_RECOMMEND_FOR_ME")
    ]

    # 8. Dosage Modification / Personal Dosing Requests
    DOSAGE_PATTERNS = [
        (r"\bwhat\s+(?:dose|dosage)(?:\s+of\s+[\w\-]+)?\s+should\s+i\s+take\b", "DOSAGE_WHAT_SHOULD_I_TAKE"),
        (r"\bhow\s+much\s+(?:mg|milligrams|dose)(?:\s+of\s+[\w\-]+)?\s+should\s+i\s+take\b", "DOSAGE_HOW_MUCH_MG"),
        (r"\bcan\s+i\s+(?:increase|decrease|double|cut|change)\s+my\s+dose\b", "DOSAGE_CHANGE_DOSE"),
        (r"\bshould\s+i\s+take\s+\d+\s*(?:mg|milligrams|pills?)\b", "DOSAGE_SPECIFIC_AMOUNT"),
        (r"\bchange\s+my\s+dosage\b", "DOSAGE_CHANGE_MY_DOSAGE")
    ]

    # 9. Drug Interaction Requests
    INTERACTION_PATTERNS = [
        (r"\bcan\s+i\s+take\s+[\w\-]+\s+(?:with|together\s+with|and)\s+[\w\-]+\b", "INTERACTION_CAN_I_TAKE"),
        (r"\binteract\w*\s+between\s+[\w\-]+\s+and\s+[\w\-]+\b", "INTERACTION_BETWEEN"),
        (r"\bdrug\s+interaction\w*(?:\s+between)?\b", "INTERACTION_GENERAL"),
        (r"\btake\s+[\w\-]+\s+while\s+on\s+[\w\-]+\b", "INTERACTION_WHILE_ON")
    ]

    # 10. Contraindication Requests
    CONTRAINDICATION_PATTERNS = [
        (r"\bis\s+[\w\-]+\s+contraindicated\b", "CONTRAINDICATION_IS_IT"),
        (r"\bcontraindications?\s+(?:for|to|of)\s+[\w\-]+\b", "CONTRAINDICATION_OF_DRUG"),
        (r"\bwho\s+should\s+not\s+take\s+[\w\-]+\b", "CONTRAINDICATION_WHO_NOT_TAKE")
    ]

    # 11. Pregnancy High-Risk Contexts
    PREGNANCY_PATTERNS = [
        (r"\b(?:pregnant|pregnancy|breastfeeding|nursing\s+mother|lactating|in\s+(?:the\s+)?(?:first|second|third)\s+trimester)\b", "PREGNANCY_CONTEXT")
    ]

    # 12. Pediatric High-Risk Contexts
    PEDIATRIC_PATTERNS = [
        (r"\b(?:infant|baby|babies|newborn|toddler|child|children|pediatric|pediatrics|\b\d+\s*(?:month|year)s?\s*old\s*(?:baby|child|infant))\b", "PEDIATRIC_CONTEXT")
    ]

    # Standard clinical advisory strings
    EMERGENCY_ADVISORY = (
        "EMERGENCY ADVISORY: If you or someone near you is experiencing severe or life-threatening symptoms "
        "(such as acute chest pain, difficulty breathing, slurred speech, sudden numbness, or heavy bleeding), "
        "please contact your local emergency service (e.g. 911 or 112) or go to the nearest emergency department immediately. "
        "This AI Healthcare Assistant cannot provide emergency medical care or clinical triage."
    )

    SELF_HARM_ADVISORY = (
        "SUPPORT NOTICE: If you are feeling overwhelmed or having thoughts of self-harm or suicide, please know that support is available right now. "
        "Reach out immediately to a confidential crisis lifeline: in the US and Canada, call or text 988; in the UK, call 111 or 999; "
        "in other countries, contact your local emergency services or a mental health crisis helpline. Please talk to someone who can help you stay safe."
    )

    POISONING_ADVISORY = (
        "URGENT TOXICOLOGY ADVISORY: If you suspect poisoning, medication overdose, or accidental ingestion of a hazardous substance, "
        "seek emergency medical assistance immediately. Contact your local emergency service (such as 911 or 112) or Poison Control (1-800-222-1222 in the US) "
        "or go to the nearest emergency department right away. Do not induce vomiting or administer home remedies unless directly instructed by medical personnel."
    )

    UNSAFE_ADVISORY = (
        "SAFETY NOTICE: This AI Healthcare Assistant cannot provide instructions, calculations, or assistance regarding "
        "hazardous, illegal, or lethal chemical substances."
    )

    DIAGNOSIS_GUIDANCE = (
        "CLINICAL BOUNDARY NOTICE: This AI Assistant provides educational research information grounded in reference documents "
        "and cannot provide personal medical diagnoses. If you are experiencing symptoms or suspect an underlying illness, "
        "please consult a qualified licensed healthcare professional for an in-person clinical evaluation and diagnosis."
    )

    TREATMENT_GUIDANCE = (
        "MEDICATION SAFETY NOTICE: You should never stop, pause, or alter the dosage of your prescribed medications "
        "without direct instruction from your treating healthcare provider. Abruptly altering prescription medications can cause "
        "adverse rebound effects. Please contact your prescribing physician or pharmacist to discuss any treatment concerns."
    )

    MEDICATION_GUIDANCE = (
        "PRESCRIPTION SAFETY NOTICE: This AI Assistant is not authorized to prescribe pharmaceutical drugs or recommend specific personal medications. "
        "Only a licensed medical practitioner who has reviewed your health history, allergies, and contraindications can safely prescribe medications. "
        "Please speak with your doctor or pharmacist."
    )

    DOSAGE_GUIDANCE = (
        "DOSAGE SAFETY NOTICE: Specific medication dosages must be individualized by a licensed healthcare provider based on clinical factors "
        "such as renal function, weight, and concurrent therapies. Do not adjust prescription dosages without physician supervision."
    )

    INTERACTION_GUIDANCE = (
        "DRUG INTERACTION ADVISORY: Medication interaction information is provided for educational and research reference. "
        "Always confirm concurrent medication safety directly with your prescribing physician or a licensed pharmacist."
    )

    PREGNANCY_GUIDANCE = (
        "PREGNANCY SAFETY ADVISORY: Medication and treatment decisions during pregnancy or lactation carry significant risks for maternal and fetal health. "
        "Always verify health decisions with an obstetrician or licensed healthcare provider."
    )

    PEDIATRIC_GUIDANCE = (
        "PEDIATRIC SAFETY ADVISORY: Pediatric healthcare and dosing require precise clinical supervision based on child weight and developmental age. "
        "Consult a pediatrician before administering or adjusting treatments for children."
    )

    @classmethod
    def classify_question(cls, user_question: Optional[str]) -> SafetyAssessment:
        """
        Executes multi-category deterministic clinical safety screening on the incoming query.
        Returns a rich SafetyAssessment object defining permissions, risk level, and advisories.
        """
        if not user_question or not user_question.strip():
            return SafetyAssessment(
                category=SafetyCategory.INSUFFICIENT_EVIDENCE,
                risk_level=RiskLevel.INFO.value,
                requires_escalation=False,
                allow_normal_rag=False,
                allow_medication_information=False,
                allow_dosage_information=False,
                reason="Inquiry cannot be empty or whitespace.",
                guidance_message="Please enter a valid medical or health research question."
            )

        q = user_question.strip()
        matched_rules: List[str] = []

        # 1. Unsafe / Malicious requests
        for pat, rule_id in cls.UNSAFE_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                matched_rules.append(rule_id)
                return SafetyAssessment(
                    category=SafetyCategory.UNSAFE_OR_UNSUPPORTED_REQUEST,
                    risk_level=RiskLevel.HIGH.value,
                    requires_escalation=True,
                    allow_normal_rag=False,
                    allow_medication_information=False,
                    allow_dosage_information=False,
                    emergency_message=cls.UNSAFE_ADVISORY,
                    reason="Inquiry requests unsafe, illegal, or hazardous substances.",
                    matched_rules=matched_rules,
                    guidance_message=cls.UNSAFE_ADVISORY
                )

        # 2. Self-Harm & Suicide
        for pat, rule_id in cls.SELF_HARM_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                matched_rules.append(rule_id)
                return SafetyAssessment(
                    category=SafetyCategory.SELF_HARM_OR_SUICIDE,
                    risk_level=RiskLevel.CRITICAL.value,
                    requires_escalation=True,
                    allow_normal_rag=False,
                    allow_medication_information=False,
                    allow_dosage_information=False,
                    emergency_message=cls.SELF_HARM_ADVISORY,
                    reason="Inquiry indicates immediate risk of self-harm or suicide.",
                    matched_rules=matched_rules,
                    guidance_message=cls.SELF_HARM_ADVISORY
                )

        # 3. Poisoning & Overdose
        for pat, rule_id in cls.POISONING_OVERDOSE_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                matched_rules.append(rule_id)
                return SafetyAssessment(
                    category=SafetyCategory.POISONING_OR_OVERDOSE,
                    risk_level=RiskLevel.CRITICAL.value,
                    requires_escalation=True,
                    allow_normal_rag=False,
                    allow_medication_information=False,
                    allow_dosage_information=False,
                    emergency_message=cls.POISONING_ADVISORY,
                    reason="Inquiry indicates acute toxic ingestion or medication overdose.",
                    matched_rules=matched_rules,
                    guidance_message=cls.POISONING_ADVISORY
                )

        # 4. Emergency Symptoms
        is_educational = any(re.search(pat, q, re.IGNORECASE) for pat in cls.EDUCATIONAL_PATTERNS)
        has_personal_acute = any(re.search(pat, q, re.IGNORECASE) for pat in cls.ACUTE_PERSONAL_INDICATORS)

        for pat, rule_id in cls.EMERGENCY_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                matched_rules.append(rule_id)
                # If question is purely educational without personal acute symptoms, let it be evaluated as normal
                if is_educational and not has_personal_acute:
                    continue

                return SafetyAssessment(
                    category=SafetyCategory.EMERGENCY_SYMPTOMS,
                    risk_level=RiskLevel.CRITICAL.value,
                    requires_escalation=True,
                    allow_normal_rag=False,
                    allow_medication_information=False,
                    allow_dosage_information=False,
                    emergency_message=cls.EMERGENCY_ADVISORY,
                    reason="Inquiry contains acute, life-threatening emergency medical symptoms.",
                    matched_rules=matched_rules,
                    guidance_message=cls.EMERGENCY_ADVISORY
                )

        # 5. Check Contextual Modifiers: Pregnancy and Pediatric Context
        is_pregnancy = False
        for pat, rule_id in cls.PREGNANCY_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                is_pregnancy = True
                matched_rules.append(rule_id)

        is_pediatric = False
        for pat, rule_id in cls.PEDIATRIC_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                is_pediatric = True
                matched_rules.append(rule_id)

        # 6. Diagnosis Requests
        for pat, rule_id in cls.DIAGNOSIS_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                matched_rules.append(rule_id)
                return SafetyAssessment(
                    category=SafetyCategory.DIAGNOSIS_REQUEST,
                    risk_level=RiskLevel.HIGH.value,
                    requires_escalation=True,
                    allow_normal_rag=True,
                    allow_medication_information=True,
                    allow_dosage_information=False,
                    reason="Inquiry requests a personal clinical diagnosis.",
                    matched_rules=matched_rules,
                    guidance_message=cls.DIAGNOSIS_GUIDANCE
                )

        # 7. Medication Prescribing Requests
        for pat, rule_id in cls.MEDICATION_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                matched_rules.append(rule_id)
                guidance = cls.MEDICATION_GUIDANCE
                if is_pediatric:
                    guidance = f"{cls.PEDIATRIC_GUIDANCE}\n\n{cls.MEDICATION_GUIDANCE}"
                elif is_pregnancy:
                    guidance = f"{cls.PREGNANCY_GUIDANCE}\n\n{cls.MEDICATION_GUIDANCE}"

                return SafetyAssessment(
                    category=SafetyCategory.MEDICATION_REQUEST,
                    risk_level=RiskLevel.HIGH.value,
                    requires_escalation=True,
                    allow_normal_rag=True,
                    allow_medication_information=True,
                    allow_dosage_information=False,
                    reason="Inquiry requests pharmaceutical prescription or personal medication advice.",
                    matched_rules=matched_rules,
                    guidance_message=guidance
                )

        # 8. Treatment / Medication Alteration Requests
        for pat, rule_id in cls.TREATMENT_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                matched_rules.append(rule_id)
                return SafetyAssessment(
                    category=SafetyCategory.TREATMENT_REQUEST,
                    risk_level=RiskLevel.HIGH.value,
                    requires_escalation=True,
                    allow_normal_rag=True,
                    allow_medication_information=True,
                    allow_dosage_information=False,
                    reason="Inquiry requests stopping or altering medical treatment.",
                    matched_rules=matched_rules,
                    guidance_message=cls.TREATMENT_GUIDANCE
                )

        # 9. Dosage Requests
        for pat, rule_id in cls.DOSAGE_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                matched_rules.append(rule_id)
                cat = SafetyCategory.DOSAGE_REQUEST
                guidance = cls.DOSAGE_GUIDANCE
                if is_pediatric:
                    guidance = f"{cls.PEDIATRIC_GUIDANCE}\n\n{cls.DOSAGE_GUIDANCE}"
                elif is_pregnancy:
                    guidance = f"{cls.PREGNANCY_GUIDANCE}\n\n{cls.DOSAGE_GUIDANCE}"

                return SafetyAssessment(
                    category=cat,
                    risk_level=RiskLevel.HIGH.value,
                    requires_escalation=True,
                    allow_normal_rag=True,
                    allow_medication_information=True,
                    allow_dosage_information=False,
                    reason="Inquiry asks for personal dosage instructions or dosage modification.",
                    matched_rules=matched_rules,
                    guidance_message=guidance
                )

        # 10. Drug Interaction Requests
        for pat, rule_id in cls.INTERACTION_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                matched_rules.append(rule_id)
                return SafetyAssessment(
                    category=SafetyCategory.DRUG_INTERACTION_REQUEST,
                    risk_level=RiskLevel.MEDIUM.value,
                    requires_escalation=False,
                    allow_normal_rag=True,
                    allow_medication_information=True,
                    allow_dosage_information=True,
                    reason="Inquiry asks about medication interactions.",
                    matched_rules=matched_rules,
                    guidance_message=cls.INTERACTION_GUIDANCE
                )

        # 11. Contraindication Requests
        for pat, rule_id in cls.CONTRAINDICATION_PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                matched_rules.append(rule_id)
                return SafetyAssessment(
                    category=SafetyCategory.CONTRAINDICATION_REQUEST,
                    risk_level=RiskLevel.MEDIUM.value,
                    requires_escalation=False,
                    allow_normal_rag=True,
                    allow_medication_information=True,
                    allow_dosage_information=True,
                    reason="Inquiry asks about medical contraindications.",
                    matched_rules=matched_rules,
                    guidance_message=None
                )

        # 12. Pregnancy High-Risk (without medication request)
        if is_pregnancy:
            return SafetyAssessment(
                category=SafetyCategory.PREGNANCY_HIGH_RISK,
                risk_level=RiskLevel.HIGH.value,
                requires_escalation=False,
                allow_normal_rag=True,
                allow_medication_information=True,
                allow_dosage_information=False,
                reason="Inquiry involves pregnancy or lactation context.",
                matched_rules=matched_rules,
                guidance_message=cls.PREGNANCY_GUIDANCE
            )

        # 13. Pediatric High-Risk (without medication request)
        if is_pediatric:
            return SafetyAssessment(
                category=SafetyCategory.PEDIATRIC_HIGH_RISK,
                risk_level=RiskLevel.HIGH.value,
                requires_escalation=False,
                allow_normal_rag=True,
                allow_medication_information=True,
                allow_dosage_information=False,
                reason="Inquiry involves pediatric context.",
                matched_rules=matched_rules,
                guidance_message=cls.PEDIATRIC_GUIDANCE
            )

        # 14. Default: Normal Medical Information
        return SafetyAssessment(
            category=SafetyCategory.NORMAL_MEDICAL_INFORMATION,
            risk_level=RiskLevel.LOW.value,
            requires_escalation=False,
            allow_normal_rag=True,
            allow_medication_information=True,
            allow_dosage_information=True,
            reason="Inquiry is a safe educational or research healthcare query.",
            matched_rules=["SAFE_INFORMATIONAL_DEFAULT"],
            guidance_message=None
        )
