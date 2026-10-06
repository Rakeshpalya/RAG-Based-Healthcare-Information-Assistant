"""
Clinical Decision Support, Uncertainty Calibration & Care Pathways Engine (Phase 6.7).

Provides deterministic, sub-millisecond post-synthesis decision support:
1. Composite Clinical Uncertainty Calibration (fusing intent confidence, retrieval similarity,
   evidence coverage, conflicts, citation precision, and grounding verification).
2. Clinical Risk Stratification (Minimal, Low, Moderate, High, Critical).
3. Guideline-Concordant Actionable Recommendations (non-prescriptive clinical action items).
4. Clinical "Red Flag" Symptoms & Escalation Triggers.
5. Context-Aware Agentic Follow-Up Exploration Pathways.
6. Provider SBAR (Situation, Background, Assessment, Recommendation) Clinical Handoff Summary.
7. Clinical Deferral & Escalation Triggering.
"""

import re
import time
from typing import List, Dict, Any, Optional

from backend.intelligence.decision_support_models import (
    ClinicalUncertaintyLevel,
    ClinicalRiskTier,
    UncertaintySourceType,
    ActionRecommendationType,
    ActionableRecommendation,
    RedFlagTrigger,
    ClinicalHandoffSummary,
    ClinicalDecisionSupportResult
)


class ClinicalDecisionSupportEngine:
    """
    Phase 6.7 Deterministic Clinical Decision Support & Uncertainty Calibration Engine.
    Executes entirely in pure Python with sub-millisecond overhead and zero external API dependencies.
    """

    # ---------------------------------------------------------------------------
    # Red Flag Reference Definitions
    # ---------------------------------------------------------------------------
    CARDIOVASCULAR_RED_FLAGS = [
        RedFlagTrigger(
            flag_id="RF_HTN_CRISIS",
            symptom_or_sign="Systolic BP > 180 mmHg or Diastolic BP > 120 mmHg",
            clinical_rationale="Signs of hypertensive crisis/urgency requiring prompt emergency evaluation.",
            action_required="Seek immediate emergency department evaluation if accompanied by headache, chest pain, or vision changes.",
            is_critical=True
        ),
        RedFlagTrigger(
            flag_id="RF_ACS",
            symptom_or_sign="Acute chest pain, substernal pressure, or pain radiating to jaw/left arm",
            clinical_rationale="Potential acute coronary syndrome (ACS) or myocardial infarction.",
            action_required="Call 911 immediately; do not drive yourself to the emergency department.",
            is_critical=True
        ),
        RedFlagTrigger(
            flag_id="RF_STROKE",
            symptom_or_sign="Sudden unilateral facial droop, arm weakness, or slurred speech (FAST signs)",
            clinical_rationale="Signs of acute ischemic stroke or intracranial hemorrhage.",
            action_required="Activate emergency medical services (911) immediately for time-sensitive thrombolysis/thrombectomy.",
            is_critical=True
        )
    ]

    METABOLIC_RED_FLAGS = [
        RedFlagTrigger(
            flag_id="RF_HYPO",
            symptom_or_sign="Severe hypoglycemia (confusion, diaphoresis, syncope, tremors)",
            clinical_rationale="Risk of neuroglycopenia, seizures, or loss of consciousness.",
            action_required="Administer fast-acting oral glucose (15-20g) if conscious, or seek emergency assistance.",
            is_critical=True
        ),
        RedFlagTrigger(
            flag_id="RF_DKA",
            symptom_or_sign="Fruity breath odor, persistent vomiting, Kussmaul respirations, altered mental status",
            clinical_rationale="Suspected diabetic ketoacidosis (DKA) or hyperosmolar hyperglycemic state (HHS).",
            action_required="Proceed immediately to nearest emergency facility for IV rehydration and insulin protocol.",
            is_critical=True
        )
    ]

    PHARMACOLOGICAL_RED_FLAGS = [
        RedFlagTrigger(
            flag_id="RF_ANAPHYLAXIS",
            symptom_or_sign="Acute urticaria accompanied by angioedema, stridor, wheezing, or hypotension",
            clinical_rationale="Severe systemic anaphylactic reaction to medication or allergen.",
            action_required="Administer intramuscular epinephrine auto-injector if available and call 911.",
            is_critical=True
        ),
        RedFlagTrigger(
            flag_id="RF_SCAR",
            symptom_or_sign="Rapidly progressive rash with mucosal blistering, skin peeling, or fever",
            clinical_rationale="Possible severe cutaneous adverse reaction (SJS/TEN or DRESS syndrome).",
            action_required="Discontinue suspected medication immediately and seek emergent specialist inpatient care.",
            is_critical=True
        ),
        RedFlagTrigger(
            flag_id="RF_RHABDO",
            symptom_or_sign="Unexplained dark tea-colored urine with profound muscle weakness and pain",
            clinical_rationale="Suspected rhabdomyolysis and risk of acute kidney injury.",
            action_required="Seek prompt emergency evaluation for serum creatine kinase (CK) and renal function testing.",
            is_critical=False
        )
    ]

    # ---------------------------------------------------------------------------
    # Main Decision Support Entrypoint
    # ---------------------------------------------------------------------------
    @classmethod
    def evaluate(
        cls,
        query: str,
        intent: Optional[str] = None,
        answer_text: Optional[str] = None,
        retrieved_sources: Optional[List[Dict[str, Any]]] = None,
        fused_evidence: Optional[Any] = None,
        attribution_report: Optional[Any] = None,
        verification_result: Optional[Any] = None,
        query_plan: Optional[Any] = None,
        safety_assessment: Optional[Any] = None,
        cumulative_profile: Optional[Any] = None
    ) -> ClinicalDecisionSupportResult:
        """
        Main Phase 6.7 decision support evaluation method.
        """
        t_start = time.perf_counter()

        norm_query = (query or "").strip()
        norm_answer = (answer_text or "").strip()
        norm_intent = (intent or "GENERAL_HEALTH").strip().upper()
        sources = retrieved_sources or []

        # 1. Uncertainty Calibration
        calibrated_conf, uncertainty_lvl, uncertainty_src = cls._calibrate_uncertainty(
            intent=norm_intent,
            sources=sources,
            fused_evidence=fused_evidence,
            attribution_report=attribution_report,
            verification_result=verification_result,
            query_plan=query_plan
        )

        # 2. Clinical Risk Stratification
        risk_tier = cls._stratify_risk(
            intent=norm_intent,
            query=norm_query,
            answer=norm_answer,
            safety_assessment=safety_assessment,
            verification_result=verification_result,
            fused_evidence=fused_evidence,
            cumulative_profile=cumulative_profile
        )

        # 3. Actionable Clinical Recommendations
        recommendations = cls._generate_actionable_recommendations(
            intent=norm_intent,
            query=norm_query,
            answer=norm_answer,
            sources=sources,
            risk_tier=risk_tier
        )

        # 4. Red Flag Trigger Analysis
        red_flags, red_flag_escalation = cls._identify_red_flags(
            query=norm_query,
            answer=norm_answer,
            intent=norm_intent,
            cumulative_profile=cumulative_profile
        )

        # 5. Escalation Decision
        escalation_required = False
        escalation_reasons: List[str] = []

        if risk_tier == ClinicalRiskTier.CRITICAL:
            escalation_required = True
            escalation_reasons.append("Critical clinical risk tier identified.")

        if red_flag_escalation:
            escalation_required = True
            escalation_reasons.append("Clinical red-flag warning criteria detected.")

        if verification_result and getattr(verification_result, "fallback_triggered", False):
            escalation_required = True
            escalation_reasons.append("Clinical verification triggered conservative fallback.")

        if calibrated_conf < 0.35 and norm_intent in ("DOSAGE_QUERY", "MEDICATION_QUERY", "DIAGNOSIS_QUERY"):
            escalation_required = True
            escalation_reasons.append("Calibrated confidence below safe clinical guidance threshold.")

        primary_escalation_reason = "; ".join(escalation_reasons) if escalation_reasons else None

        # 6. Suggested Agentic Follow-Up Inquiries
        follow_ups = cls._generate_follow_up_inquiries(
            intent=norm_intent,
            query=norm_query,
            answer=norm_answer
        )

        # 7. Provider SBAR Clinical Handoff Summary
        handoff = cls._generate_sbar_handoff(
            intent=norm_intent,
            query=norm_query,
            answer=norm_answer,
            sources=sources,
            calibrated_conf=calibrated_conf,
            risk_tier=risk_tier,
            uncertainty_lvl=uncertainty_lvl,
            fused_evidence=fused_evidence,
            escalation_required=escalation_required,
            escalation_reason=primary_escalation_reason
        )

        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        return ClinicalDecisionSupportResult(
            calibrated_confidence=calibrated_conf,
            uncertainty_level=uncertainty_lvl,
            primary_uncertainty_source=uncertainty_src,
            risk_tier=risk_tier,
            escalation_required=escalation_required,
            escalation_reason=primary_escalation_reason,
            actionable_recommendations=recommendations,
            red_flag_triggers=red_flags,
            suggested_follow_up_inquiries=follow_ups,
            clinical_handoff=handoff,
            latency_ms=latency_ms,
            metadata={
                "intent": norm_intent,
                "sources_evaluated": len(sources),
                "red_flags_count": len(red_flags),
                "recommendations_count": len(recommendations)
            }
        )

    # ---------------------------------------------------------------------------
    # 1. Uncertainty Calibration Internal Logic
    # ---------------------------------------------------------------------------
    @classmethod
    def _calibrate_uncertainty(
        cls,
        intent: str,
        sources: List[Dict[str, Any]],
        fused_evidence: Optional[Any],
        attribution_report: Optional[Any],
        verification_result: Optional[Any],
        query_plan: Optional[Any]
    ) -> tuple[float, ClinicalUncertaintyLevel, UncertaintySourceType]:
        """
        Computes composite calibrated confidence and categorizes clinical uncertainty.
        """
        # If no sources or fallback triggered in verification, confidence is zero
        if not sources or (verification_result and getattr(verification_result, "fallback_triggered", False)):
            return 0.0, ClinicalUncertaintyLevel.INDETERMINATE, UncertaintySourceType.EPISTEMIC

        # Component 1: Similarity strength (max cosine similarity)
        sim_scores = [float(s.get("similarity_score", 0.0)) for s in sources if s.get("similarity_score") is not None]
        max_sim = max(sim_scores) if sim_scores else 0.50

        # Component 2: Intent classification baseline confidence
        intent_conf = 0.85
        if query_plan and hasattr(query_plan, "target_confidence") and query_plan.target_confidence is not None:
            intent_conf = float(query_plan.target_confidence)

        # Component 3: Evidence coverage factor
        coverage_factor = 0.80
        if fused_evidence and hasattr(fused_evidence, "coverage") and fused_evidence.coverage:
            cov_status = getattr(fused_evidence.coverage, "status", None)
            cov_val = cov_status.value if hasattr(cov_status, "value") else str(cov_status).upper()
            if "FULL" in cov_val:
                coverage_factor = 1.0
            elif "PARTIAL" in cov_val:
                coverage_factor = 0.65
            else:
                coverage_factor = 0.30

        # Component 4: Citation attribution precision
        attr_prec = 0.85
        if attribution_report and hasattr(attribution_report, "citation_precision"):
            attr_prec = float(attribution_report.citation_precision)

        # Component 5: Grounding verification score
        grounding_score = 0.85
        if verification_result:
            if hasattr(verification_result, "overall_grounding_score"):
                grounding_score = float(verification_result.overall_grounding_score)
            elif hasattr(verification_result, "grounding_score"):
                grounding_score = float(verification_result.grounding_score)

        # Component 6: Conflict penalty
        has_conflicts = False
        if fused_evidence and getattr(fused_evidence, "has_conflicts", False):
            has_conflicts = True

        # Multi-factor weighted composite calculation:
        # 15% Intent + 20% Retrieval Sim + 20% Coverage + 20% Citation Prec + 25% Grounding
        raw_composite = (
            0.15 * intent_conf +
            0.20 * max_sim +
            0.20 * coverage_factor +
            0.20 * attr_prec +
            0.25 * grounding_score
        )

        if has_conflicts:
            raw_composite -= 0.22  # Significant penalty for contradictory guidelines

        calibrated = max(0.0, min(1.0, raw_composite))

        # Categorize Uncertainty Level
        if calibrated >= 0.80:
            level = ClinicalUncertaintyLevel.LOW
        elif calibrated >= 0.60:
            level = ClinicalUncertaintyLevel.MODERATE
        elif calibrated >= 0.30:
            level = ClinicalUncertaintyLevel.HIGH
        else:
            level = ClinicalUncertaintyLevel.INDETERMINATE

        # Identify Primary Source of Uncertainty
        has_unsupported = False
        if verification_result:
            if getattr(verification_result, "ungrounded_claims_count", 0) > 0 or getattr(verification_result, "hallucinations_detected", 0) > 0:
                has_unsupported = True
            elif getattr(verification_result, "hallucinatory_claims_count", 0) > 0:
                has_unsupported = True
        if attribution_report and getattr(attribution_report, "unsupported_claims_count", 0) > 0:
            has_unsupported = True

        if has_conflicts:
            source_type = UncertaintySourceType.ALEATORIC
        elif max_sim < 0.35 or coverage_factor < 0.50:
            source_type = UncertaintySourceType.EPISTEMIC
        elif has_unsupported:
            source_type = UncertaintySourceType.VERIFICATION_DEFICIT
        else:
            source_type = UncertaintySourceType.NONE

        return round(calibrated, 4), level, source_type

    # ---------------------------------------------------------------------------
    # 2. Clinical Risk Stratification
    # ---------------------------------------------------------------------------
    @classmethod
    def _stratify_risk(
        cls,
        intent: str,
        query: str,
        answer: str,
        safety_assessment: Optional[Any] = None,
        verification_result: Optional[Any] = None,
        fused_evidence: Optional[Any] = None,
        cumulative_profile: Optional[Any] = None
    ) -> ClinicalRiskTier:
        """
        Determines the clinical risk tier based on intent, safety signals, and verification outcomes.
        """
        combined_text = (query + " " + answer).lower()

        # Immediate Critical Precedence: Emergency, Self-harm, Poisoning
        if intent in ("EMERGENCY", "SELF_HARM", "POISONING"):
            return ClinicalRiskTier.CRITICAL

        if safety_assessment:
            cat = str(getattr(safety_assessment, "category", "")).lower()
            if any(k in cat for k in ("emergency", "self_harm", "crisis", "suicide", "poisoning", "overdose")):
                return ClinicalRiskTier.CRITICAL
            risk_lvl = str(getattr(safety_assessment, "risk_level", "")).lower()
            if "critical" in risk_lvl:
                return ClinicalRiskTier.CRITICAL

        # Critical Contradictions from Phase 6.6
        if verification_result:
            if getattr(verification_result, "contradictions_count", 0) > 0:
                return ClinicalRiskTier.CRITICAL
            claims = getattr(verification_result, "claim_verifications", []) or getattr(verification_result, "claims", [])
            for c in claims:
                h_type = getattr(c, "hallucination_type", None)
                h_name = h_type.value if hasattr(h_type, "value") else str(h_type)
                if h_name in ("DIRECTIONAL_INVERSION", "NEGATION_CONFLICT", "DIRECTIONAL_CONTRADICTION"):
                    return ClinicalRiskTier.CRITICAL
            hallucinations = getattr(verification_result, "detected_hallucinations", [])
            h_names = [h.value if hasattr(h, "value") else str(h) for h in hallucinations]
            if "DIRECTIONAL_CONTRADICTION" in h_names or "NEGATION_CONFLICT" in h_names or "DIRECTIONAL_INVERSION" in h_names:
                return ClinicalRiskTier.CRITICAL


        # High Risk: Medication dosage, prescription therapy, drug interactions, or high-risk comorbidity
        if intent in ("DOSAGE_QUERY", "MEDICATION_QUERY"):
            return ClinicalRiskTier.HIGH

        if any(w in combined_text for w in ("prescribe", "dosage", "contraindication", "black box", "adverse reaction", "drug interaction")):
            return ClinicalRiskTier.HIGH

        # Cumulative comorbidity risk elevation (Phase 6.9)
        if cumulative_profile:
            conds = getattr(cumulative_profile, "active_conditions", [])
            conds_lower = {str(c).lower() for c in conds}
            risks = getattr(cumulative_profile, "risk_factors", [])
            high_risk_conds = {
                "chronic kidney disease", "ckd", "renal failure", "renal impairment",
                "heart failure", "cirrhosis", "liver failure", "coronary artery disease", "cad"
            }
            if bool(conds_lower & high_risk_conds) or "Pregnancy" in risks or len(conds) >= 2:
                return ClinicalRiskTier.HIGH

        # Moderate Risk: Symptom queries, diagnostic stages, lab result evaluations
        if intent in ("DIAGNOSIS_QUERY", "TREATMENT_QUERY", "LAB_RESULT_QUERY", "SYMPTOM_QUERY", "SYMPTOM_ASSESSMENT"):
            return ClinicalRiskTier.MODERATE

        if any(w in combined_text for w in ("symptom", "diagnosis", "stage", "hypertension", "diabetes", "lab", "creatinine", "glucose")):
            return ClinicalRiskTier.MODERATE

        if cumulative_profile:
            conds = getattr(cumulative_profile, "active_conditions", [])
            if len(conds) >= 1:
                return ClinicalRiskTier.MODERATE

        # Low Risk: Document comparisons, prevention, lifestyle
        if intent in ("PREVENTION_QUERY", "DOCUMENT_COMPARISON"):
            return ClinicalRiskTier.LOW

        # Minimal Risk: General educational summaries
        return ClinicalRiskTier.MINIMAL

    # ---------------------------------------------------------------------------
    # 3. Actionable Clinical Recommendations
    # ---------------------------------------------------------------------------
    @classmethod
    def _generate_actionable_recommendations(
        cls,
        intent: str,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        risk_tier: ClinicalRiskTier
    ) -> List[ActionableRecommendation]:
        """
        Extracts guideline-concordant, non-prescriptive actionable recommendations.
        """
        recommendations: List[ActionableRecommendation] = []
        combined_lower = (query + " " + answer).lower()
        src_doc = sources[0].get("document_name", "Clinical Guideline") if sources else None

        # A. Cardiovascular / Blood Pressure Specific Care Pathway
        if any(w in combined_lower for w in ("hypertens", "blood pressure", "systolic", "diastolic", "amlodipine", "lisinopril")):
            recommendations.append(ActionableRecommendation(
                recommendation_id="REC_HTN_BP_LOG",
                category=ActionRecommendationType.DIAGNOSTIC_MONITORING,
                text="Maintain a structured daily home blood pressure log (recording morning and evening measurements) for physician review.",
                urgency="ROUTINE",
                evidence_source=src_doc
            ))
            recommendations.append(ActionableRecommendation(
                recommendation_id="REC_HTN_LIFESTYLE",
                category=ActionRecommendationType.LIFESTYLE_MODIFICATION,
                text="Implement sodium restriction (< 2,300 mg/day) and DASH dietary patterns consistent with established clinical guidelines.",
                urgency="ROUTINE",
                evidence_source=src_doc
            ))

        # B. Medication & Dosage Management Care Pathway
        if intent in ("DOSAGE_QUERY", "MEDICATION_QUERY") or any(w in combined_lower for w in ("dose", "dosage", "mg", "medication", "drug")):
            recommendations.append(ActionableRecommendation(
                recommendation_id="REC_MED_REVIEW",
                category=ActionRecommendationType.MEDICATION_REVIEW,
                text="Consult with the prescribing clinician or pharmacist before modifying medication dosage, schedule, or formulation.",
                urgency="PROMPT" if risk_tier in (ClinicalRiskTier.HIGH, ClinicalRiskTier.CRITICAL) else "ROUTINE",
                evidence_source=src_doc
            ))
            recommendations.append(ActionableRecommendation(
                recommendation_id="REC_LAB_MONITORING",
                category=ActionRecommendationType.DIAGNOSTIC_MONITORING,
                text="Ensure routine baseline and periodic laboratory monitoring (e.g., serum electrolytes, renal function) as clinically indicated.",
                urgency="ROUTINE",
                evidence_source=src_doc
            ))

        # C. Symptom & Diagnostic Evaluation Care Pathway
        if intent in ("SYMPTOM_QUERY", "DIAGNOSIS_QUERY"):
            recommendations.append(ActionableRecommendation(
                recommendation_id="REC_CLINICIAN_VISIT",
                category=ActionRecommendationType.CLINICIAN_CONSULTATION,
                text="Schedule an in-person clinical assessment to obtain a formal physical examination and definitive diagnosis.",
                urgency="PROMPT" if risk_tier in (ClinicalRiskTier.HIGH, ClinicalRiskTier.CRITICAL) else "ROUTINE",
                evidence_source=src_doc
            ))

        # D. General Preventive Fallback
        if not recommendations:
            recommendations.append(ActionableRecommendation(
                recommendation_id="REC_PREVENTIVE_CONSULT",
                category=ActionRecommendationType.PREVENTIVE_MEASURE,
                text="Discuss preventive screening and personalized health maintenance goals with your primary care provider.",
                urgency="ROUTINE",
                evidence_source=src_doc
            ))

        return recommendations

    # ---------------------------------------------------------------------------
    # 4. Red Flag Trigger Analysis
    # ---------------------------------------------------------------------------
    @classmethod
    def _identify_red_flags(
        cls,
        query: str,
        answer: str,
        intent: str,
        cumulative_profile: Optional[Any] = None
    ) -> tuple[List[RedFlagTrigger], bool]:
        """
        Scans inquiry and clinical answer for red-flag warning criteria.
        """
        combined = (query + " " + answer).lower()
        matched_flags: List[RedFlagTrigger] = []
        has_critical = False

        # Check Cardiovascular Red Flags
        if any(w in combined for w in ("hypertens", "blood pressure", "bp", "180", "120", "chest pain", "angina", "stroke", "droop")):
            for rf in cls.CARDIOVASCULAR_RED_FLAGS:
                if rf.flag_id == "RF_HTN_CRISIS" and any(k in combined for k in ("180", "120", "crisis", "urgency")):
                    matched_flags.append(rf)
                    has_critical = True
                elif rf.flag_id == "RF_ACS" and any(k in combined for k in ("chest pain", "substernal", "radiat", "heart attack", "myocardial")):
                    matched_flags.append(rf)
                    has_critical = True
                elif rf.flag_id == "RF_STROKE" and any(k in combined for k in ("facial droop", "slurred speech", "stroke", "arm weakness", "hemiparesis")):
                    matched_flags.append(rf)
                    has_critical = True

        # Check Metabolic Red Flags
        if any(w in combined for w in ("glucose", "diabetes", "hypoglycemia", "ketoacidosis", "dka")):
            for rf in cls.METABOLIC_RED_FLAGS:
                if rf.flag_id == "RF_HYPO" and any(k in combined for k in ("hypoglycem", "low sugar", "confusion", "tremor", "diaphoresis")):
                    matched_flags.append(rf)
                    has_critical = True
                elif rf.flag_id == "RF_DKA" and any(k in combined for k in ("ketoacidosis", "dka", "fruity breath", "vomiting")):
                    matched_flags.append(rf)
                    has_critical = True

        # Check Pharmacological Red Flags
        if any(w in combined for w in ("anaphylaxis", "allergic", "rash", "blister", "peeling", "stevens-johnson", "sjs", "rhabdo", "dark urine")):
            for rf in cls.PHARMACOLOGICAL_RED_FLAGS:
                if rf.flag_id == "RF_ANAPHYLAXIS" and any(k in combined for k in ("anaphylaxis", "angioedema", "stridor", "wheez")):
                    matched_flags.append(rf)
                    has_critical = True
                elif rf.flag_id == "RF_SCAR" and any(k in combined for k in ("rash", "blister", "peeling", "sjs", "mucosal")):
                    matched_flags.append(rf)
                    has_critical = True
                elif rf.flag_id == "RF_RHABDO" and any(k in combined for k in ("rhabdo", "dark urine", "tea-colored", "muscle pain")):
                    matched_flags.append(rf)

        # Check Cross-Turn Comorbidity Red Flags (Phase 6.9)
        if cumulative_profile:
            conds = getattr(cumulative_profile, "active_conditions", [])
            conds_lower = {str(c).lower() for c in conds}
            has_cardio = bool(conds_lower & {"coronary artery disease", "heart failure", "cad", "myocardial infarction", "angina"})
            has_renal = bool(conds_lower & {"chronic kidney disease", "ckd", "renal failure", "renal impairment"})
            has_hepatic = bool(conds_lower & {"cirrhosis", "liver failure", "hepatic impairment"})
            if has_cardio or has_renal or has_hepatic or len(conds) >= 2:
                matched_flags.append(RedFlagTrigger(
                    flag_id="RF_COMORBIDITY",
                    symptom_or_sign="Cumulative comorbidity: " + ", ".join(conds),
                    clinical_rationale=f"Cumulative comorbidity identified: Patient with {', '.join(conds)} presenting with clinical symptoms.",
                    action_required="Expedited physician evaluation and closer clinical monitoring recommended.",
                    is_critical=False
                ))

        # De-duplicate flags by flag_id
        unique_flags = {f.flag_id: f for f in matched_flags}
        final_flags = list(unique_flags.values())

        return final_flags, has_critical

    # ---------------------------------------------------------------------------
    # 5. Agentic Follow-Up Exploration Pathways
    # ---------------------------------------------------------------------------
    @classmethod
    def _generate_follow_up_inquiries(
        cls,
        intent: str,
        query: str,
        answer: str
    ) -> List[str]:
        """
        Generates 2 to 4 contextual follow-up inquiries to assist clinician/patient workflow.
        """
        combined = (query + " " + answer).lower()

        if "hypertens" in combined or "blood pressure" in combined:
            return [
                "What diagnostic tests are recommended to evaluate secondary causes of hypertension?",
                "What are the target blood pressure thresholds for patients with diabetes or CKD?",
                "What non-pharmacological lifestyle interventions have the greatest impact on systolic blood pressure?"
            ]

        if intent in ("DOSAGE_QUERY", "MEDICATION_QUERY") or "dose" in combined or "drug" in combined:
            return [
                "What are the most common drug-drug interactions associated with this medication?",
                "What baseline laboratory tests should be checked before starting therapy?",
                "What adverse symptoms indicate a need to stop this medication immediately?"
            ]

        if intent in ("DIAGNOSIS_QUERY", "SYMPTOM_QUERY") or "stage" in combined:
            return [
                "What clinical criteria differentiate between mild, moderate, and severe stages?",
                "What differential diagnoses should be ruled out for these symptoms?",
                "When is referral to a subspecialist recommended?"
            ]

        # General Clinical Follow-ups
        return [
            "What preventive measures are recommended by clinical guidelines for this condition?",
            "What questions should I ask my healthcare provider during my next appointment?",
            "What monitoring parameters are most useful for tracking disease progression?"
        ]

    # ---------------------------------------------------------------------------
    # 6. SBAR Clinical Handoff Summary
    # ---------------------------------------------------------------------------
    @classmethod
    def _generate_sbar_handoff(
        cls,
        intent: str,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        calibrated_conf: float,
        risk_tier: ClinicalRiskTier,
        uncertainty_lvl: ClinicalUncertaintyLevel,
        fused_evidence: Optional[Any],
        escalation_required: bool,
        escalation_reason: Optional[str]
    ) -> ClinicalHandoffSummary:
        """
        Constructs a structured SBAR (Situation, Background, Assessment, Recommendation) provider handoff.
        """
        # S: Situation
        situation = f"Patient/user submitted a {intent.replace('_', ' ').title()} regarding: '{query[:80]}...'."

        # B: Background
        doc_names = list(dict.fromkeys(s.get("document_name", "Clinical Reference") for s in sources if s.get("document_name")))
        doc_summary = ", ".join(doc_names[:3]) if doc_names else "Available clinical guidelines"
        background = f"Queried authoritative clinical documentation ({len(sources)} source passages from {doc_summary})."

        # A: Assessment
        conf_pct = int(calibrated_conf * 100)
        has_conflicts = bool(fused_evidence and getattr(fused_evidence, "has_conflicts", False))
        conflict_note = " Guideline discrepancies detected." if has_conflicts else ""
        assessment = (
            f"Evidence-grounded synthesis generated with {conf_pct}% calibrated confidence, "
            f"{uncertainty_lvl.value} uncertainty, and {risk_tier.value} clinical risk tier.{conflict_note}"
        )

        # R: Recommendation
        if escalation_required:
            recommendation = (
                f"Escalation required ({escalation_reason or 'Risk criteria met'}). "
                "Recommend direct clinician consultation and formal medical evaluation."
            )
        else:
            recommendation = (
                "Guideline-concordant educational summary provided. "
                "Recommend regular monitoring and routine clinical follow-up as scheduled."
            )

        return ClinicalHandoffSummary(
            situation=situation,
            background=background,
            assessment=assessment,
            recommendation=recommendation
        )
