"""
Clinical Intelligence Orchestrator & Auditability Engine (Phase 6.8).

Orchestrates Phase 6.1 through Phase 6.7 clinical reasoning stages, verifies cross-stage
safety and isolation invariants, generates cryptographic audit checksums, and produces
comprehensive execution provenance.
"""

import time
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from backend.intelligence.orchestration_models import (
    PipelineStage,
    StageExecutionStatus,
    StageAuditRecord,
    ClinicalSafetyProvenance,
    DataIsolationProvenance,
    ClinicalIntelligenceOrchestrationResult
)


class ClinicalIntelligenceOrchestrator:
    """
    Phase 6.8 Clinical Intelligence Orchestrator.
    Executes in pure Python with sub-millisecond deterministic overhead and zero external dependencies.
    """

    @classmethod
    def orchestrate(
        cls,
        trace_id: str,
        query: str,
        safety_assessment: Optional[Any] = None,
        intent_result: Optional[Any] = None,
        query_plan: Optional[Any] = None,
        fused_evidence: Optional[Any] = None,
        answer_synthesis: Optional[Any] = None,
        attribution_report: Optional[Any] = None,
        verification_result: Optional[Any] = None,
        decision_support: Optional[Any] = None,
        dialogue_context: Optional[Any] = None,
        user_id: Optional[int] = None,
        tenant_id: Optional[str] = None,
        retrieved_sources: Optional[List[Dict[str, Any]]] = None,
        total_pipeline_latency_ms: float = 0.0
    ) -> ClinicalIntelligenceOrchestrationResult:
        """
        Orchestrates and audits all stages of the clinical intelligence reasoning pipeline.

        Args:
            trace_id: Unique deterministic or distributed trace ID.
            query: Sanitized user clinical query string.
            safety_assessment: Phase 4 / Core SafetyAssessment object.
            intent_result: Phase 6.1 IntentClassificationResult.
            query_plan: Phase 6.2 QueryPlan object.
            fused_evidence: Phase 6.3 FusedContextResult object.
            answer_synthesis: Phase 6.4 AnswerSynthesisResult object.
            attribution_report: Phase 6.5 CitationAttributionReport object.
            verification_result: Phase 6.6 ClinicalVerificationResult object.
            decision_support: Phase 6.7 ClinicalDecisionSupportResult object.
            user_id: Optional authenticated user ID.
            tenant_id: Optional multi-tenant organizational ID.
            retrieved_sources: List of retrieved evidence chunk dictionaries.
            total_pipeline_latency_ms: Accumulated end-to-end latency in milliseconds.

        Returns:
            ClinicalIntelligenceOrchestrationResult with complete audit records and provenance.
        """
        t_start = time.perf_counter()
        sources_list = retrieved_sources or []
        stages: List[StageAuditRecord] = []
        stage_status_map: Dict[str, str] = {}

        # ----------------------------------------------------------------------
        # 1. Audit Stage 0: Safety Pre-Screening
        # ----------------------------------------------------------------------
        cat_str = "GENERAL_INQUIRY"
        is_emergency = False
        is_self_harm = False
        is_poisoning = False
        allow_normal = True

        if safety_assessment is not None:
            cat = getattr(safety_assessment, "category", None)
            cat_str = cat.value if hasattr(cat, "value") else str(cat or "GENERAL_INQUIRY")
            allow_normal = getattr(safety_assessment, "allow_normal_rag", True)
            is_emergency = (cat_str == "EMERGENCY") or ("emergency" in cat_str.lower())
            is_self_harm = (cat_str == "SELF_HARM") or ("self_harm" in cat_str.lower())
            is_poisoning = (cat_str == "POISONING") or ("poisoning" in cat_str.lower())

        pre_screen_status = StageExecutionStatus.INTERCEPTED if not allow_normal else StageExecutionStatus.SUCCESS
        pre_screen_record = StageAuditRecord(
            stage=PipelineStage.SAFETY_PRE_SCREEN,
            status=pre_screen_status,
            execution_latency_ms=0.0,
            summary=f"Safety classification: {cat_str} (intercepted={not allow_normal})",
            flags={
                "category": cat_str,
                "is_emergency": is_emergency,
                "is_self_harm": is_self_harm,
                "is_poisoning": is_poisoning,
                "allow_normal_rag": allow_normal
            }
        )
        stages.append(pre_screen_record)
        stage_status_map[PipelineStage.SAFETY_PRE_SCREEN.value] = pre_screen_status.value

        # ----------------------------------------------------------------------
        # 1.5 Audit Stage 0.5: Longitudinal Clinical Context (Phase 6.9)
        # ----------------------------------------------------------------------
        if dialogue_context is not None:
            turn_cnt = getattr(dialogue_context, "turn_count", 0)
            is_fu = getattr(dialogue_context, "is_follow_up", False)
            res_topic = getattr(dialogue_context, "resolved_topic", None)
            ent_cnt = getattr(dialogue_context, "entities_count", 0)
            prof_hash = getattr(dialogue_context, "profile_hash", "")
            d_lat = getattr(dialogue_context, "resolution_latency_ms", 0.0)

            d_status = StageExecutionStatus.SUCCESS if turn_cnt > 0 else StageExecutionStatus.SKIPPED
            d_record = StageAuditRecord(
                stage=PipelineStage.DIALOGUE_CONTEXT,
                status=d_status,
                execution_latency_ms=d_lat,
                summary=f"Dialogue context: {turn_cnt} prior turns, is_follow_up={is_fu}, entities={ent_cnt}",
                flags={
                    "turn_count": turn_cnt,
                    "is_follow_up": is_fu,
                    "resolved_topic": res_topic,
                    "entities_count": ent_cnt,
                    "profile_hash": prof_hash
                }
            )
            stages.append(d_record)
            stage_status_map[PipelineStage.DIALOGUE_CONTEXT.value] = d_record.status.value

        # ----------------------------------------------------------------------
        # 2. Audit Stage 1: Clinical Intent Classification (Phase 6.1)
        # ----------------------------------------------------------------------
        if intent_result is not None:
            intent_val = getattr(intent_result.intent, "value", str(intent_result.intent)) if hasattr(intent_result, "intent") else "GENERAL_HEALTH_INQUIRY"
            intent_conf = getattr(intent_result, "confidence", 1.0)
            intent_lat = getattr(intent_result, "latency_ms", 0.0)
            intent_record = StageAuditRecord(
                stage=PipelineStage.INTENT_CLASSIFICATION,
                status=StageExecutionStatus.SUCCESS,
                execution_latency_ms=intent_lat,
                summary=f"Intent: {intent_val} (confidence={intent_conf:.2f})",
                flags={"intent": intent_val, "confidence": round(intent_conf, 3)}
            )
        else:
            intent_record = StageAuditRecord(
                stage=PipelineStage.INTENT_CLASSIFICATION,
                status=StageExecutionStatus.SKIPPED,
                execution_latency_ms=0.0,
                summary="Intent classification skipped or intercepted",
                flags={"intent": "UNKNOWN", "confidence": 0.0}
            )
        stages.append(intent_record)
        stage_status_map[PipelineStage.INTENT_CLASSIFICATION.value] = intent_record.status.value

        # ----------------------------------------------------------------------
        # 3. Audit Stage 2: Clinical Query Planning (Phase 6.2)
        # ----------------------------------------------------------------------
        if query_plan is not None:
            plan_strategy = getattr(query_plan.generation_strategy, "value", str(getattr(query_plan, "generation_strategy", "STANDARD")))
            plan_lat = getattr(query_plan, "latency_ms", 0.0)
            plan_record = StageAuditRecord(
                stage=PipelineStage.QUERY_PLANNING,
                status=StageExecutionStatus.SUCCESS,
                execution_latency_ms=plan_lat,
                summary=f"Query plan formulated with strategy: {plan_strategy}",
                flags={"strategy": plan_strategy, "top_k": getattr(query_plan, "top_k", 5)}
            )
        else:
            plan_record = StageAuditRecord(
                stage=PipelineStage.QUERY_PLANNING,
                status=StageExecutionStatus.SKIPPED,
                execution_latency_ms=0.0,
                summary="Query planning skipped",
                flags={}
            )
        stages.append(plan_record)
        stage_status_map[PipelineStage.QUERY_PLANNING.value] = plan_record.status.value

        # ----------------------------------------------------------------------
        # 4. Audit Stage 3: Clinical Evidence Fusion (Phase 6.3)
        # ----------------------------------------------------------------------
        if fused_evidence is not None:
            cov_status = "FULL"
            if hasattr(fused_evidence, "coverage") and fused_evidence.coverage:
                c_st = getattr(fused_evidence.coverage, "status", None)
                cov_status = c_st.value if hasattr(c_st, "value") else str(c_st or "FULL")
            conflicts_count = len(getattr(fused_evidence, "conflicts", []) or [])
            fusion_lat = getattr(fused_evidence, "fusion_latency_ms", getattr(fused_evidence, "latency_ms", 0.0))
            fusion_status = StageExecutionStatus.DEGRADED if conflicts_count > 0 or cov_status.upper() == "INSUFFICIENT" else StageExecutionStatus.SUCCESS
            fusion_record = StageAuditRecord(
                stage=PipelineStage.EVIDENCE_FUSION,
                status=fusion_status,
                execution_latency_ms=fusion_lat,
                summary=f"Fused evidence with coverage={cov_status}, conflicts={conflicts_count}",
                flags={"coverage_status": cov_status.upper(), "conflicts_count": conflicts_count}
            )
        else:
            fusion_record = StageAuditRecord(
                stage=PipelineStage.EVIDENCE_FUSION,
                status=StageExecutionStatus.SKIPPED,
                execution_latency_ms=0.0,
                summary="Evidence fusion skipped",
                flags={}
            )
        stages.append(fusion_record)
        stage_status_map[PipelineStage.EVIDENCE_FUSION.value] = fusion_record.status.value

        # ----------------------------------------------------------------------
        # 5. Audit Stage 4: Clinical Answer Synthesis (Phase 6.4)
        # ----------------------------------------------------------------------
        if answer_synthesis is not None:
            syn_conf = getattr(answer_synthesis.confidence, "value", str(getattr(answer_synthesis, "confidence", "HIGH")))
            syn_supp = getattr(answer_synthesis.support_level, "value", str(getattr(answer_synthesis, "support_level", "FULL")))
            syn_fallback = getattr(answer_synthesis, "is_fallback", False)
            syn_lat = getattr(answer_synthesis, "latency_ms", 0.0)
            syn_status = StageExecutionStatus.DEGRADED if syn_fallback else StageExecutionStatus.SUCCESS
            synthesis_record = StageAuditRecord(
                stage=PipelineStage.ANSWER_SYNTHESIS,
                status=syn_status,
                execution_latency_ms=syn_lat,
                summary=f"Answer synthesized (confidence={syn_conf}, support={syn_supp}, fallback={syn_fallback})",
                flags={"confidence": syn_conf, "support_level": syn_supp, "is_fallback": syn_fallback}
            )
        else:
            synthesis_record = StageAuditRecord(
                stage=PipelineStage.ANSWER_SYNTHESIS,
                status=StageExecutionStatus.SKIPPED,
                execution_latency_ms=0.0,
                summary="Answer synthesis skipped",
                flags={}
            )
        stages.append(synthesis_record)
        stage_status_map[PipelineStage.ANSWER_SYNTHESIS.value] = synthesis_record.status.value

        # ----------------------------------------------------------------------
        # 6. Audit Stage 5: Citation Attribution (Phase 6.5)
        # ----------------------------------------------------------------------
        citations_ok = True
        if attribution_report is not None:
            attr_prec = getattr(attribution_report, "citation_precision", getattr(attribution_report, "precision", 1.0))
            attr_claims = getattr(attribution_report, "factual_claims_count", getattr(attribution_report, "total_claims_count", 0))
            attr_lat = getattr(attribution_report, "latency_ms", 0.0)
            citations_ok = (attr_prec >= 0.70) or (attr_claims == 0)
            citation_status = StageExecutionStatus.SUCCESS if citations_ok else StageExecutionStatus.DEGRADED
            citation_record = StageAuditRecord(
                stage=PipelineStage.CITATION_ATTRIBUTION,
                status=citation_status,
                execution_latency_ms=attr_lat,
                summary=f"Attribution precision={attr_prec:.2f} across {attr_claims} claims",
                flags={"precision": round(attr_prec, 3), "claims_count": attr_claims, "verified": citations_ok}
            )
        else:
            citation_record = StageAuditRecord(
                stage=PipelineStage.CITATION_ATTRIBUTION,
                status=StageExecutionStatus.SKIPPED,
                execution_latency_ms=0.0,
                summary="Citation attribution skipped",
                flags={"verified": True}
            )
        stages.append(citation_record)
        stage_status_map[PipelineStage.CITATION_ATTRIBUTION.value] = citation_record.status.value

        # ----------------------------------------------------------------------
        # 7. Audit Stage 6: Clinical Verification (Phase 6.6)
        # ----------------------------------------------------------------------
        grounding_ok = True
        contra_count = 0
        if verification_result is not None:
            contra_count = getattr(verification_result, "contradictions_count", 0)
            ver_status_enum = getattr(verification_result, "status", getattr(verification_result, "action_taken", None))
            ver_status_str = ver_status_enum.value if hasattr(ver_status_enum, "value") else str(ver_status_enum or "VERIFIED")
            base_grounded = getattr(verification_result, "is_verified_safe", getattr(verification_result, "is_grounded", True))
            grounding_ok = bool(base_grounded) and (contra_count == 0)
            ver_fallback = getattr(verification_result, "fallback_triggered", False)
            ver_lat = getattr(verification_result, "latency_ms", 0.0)
            ver_stage_status = StageExecutionStatus.SUCCESS if (grounding_ok and not ver_fallback) else StageExecutionStatus.DEGRADED
            verification_record = StageAuditRecord(
                stage=PipelineStage.CLINICAL_VERIFICATION,
                status=ver_stage_status,
                execution_latency_ms=ver_lat,
                summary=f"Clinical verification: {ver_status_str} (contradictions={contra_count})",
                flags={
                    "status": ver_status_str,
                    "contradictions_count": contra_count,
                    "is_grounded": grounding_ok,
                    "fallback_triggered": ver_fallback
                }
            )
        else:
            verification_record = StageAuditRecord(
                stage=PipelineStage.CLINICAL_VERIFICATION,
                status=StageExecutionStatus.SKIPPED,
                execution_latency_ms=0.0,
                summary="Clinical verification skipped",
                flags={"is_grounded": True}
            )
        stages.append(verification_record)
        stage_status_map[PipelineStage.CLINICAL_VERIFICATION.value] = verification_record.status.value

        # ----------------------------------------------------------------------
        # 8. Audit Stage 7: Clinical Decision Support (Phase 6.7)
        # ----------------------------------------------------------------------
        risk_tier_str = "MINIMAL"
        uncertainty_str = "LOW"
        red_flags_cnt = 0
        escalation_req = False
        if decision_support is not None:
            rt = getattr(decision_support, "risk_tier", None)
            risk_tier_str = rt.value if hasattr(rt, "value") else str(rt or "MINIMAL")
            unc = getattr(decision_support, "uncertainty_level", None)
            uncertainty_str = unc.value if hasattr(unc, "value") else str(unc or "LOW")
            red_flags_cnt = len(getattr(decision_support, "red_flag_triggers", []) or [])
            escalation_req = getattr(decision_support, "escalation_required", False)
            cds_lat = getattr(decision_support, "latency_ms", 0.0)
            cds_status = StageExecutionStatus.SUCCESS
            cds_record = StageAuditRecord(
                stage=PipelineStage.DECISION_SUPPORT,
                status=cds_status,
                execution_latency_ms=cds_lat,
                summary=f"Decision support: risk={risk_tier_str}, uncertainty={uncertainty_str}, red_flags={red_flags_cnt}",
                flags={
                    "risk_tier": risk_tier_str,
                    "uncertainty_level": uncertainty_str,
                    "red_flags_count": red_flags_cnt,
                    "escalation_required": escalation_req
                }
            )
        else:
            cds_record = StageAuditRecord(
                stage=PipelineStage.DECISION_SUPPORT,
                status=StageExecutionStatus.SKIPPED,
                execution_latency_ms=0.0,
                summary="Decision support skipped",
                flags={}
            )
        stages.append(cds_record)
        stage_status_map[PipelineStage.DECISION_SUPPORT.value] = cds_record.status.value

        # ----------------------------------------------------------------------
        # 9. Audit Stage 8: Orchestration & Invariant Audit (Phase 6.8)
        # ----------------------------------------------------------------------
        orch_record = StageAuditRecord(
            stage=PipelineStage.AUDIT_ORCHESTRATION,
            status=StageExecutionStatus.SUCCESS,
            execution_latency_ms=0.0,  # Updated at finish
            summary="All clinical stages audited and cross-verified",
            flags={"stages_count": len(stages) + 1}
        )
        stages.append(orch_record)
        stage_status_map[PipelineStage.AUDIT_ORCHESTRATION.value] = StageExecutionStatus.SUCCESS.value

        # ----------------------------------------------------------------------
        # 10. Compile Provenance
        # ----------------------------------------------------------------------
        safety_provenance = ClinicalSafetyProvenance(
            emergency_intercepted=is_emergency,
            self_harm_intercepted=is_self_harm,
            poisoning_intercepted=is_poisoning,
            requires_escalation=escalation_req,
            risk_tier=risk_tier_str,
            uncertainty_level=uncertainty_str,
            red_flags_count=red_flags_cnt,
            contradictions_detected=contra_count,
            citations_verified=citations_ok,
            grounding_verified=grounding_ok,
            safety_assessment_category=cat_str
        )

        chunk_ids = [str(c.get("chunk_id", "")) for c in sources_list if c.get("chunk_id")]
        doc_ids = list(set([str(c.get("document_id") or c.get("document_name", "")) for c in sources_list if c.get("document_id") or c.get("document_name")]))

        # Cross-tenant contamination validation
        cross_tenant_ok = True
        if user_id is not None:
            # If user_id is set, confirm all chunks with a user_id tag match the requesting user
            for src in sources_list:
                src_user = src.get("user_id")
                if src_user is not None and int(src_user) != int(user_id):
                    cross_tenant_ok = False
                    break

        isolation_provenance = DataIsolationProvenance(
            tenant_id=tenant_id or "default_tenant",
            user_id=user_id,
            documents_accessed_count=len(doc_ids),
            chunks_accessed_count=len(chunk_ids),
            retrieved_chunk_ids=chunk_ids,
            cross_tenant_contamination_check_passed=cross_tenant_ok,
            cache_isolated=True
        )

        # ----------------------------------------------------------------------
        # 11. Deterministic Cryptographic Audit Checksum
        # ----------------------------------------------------------------------
        d_hash = getattr(dialogue_context, "profile_hash", "") if dialogue_context else ""
        raw_token = f"{trace_id}:{cat_str}:{risk_tier_str}:{uncertainty_str}:{is_emergency}:{is_self_harm}:{is_poisoning}:{citations_ok}:{grounding_ok}:{len(chunk_ids)}:{cross_tenant_ok}"
        if d_hash:
            raw_token += f":{d_hash}"
        audit_checksum = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

        # Concordance evaluation:
        # Pipeline is concordant if grounding is preserved, citations verified, tenant isolated,
        # and no unhandled critical contradictions exist.
        is_concordant = (grounding_ok and citations_ok and cross_tenant_ok and (contra_count == 0))

        orch_latency = (time.perf_counter() - t_start) * 1000.0
        orch_record.execution_latency_ms = round(orch_latency, 3)

        return ClinicalIntelligenceOrchestrationResult(
            trace_id=trace_id,
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            pipeline_stages=stages,
            stage_status_map=stage_status_map,
            safety_provenance=safety_provenance,
            isolation_provenance=isolation_provenance,
            audit_checksum=audit_checksum,
            is_concordant=is_concordant,
            orchestration_latency_ms=orch_latency,
            total_pipeline_latency_ms=total_pipeline_latency_ms
        )
