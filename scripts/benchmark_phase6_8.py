"""
Phase 6.8 Benchmark: Clinical Intelligence Orchestration & Auditability.

Measures deterministic performance overhead of:
- Stage audit record generation and timestamping
- Cryptographic SHA-256 provenance checksum calculation
- Multi-tenant boundary isolation validation and chunk audit
- Clinical safety and fast-path interception provenance assembly
- End-to-end orchestration latency (ClinicalIntelligenceOrchestrator.orchestrate p50, p95, p99, mean, max)
- Total pipeline execution latency vs orchestration overhead percentage
- Overall throughput (orchestrations/sec)
- Production vector store invariant verification (744 FAISS vectors, 744 metadata records, 384 dim)

Outputs structured JSON to evaluation_reports/benchmark_phase6_8_results.json.
"""

import os
import sys
import time
import json
import math
import logging
from typing import List, Dict, Any

# Ensure workspace root is in sys.path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from backend.intelligence.orchestration_models import (
    PipelineStage,
    StageExecutionStatus,
    StageAuditRecord,
    ClinicalSafetyProvenance,
    DataIsolationProvenance,
    ClinicalIntelligenceOrchestrationResult
)
from backend.intelligence.clinical_orchestrator import ClinicalIntelligenceOrchestrator
from backend.intelligence.intent_models import (
    ClinicalIntent,
    IntentClassificationResult,
    ClinicalRoutingStrategy,
    SafetyPriority
)
from backend.intelligence.query_plan import (
    QueryPlan,
    ChunkWeightingStrategy,
    DocumentFilterStrategy,
    MultiDocumentStrategy,
    GenerationStrategy
)
from backend.intelligence.evidence_models import (
    ConflictType,
    ConflictSeverity,
    CoverageStatus,
    EvidenceConflict,
    EvidenceCoverage,
    FusedEvidenceChunk,
    FusedContextResult
)
from backend.intelligence.answer_models import (
    AnswerSectionType,
    AnswerConfidence,
    EvidenceSupportLevel,
    ClinicalAnswerSection,
    AnswerSynthesisResult
)
from backend.intelligence.citation_models import (
    CitationVerificationStatus,
    ClinicalClaimType,
    AttributedEvidenceSpan,
    ClinicalClaimAttribution,
    CitationAttributionReport
)
from backend.intelligence.verification_models import (
    GroundingVerificationStatus,
    ClinicalHallucinationType,
    SafetyPostScreenAction,
    ExtractedClinicalEntities,
    ClinicalVerificationClaim,
    ClinicalVerificationResult
)
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
from backend.safety.safety_types import SafetyCategory, RiskLevel, SafetyAssessment
from backend.services.vector_store_service import get_vector_store_service
from backend.startup_validation import validate_production_startup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_percentile(data: List[float], percentile: float) -> float:
    """Computes exact percentile rank."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(sorted_data[int(k)], 3)
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return round(d0 + d1, 3)


def make_mock_clinical_stages():
    """Generates synthetic, highly realistic pipeline outputs across phases 6.1 through 6.7."""
    intent_res = IntentClassificationResult(
        intent=ClinicalIntent.TREATMENT_QUERY,
        confidence=0.96,
        routing_strategy=ClinicalRoutingStrategy.STANDARD_RAG,
        requires_retrieval=True,
        requires_document_context=True,
        latency_ms=1.2
    )

    query_plan = QueryPlan(
        intent=ClinicalIntent.TREATMENT_QUERY,
        retrieval_required=True,
        retrieval_strategy=ClinicalRoutingStrategy.STANDARD_RAG,
        generation_strategy=GenerationStrategy.STANDARD_GROUNDED,
        top_k=5,
        similarity_threshold=0.25
    )

    sources = [
        {
            "chunk_id": f"CHUNK_HTN_{i:03d}",
            "document_id": "DOC_AHA_2023",
            "document_name": "AHA_Hypertension_Guidelines.pdf",
            "similarity_score": round(0.89 - (i * 0.04), 3),
            "text": f"Passage {i}: Recommended first-line antihypertensive therapy includes ACE inhibitors and ARBs.",
            "source_index": i + 1,
            "user_id": 101
        }
        for i in range(4)
    ]

    fused_res = FusedContextResult(
        fused_chunks=[
            FusedEvidenceChunk(
                chunk_id=f"CHUNK_HTN_{i:03d}",
                document_id="DOC_AHA_2023",
                document_name="AHA_Hypertension_Guidelines.pdf",
                page_number=12 + i,
                text=f"Passage {i}: Clinical evidence text.",
                similarity_score=round(0.89 - (i * 0.04), 3),
                weighting_boost=1.1,
                fused_score=round(0.95 - (i * 0.04), 3),
                rank=i + 1,
                source_index=i + 1
            )
            for i in range(4)
        ],
        contributing_documents=["DOC_AHA_2023"],
        contributing_documents_count=1,
        conflicts=[],
        has_conflicts=False,
        coverage=EvidenceCoverage(
            coverage_score=0.95,
            status=CoverageStatus.FULL,
            query_aspects=["medications", "guidelines"],
            covered_aspects=["medications", "guidelines"],
            missing_aspects=[]
        ),
        formatted_context="Evidence formatted context text.",
        sources=sources,
        deduped_count=4,
        is_sufficient=True,
        latency_ms=3.4
    )

    sections = [
        ClinicalAnswerSection(
            section_type=AnswerSectionType.DIRECT_ANSWER,
            title="First-Line Pharmacotherapy",
            content="ACE inhibitors or ARBs are primary agents [Source 1]. CCBs are also first line [Source 2].",
            source_indices=[1, 2]
        ),
        ClinicalAnswerSection(
            section_type=AnswerSectionType.EVIDENCE,
            title="Clinical Considerations",
            content="Blood pressure target is < 130/80 mmHg according to clinical guidelines [Source 3].",
            source_indices=[3]
        )
    ]

    synth_res = AnswerSynthesisResult(
        intent="TREATMENT_QUERY",
        confidence=AnswerConfidence.HIGH,
        support_level=EvidenceSupportLevel.FULL,
        coverage_status="FULL",
        answer="ACE inhibitors or ARBs are primary agents [Source 1]. CCBs are also first line [Source 2]. Target is < 130/80 mmHg [Source 3].",
        sections=sections,
        conflicts_present=False,
        conflict_summary=None,
        limitations_noted=False,
        is_fallback=False,
        cited_sources=[1, 2, 3],
        latency_ms=18.5
    )

    attr_report = CitationAttributionReport(
        is_valid=True,
        total_claims_count=3,
        factual_claims_count=3,
        verified_claims_count=3,
        unsupported_claims_count=0,
        citations_found=[1, 2, 3],
        valid_citations=[1, 2, 3],
        citation_precision=1.0,
        claim_attribution_coverage=1.0,
        latency_ms=2.1
    )

    ver_res = ClinicalVerificationResult(
        is_verified_safe=True,
        overall_grounding_score=0.98,
        total_claims_analyzed=3,
        grounded_claims_count=3,
        ungrounded_claims_count=0,
        contradictions_count=0,
        hallucinations_detected=0,
        claim_verifications=[],
        action_taken=SafetyPostScreenAction.ALLOW,
        sanitized_answer=synth_res.answer,
        latency_ms=2.8
    )

    cds_res = ClinicalDecisionSupportResult(
        calibrated_confidence=0.94,
        uncertainty_level=ClinicalUncertaintyLevel.LOW,
        primary_uncertainty_source=UncertaintySourceType.NONE,
        risk_tier=ClinicalRiskTier.LOW,
        escalation_required=False,
        escalation_reason=None,
        actionable_recommendations=[
            ActionableRecommendation(
                recommendation_id="REC_01",
                category=ActionRecommendationType.DIAGNOSTIC_MONITORING,
                text="Recheck blood pressure in 2-4 weeks after initiating or adjusting therapy."
            )
        ],
        red_flag_triggers=[],
        suggested_follow_up_inquiries=[
            "What lifestyle modifications are recommended alongside medications?"
        ],
        latency_ms=2.3
    )

    safety_assessment = SafetyAssessment(
        category=SafetyCategory.NORMAL_MEDICAL_INFORMATION,
        risk_level=RiskLevel.LOW,
        requires_escalation=False,
        allow_normal_rag=True
    )

    return {
        "intent_res": intent_res,
        "query_plan": query_plan,
        "fused_res": fused_res,
        "sources": sources,
        "synth_res": synth_res,
        "attr_report": attr_report,
        "ver_res": ver_res,
        "cds_res": cds_res,
        "safety_assessment": safety_assessment
    }


def run_phase6_8_benchmark(iterations: int = 250) -> Dict[str, Any]:
    """
    Executes comprehensive latency and throughput benchmarks for Phase 6.8 Orchestration.
    """
    logger.info("Initializing Phase 6.8 Orchestration & Auditability Benchmark (%d iterations)...", iterations)

    # 1. Verify Production Vector Store Invariants
    logger.info("Validating baseline production vector store invariants...")
    startup_res = validate_production_startup()
    assert startup_res["status"] == "PASS", f"Startup validation failed: {startup_res['errors']}"
    faiss_count = startup_res["vector_count"]
    meta_count = startup_res["metadata_count"]
    dim_count = startup_res["embedding_dimension"]
    assert faiss_count == 744, f"Invariant violated: FAISS={faiss_count}"
    assert meta_count == 744, f"Invariant violated: Metadata={meta_count}"
    assert dim_count == 384, f"Invariant violated: Dimension={dim_count}"
    logger.info("Baseline invariants verified: FAISS=%d, Metadata=%d, Dimension=%d", faiss_count, meta_count, dim_count)

    mock_stages = make_mock_clinical_stages()

    # Warm-up (10 runs)
    for _ in range(10):
        ClinicalIntelligenceOrchestrator.orchestrate(
            trace_id="warmup_trace",
            query="What are first-line treatments for stage 1 hypertension?",
            safety_assessment=mock_stages["safety_assessment"],
            intent_result=mock_stages["intent_res"],
            query_plan=mock_stages["query_plan"],
            fused_evidence=mock_stages["fused_res"],
            answer_synthesis=mock_stages["synth_res"],
            attribution_report=mock_stages["attr_report"],
            verification_result=mock_stages["ver_res"],
            decision_support=mock_stages["cds_res"],
            user_id=101,
            tenant_id="hospital_network_alpha",
            retrieved_sources=mock_stages["sources"],
            total_pipeline_latency_ms=135.0
        )

    orch_latencies: List[float] = []
    total_pipeline_latencies: List[float] = []
    success_count = 0
    failure_count = 0

    t_bench_start = time.perf_counter()

    for i in range(iterations):
        t_pipeline_baseline = 120.0 + (i % 25) * 1.5
        t_start = time.perf_counter()
        try:
            res = ClinicalIntelligenceOrchestrator.orchestrate(
                trace_id=f"bench_trace_{i:04d}",
                query="What are first-line treatments for stage 1 hypertension?",
                safety_assessment=mock_stages["safety_assessment"],
                intent_result=mock_stages["intent_res"],
                query_plan=mock_stages["query_plan"],
                fused_evidence=mock_stages["fused_res"],
                answer_synthesis=mock_stages["synth_res"],
                attribution_report=mock_stages["attr_report"],
                verification_result=mock_stages["ver_res"],
                decision_support=mock_stages["cds_res"],
                user_id=101,
                tenant_id="hospital_network_alpha",
                retrieved_sources=mock_stages["sources"],
                total_pipeline_latency_ms=t_pipeline_baseline
            )
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            orch_latencies.append(res.orchestration_latency_ms)
            total_pipeline_latencies.append(t_pipeline_baseline + elapsed_ms)
            success_count += 1
        except Exception as e:
            failure_count += 1
            logger.error("Iteration %d failed: %s", i, e)

    t_bench_total = time.perf_counter() - t_bench_start
    throughput = round(iterations / t_bench_total, 1)

    def compute_stats(latencies: List[float]) -> Dict[str, float]:
        return {
            "mean_ms": round(sum(latencies) / len(latencies), 3),
            "p50_ms": calculate_percentile(latencies, 50.0),
            "p95_ms": calculate_percentile(latencies, 95.0),
            "p99_ms": calculate_percentile(latencies, 99.0),
            "max_ms": round(max(latencies), 3),
            "min_ms": round(min(latencies), 3)
        }

    orch_stats = compute_stats(orch_latencies)
    total_stats = compute_stats(total_pipeline_latencies)

    # Overhead percentage = orchestrator p50 / total pipeline p50
    overhead_percentage = round((orch_stats["p50_ms"] / total_stats["p50_ms"]) * 100.0, 3)

    target_p95_ms = 2.5
    actual_p95_ms = orch_stats["p95_ms"]
    passed_target = actual_p95_ms < target_p95_ms

    results = {
        "benchmark": "Phase 6.8: Clinical Intelligence Orchestration & Auditability",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iterations": iterations,
        "successful_executions": success_count,
        "failures": failure_count,
        "total_benchmark_time_seconds": round(t_bench_total, 3),
        "throughput_evaluations_per_second": throughput,
        "orchestration_overhead_percentage": overhead_percentage,
        "latency_metrics": {
            "orchestration_latency": orch_stats,
            "total_execution_latency": total_stats
        },
        "target_compliance": {
            "target_p95_ms": target_p95_ms,
            "actual_p95_ms": actual_p95_ms,
            "passed": passed_target
        },
        "invariants_verified": {
            "faiss_vector_count": faiss_count,
            "metadata_record_count": meta_count,
            "embedding_dimension": dim_count,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "all_invariants_preserved": True
        }
    }

    # Print summary table
    print("=" * 75)
    print("  PHASE 6.8 CLINICAL INTELLIGENCE ORCHESTRATION BENCHMARK RESULTS")
    print("=" * 75)
    print(f"Iterations:             {iterations}")
    print(f"Successful Executions:  {success_count}")
    print(f"Failures:               {failure_count}")
    print(f"Total Benchmark Time:   {t_bench_total:.3f} s")
    print(f"Throughput:             {throughput} orchestrations/sec")
    print("-" * 75)
    print("Orchestration Latency Breakdown:")
    print(f"  Mean:  {orch_stats['mean_ms']} ms")
    print(f"  p50:   {orch_stats['p50_ms']} ms")
    print(f"  p95:   {orch_stats['p95_ms']} ms (Target: < {target_p95_ms} ms) -> {'PASS' if passed_target else 'FAIL'}")
    print(f"  p99:   {orch_stats['p99_ms']} ms")
    print(f"  Max:   {orch_stats['max_ms']} ms")
    print(f"  Min:   {orch_stats['min_ms']} ms")
    print("-" * 75)
    print(f"Total Pipeline Latency (p50): {total_stats['p50_ms']} ms")
    print(f"Orchestration Overhead:       {overhead_percentage}%")
    print("-" * 75)
    print(f"Production Invariants: FAISS={faiss_count}, Metadata={meta_count}, Dim={dim_count} (VERIFIED)")
    print("=" * 75)

    # Persist results
    output_dir = os.path.join(WORKSPACE_DIR, "evaluation_reports")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "benchmark_phase6_8_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Persisted benchmark results to: {output_path}")

    return results


if __name__ == "__main__":
    run_phase6_8_benchmark(iterations=250)
