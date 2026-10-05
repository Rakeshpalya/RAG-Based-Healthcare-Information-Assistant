"""
Phase 6.7 Benchmark: Clinical Decision Support, Uncertainty Calibration & Care Pathways.

Measures deterministic performance overhead of:
- Uncertainty calibration latency (multi-factor composite uncertainty)
- Clinical risk stratification latency (CRITICAL, HIGH, MODERATE, LOW, MINIMAL)
- Clinical red-flag screening latency (cardiovascular, metabolic, pharmacological)
- Actionable recommendation generation latency (guideline-concordant non-prescriptive pathways)
- Provider SBAR clinical handoff summary generation latency
- Agentic follow-up inquiries generation latency
- End-to-end evaluation overhead (ClinicalDecisionSupportEngine.evaluate p50, p95, p99, mean, max)
- Full synthesis integration overhead (ClinicalAnswerSynthesisEngine.synthesize with Phase 6.7)
- Overall throughput (evaluations/sec)
- Production vector store invariant verification (744 FAISS vectors, 744 metadata records, 384 dim)

Outputs structured JSON to evaluation_reports/benchmark_phase6_7_results.json.
"""

import os
import sys
import time
import json
import math
import logging
from typing import List, Dict, Any
from unittest.mock import MagicMock

# Ensure workspace root is in sys.path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

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
from backend.intelligence.clinical_decision_support import ClinicalDecisionSupportEngine
from backend.intelligence.verification_models import (
    GroundingVerificationStatus,
    ClinicalHallucinationType,
    SafetyPostScreenAction,
    ExtractedClinicalEntities,
    ClinicalVerificationClaim,
    ClinicalVerificationResult
)
from backend.intelligence.citation_models import (
    CitationVerificationStatus,
    ClinicalClaimType,
    AttributedEvidenceSpan,
    ClinicalClaimAttribution,
    CitationAttributionReport
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
from backend.intelligence.answer_synthesis import ClinicalAnswerSynthesisEngine
from backend.intelligence.query_plan import (
    QueryPlan,
    ChunkWeightingStrategy,
    DocumentFilterStrategy,
    MultiDocumentStrategy,
    GenerationStrategy
)
from backend.intelligence.intent_models import (
    ClinicalIntent,
    ClinicalRoutingStrategy
)
from backend.services.vector_store_service import get_vector_store_service


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


def run_phase6_7_benchmark(iterations: int = 200) -> Dict[str, Any]:
    """Executes the Phase 6.7 performance benchmark."""
    print("=" * 70)
    print("  PHASE 6.7 BENCHMARK: CLINICAL DECISION SUPPORT & CARE PATHWAYS  ")
    print("=" * 70)
    print(f"Iterations: {iterations}")
    print(f"Target: Deterministic Local Overhead p95 < 5.0 ms\n")

    # Representative multi-source clinical synthesis setup
    sources = [
        {
            "source_index": 1,
            "source_number": 1,
            "chunk_id": "chunk_htn_001",
            "document_id": "doc_aha_2024",
            "document_name": "AHA_Hypertension_2024.pdf",
            "page_number": 12,
            "text": "First-line pharmacotherapy for stage 1 hypertension includes thiazide diuretics, calcium channel blockers, and ACE inhibitors. The standard starting dose for amlodipine is 5 mg daily, which reduces systolic blood pressure by 8 to 12 mmHg.",
            "preview_text": "First-line pharmacotherapy for stage 1 hypertension includes thiazide diuretics...",
            "similarity_score": 0.89
        },
        {
            "source_index": 2,
            "source_number": 2,
            "chunk_id": "chunk_htn_002",
            "document_id": "doc_aha_2024",
            "document_name": "AHA_Hypertension_2024.pdf",
            "page_number": 14,
            "text": "Lifestyle modifications including dietary sodium restriction (< 2,300 mg/day) and DASH dietary pattern should be initiated alongside pharmacotherapy.",
            "preview_text": "Lifestyle modifications including dietary sodium restriction...",
            "similarity_score": 0.82
        }
    ]

    attribution_report = CitationAttributionReport(
        is_valid=True,
        claims=[
            ClinicalClaimAttribution(
                claim_id="claim_001",
                claim_text="First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily.",
                raw_sentence="First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily [Source 1].",
                claim_type=ClinicalClaimType.FACTUAL_MEDICAL,
                cited_source_indices=[1],
                verification_status=CitationVerificationStatus.VERIFIED,
                best_support_score=0.92,
                is_supported=True
            ),
            ClinicalClaimAttribution(
                claim_id="claim_002",
                claim_text="Lifestyle modifications including dietary sodium restriction should be initiated.",
                raw_sentence="Lifestyle modifications including dietary sodium restriction should be initiated [Source 2].",
                claim_type=ClinicalClaimType.FACTUAL_MEDICAL,
                cited_source_indices=[2],
                verification_status=CitationVerificationStatus.VERIFIED,
                best_support_score=0.88,
                is_supported=True
            )
        ],
        total_claims_count=2,
        factual_claims_count=2,
        verified_claims_count=2,
        unsupported_claims_count=0,
        citations_found=[1, 2],
        valid_citations=[1, 2],
        invalid_citations=[],
        duplicate_citations=[],
        citation_precision=1.0,
        claim_attribution_coverage=1.0,
        spoofed_citations_detected=False,
        spoofed_citation_tags=[],
        unsupported_claims=[],
        cleaned_attributed_answer="First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily [Source 1]. Lifestyle modifications including dietary sodium restriction should be initiated [Source 2].",
        latency_ms=1.1,
        metadata={}
    )

    verification_result = ClinicalVerificationResult(
        is_verified_safe=True,
        overall_grounding_score=0.96,
        total_claims_analyzed=2,
        grounded_claims_count=2,
        ungrounded_claims_count=0,
        contradictions_count=0,
        hallucinations_detected=0,
        claim_verifications=[
            ClinicalVerificationClaim(
                claim_id="clm_1",
                claim_text="First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily.",
                raw_sentence="First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily [Source 1].",
                verification_status=GroundingVerificationStatus.GROUNDED,
                hallucination_type=ClinicalHallucinationType.NONE,
                is_grounded=True,
                confidence_score=0.96,
                supporting_sources=[1],
                entities=ExtractedClinicalEntities(
                    medications=["amlodipine"],
                    dosages=["5 mg daily"],
                    diagnoses=["stage 1"]
                )
            ),
            ClinicalVerificationClaim(
                claim_id="clm_2",
                claim_text="Lifestyle modifications including dietary sodium restriction should be initiated.",
                raw_sentence="Lifestyle modifications including dietary sodium restriction should be initiated [Source 2].",
                verification_status=GroundingVerificationStatus.GROUNDED,
                hallucination_type=ClinicalHallucinationType.NONE,
                is_grounded=True,
                confidence_score=0.92,
                supporting_sources=[2],
                entities=ExtractedClinicalEntities(
                    recommendations=["dietary sodium restriction"]
                )
            )
        ],
        action_taken=SafetyPostScreenAction.ALLOW,
        sanitized_answer="First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily [Source 1]. Lifestyle modifications including dietary sodium restriction should be initiated [Source 2].",
        fallback_triggered=False,
        disclaimer_enforced=True,
        latency_ms=1.2,
        metadata={}
    )

    fused_res = FusedContextResult(
        fused_chunks=[
            FusedEvidenceChunk(
                chunk_id="chunk_htn_001",
                document_id="doc_aha_2024",
                document_name="AHA_Hypertension_2024.pdf",
                page_number=12,
                text=sources[0]["text"],
                similarity_score=0.89,
                weighting_boost=1.0,
                fused_score=0.89,
                rank=1,
                source_index=1
            )
        ],
        contributing_documents=["AHA_Hypertension_2024.pdf"],
        contributing_documents_count=1,
        conflicts=[],
        has_conflicts=False,
        coverage=EvidenceCoverage(coverage_score=1.0, status=CoverageStatus.FULL, covered_aspects=["dosing", "lifestyle"]),
        formatted_context="[SOURCE 1] AHA_Hypertension_2024.pdf: First-line pharmacotherapy for stage 1 hypertension includes amlodipine 5 mg daily.",
        sources=sources,
        deduped_count=0,
        is_sufficient=True,
        latency_ms=1.5,
        metadata={}
    )

    query_plan = QueryPlan(
        intent=ClinicalIntent.DOSAGE_QUERY,
        retrieval_required=True,
        retrieval_strategy=ClinicalRoutingStrategy.STANDARD_RAG,
        top_k=5,
        similarity_threshold=0.25,
        multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
        chunk_weighting_strategy=ChunkWeightingStrategy.STANDARD,
        generation_strategy=GenerationStrategy.STANDARD_GROUNDED
    )

    query_text = "What is the recommended dose of amlodipine for stage 1 hypertension?"
    answer_text = "First-line therapy for stage 1 hypertension includes amlodipine 5 mg daily [Source 1]. Lifestyle modifications including dietary sodium restriction should be initiated [Source 2]."

    mock_llm = MagicMock()
    mock_llm.generate_answer_from_prompt.return_value = {
        "answer": answer_text,
        "model": "gemini-3.5-flash-lite",
        "api_request_time_ms": 110.0,
        "input_tokens": 300,
        "output_tokens": 120,
        "disclaimer": "Consult a healthcare professional."
    }

    # Latency tracking arrays
    calib_latencies: List[float] = []
    risk_latencies: List[float] = []
    redflag_latencies: List[float] = []
    recom_latencies: List[float] = []
    followup_latencies: List[float] = []
    sbar_latencies: List[float] = []
    e2e_ds_latencies: List[float] = []
    synthesis_e2e_latencies: List[float] = []

    # Warmup (25 iterations)
    for _ in range(25):
        ClinicalDecisionSupportEngine._calibrate_uncertainty(
            intent="DOSAGE_QUERY",
            sources=sources,
            fused_evidence=fused_res,
            attribution_report=attribution_report,
            verification_result=verification_result,
            query_plan=query_plan
        )
        ClinicalDecisionSupportEngine._stratify_risk(
            intent="DOSAGE_QUERY",
            query=query_text,
            answer=answer_text
        )
        ClinicalDecisionSupportEngine._identify_red_flags(
            query=query_text,
            answer=answer_text,
            intent="DOSAGE_QUERY"
        )
        ClinicalDecisionSupportEngine._generate_actionable_recommendations(
            intent="DOSAGE_QUERY",
            query=query_text,
            answer=answer_text,
            sources=sources,
            risk_tier=ClinicalRiskTier.HIGH
        )
        ClinicalDecisionSupportEngine._generate_follow_up_inquiries(
            intent="DOSAGE_QUERY",
            query=query_text,
            answer=answer_text
        )
        ClinicalDecisionSupportEngine.evaluate(
            query=query_text,
            intent="DOSAGE_QUERY",
            answer_text=answer_text,
            retrieved_sources=sources,
            fused_evidence=fused_res,
            attribution_report=attribution_report,
            verification_result=verification_result,
            query_plan=query_plan
        )

    t_bench_start = time.perf_counter()

    for _ in range(iterations):
        # 1. Uncertainty Calibration Latency
        t0 = time.perf_counter()
        calib_conf, uncert_lvl, uncert_src = ClinicalDecisionSupportEngine._calibrate_uncertainty(
            intent="DOSAGE_QUERY",
            sources=sources,
            fused_evidence=fused_res,
            attribution_report=attribution_report,
            verification_result=verification_result,
            query_plan=query_plan
        )
        t1 = time.perf_counter()
        calib_latencies.append((t1 - t0) * 1000.0)

        # 2. Clinical Risk Stratification Latency
        t0 = time.perf_counter()
        risk_tier = ClinicalDecisionSupportEngine._stratify_risk(
            intent="DOSAGE_QUERY",
            query=query_text,
            answer=answer_text
        )
        t1 = time.perf_counter()
        risk_latencies.append((t1 - t0) * 1000.0)

        # 3. Clinical Red-Flag Screening Latency
        t0 = time.perf_counter()
        red_flags, rf_escalation = ClinicalDecisionSupportEngine._identify_red_flags(
            query=query_text,
            answer=answer_text,
            intent="DOSAGE_QUERY"
        )
        t1 = time.perf_counter()
        redflag_latencies.append((t1 - t0) * 1000.0)

        # 4. Actionable Recommendations Latency
        t0 = time.perf_counter()
        recs = ClinicalDecisionSupportEngine._generate_actionable_recommendations(
            intent="DOSAGE_QUERY",
            query=query_text,
            answer=answer_text,
            sources=sources,
            risk_tier=risk_tier
        )
        t1 = time.perf_counter()
        recom_latencies.append((t1 - t0) * 1000.0)

        # 5. Agentic Follow-Up Inquiries Latency
        t0 = time.perf_counter()
        follow_ups = ClinicalDecisionSupportEngine._generate_follow_up_inquiries(
            intent="DOSAGE_QUERY",
            query=query_text,
            answer=answer_text
        )
        t1 = time.perf_counter()
        followup_latencies.append((t1 - t0) * 1000.0)

        # 6. SBAR Clinical Handoff Latency
        t0 = time.perf_counter()
        handoff = ClinicalDecisionSupportEngine._generate_sbar_handoff(
            intent="DOSAGE_QUERY",
            query=query_text,
            answer=answer_text,
            sources=sources,
            calibrated_conf=calib_conf,
            risk_tier=risk_tier,
            uncertainty_lvl=uncert_lvl,
            fused_evidence=fused_res,
            escalation_required=False,
            escalation_reason=None
        )
        t1 = time.perf_counter()
        sbar_latencies.append((t1 - t0) * 1000.0)

        # 7. End-to-End Decision Support Engine Overhead (evaluate)
        t0 = time.perf_counter()
        ds_res = ClinicalDecisionSupportEngine.evaluate(
            query=query_text,
            intent="DOSAGE_QUERY",
            answer_text=answer_text,
            retrieved_sources=sources,
            fused_evidence=fused_res,
            attribution_report=attribution_report,
            verification_result=verification_result,
            query_plan=query_plan
        )
        t1 = time.perf_counter()
        e2e_ds_latencies.append((t1 - t0) * 1000.0)

        # 8. Full Synthesis Integration Overhead
        t0 = time.perf_counter()
        synth_res = ClinicalAnswerSynthesisEngine.synthesize(
            query=query_text,
            fused_result=fused_res,
            query_plan=query_plan,
            gemini_service=mock_llm
        )
        t1 = time.perf_counter()
        synthesis_e2e_latencies.append((t1 - t0) * 1000.0)

    t_bench_total = time.perf_counter() - t_bench_start
    throughput = round(iterations / t_bench_total, 1)

    # Invariants Verification
    vs = get_vector_store_service()
    faiss_count = vs.index.ntotal if vs.index is not None else 0
    meta_count = len(getattr(vs, "metadata_store", []))
    dim_count = vs.index.d if vs.index is not None else 0

    assert faiss_count == 744, f"FAISS invariant broken: {faiss_count} != 744"
    assert meta_count == 744, f"Metadata invariant broken: {meta_count} != 744"
    assert dim_count == 384, f"Dimension invariant broken: {dim_count} != 384"

    def compute_stats(latencies: List[float]) -> Dict[str, float]:
        return {
            "mean_ms": round(sum(latencies) / len(latencies), 3),
            "p50_ms": calculate_percentile(latencies, 50.0),
            "p95_ms": calculate_percentile(latencies, 95.0),
            "p99_ms": calculate_percentile(latencies, 99.0),
            "max_ms": round(max(latencies), 3),
            "min_ms": round(min(latencies), 3)
        }

    results = {
        "benchmark": "Phase 6.7: Clinical Decision Support, Uncertainty Calibration & Care Pathways",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iterations": iterations,
        "total_benchmark_time_seconds": round(t_bench_total, 3),
        "throughput_evaluations_per_second": throughput,
        "latency_metrics": {
            "uncertainty_calibration": compute_stats(calib_latencies),
            "risk_stratification": compute_stats(risk_latencies),
            "red_flag_screening": compute_stats(redflag_latencies),
            "actionable_recommendations": compute_stats(recom_latencies),
            "agentic_follow_ups": compute_stats(followup_latencies),
            "sbar_clinical_handoff": compute_stats(sbar_latencies),
            "end_to_end_decision_support_evaluate": compute_stats(e2e_ds_latencies),
            "full_synthesis_integration": compute_stats(synthesis_e2e_latencies)
        },
        "target_compliance": {
            "target_p95_ms": 5.0,
            "actual_p95_ms": calculate_percentile(e2e_ds_latencies, 95.0),
            "passed": calculate_percentile(e2e_ds_latencies, 95.0) < 5.0
        },
        "invariants_verified": {
            "faiss_vector_count": faiss_count,
            "metadata_record_count": meta_count,
            "embedding_dimension": dim_count,
            "all_invariants_preserved": True
        }
    }

    # Print summary
    print("-" * 70)
    print("  PHASE 6.7 BENCHMARK RESULTS SUMMARY")
    print("-" * 70)
    print(f"End-to-End Decision Support evaluate():")
    print(f"  Mean: {results['latency_metrics']['end_to_end_decision_support_evaluate']['mean_ms']} ms")
    print(f"  p50:  {results['latency_metrics']['end_to_end_decision_support_evaluate']['p50_ms']} ms")
    print(f"  p95:  {results['latency_metrics']['end_to_end_decision_support_evaluate']['p95_ms']} ms")
    print(f"  p99:  {results['latency_metrics']['end_to_end_decision_support_evaluate']['p99_ms']} ms")
    print(f"  Max:  {results['latency_metrics']['end_to_end_decision_support_evaluate']['max_ms']} ms")
    print(f"Throughput: {throughput} evals/sec")
    print(f"Target Compliance: {'PASS' if results['target_compliance']['passed'] else 'FAIL'} (p95={results['target_compliance']['actual_p95_ms']}ms < 5.0ms)")
    print(f"Vector Store Invariants: FAISS={faiss_count}, Metadata={meta_count}, Dim={dim_count} (VERIFIED)")
    print("-" * 70)

    # Save to evaluation_reports/benchmark_phase6_7_results.json
    output_dir = os.path.join(WORKSPACE_DIR, "evaluation_reports")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "benchmark_phase6_7_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Persisted results to: {output_path}")

    return results


if __name__ == "__main__":
    run_phase6_7_benchmark(iterations=200)
