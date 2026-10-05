"""
Phase 6.5 Benchmark: Clinical Citation & Attribution Engine.

Measures deterministic performance overhead of:
- Bracket parsing & anti-spoofing detection latency
- Sentence & claim segmentation latency
- Claim classification latency
- Claim-level evidence attribution & support scoring latency
- Contradiction & dosage hallucination detection latency
- Unsupported claim pruning latency
- End-to-end attribution report generation overhead (p50, p95, p99, mean, max)
- Synthesis integration overhead (Answer synthesis with full Phase 6.5 attribution pipeline)
- Overall throughput (evaluations/sec)
- Production vector store invariant verification (744 FAISS vectors, 744 metadata records, 384 dim)

Outputs structured JSON to evaluation_reports/benchmark_phase6_5_results.json.
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

from backend.intelligence.citation_models import (
    CitationVerificationStatus,
    ClinicalClaimType,
    AttributedEvidenceSpan,
    ClinicalClaimAttribution,
    CitationAttributionReport
)
from backend.intelligence.citation_attribution import ClinicalCitationAttributionEngine
from backend.intelligence.answer_synthesis import ClinicalAnswerSynthesisEngine
from backend.intelligence.evidence_models import (
    FusedEvidenceChunk,
    FusedContextResult,
    EvidenceCoverage,
    CoverageStatus
)
from backend.intelligence.query_plan import (
    QueryPlan,
    ChunkWeightingStrategy,
    MultiDocumentStrategy,
    GenerationStrategy
)
from backend.intelligence.intent_models import ClinicalIntent, ClinicalRoutingStrategy
from backend.services.vector_store_service import get_vector_store_service


def calculate_percentile(data: List[float], p: float) -> float:
    """Calculates percentile with linear interpolation."""
    if not data:
        return 0.0
    sorted_d = sorted(data)
    k = (len(sorted_d) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(sorted_d[int(k)], 3)
    return round(sorted_d[f] * (c - k) + sorted_d[c] * (k - f), 3)


def build_benchmark_test_data() -> tuple[str, List[Dict[str, Any]], FusedContextResult, QueryPlan]:
    """Assembles realistic multi-source clinical synthesis text and evidence for benchmarking."""
    chunks = [
        FusedEvidenceChunk(
            chunk_id="chunk_htn_001",
            document_id="guideline_aha_2024",
            document_name="AHA_Hypertension_2024.pdf",
            page_number=12,
            text=(
                "First-line pharmacological management of Stage 1 hypertension involves thiazide diuretics, "
                "calcium channel blockers, or ACE inhibitors. Lisinopril initial dosage is 10 mg orally once daily. "
                "Target blood pressure goal is < 130/80 mmHg in adult patients."
            ),
            similarity_score=0.91,
            weighting_boost=1.0,
            fused_score=0.91,
            rank=1,
            source_index=1
        ),
        FusedEvidenceChunk(
            chunk_id="chunk_cvd_002",
            document_id="guideline_esc_cvd",
            document_name="ESC_Cardiovascular_Prevention.pdf",
            page_number=24,
            text=(
                "Lifestyle interventions including dietary sodium restriction (< 2 g/day), regular aerobic exercise "
                "(at least 150 minutes weekly), and alcohol moderation provide additive blood pressure reduction."
            ),
            similarity_score=0.85,
            weighting_boost=0.95,
            fused_score=0.81,
            rank=2,
            source_index=2
        ),
        FusedEvidenceChunk(
            chunk_id="chunk_drug_003",
            document_id="fda_lisinopril_monograph",
            document_name="FDA_Lisinopril_Package_Insert.pdf",
            page_number=3,
            text=(
                "Lisinopril starting dosage is 10 mg orally once daily. Maintenance dosage ranges from 20 to 40 mg daily. "
                "Contraindicated in angioedema and pregnancy. Renal function and potassium should be monitored periodically."
            ),
            similarity_score=0.88,
            weighting_boost=0.92,
            fused_score=0.81,
            rank=3,
            source_index=3
        )
    ]

    sources = [
        {
            "source_index": c.source_index,
            "source_label": f"[Source {c.source_index}]",
            "document_id": c.document_id,
            "document_name": c.document_name,
            "page_number": c.page_number,
            "similarity_score": c.similarity_score,
            "text": c.text,
            "chunk_id": c.chunk_id
        }
        for c in chunks
    ]

    cov = EvidenceCoverage(
        coverage_score=0.94,
        status=CoverageStatus.FULL,
        query_aspects=["first-line therapy", "dosage", "lifestyle"],
        covered_aspects=["first-line therapy", "dosage", "lifestyle"],
        missing_aspects=[]
    )

    fused_res = FusedContextResult(
        fused_chunks=chunks,
        contributing_documents=[c.document_id for c in chunks],
        contributing_documents_count=len(chunks),
        conflicts=[],
        has_conflicts=False,
        coverage=cov,
        formatted_context="\n\n".join(f"[SOURCE {c.source_index}]: {c.text}" for c in chunks),
        sources=sources,
        deduped_count=0,
        is_sufficient=True,
        latency_ms=1.1
    )

    query_plan = QueryPlan(
        intent=ClinicalIntent.TREATMENT_QUERY,
        retrieval_required=True,
        retrieval_strategy=ClinicalRoutingStrategy.TREATMENT_RAG,
        top_k=5,
        similarity_threshold=0.25,
        multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
        chunk_weighting_strategy=ChunkWeightingStrategy.STANDARD,
        generation_strategy=GenerationStrategy.STANDARD_GROUNDED
    )

    answer_text = (
        "First-line pharmacological management of Stage 1 hypertension includes thiazide diuretics, calcium channel blockers, "
        "or ACE inhibitors [Source 1]. Initial lisinopril therapy begins at 10 mg orally once daily [Source 1, Source 3]. "
        "Lifestyle interventions including sodium restriction to less than 2 grams daily and regular aerobic exercise provide "
        "meaningful blood pressure reduction [Source 2]. Lisinopril is contraindicated in patients with a history of angioedema [Source 3]."
    )

    return answer_text, sources, fused_res, query_plan


def run_benchmark(iterations: int = 200) -> Dict[str, Any]:
    """Executes the Phase 6.5 performance benchmark."""
    print(f"\n==================================================================")
    print(f"  PHASE 6.5 BENCHMARK: CLINICAL CITATION & ATTRIBUTION ENGINE")
    print(f"==================================================================")
    print(f"Iterations: {iterations}")
    print(f"Target: Deterministic Local Overhead p95 < 5.0 ms")
    print(f"Note: Evaluates bracket parsing, anti-spoofing, claim segmentation,")
    print(f"      evidence attribution, contradiction checking, and pruning.")

    answer_text, sources, fused_res, query_plan = build_benchmark_test_data()

    # Pre-configured mock LLM returning instant deterministic response for end-to-end synthesis
    mock_llm = MagicMock()
    mock_llm.generate_answer_from_prompt.return_value = {
        "answer": answer_text,
        "model": "gemini-3.5-flash-lite",
        "elapsed_ms": 0.0
    }
    mock_llm.generate_answer.return_value = mock_llm.generate_answer_from_prompt.return_value

    bracket_latencies: List[float] = []
    segment_latencies: List[float] = []
    attrib_latencies: List[float] = []
    pruning_latencies: List[float] = []
    e2e_attribution_latencies: List[float] = []
    synthesis_e2e_latencies: List[float] = []

    # Warmup
    for _ in range(25):
        ClinicalCitationAttributionEngine.extract_citations(answer_text)
        ClinicalCitationAttributionEngine.detect_spoofed_citations(answer_text)
        ClinicalCitationAttributionEngine.segment_claims(answer_text)
        ClinicalCitationAttributionEngine.attribute_and_validate(
            answer_text=answer_text,
            retrieved_sources=sources
        )

    t_bench_start = time.perf_counter()

    for _ in range(iterations):
        # 1. Bracket Parsing & Anti-Spoofing Detection Latency
        t0 = time.perf_counter()
        brackets = ClinicalCitationAttributionEngine.extract_citations(answer_text)
        spoofed = ClinicalCitationAttributionEngine.detect_spoofed_citations(answer_text)
        t1 = time.perf_counter()
        bracket_latencies.append((t1 - t0) * 1000.0)

        # 2. Claim Segmentation Latency
        t0 = time.perf_counter()
        claim_segments = ClinicalCitationAttributionEngine.segment_claims(answer_text)
        t1 = time.perf_counter()
        segment_latencies.append((t1 - t0) * 1000.0)

        # 3. Evidence Attribution & Validation Latency (attribute_and_validate)
        t0 = time.perf_counter()
        report = ClinicalCitationAttributionEngine.attribute_and_validate(
            answer_text=answer_text,
            retrieved_sources=sources
        )
        t1 = time.perf_counter()
        attrib_latencies.append((t1 - t0) * 1000.0)
        e2e_attribution_latencies.append((t1 - t0) * 1000.0)

        # 4. Pruning Latency
        t0 = time.perf_counter()
        pruned_text = ClinicalCitationAttributionEngine.prune_unsupported_claims(
            answer_text=answer_text,
            report=report
        )
        t1 = time.perf_counter()
        pruning_latencies.append((t1 - t0) * 1000.0)

        # 5. Full Synthesis Engine Integration Overhead
        t0 = time.perf_counter()
        synth_res = ClinicalAnswerSynthesisEngine.synthesize(
            query="What is the first-line treatment and starting dose for hypertension?",
            fused_result=fused_res,
            query_plan=query_plan,
            intent="TREATMENT_QUERY",
            gemini_service=mock_llm
        )
        t1 = time.perf_counter()
        synthesis_e2e_latencies.append((t1 - t0) * 1000.0)

    total_bench_duration = time.perf_counter() - t_bench_start
    throughput = round(iterations / total_bench_duration, 1)

    def calc_stats(lat_list: List[float]) -> Dict[str, float]:
        return {
            "mean_ms": round(sum(lat_list) / len(lat_list), 3),
            "p50_ms": calculate_percentile(lat_list, 50.0),
            "p95_ms": calculate_percentile(lat_list, 95.0),
            "p99_ms": calculate_percentile(lat_list, 99.0),
            "min_ms": round(min(lat_list), 3),
            "max_ms": round(max(lat_list), 3)
        }

    bracket_stats = calc_stats(bracket_latencies)
    segment_stats = calc_stats(segment_latencies)
    attrib_stats = calc_stats(attrib_latencies)
    pruning_stats = calc_stats(pruning_latencies)
    e2e_attrib_stats = calc_stats(e2e_attribution_latencies)
    synthesis_e2e_stats = calc_stats(synthesis_e2e_latencies)

    # Verify Production Vector Store Invariants
    vs = get_vector_store_service()
    faiss_count = vs.index.ntotal if vs.index is not None else 0
    meta_count = len(getattr(vs, "metadata_store", []))
    dim_count = vs.index.d if vs.index is not None else 0

    invariants_intact = (faiss_count == 744 and meta_count == 744 and dim_count == 384)
    target_met = (e2e_attrib_stats["p95_ms"] < 5.0)

    print("\n--- BENCHMARK RESULTS ---")
    print(f"Citation Attribution Engine Overhead (attribute_and_validate):")
    print(f"  Mean:  {e2e_attrib_stats['mean_ms']} ms")
    print(f"  p50:   {e2e_attrib_stats['p50_ms']} ms")
    print(f"  p95:   {e2e_attrib_stats['p95_ms']} ms (Target: < 5.0 ms - {'PASSED' if target_met else 'FAILED'})")
    print(f"  p99:   {e2e_attrib_stats['p99_ms']} ms")
    print(f"  Max:   {e2e_attrib_stats['max_ms']} ms")
    print(f"Throughput: {throughput} attribution cycles/sec")
    print(f"\nSub-Component Latencies (p95):")
    print(f"  Bracket & Spoofing Parsing: {bracket_stats['p95_ms']} ms")
    print(f"  Claim Segmentation:         {segment_stats['p95_ms']} ms")
    print(f"  Evidence Attribution:       {attrib_stats['p95_ms']} ms")
    print(f"  Unsupported Claim Pruning:  {pruning_stats['p95_ms']} ms")
    print(f"  Full Synthesis Integration: {synthesis_e2e_stats['p95_ms']} ms")
    print(f"\nProduction Invariants:")
    print(f"  FAISS Vectors:              {faiss_count} (Expected: 744)")
    print(f"  Metadata Records:           {meta_count} (Expected: 744)")
    print(f"  Embedding Dimension:        {dim_count} (Expected: 384)")
    print(f"  Invariants Intact:          {invariants_intact}")

    results = {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_type": "Phase 6.5 Clinical Citation & Attribution",
        "iterations": iterations,
        "throughput_ops_per_sec": throughput,
        "citation_attribution_overhead": e2e_attrib_stats,
        "component_breakdown": {
            "bracket_and_spoofing_detection": bracket_stats,
            "claim_segmentation": segment_stats,
            "evidence_attribution": attrib_stats,
            "unsupported_claim_pruning": pruning_stats,
            "synthesis_engine_integration": synthesis_e2e_stats
        },
        "target_p95_threshold_ms": 5.0,
        "target_met": target_met,
        "production_invariants": {
            "faiss_vector_count": faiss_count,
            "metadata_record_count": meta_count,
            "embedding_dimension": dim_count,
            "invariants_intact": invariants_intact
        }
    }

    report_dir = os.path.join(WORKSPACE_DIR, "evaluation_reports")
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, "benchmark_phase6_5_results.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nReport saved to: {report_file}")
    return results


if __name__ == "__main__":
    run_benchmark(iterations=200)
