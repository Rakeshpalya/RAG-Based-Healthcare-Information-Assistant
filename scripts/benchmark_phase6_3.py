"""
Phase 6.3 Benchmark: Clinical Evidence Fusion & Multi-Document Reasoning.

Measures:
- Fusion latency (p50, p95, p99, mean, max)
- Ranking and balancing latency
- Deduplication overhead
- Conflict detection overhead
- Context construction overhead
- Evidence coverage analysis overhead
- Overall throughput (fusions/sec)
- Production invariant verification (744 FAISS vectors, 744 metadata records, 384 dim)

Outputs structured JSON to evaluation_reports/benchmark_phase6_3_results.json.
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

from backend.intelligence.evidence_models import (
    ConflictType,
    ConflictSeverity,
    CoverageStatus,
    EvidenceConflict,
    EvidenceCoverage,
    FusedEvidenceChunk,
    FusedContextResult
)
from backend.intelligence.evidence_fusion import ClinicalEvidenceFusionEngine
from backend.intelligence.query_plan import (
    QueryPlan,
    ChunkWeightingStrategy,
    DocumentFilterStrategy,
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
    d0 = sorted_d[int(f)] * (c - k)
    d1 = sorted_d[int(c)] * (k - f)
    return round(d0 + d1, 3)


def run_phase6_3_benchmarks(iterations: int = 500) -> Dict[str, Any]:
    print(f"============================================================")
    print(f"Starting Phase 6.3 Evidence Fusion Benchmark ({iterations} iterations)")
    print(f"============================================================")

    # 1. Prepare benchmark scenarios
    benchmark_queries = [
        (
            "What is the starting dose and titration schedule for lisinopril in hypertension?",
            QueryPlan(
                intent=ClinicalIntent.DOSAGE_QUERY,
                retrieval_required=True,
                retrieval_strategy=ClinicalRoutingStrategy.DOSAGE_RAG,
                chunk_weighting_strategy=ChunkWeightingStrategy.DOSAGE_PRIORITY,
                multi_document_strategy=MultiDocumentStrategy.SINGLE_DOCUMENT,
                top_k=5
            ),
            [
                {
                    "chunk_id": f"CHUNK_LIS_{i}",
                    "document_id": "DOC_CARDIO_2024",
                    "filename": "cardiology_guidelines_2024.pdf",
                    "page_number": i + 1,
                    "text": f"Lisinopril starting dose is 10 mg daily for hypertension, titration schedule up to 40 mg daily. Monitoring blood pressure every 2 weeks.",
                    "similarity_score": 0.88 - (i * 0.02),
                    "metadata": {"filename": "cardiology_guidelines_2024.pdf"}
                }
                for i in range(5)
            ]
        ),
        (
            "Compare lisinopril and amlodipine for blood pressure control and adverse effects.",
            QueryPlan(
                intent=ClinicalIntent.DOCUMENT_COMPARISON,
                retrieval_required=True,
                retrieval_strategy=ClinicalRoutingStrategy.DOCUMENT_COMPARISON_RAG,
                chunk_weighting_strategy=ChunkWeightingStrategy.BALANCED_MULTI_DOCUMENT,
                multi_document_strategy=MultiDocumentStrategy.BALANCED_DOCUMENT_RETRIEVAL,
                top_k=6
            ),
            [
                {
                    "chunk_id": f"CHUNK_COMP_A_{i}",
                    "document_id": "DOC_ACE",
                    "filename": "ace_inhibitors_study.pdf",
                    "page_number": i + 1,
                    "text": f"Lisinopril effectively controls blood pressure. Side effects include dry cough and hyperkalemia.",
                    "similarity_score": 0.87 - (i * 0.02),
                    "metadata": {"filename": "ace_inhibitors_study.pdf"}
                }
                for i in range(3)
            ] + [
                {
                    "chunk_id": f"CHUNK_COMP_B_{i}",
                    "document_id": "DOC_CCB",
                    "filename": "calcium_channel_blockers.pdf",
                    "page_number": i + 1,
                    "text": f"Amlodipine lowers peripheral vascular resistance. Main adverse effect is peripheral edema.",
                    "similarity_score": 0.85 - (i * 0.02),
                    "metadata": {"filename": "calcium_channel_blockers.pdf"}
                }
                for i in range(3)
            ]
        ),
        (
            "What is the starting dose of metformin in adults?",
            QueryPlan(
                intent=ClinicalIntent.DOSAGE_QUERY,
                retrieval_required=True,
                retrieval_strategy=ClinicalRoutingStrategy.DOSAGE_RAG,
                chunk_weighting_strategy=ChunkWeightingStrategy.DOSAGE_PRIORITY,
                multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
                top_k=4
            ),
            [
                {
                    "chunk_id": "MET_1",
                    "document_id": "DOC_HOSP_A",
                    "filename": "hospital_a_formulary.pdf",
                    "page_number": 2,
                    "text": "Metformin starting dose is 500 mg once daily with food.",
                    "similarity_score": 0.89,
                    "metadata": {"filename": "hospital_a_formulary.pdf"}
                },
                {
                    "chunk_id": "MET_2",
                    "document_id": "DOC_HOSP_B",
                    "filename": "hospital_b_guideline.pdf",
                    "page_number": 6,
                    "text": "Metformin initial starting dose is 1000 mg twice daily with food.",
                    "similarity_score": 0.86,
                    "metadata": {"filename": "hospital_b_guideline.pdf"}
                },
                {
                    "chunk_id": "MET_1_NEAR_DUP",
                    "document_id": "DOC_HOSP_A",
                    "filename": "hospital_a_formulary.pdf",
                    "page_number": 3,
                    "text": "Metformin starting dose is 500 mg once daily with food and meals.",
                    "similarity_score": 0.84,
                    "metadata": {"filename": "hospital_a_formulary.pdf"}
                }
            ]
        )
    ]

    # Warmup
    for query, plan, chunks in benchmark_queries:
        for _ in range(20):
            ClinicalEvidenceFusionEngine.fuse_evidence(query, chunks, plan)

    # 2. Measure Component Latencies
    fusion_latencies_ms: List[float] = []
    dedup_latencies_ms: List[float] = []
    ranking_latencies_ms: List[float] = []
    conflict_latencies_ms: List[float] = []
    coverage_latencies_ms: List[float] = []
    context_latencies_ms: List[float] = []

    t_bench_start = time.perf_counter()

    for i in range(iterations):
        query, plan, chunks = benchmark_queries[i % len(benchmark_queries)]

        # Benchmark Full End-to-End Fusion
        t0 = time.perf_counter()
        fused = ClinicalEvidenceFusionEngine.fuse_evidence(query, chunks, plan)
        fusion_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        # Benchmark Isolated Deduplication
        t_d = time.perf_counter()
        deduped, _ = ClinicalEvidenceFusionEngine.deduplicate_evidence(chunks)
        dedup_latencies_ms.append((time.perf_counter() - t_d) * 1000.0)

        # Benchmark Isolated Ranking & Diversity
        t_r = time.perf_counter()
        fused_chunks = ClinicalEvidenceFusionEngine.rank_and_balance_evidence(deduped, plan, top_k=plan.top_k)
        ranking_latencies_ms.append((time.perf_counter() - t_r) * 1000.0)

        # Benchmark Isolated Conflict Detection
        t_c = time.perf_counter()
        conflicts = ClinicalEvidenceFusionEngine.detect_conflicts(fused_chunks, query=query)
        conflict_latencies_ms.append((time.perf_counter() - t_c) * 1000.0)

        # Benchmark Isolated Coverage Assessment
        t_cov = time.perf_counter()
        coverage = ClinicalEvidenceFusionEngine.assess_coverage(query, fused_chunks, plan)
        coverage_latencies_ms.append((time.perf_counter() - t_cov) * 1000.0)

        # Benchmark Isolated Context Construction
        t_ctx = time.perf_counter()
        ClinicalEvidenceFusionEngine.format_multi_document_context(fused_chunks, conflicts, coverage)
        context_latencies_ms.append((time.perf_counter() - t_ctx) * 1000.0)

    total_bench_duration_s = time.perf_counter() - t_bench_start
    throughput_fps = round(iterations / total_bench_duration_s, 1)

    # 3. Verify Production Vector Store Invariants
    vs = get_vector_store_service()
    faiss_count = vs.index.ntotal if vs.index is not None else 0
    metadata_count = len(vs.metadata_store) if vs.metadata_store is not None else 0
    embedding_dim = vs.index.d if vs.index is not None else 0

    results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iterations": iterations,
        "total_duration_seconds": round(total_bench_duration_s, 3),
        "throughput_fusions_per_sec": throughput_fps,
        "production_invariants": {
            "faiss_vector_count": faiss_count,
            "metadata_record_count": metadata_count,
            "embedding_dimension": embedding_dim,
            "invariants_satisfied": (faiss_count == 744 and metadata_count == 744 and embedding_dim == 384)
        },
        "latencies_ms": {
            "end_to_end_fusion": {
                "p50": calculate_percentile(fusion_latencies_ms, 50),
                "p95": calculate_percentile(fusion_latencies_ms, 95),
                "p99": calculate_percentile(fusion_latencies_ms, 99),
                "mean": round(sum(fusion_latencies_ms) / len(fusion_latencies_ms), 3),
                "max": round(max(fusion_latencies_ms), 3),
                "min": round(min(fusion_latencies_ms), 3)
            },
            "deduplication_overhead": {
                "p50": calculate_percentile(dedup_latencies_ms, 50),
                "p95": calculate_percentile(dedup_latencies_ms, 95),
                "p99": calculate_percentile(dedup_latencies_ms, 99),
                "mean": round(sum(dedup_latencies_ms) / len(dedup_latencies_ms), 3)
            },
            "ranking_and_diversity": {
                "p50": calculate_percentile(ranking_latencies_ms, 50),
                "p95": calculate_percentile(ranking_latencies_ms, 95),
                "p99": calculate_percentile(ranking_latencies_ms, 99),
                "mean": round(sum(ranking_latencies_ms) / len(ranking_latencies_ms), 3)
            },
            "conflict_detection": {
                "p50": calculate_percentile(conflict_latencies_ms, 50),
                "p95": calculate_percentile(conflict_latencies_ms, 95),
                "p99": calculate_percentile(conflict_latencies_ms, 99),
                "mean": round(sum(conflict_latencies_ms) / len(conflict_latencies_ms), 3)
            },
            "evidence_coverage": {
                "p50": calculate_percentile(coverage_latencies_ms, 50),
                "p95": calculate_percentile(coverage_latencies_ms, 95),
                "p99": calculate_percentile(coverage_latencies_ms, 99),
                "mean": round(sum(coverage_latencies_ms) / len(coverage_latencies_ms), 3)
            },
            "context_construction": {
                "p50": calculate_percentile(context_latencies_ms, 50),
                "p95": calculate_percentile(context_latencies_ms, 95),
                "p99": calculate_percentile(context_latencies_ms, 99),
                "mean": round(sum(context_latencies_ms) / len(context_latencies_ms), 3)
            }
        }
    }

    # Print summary
    print(f"\nBenchmark Results Summary:")
    print(f"------------------------------------------------------------")
    print(f"End-to-End Fusion Latency:")
    print(f"  p50:  {results['latencies_ms']['end_to_end_fusion']['p50']} ms")
    print(f"  p95:  {results['latencies_ms']['end_to_end_fusion']['p95']} ms")
    print(f"  p99:  {results['latencies_ms']['end_to_end_fusion']['p99']} ms")
    print(f"  Mean: {results['latencies_ms']['end_to_end_fusion']['mean']} ms")
    print(f"Throughput: {throughput_fps} fusions/sec")
    print(f"\nSub-Component Latencies (p50):")
    print(f"  Deduplication:        {results['latencies_ms']['deduplication_overhead']['p50']} ms")
    print(f"  Ranking & Diversity:  {results['latencies_ms']['ranking_and_diversity']['p50']} ms")
    print(f"  Conflict Detection:   {results['latencies_ms']['conflict_detection']['p50']} ms")
    print(f"  Coverage Assessment:  {results['latencies_ms']['evidence_coverage']['p50']} ms")
    print(f"  Context Construction: {results['latencies_ms']['context_construction']['p50']} ms")
    print(f"\nProduction Invariants:")
    print(f"  FAISS Vectors:   {faiss_count} (expected 744)")
    print(f"  Metadata Store:  {metadata_count} (expected 744)")
    print(f"  Embedding Dim:   {embedding_dim} (expected 384)")
    print(f"  Status:          {'PASS' if results['production_invariants']['invariants_satisfied'] else 'FAIL'}")
    print(f"============================================================\n")

    # Save to report file
    report_path = os.path.join(WORKSPACE_DIR, "evaluation_reports", "benchmark_phase6_3_results.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results saved to: {report_path}")

    return results


if __name__ == "__main__":
    run_phase6_3_benchmarks()
