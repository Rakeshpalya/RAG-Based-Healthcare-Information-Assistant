"""
Phase 6.2 Performance Benchmark: Clinical Query Planning & Adaptive Retrieval Strategies.

Measures:
- Planning latency (p50, p95, p99, min, max, avg) across representative queries
- Query expansion latency and boundedness
- Chunk weighting latency and reordering impact
- Multi-document retrieval vs single-document retrieval latency
- Overall pipeline query planning overhead (< 5ms target)

Saves results to evaluation_reports/benchmark_phase6_2_results.json.
"""

import os
import sys
import time
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.intelligence.intent_models import ClinicalIntent, SafetyPriority, ClinicalRoutingStrategy
from backend.intelligence.intent_classifier import ClinicalIntentClassifier
from backend.intelligence.query_plan import (
    QueryPlan,
    ChunkWeightingStrategy,
    MultiDocumentStrategy
)
from backend.intelligence.query_planner import ClinicalQueryPlanner
from backend.rag.rag_service import (
    RAGService,
    apply_chunk_weighting,
    select_multi_document_evidence
)

BENCHMARK_QUERIES = [
    ("What are the side effects and contraindications of metformin?", ClinicalIntent.MEDICATION_QUERY),
    ("What is the typical starting dose of lisinopril for hypertension?", ClinicalIntent.DOSAGE_QUERY),
    ("What does an elevated HbA1c level of 8.2% indicate?", ClinicalIntent.LAB_RESULT_QUERY),
    ("Do my symptoms of polydipsia and fatigue indicate Type 2 diabetes?", ClinicalIntent.DIAGNOSIS_QUERY),
    ("What is the first-line treatment for acute bacterial sinusitis?", ClinicalIntent.TREATMENT_QUERY),
    ("What lifestyle modifications help prevent cardiovascular disease?", ClinicalIntent.PREVENTION_QUERY),
    ("What are the early warning symptoms of acute myocardial infarction?", ClinicalIntent.SYMPTOM_QUERY),
    ("Summarize the attached clinical discharge summary document.", ClinicalIntent.DOCUMENT_SUMMARY),
    ("Compare the treatment guidelines between ACC and AHA for hypertension.", ClinicalIntent.DOCUMENT_COMPARISON),
    ("How does sleep affect immune function?", ClinicalIntent.GENERAL_HEALTH),
    ("Who won the soccer world championship in 2022?", ClinicalIntent.OUT_OF_SCOPE),
    ("Severe crushing chest pain and shortness of breath right now", ClinicalIntent.EMERGENCY),
]


def benchmark_planning_latency(iterations: int = 50) -> Dict[str, Any]:
    """Measures ClinicalQueryPlanner.plan execution latency across all query types."""
    print("\n--- 1. Clinical Query Planning Latency Benchmark ---")
    all_latencies: List[float] = []
    per_intent_latencies: Dict[str, List[float]] = {}

    # Pre-classify intents to isolate pure planning latency
    classified = []
    for q, _ in BENCHMARK_QUERIES:
        res = ClinicalIntentClassifier.classify(q)
        classified.append((q, res))

    for _ in range(iterations):
        for q, intent_res in classified:
            t0 = time.perf_counter()
            plan = ClinicalQueryPlanner.plan(q, intent_res)
            dur = (time.perf_counter() - t0) * 1000.0  # ms
            all_latencies.append(dur)
            intent_name = plan.intent.value if hasattr(plan.intent, "value") else str(plan.intent)
            per_intent_latencies.setdefault(intent_name, []).append(dur)

    all_latencies.sort()
    p50 = statistics.median(all_latencies)
    p95 = all_latencies[int(len(all_latencies) * 0.95)]
    p99 = all_latencies[int(len(all_latencies) * 0.99)]
    avg = statistics.mean(all_latencies)
    min_lat = min(all_latencies)
    max_lat = max(all_latencies)

    print(f"Total Planning Invocations: {len(all_latencies)}")
    print(f"p50: {p50:.3f} ms | p95: {p95:.3f} ms | p99: {p99:.3f} ms | Avg: {avg:.3f} ms")
    print(f"Target < 5.0 ms: {'PASSED [OK]' if p95 < 5.0 else 'FAILED'}")

    return {
        "iterations": iterations,
        "total_invocations": len(all_latencies),
        "p50_ms": round(p50, 4),
        "p95_ms": round(p95, 4),
        "p99_ms": round(p99, 4),
        "avg_ms": round(avg, 4),
        "min_ms": round(min_lat, 4),
        "max_ms": round(max_lat, 4),
        "meets_target_5ms": p95 < 5.0
    }


def benchmark_query_expansion(iterations: int = 50) -> Dict[str, Any]:
    """Measures query expansion latency, determinism, and bound compliance."""
    print("\n--- 2. Query Expansion Latency & Boundedness Benchmark ---")
    expansion_latencies: List[float] = []
    expansion_counts: List[int] = []

    for _ in range(iterations):
        for q, intent in BENCHMARK_QUERIES:
            t0 = time.perf_counter()
            expansions = ClinicalQueryPlanner.generate_expansions(q, intent)
            dur = (time.perf_counter() - t0) * 1000.0
            expansion_latencies.append(dur)
            expansion_counts.append(len(expansions))

    expansion_latencies.sort()
    p50 = statistics.median(expansion_latencies)
    p95 = expansion_latencies[int(len(expansion_latencies) * 0.95)]
    max_count = max(expansion_counts)

    print(f"Total Expansion Invocations: {len(expansion_latencies)}")
    print(f"p50: {p50:.3f} ms | p95: {p95:.3f} ms | Max Expansions: {max_count}")
    print(f"Bounded <= 4: {'PASSED [OK]' if max_count <= 4 else 'FAILED'}")
    print(f"Latency Target < 2.0 ms: {'PASSED [OK]' if p95 < 2.0 else 'FAILED'}")

    return {
        "iterations": iterations,
        "p50_ms": round(p50, 4),
        "p95_ms": round(p95, 4),
        "max_expansions_count": max_count,
        "bounded_under_4": max_count <= 4,
        "meets_target_2ms": p95 < 2.0
    }


def benchmark_chunk_weighting_and_multi_doc(iterations: int = 50) -> Dict[str, Any]:
    """Measures chunk weighting and multi-document distribution algorithm speed."""
    print("\n--- 3. Adaptive Chunk Weighting & Multi-Doc Selection Benchmark ---")
    mock_chunks = [
        {"chunk_id": f"c_{i}", "document_id": f"doc_{i % 4}", "text": f"Clinical chunk {i} discussing metformin dosage 500 mg and renal contraindications.", "similarity_score": 0.80 - (i * 0.01), "metadata": {}}
        for i in range(16)
    ]

    weighting_latencies: List[float] = []
    selection_latencies: List[float] = []

    for _ in range(iterations):
        # Weighting
        t0 = time.perf_counter()
        _ = apply_chunk_weighting(mock_chunks, ChunkWeightingStrategy.DOSAGE_PRIORITY, "metformin dose")
        weighting_latencies.append((time.perf_counter() - t0) * 1000.0)

        # Multi-doc balanced selection
        t1 = time.perf_counter()
        _ = select_multi_document_evidence(mock_chunks, MultiDocumentStrategy.BALANCED_DOCUMENT_RETRIEVAL, top_k=6)
        selection_latencies.append((time.perf_counter() - t1) * 1000.0)

    weighting_p95 = sorted(weighting_latencies)[int(len(weighting_latencies) * 0.95)]
    selection_p95 = sorted(selection_latencies)[int(len(selection_latencies) * 0.95)]

    print(f"Chunk Weighting p95: {weighting_p95:.4f} ms")
    print(f"Multi-Doc Balanced Selection p95: {selection_p95:.4f} ms")

    return {
        "chunk_weighting_p95_ms": round(weighting_p95, 4),
        "multi_doc_selection_p95_ms": round(selection_p95, 4)
    }


def main():
    print("======================================================================")
    print("  PHASE 6.2 BENCHMARK: CLINICAL QUERY PLANNING & ADAPTIVE RETRIEVAL   ")
    print("======================================================================")

    res_planning = benchmark_planning_latency(iterations=100)
    res_expansion = benchmark_query_expansion(iterations=100)
    res_adaptive = benchmark_chunk_weighting_and_multi_doc(iterations=100)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_environment": {
            "os": sys.platform,
            "python_version": sys.version
        },
        "query_planning": res_planning,
        "query_expansion": res_expansion,
        "adaptive_retrieval": res_adaptive,
        "summary": {
            "all_criteria_passed": (
                res_planning["meets_target_5ms"] and
                res_expansion["meets_target_2ms"] and
                res_expansion["bounded_under_4"]
            )
        }
    }

    out_dir = Path("evaluation_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_phase6_2_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n======================================================================")
    print(f"Results successfully saved to: {out_file}")
    print(f"All benchmarks passed: {report['summary']['all_criteria_passed']}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
