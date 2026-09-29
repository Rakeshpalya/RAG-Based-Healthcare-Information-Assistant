"""
Phase 2F: Granular Retrieval Stage Latency Benchmark Runner.

Measures latency across 10 distinct retrieval stages:
1. Query Normalization
2. Query Expansion & Sub-Query Decomposition
3. Embedding Generation (all-MiniLM-L6-v2)
4. FAISS Vector Retrieval (Candidate pool search)
5. Candidate Merging
6. Content Deduplication (SHA-256 identity)
7. Dynamic Relative Precision Filtering (Phase 2C & 2E)
8. Multi-Document Diversity Selection (Phase 2B)
9. Pre-LLM Safety Gate (verify_relevance_and_sufficiency)
10. Total End-to-End Retrieval Latency

Reports:
- Cold latency (first un-cached execution)
- Warm latency (repeated executions)
- Average, Median, P95, Maximum
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath("."))

from backend.rag.rag_service import RAGService
from backend.rag.query_expander import MedicalQueryExpander
from backend.services.vector_store_service import get_vector_store_service
from backend.services.embedding_service import EmbeddingService


BENCHMARK_QUERIES = [
    # 1. Single-Document Query
    "What are the common risk factors for hypertension according to the synthetic hypertension document?",
    # 2. Synonym Query
    "What causes elevated arterial blood pressure?",
    # 3. Multi-Aspect Query
    "Provide an overview of hypertension definition, risk factors, lifestyle measures, and complications.",
    # 4. Multi-Document Query
    "How do the lifestyle measures in the synthetic document relate to patient John Doe's recorded vitals?",
    # 5. Out-of-Scope Query (Rejected)
    "What are the chemotherapy guidelines for pancreatic carcinoma in the general hypertension guideline?"
]


def benchmark_single_query(rag: RAGService, query: str, top_k: int = 5) -> Dict[str, float]:
    """Measures precise execution times across all 10 stages for one query execution."""
    stage_times = {}

    # Stage 1: Query Normalization
    t0 = time.perf_counter()
    norm_q = MedicalQueryExpander.normalize_query(query)
    stage_times["1_normalization"] = (time.perf_counter() - t0) * 1000.0

    # Stage 2: Query Expansion & Sub-query Decomposition
    t0 = time.perf_counter()
    is_multi = MedicalQueryExpander.is_multi_aspect_query(query)
    if is_multi:
        sub_qs = MedicalQueryExpander.decompose_multi_aspect_query(query)
        exp_terms = []
    else:
        sub_qs = []
        exp_terms = MedicalQueryExpander.get_expanded_terms(query)
    stage_times["2_expansion_decomposition"] = (time.perf_counter() - t0) * 1000.0

    # Stage 3: Embedding Generation
    t0 = time.perf_counter()
    main_vec = EmbeddingService.embed_query(query.strip())
    sub_vecs = [EmbeddingService.embed_query(sq) for sq in sub_qs] if is_multi else []
    stage_times["3_embedding"] = (time.perf_counter() - t0) * 1000.0

    # Stage 4: FAISS Vector Retrieval (Candidate pool search)
    candidate_k = max(top_k * 4, 20)
    t0 = time.perf_counter()
    raw_results = rag.vector_store.search(main_vec, top_k=candidate_k)
    for sv in sub_vecs:
        raw_results.extend(rag.vector_store.search(sv, top_k=candidate_k))
    stage_times["4_faiss_search"] = (time.perf_counter() - t0) * 1000.0

    # Stage 5: Candidate Merging & Threshold Filter
    t0 = time.perf_counter()
    threshold = rag.default_similarity_threshold
    filtered_results = [c for c in raw_results if c.get("similarity_score", 0.0) >= threshold]
    stage_times["5_candidate_merging"] = (time.perf_counter() - t0) * 1000.0

    # Stage 6: Content Deduplication
    t0 = time.perf_counter()
    deduped = rag.deduplicate_chunks(filtered_results)
    stage_times["6_deduplication"] = (time.perf_counter() - t0) * 1000.0

    # Stage 7: Precision Filtering
    t0 = time.perf_counter()
    precision_chunks = rag.filter_candidate_precision(query=query, chunks=deduped, threshold=threshold)
    stage_times["7_precision_filtering"] = (time.perf_counter() - t0) * 1000.0

    # Stage 8: Diversity Selection
    t0 = time.perf_counter()
    diverse_chunks = rag.select_diverse_evidence(precision_chunks, top_k=top_k, max_per_doc=2)
    stage_times["8_diversity_selection"] = (time.perf_counter() - t0) * 1000.0

    # Stage 9: Safety Gate (verify_relevance_and_sufficiency)
    t0 = time.perf_counter()
    is_rel, reason = rag.verify_relevance_and_sufficiency(question=query, retrieved_chunks=diverse_chunks, similarity_threshold=threshold)
    stage_times["9_safety_gate"] = (time.perf_counter() - t0) * 1000.0

    # Stage 10: Total End-to-End
    stage_times["10_total_retrieval"] = sum(stage_times.values())

    return stage_times


def run_latency_benchmark(num_repetitions: int = 5):
    print("=" * 75)
    print("=== PHASE 2F: GRANULAR RETRIEVAL STAGE LATENCY BENCHMARK ===")
    print("=" * 75)

    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)

    stages = [
        "1_normalization",
        "2_expansion_decomposition",
        "3_embedding",
        "4_faiss_search",
        "5_candidate_merging",
        "6_deduplication",
        "7_precision_filtering",
        "8_diversity_selection",
        "9_safety_gate",
        "10_total_retrieval"
    ]

    all_runs: Dict[str, List[float]] = {st: [] for st in stages}
    cold_runs: Dict[str, float] = {}
    warm_runs: Dict[str, List[float]] = {st: [] for st in stages}

    # Warm up SentenceTransformer once before benchmark to isolate model load from cold query time
    EmbeddingService.embed_query("warmup query")

    is_first_overall = True

    for q_idx, query in enumerate(BENCHMARK_QUERIES, 1):
        print(f"\nBenchmarking Query [{q_idx}/{len(BENCHMARK_QUERIES)}]: \"{query[:55]}...\"")
        for rep in range(num_repetitions):
            timings = benchmark_single_query(rag, query)

            if is_first_overall and rep == 0:
                cold_runs = dict(timings)
                is_first_overall = False
            else:
                for st in stages:
                    warm_runs[st].append(timings[st])

            for st in stages:
                all_runs[st].append(timings[st])

    print("\n" + "=" * 75)
    print("--- LATENCY BENCHMARK RESULTS (All measurements in milliseconds ms) ---")
    print("=" * 75)
    header = f"{'Stage':<28} | {'Cold':<8} | {'Warm Avg':<9} | {'Median':<8} | {'P95':<8} | {'Max':<8}"
    print(header)
    print("-" * len(header))

    stage_display_names = {
        "1_normalization": "1. Normalization",
        "2_expansion_decomposition": "2. Expansion/Decomposition",
        "3_embedding": "3. Embedding Generation",
        "4_faiss_search": "4. FAISS Search",
        "5_candidate_merging": "5. Merging & Threshold",
        "6_deduplication": "6. Deduplication (SHA)",
        "7_precision_filtering": "7. Precision Filtering",
        "8_diversity_selection": "8. Diversity Selection",
        "9_safety_gate": "9. Safety Gate",
        "10_total_retrieval": "10. Total Latency"
    }

    report_table = {}
    for st in stages:
        c_val = cold_runs.get(st, 0.0)
        w_vals = warm_runs[st]
        w_avg = float(np.mean(w_vals)) if w_vals else c_val
        w_med = float(np.median(all_runs[st]))
        w_p95 = float(np.percentile(all_runs[st], 95))
        w_max = float(np.max(all_runs[st]))

        report_table[st] = {
            "cold": c_val,
            "warm_avg": w_avg,
            "median": w_med,
            "p95": w_p95,
            "max": w_max
        }
        print(f"{stage_display_names[st]:<28} | {c_val:<8.2f} | {w_avg:<9.2f} | {w_med:<8.2f} | {w_p95:<8.2f} | {w_max:<8.2f}")

    print("=" * 75)
    return report_table


if __name__ == "__main__":
    run_latency_benchmark()
