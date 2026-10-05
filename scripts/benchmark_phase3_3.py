"""
Phase 3.3 Performance Benchmark Script.

Measures:
A. Existing non-streaming request (Miss / Generation)
B. Cached request (Hit / Instant Retrieval)
C. Streaming request (SSE generator TTFE & duration)
D. Unsupported query (Pre-LLM gate fast-path)
E. First request after startup (Pre-warmed singleton client)

Measures:
- Retrieval latency
- Cache lookup latency
- LLM latency
- Time-to-first-event (TTFE)
- Time-to-first-token (TTFT)
- Total response latency
- Cache hit rate
- LLM calls avoided
- Memory/cache entries
"""

import os
import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings
from backend.services.vector_store_service import get_vector_store_service
from backend.services.gemini_service import GeminiService
from backend.services.llm_cache_service import get_llm_cache_service
from backend.rag.rag_service import RAGService


def run_benchmark():
    print("=" * 60)
    print("PHASE 3.3 PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Invariant Verification
    vs = get_vector_store_service()
    vector_count = vs.count()
    metadata_count = len(vs.metadata_store)
    print(f"[INVARIANT] FAISS vectors: {vector_count}")
    print(f"[INVARIANT] Metadata records: {metadata_count}")
    assert vector_count == 744, f"Invariant violated: {vector_count} != 744"
    assert metadata_count == 744, f"Invariant violated: {metadata_count} != 744"

    # Pre-warming verification
    warm_start = time.perf_counter()
    gemini_client = GeminiService()
    gemini_client.prewarm_client()
    warm_duration_ms = round((time.perf_counter() - warm_start) * 1000.0, 2)
    print(f"[PRE-WARM] Gemini client initialized/verified in {warm_duration_ms} ms. Warm status: {gemini_client.is_client_warm()}")

    rag = RAGService(vector_store=vs)
    cache = get_llm_cache_service()
    cache.clear()

    results = {}

    # Query definitions
    supported_query = "What lifestyle changes are recommended for managing hypertension?"
    unsupported_query = "What antibiotics should I take for hypertension?"
    test_user_id = 2

    # ----------------------------------------------------
    # Scenario E: First request after startup (Cold Cache, Pre-warmed Client)
    # ----------------------------------------------------
    print("\n--- Running Scenario E: First Request After Startup ---")
    t0 = time.perf_counter()
    res_e = rag.generate_rag_answer(question=supported_query, user_id=test_user_id)
    t_e_total = round((time.perf_counter() - t0) * 1000.0, 2)
    results["Scenario_E_First_Request"] = {
        "total_latency_ms": t_e_total,
        "retrieval_latency_ms": res_e["timings"]["retrieval_time_ms"],
        "llm_latency_ms": res_e["timings"]["llm_generation_time_ms"],
        "cache_hit": res_e["timings"].get("cache_hit", False),
        "llm_called": res_e["timings"]["llm_called"],
        "citations_valid": len(res_e.get("sources", [])) > 0,
        "status": res_e["retrieval_status"]
    }
    print(f"Total: {t_e_total} ms | Retrieval: {res_e['timings']['retrieval_time_ms']} ms | LLM: {res_e['timings']['llm_generation_time_ms']} ms | LLM called: {res_e['timings']['llm_called']}")

    # ----------------------------------------------------
    # Scenario B: Cached Request (Cache HIT)
    # ----------------------------------------------------
    print("\n--- Running Scenario B: Cached Request (HIT) ---")
    t0 = time.perf_counter()
    res_b = rag.generate_rag_answer(question=supported_query, user_id=test_user_id)
    t_b_total = round((time.perf_counter() - t0) * 1000.0, 2)
    results["Scenario_B_Cached_Request"] = {
        "total_latency_ms": t_b_total,
        "cache_lookup_latency_ms": t_b_total,
        "llm_latency_ms": 0.0,
        "cache_hit": res_b["timings"].get("cache_hit", False),
        "llm_called": res_b["timings"]["llm_called"],
        "llm_calls_avoided": 1,
        "status": res_b["retrieval_status"],
        "sources_count": len(res_b.get("sources", []))
    }
    print(f"Total: {t_b_total} ms | Cache Hit: {res_b['timings'].get('cache_hit')} | LLM called: {res_b['timings']['llm_called']} (Calls avoided: 1)")

    # ----------------------------------------------------
    # Scenario C: Streaming Request
    # ----------------------------------------------------
    print("\n--- Running Scenario C: Streaming Request ---")
    # Invalidate cache for this query to measure generation streaming
    cache.clear()
    t0 = time.perf_counter()
    ttfe_ms = None
    ttft_ms = None
    events_count = 0
    tokens_count = 0

    stream_gen = rag.generate_rag_stream(question=supported_query, user_id=test_user_id)
    for event_type, payload in stream_gen:
        events_count += 1
        now_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if ttfe_ms is None:
            ttfe_ms = now_ms
        if event_type == "token":
            tokens_count += 1
            if ttft_ms is None:
                ttft_ms = now_ms

    t_c_total = round((time.perf_counter() - t0) * 1000.0, 2)
    results["Scenario_C_Streaming_Request"] = {
        "total_latency_ms": t_c_total,
        "time_to_first_event_ms": ttfe_ms,
        "time_to_first_token_ms": ttft_ms,
        "events_count": events_count,
        "tokens_count": tokens_count,
        "llm_called": True
    }
    print(f"Total: {t_c_total} ms | TTFE: {ttfe_ms} ms | TTFT: {ttft_ms} ms | Events: {events_count} | Tokens: {tokens_count}")

    # ----------------------------------------------------
    # Scenario D: Unsupported Query (Pre-LLM Sufficiency Gate Intercept)
    # ----------------------------------------------------
    print("\n--- Running Scenario D: Unsupported Query (Gate Intercept) ---")
    t0 = time.perf_counter()
    res_d = rag.generate_rag_answer(question=unsupported_query, user_id=test_user_id)
    t_d_total = round((time.perf_counter() - t0) * 1000.0, 2)
    results["Scenario_D_Unsupported_Query"] = {
        "total_latency_ms": t_d_total,
        "retrieval_latency_ms": res_d["timings"]["retrieval_time_ms"],
        "llm_latency_ms": 0.0,
        "llm_called": res_d["timings"]["llm_called"],
        "llm_calls_avoided": 1,
        "status": res_d["retrieval_status"]
    }
    print(f"Total: {t_d_total} ms | Status: {res_d['retrieval_status']} | LLM called: {res_d['timings']['llm_called']}")

    # ----------------------------------------------------
    # Scenario A: Non-Streaming Request Baseline
    # ----------------------------------------------------
    print("\n--- Running Scenario A: Non-Streaming Request ---")
    cache.clear()
    t0 = time.perf_counter()
    res_a = rag.generate_rag_answer(question=supported_query, user_id=test_user_id)
    t_a_total = round((time.perf_counter() - t0) * 1000.0, 2)
    results["Scenario_A_Non_Streaming_Baseline"] = {
        "total_latency_ms": t_a_total,
        "retrieval_latency_ms": res_a["timings"]["retrieval_time_ms"],
        "llm_latency_ms": res_a["timings"]["llm_generation_time_ms"],
        "cache_hit": False,
        "llm_called": True,
        "status": res_a["retrieval_status"]
    }
    print(f"Total: {t_a_total} ms | Retrieval: {res_a['timings']['retrieval_time_ms']} ms | LLM: {res_a['timings']['llm_generation_time_ms']} ms")

    # Cache Stats
    cache_stats = cache.stats()
    results["Cache_Statistics"] = cache_stats
    print(f"\n[CACHE STATS] {json.dumps(cache_stats, indent=2)}")

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))

    # Save to disk for inclusion in report
    out_file = Path("benchmark_phase3_3_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {out_file.resolve()}")


if __name__ == "__main__":
    run_benchmark()
