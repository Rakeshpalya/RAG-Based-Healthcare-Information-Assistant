"""
Phase 3.4 Production Load Testing Script

Executes concurrent load testing across 10, 25, 50, and 100 simulated concurrent users.
Measures:
- Throughput (requests/sec)
- Total, successful, failed, and blocked (rate-limited / throttled) requests
- Latency metrics: p50, p95, p99, average, and maximum latency
- Cache hit rate and LLM execution statistics
- CPU and process memory utilization (via psutil if available)

Two operating modes:
- OFFLINE / MOCK (DEFAULT): Uses mock Gemini responses for safe, zero-cost CI load verification.
- REAL LLM: Requires explicit environment variable RUN_REAL_LLM_LOAD_TEST=true.

Usage:
  python scripts/load_test_phase3_4.py
  python scripts/load_test_phase3_4.py --concurrency 10 25 50 100 --requests-per-user 2
"""

import os
import sys
import time
import math
import argparse
import statistics
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from fastapi.testclient import TestClient
from backend.main import app
from backend.evaluation.observability import get_metrics_collector
from backend.services.llm_cache_service import get_llm_cache_service


def get_system_utilization() -> Dict[str, Any]:
    """Measures current process memory and CPU if psutil is available."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "rss_mb": round(mem_info.rss / (1024 * 1024), 2),
            "cpu_percent": process.cpu_percent(interval=None)
        }
    except Exception:
        return {"rss_mb": 0.0, "cpu_percent": 0.0}


def run_load_level(
    client: TestClient,
    concurrency: int,
    requests_per_user: int = 2,
    mode: str = "OFFLINE/MOCK"
) -> Dict[str, Any]:
    """Runs a single load test tier with a given number of concurrent simulated users."""
    total_requests = concurrency * requests_per_user
    latencies_ms: List[float] = []
    status_codes: Dict[int, int] = {}
    successful = 0
    failed = 0
    blocked = 0

    benchmark_queries = [
        "What are the symptoms and diagnostic criteria for Type 2 Diabetes?",
        "What are the lifestyle measures for hypertension management?",
        "What are the common risk factors for cardiovascular disease?",
        "How is asthma monitored and treated in primary care?"
    ]

    initial_metrics = get_metrics_collector().get_metrics_snapshot()
    initial_cache_hits = initial_metrics["cache"]["hits"]
    initial_llm_calls = initial_metrics["llm"]["calls"]
    initial_llm_failures = initial_metrics["llm"]["failures"]

    mem_before = get_system_utilization()
    start_wall = time.perf_counter()

    def _execute_user_session(user_idx: int) -> List[tuple[int, float]]:
        session_results = []
        user_headers = {
            "X-Forwarded-For": f"10.0.{user_idx // 200}.{(user_idx % 200) + 1}",
            "X-User-ID": str(1000 + user_idx),
            "X-Request-ID": f"load-u{user_idx}-{time.time_ns()}"
        }
        for req_idx in range(requests_per_user):
            q = benchmark_queries[(user_idx + req_idx) % len(benchmark_queries)]
            t0 = time.perf_counter()
            resp = client.post(
                "/rag/query",
                json={"question": q, "top_k": 3},
                headers=user_headers
            )
            lat = (time.perf_counter() - t0) * 1000.0
            session_results.append((resp.status_code, lat))
        return session_results

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_execute_user_session, i) for i in range(concurrency)]
        for fut in concurrent.futures.as_completed(futures):
            results = fut.result()
            for code, lat in results:
                latencies_ms.append(lat)
                status_codes[code] = status_codes.get(code, 0) + 1
                if 200 <= code < 400:
                    successful += 1
                elif code in (429, 503):
                    blocked += 1
                else:
                    failed += 1

    wall_duration = time.perf_counter() - start_wall
    throughput_rps = round(total_requests / wall_duration, 2) if wall_duration > 0 else 0.0

    mem_after = get_system_utilization()
    final_metrics = get_metrics_collector().get_metrics_snapshot()

    cache_hits = final_metrics["cache"]["hits"] - initial_cache_hits
    llm_calls = final_metrics["llm"]["calls"] - initial_llm_calls
    llm_failures = final_metrics["llm"]["failures"] - initial_llm_failures
    cache_hit_rate = round(cache_hits / total_requests, 4) if total_requests > 0 else 0.0

    # Latency percentiles
    sorted_lat = sorted(latencies_ms) if latencies_ms else [0.0]

    def _p(p: float) -> float:
        if not sorted_lat:
            return 0.0
        k = (len(sorted_lat) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return round(sorted_lat[int(k)], 2)
        return round(sorted_lat[int(f)] * (c - k) + sorted_lat[int(c)] * (k - f), 2)

    return {
        "concurrency": concurrency,
        "mode": mode,
        "total_requests": total_requests,
        "successful_requests": successful,
        "failed_requests": failed,
        "blocked_requests": blocked,
        "throughput_rps": throughput_rps,
        "latency_ms": {
            "p50": _p(50),
            "p95": _p(95),
            "p99": _p(99),
            "avg": round(sum(sorted_lat) / len(sorted_lat), 2) if sorted_lat else 0.0,
            "max": round(sorted_lat[-1], 2) if sorted_lat else 0.0
        },
        "cache_hits": cache_hits,
        "cache_hit_rate": cache_hit_rate,
        "llm_calls": llm_calls,
        "llm_failures": llm_failures,
        "memory_rss_mb": mem_after.get("rss_mb", 0.0),
        "status_breakdown": status_codes,
        "duration_sec": round(wall_duration, 3)
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3.4 Production Load Testing")
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[10, 25, 50, 100],
        help="List of concurrent user levels to test"
    )
    parser.add_argument(
        "--requests-per-user",
        type=int,
        default=2,
        help="Number of requests sent by each simulated user"
    )
    args = parser.parse_args()

    run_real = os.getenv("RUN_REAL_LLM_LOAD_TEST", "false").lower() in ("true", "1", "yes")
    mode = "REAL LLM" if run_real else "OFFLINE/MOCK"

    print("=" * 70)
    print("PHASE 3.4 PRODUCTION LOAD TEST HARNESS")
    print(f"Mode:               {mode}")
    print(f"Concurrency Tiers:  {args.concurrency}")
    print(f"Requests Per User:  {args.requests_per_user}")
    print("=" * 70)

    client = TestClient(app)

    # If in mock mode, mock GeminiService.generate_answer to prevent external network calls
    if not run_real:
        from unittest.mock import MagicMock
        from backend.services.gemini_service import GeminiService
        mock_gen = MagicMock(return_value={
            "answer": "Mock grounded clinical evidence for hypertension and diabetes [Source 1].",
            "model": "gemini-3.5-flash-lite",
            "generation_time_ms": 35.0,
            "gemini_calls_count": 1
        })
        GeminiService.generate_answer = mock_gen

    print("\n[INFO] Pre-warming embedding model and client cache...")
    from backend.services.embedding_service import EmbeddingService
    EmbeddingService.embed_query("Hypertension clinical management prewarm query")
    print("[INFO] Embedding model pre-warmed successfully.")

    results = []
    for c in args.concurrency:
        print(f"\n---> Testing {c} Concurrent Users (Total: {c * args.requests_per_user} requests)...")
        tier_res = run_load_level(
            client=client,
            concurrency=c,
            requests_per_user=args.requests_per_user,
            mode=mode
        )
        results.append(tier_res)

        lat = tier_res["latency_ms"]
        print(f"     Throughput: {tier_res['throughput_rps']} req/s | Wall Time: {tier_res['duration_sec']}s")
        print(f"     Success: {tier_res['successful_requests']} | Blocked: {tier_res['blocked_requests']} | Failed: {tier_res['failed_requests']}")
        print(f"     Latencies: p50={lat['p50']}ms | p95={lat['p95']}ms | p99={lat['p99']}ms | max={lat['max']}ms | avg={lat['avg']}ms")
        print(f"     Cache Hits: {tier_res['cache_hits']} | Hit Rate: {tier_res['cache_hit_rate'] * 100:.1f}%")
        print(f"     Memory RSS: {tier_res['memory_rss_mb']} MB")

    print("\n" + "=" * 70)
    print("LOAD TEST SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Users':<8}{'Reqs':<8}{'RPS':<10}{'p50 (ms)':<12}{'p95 (ms)':<12}{'p99 (ms)':<12}{'Hit Rate':<10}{'Status'}")
    print("-" * 70)
    for r in results:
        l = r["latency_ms"]
        status = "PASS" if r["failed_requests"] == 0 else "DEGRADED"
        print(
            f"{r['concurrency']:<8}{r['total_requests']:<8}{r['throughput_rps']:<10}"
            f"{l['p50']:<12}{l['p95']:<12}{l['p99']:<12}"
            f"{r['cache_hit_rate']*100:>5.1f}%    {status}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
