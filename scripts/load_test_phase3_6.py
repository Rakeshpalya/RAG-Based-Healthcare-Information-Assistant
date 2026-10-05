"""
Phase 3.6 — Production Sustained Load Testing Script.

Evaluates system throughput, latency percentiles (p50, p95, p99), cache performance,
error rates, rate-limiting, and resource utilization under sustained concurrency:
- Concurrency levels: 10, 25, 50, and 100 simulated users.
- Sustained durations: 30s, 60s, and 300s (5 minutes).
- Dual Modes:
    MODE A: Mock/offline LLM (default, zero external cost, safe for CI)
    MODE B: Real Gemini (explicitly opt-in via --live-llm or LIVE_LLM=1)

Identifies:
- Saturation point
- Concurrency bottleneck
- Rate-limit bottleneck
- LLM bottleneck
- Memory bottleneck
- CPU bottleneck
"""

import os
import sys
import time
import json
import math
import argparse
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.llm_cache_service import get_llm_cache_service
from backend.evaluation.observability import get_metrics_collector

try:
    import psutil
except ImportError:
    psutil = None


BENCHMARK_QUERIES = [
    "What are the symptoms and diagnostic criteria for Type 2 Diabetes?",
    "What is the first-line pharmacotherapy for hypertension?",
    "What are the clinical indications and contraindications for metformin?",
    "How is acute asthma exacerbation managed in primary care?",
    "What are the diagnostic thresholds for prediabetes with HbA1c?"
]


def get_process_metrics() -> Dict[str, float]:
    """Captures process RSS memory in MB and CPU percentage."""
    if psutil:
        try:
            proc = psutil.Process(os.getpid())
            mem_mb = proc.memory_info().rss / (1024 * 1024)
            cpu_pct = proc.cpu_percent(interval=None)
            return {"rss_mb": round(mem_mb, 2), "cpu_percent": round(cpu_pct, 1)}
        except Exception:
            pass
    return {"rss_mb": 0.0, "cpu_percent": 0.0}


class SustainedLoadRunner:
    def __init__(
        self,
        duration_sec: int = 30,
        mode: str = "mock",
        output_file: Optional[str] = None
    ):
        self.duration_sec = duration_sec
        self.mode = mode.lower()
        self.output_file = output_file or "evaluation_reports/load_test_phase3_6_results.json"

        from backend.api.auth_dependencies import get_optional_current_db_user
        from backend.database.models import User
        app.dependency_overrides[get_optional_current_db_user] = lambda: User(id=1, email="loadtest@clinic.org", role="doctor")
        self.client = TestClient(app)

    def _worker_loop(
        self,
        worker_id: int,
        stop_event: threading.Event,
        latencies: List[float],
        status_counts: Dict[str, int],
        lock: threading.Lock
    ):
        """Worker thread executing queries until stop_event is signaled."""
        query_idx = worker_id % len(BENCHMARK_QUERIES)
        user_id = (worker_id % 10) + 1  # Distribute across 10 users to test user isolation

        while not stop_event.is_set():
            query = BENCHMARK_QUERIES[query_idx]
            query_idx = (query_idx + 1) % len(BENCHMARK_QUERIES)

            t0 = time.perf_counter()
            try:
                resp = self.client.post(
                    "/rag/query",
                    json={"question": query},
                    headers={"X-Forwarded-For": f"10.0.0.{user_id}", "X-User-Id": str(user_id)}
                )
                lat_ms = (time.perf_counter() - t0) * 1000.0
                st = resp.status_code
            except Exception as exc:
                lat_ms = (time.perf_counter() - t0) * 1000.0
                st = 500

            with lock:
                latencies.append(lat_ms)
                if st == 200:
                    status_counts["success"] = status_counts.get("success", 0) + 1
                elif st == 429:
                    status_counts["rate_limited"] = status_counts.get("rate_limited", 0) + 1
                elif st == 504:
                    status_counts["timeout"] = status_counts.get("timeout", 0) + 1
                else:
                    status_counts["error"] = status_counts.get("error", 0) + 1

    def run_tier(self, concurrency: int) -> Dict[str, Any]:
        """Runs sustained load for duration_sec at given concurrency."""
        print(f"\n[LOAD TIER] Starting concurrency={concurrency}, duration={self.duration_sec}s, mode={self.mode}...")

        initial_sys = get_process_metrics()
        collector = get_metrics_collector()
        initial_collector = collector.get_metrics_snapshot()

        latencies: List[float] = []
        status_counts: Dict[str, int] = {}
        lock = threading.Lock()
        stop_event = threading.Event()

        start_time = time.perf_counter()

        # In mock mode, patch GeminiService to return realistic clinical text with 30ms simulation delay
        if self.mode == "mock":
            mock_generate = MagicMock()
            def mock_gen_answer(*args, **kwargs):
                time.sleep(0.03)  # 30ms simulated downstream LLM latency
                return {
                    "answer": "Metformin reduces hepatic gluconeogenesis and is first-line pharmacotherapy for type 2 diabetes.",
                    "model": "gemini-3.5-flash-lite",
                    "disclaimer": "Consult a healthcare provider.",
                    "generation_time_ms": 30.0,
                    "api_request_time_ms": 30.0,
                    "request_start_time": "",
                    "gemini_calls_count": 1,
                    "input_tokens": 120,
                    "output_tokens": 40,
                    "time_to_first_token_ms": None,
                    "status": "success"
                }
            mock_generate.side_effect = mock_gen_answer
            patcher = patch("backend.services.gemini_service.GeminiService.generate_answer", mock_generate)
            patcher.start()
        else:
            patcher = None

        try:
            threads = []
            for i in range(concurrency):
                t = threading.Thread(
                    target=self._worker_loop,
                    args=(i, stop_event, latencies, status_counts, lock),
                    daemon=True
                )
                threads.append(t)
                t.start()

            # Wait for duration
            time.sleep(self.duration_sec)
            stop_event.set()

            for t in threads:
                t.join(timeout=3.0)

        finally:
            if patcher:
                patcher.stop()

        actual_duration = time.perf_counter() - start_time
        final_sys = get_process_metrics()
        final_collector = collector.get_metrics_snapshot()

        total_reqs = len(latencies)
        success_reqs = status_counts.get("success", 0)
        rate_limited_reqs = status_counts.get("rate_limited", 0)
        error_reqs = status_counts.get("error", 0)
        timeout_reqs = status_counts.get("timeout", 0)

        rps = round(total_reqs / actual_duration, 2) if actual_duration > 0 else 0.0

        if latencies:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            p50 = round(sorted_lat[int(0.50 * (n - 1))], 2)
            p95 = round(sorted_lat[int(0.95 * (n - 1))], 2)
            p99 = round(sorted_lat[int(0.99 * (n - 1))], 2)
            avg_lat = round(sum(latencies) / n, 2)
        else:
            p50 = p95 = p99 = avg_lat = 0.0

        cache_hits_delta = final_collector["cache"]["hits"] - initial_collector["cache"]["hits"]
        cache_misses_delta = final_collector["cache"]["misses"] - initial_collector["cache"]["misses"]
        llm_calls_delta = final_collector["llm"]["calls"] - initial_collector["llm"]["calls"]

        result = {
            "concurrency": concurrency,
            "duration_sec": round(actual_duration, 1),
            "total_requests": total_reqs,
            "successful_requests": success_reqs,
            "rate_limited_requests": rate_limited_reqs,
            "error_requests": error_reqs,
            "timeout_requests": timeout_reqs,
            "throughput_rps": rps,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "avg_latency_ms": avg_lat,
            "cache_hits": cache_hits_delta,
            "cache_misses": cache_misses_delta,
            "llm_calls": llm_calls_delta,
            "initial_rss_mb": initial_sys["rss_mb"],
            "final_rss_mb": final_sys["rss_mb"],
            "rss_delta_mb": round(final_sys["rss_mb"] - initial_sys["rss_mb"], 2),
            "cpu_percent": final_sys["cpu_percent"]
        }

        print(f"  Throughput: {rps} req/sec ({total_reqs} total in {actual_duration:.1f}s)")
        print(f"  Latency: p50={p50}ms, p95={p95}ms, p99={p99}ms")
        print(f"  Status: {success_reqs} success, {rate_limited_reqs} rate-limited (429), {error_reqs} errors")
        print(f"  Memory: RSS={final_sys['rss_mb']} MB (delta: {result['rss_delta_mb']} MB)")
        return result

    def analyze_bottlenecks(self, tier_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes degradation points, saturation, and bottlenecks across tiers."""
        analysis = {
            "saturation_point": None,
            "concurrency_bottleneck": None,
            "rate_limit_bottleneck": None,
            "llm_bottleneck": None,
            "memory_bottleneck": None,
            "cpu_bottleneck": None,
            "degradation_begins_at": None,
            "verdict": "ANALYSIS_COMPLETE"
        }

        # Rate-limiting bottleneck
        rate_limited_tiers = [r for r in tier_results if r["rate_limited_requests"] > 0]
        if rate_limited_tiers:
            first_rl = rate_limited_tiers[0]
            analysis["rate_limit_bottleneck"] = (
                f"Rate limiting active starting at concurrency={first_rl['concurrency']} "
                f"({first_rl['rate_limited_requests']} requests throttled with HTTP 429)."
            )
            analysis["degradation_begins_at"] = f"Concurrency {first_rl['concurrency']}"
        else:
            analysis["rate_limit_bottleneck"] = "No rate-limit throttling observed at tested loads."

        # Concurrency / Latency Saturation
        for i in range(1, len(tier_results)):
            prev = tier_results[i - 1]
            curr = tier_results[i]
            # If p99 doubles or latency spikes > 3x
            if prev["p99_latency_ms"] > 0 and curr["p99_latency_ms"] > (prev["p99_latency_ms"] * 2.5):
                analysis["concurrency_bottleneck"] = (
                    f"Latency degradation detected between concurrency {prev['concurrency']} "
                    f"and {curr['concurrency']} (p99 increased from {prev['p99_latency_ms']}ms to {curr['p99_latency_ms']}ms)."
                )
                if not analysis["degradation_begins_at"]:
                    analysis["degradation_begins_at"] = f"Concurrency {curr['concurrency']}"
                break

        # Memory Bottleneck
        max_rss_delta = max((r["rss_delta_mb"] for r in tier_results), default=0.0)
        if max_rss_delta > 150.0:
            analysis["memory_bottleneck"] = f"Significant RSS growth detected ({max_rss_delta} MB delta)."
        else:
            analysis["memory_bottleneck"] = f"Memory remains stable and bounded (max delta: {max_rss_delta} MB)."

        # CPU Bottleneck
        max_cpu = max((r["cpu_percent"] for r in tier_results), default=0.0)
        if max_cpu > 90.0:
            analysis["cpu_bottleneck"] = f"High CPU utilization observed ({max_cpu}%)."
        else:
            analysis["cpu_bottleneck"] = f"CPU utilization within operational margins ({max_cpu}%)."

        # Saturation Point
        max_rps_tier = max(tier_results, key=lambda r: r["throughput_rps"])
        analysis["saturation_point"] = f"Max throughput achieved at concurrency {max_rps_tier['concurrency']} ({max_rps_tier['throughput_rps']} req/s)."

        return analysis

    def run_suite(self, concurrencies: List[int]) -> Dict[str, Any]:
        """Runs all specified concurrency tiers and saves detailed report."""
        tier_results = []
        for c in concurrencies:
            res = self.run_tier(c)
            tier_results.append(res)
            time.sleep(1.0)  # Brief cooldown between tiers

        analysis = self.analyze_bottlenecks(tier_results)

        full_report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": self.mode,
            "duration_per_tier_sec": self.duration_sec,
            "concurrency_tiers": concurrencies,
            "tier_results": tier_results,
            "analysis": analysis
        }

        out_path = Path(self.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)

        print(f"\n[REPORT SAVED] Full sustained load test results written to: {out_path.resolve()}")
        print(f"Saturation Point: {analysis['saturation_point']}")
        print(f"Degradation Begins At: {analysis['degradation_begins_at']}")
        print(f"Rate Limiting: {analysis['rate_limit_bottleneck']}")
        print(f"Memory: {analysis['memory_bottleneck']}")
        return full_report


def main():
    parser = argparse.ArgumentParser(description="AI Healthcare Agent Phase 3.6 Sustained Load Test")
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[10, 25, 50, 100],
        help="Concurrency levels to test (default: 10 25 50 100)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Sustained test duration per tier in seconds (default: 30)"
    )
    parser.add_argument(
        "--live-llm",
        action="store_true",
        default=False,
        help="Opt-in to real Gemini calls (MODE B). Default is Mock (MODE A)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_reports/load_test_phase3_6_results.json",
        help="Output JSON file path"
    )

    args = parser.parse_args()
    mode = "real" if (args.live_llm or os.getenv("RUN_REAL_LLM_LOAD_TEST", "").lower() in ("1", "true")) else "mock"

    if mode == "real":
        print("[WARNING] REAL GEMINI MODE ACTIVATED! Real API calls will be executed against Google GenAI.")
    else:
        print("[INFO] MOCK MODE ACTIVATED. Offline simulation with 30ms network delay. Safe for CI.")

    runner = SustainedLoadRunner(
        duration_sec=args.duration,
        mode=mode,
        output_file=args.output
    )
    runner.run_suite(args.concurrency)


if __name__ == "__main__":
    main()
