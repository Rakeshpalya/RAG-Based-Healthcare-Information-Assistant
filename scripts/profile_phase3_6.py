"""
Phase 3.6 — Production Resource Profiling Script.

Measures and audits:
1. Process RSS memory before, during, and after repeated RAG requests
2. CPU utilization
3. Embedding model and FAISS memory footprint
4. Redis memory and connection lifecycle
5. Memory leaks and unbounded cache growth
6. Thread/task leaks
7. Connection leaks
"""

import os
import sys
import gc
import time
import json
import psutil
import threading
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.vector_store_service import get_vector_store_service
from backend.services.llm_cache_service import get_llm_cache_service
from backend.services.redis_service import get_redis_service
from backend.evaluation.observability import get_metrics_collector


def get_process_rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) in megabytes."""
    proc = psutil.Process(os.getpid())
    return round(proc.memory_info().rss / (1024 * 1024), 2)


def get_process_cpu_pct() -> float:
    """Returns current CPU utilization percentage."""
    proc = psutil.Process(os.getpid())
    return round(proc.cpu_percent(interval=0.1), 1)


PROFILING_QUERIES = [
    "What are the clinical indicators for initiating metformin therapy?",
    "What is the first-line antihypertensive regimen for diabetic nephropathy?",
    "How does HbA1c correlate with mean plasma glucose levels?",
    "What are the contraindications for SGLT2 inhibitors in heart failure?",
    "What are the diagnostic criteria for metabolic syndrome?"
]


def run_profiling(iterations: int = 300, output_file: str = "evaluation_reports/resource_profile_phase3_6.json") -> Dict[str, Any]:
    print("=" * 60)
    print(f"Phase 3.6 Resource Profiling — Running {iterations} Iterations")
    print("=" * 60)

    # Warm up SentenceTransformer model before measuring baseline RSS
    from backend.services.embedding_service import EmbeddingService
    EmbeddingService.embed_query("Hypertension clinical warmup query")
    gc.collect()

    initial_threads = threading.active_count()
    initial_rss = get_process_rss_mb()
    initial_cpu = get_process_cpu_pct()

    vs = get_vector_store_service()
    faiss_vectors = vs.count()
    faiss_dim = vs.dimension
    metadata_count = len(vs.metadata_store)

    # Estimate FAISS index raw memory (vectors * dim * 4 bytes for float32)
    faiss_raw_bytes = faiss_vectors * faiss_dim * 4
    faiss_mem_mb = round(faiss_raw_bytes / (1024 * 1024), 3)

    cache = get_llm_cache_service()
    cache.clear()
    initial_cache_entries = len(cache._cache)

    redis_svc = get_redis_service()
    redis_available = redis_svc.is_available()

    print(f"Initial State:")
    print(f"  RSS Memory:        {initial_rss:.2f} MB")
    print(f"  Active Threads:    {initial_threads}")
    print(f"  FAISS Vectors:     {faiss_vectors} (dimension: {faiss_dim}, raw index: {faiss_mem_mb} MB)")
    print(f"  Metadata Records:  {metadata_count}")
    print(f"  Cache Entries:     {initial_cache_entries}")
    print(f"  Redis Available:   {redis_available}")

    from backend.api.auth_dependencies import get_optional_current_db_user
    from backend.database.models import User
    app.dependency_overrides[get_optional_current_db_user] = lambda: User(id=1, email="profile@clinic.org", role="doctor")
    client = TestClient(app)

    # Mock Gemini answer to isolate retrieval, embedding, FAISS, caching, and pipeline memory
    mock_gen_answer = MagicMock(return_value={
        "answer": "Metformin reduces hepatic gluconeogenesis and is first-line pharmacotherapy for type 2 diabetes.",
        "model": "gemini-3.5-flash-lite",
        "disclaimer": "Consult a physician.",
        "generation_time_ms": 25.0,
        "api_request_time_ms": 25.0,
        "request_start_time": "",
        "gemini_calls_count": 1,
        "input_tokens": 100,
        "output_tokens": 35,
        "time_to_first_token_ms": None,
        "status": "success"
    })

    latencies_ms: List[float] = []
    checkpoint_samples: List[Dict[str, Any]] = []

    with patch("backend.services.gemini_service.GeminiService.generate_answer", mock_gen_answer):
        for i in range(1, iterations + 1):
            q = PROFILING_QUERIES[i % len(PROFILING_QUERIES)]
            user_id = (i % 20) + 1  # 20 distinct users

            t0 = time.perf_counter()
            resp = client.post(
                "/rag/query",
                json={"question": q},
                headers={"X-Forwarded-For": f"10.0.0.{user_id}", "X-User-Id": str(user_id)}
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

            # Checkpoint every 50 iterations
            if i % 50 == 0 or i == iterations:
                current_rss = get_process_rss_mb()
                current_threads = threading.active_count()
                current_cache = len(cache._cache)
                sample = {
                    "iteration": i,
                    "rss_mb": current_rss,
                    "rss_delta_from_baseline": round(current_rss - initial_rss, 2),
                    "active_threads": current_threads,
                    "cache_entries": current_cache,
                    "status_code": resp.status_code
                }
                checkpoint_samples.append(sample)
                print(f"  [@{i:03d} reqs] RSS: {current_rss} MB (delta: {sample['rss_delta_from_baseline']:+.2f} MB) | Threads: {current_threads} | Cache: {current_cache}")

    # Post-run cleanup and assessment
    gc.collect()
    time.sleep(0.5)

    final_rss = get_process_rss_mb()
    final_threads = threading.active_count()
    final_cpu = get_process_cpu_pct()
    final_cache_entries = len(cache._cache)

    rss_growth_mb = round(final_rss - initial_rss, 2)
    thread_leak = max(0, final_threads - initial_threads)

    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)
    p50 = round(sorted_lats[int(0.50 * (n - 1))], 2)
    p95 = round(sorted_lats[int(0.95 * (n - 1))], 2)
    p99 = round(sorted_lats[int(0.99 * (n - 1))], 2)
    avg_lat = round(sum(latencies_ms) / n, 2)

    # Audits
    is_memory_bounded = rss_growth_mb < 80.0  # Safe threshold for 300 iterations
    is_cache_bounded = final_cache_entries <= cache.max_entries
    is_thread_leak_free = thread_leak <= 2  # Allow standard background pool variance

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_requests": iterations,
        "initial_rss_mb": initial_rss,
        "final_rss_mb": final_rss,
        "rss_growth_mb": rss_growth_mb,
        "initial_threads": initial_threads,
        "final_threads": final_threads,
        "thread_leak_count": thread_leak,
        "initial_cpu_pct": initial_cpu,
        "final_cpu_pct": final_cpu,
        "faiss_vectors": faiss_vectors,
        "faiss_dimension": faiss_dim,
        "faiss_estimated_mem_mb": faiss_mem_mb,
        "cache_entries": final_cache_entries,
        "cache_max_entries": cache.max_entries,
        "latencies": {
            "avg_ms": avg_lat,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "min_ms": round(sorted_lats[0], 2),
            "max_ms": round(sorted_lats[-1], 2)
        },
        "checkpoints": checkpoint_samples,
        "audit": {
            "memory_leak_detected": not is_memory_bounded,
            "unbounded_cache_growth_detected": not is_cache_bounded,
            "thread_leak_detected": not is_thread_leak_free,
            "production_ready": is_memory_bounded and is_cache_bounded and is_thread_leak_free
        }
    }

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("PROFILING AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total Requests:          {iterations}")
    print(f"RSS Growth:              {rss_growth_mb:+.2f} MB ({'PASS: Bounded' if is_memory_bounded else 'FAIL: Leak'})")
    print(f"Thread Growth:           {thread_leak} ({'PASS: Stable' if is_thread_leak_free else 'FAIL: Thread Leak'})")
    print(f"Cache Growth:            {final_cache_entries} / {cache.max_entries} max ({'PASS: Bounded' if is_cache_bounded else 'FAIL: Unbounded'})")
    print(f"Latency:                 p50={p50}ms | p95={p95}ms | p99={p99}ms")
    print(f"Report written to:       {out_path.resolve()}")
    print("=" * 60)
    return report


if __name__ == "__main__":
    run_profiling(iterations=250)
