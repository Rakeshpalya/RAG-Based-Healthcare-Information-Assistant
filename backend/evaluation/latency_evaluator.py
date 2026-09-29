from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class StageLatencyStats:
    """Statistical summary of latencies for a specific pipeline stage in milliseconds."""
    stage_name: str
    sample_count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    @property
    def count(self) -> int:
        return self.sample_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "sample_count": self.sample_count,
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


# Type alias for Phase 5 consistency
LatencySummary = StageLatencyStats


@dataclass
class LatencyMetricsReport:
    """Comprehensive statistical report of system latency across RAG pipeline stages."""
    total_samples: int
    stages: Dict[str, StageLatencyStats] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "summary": self.summary
        }


PipelineLatencyBenchmark = LatencyMetricsReport


class LatencyEvaluator:
    """
    Lightweight deterministic latency and performance profiler (Phase 5.9).
    Calculates actual measured Mean, Median, P95, and P99 percentiles across RAG pipeline stages.
    """

    STAGES_TO_PROFILE = [
        ("embedding_time_ms", "Embedding Generation"),
        ("faiss_retrieval_time_ms", "FAISS Vector Search"),
        ("context_construction_time_ms", "Context Construction"),
        ("llm_generation_time_ms", "Gemini LLM Generation"),
        ("total_time_ms", "Total End-to-End Latency")
    ]

    @classmethod
    def compute_summary(cls, stage_name: str, values: List[float]) -> StageLatencyStats:
        """Computes statistical summary for a single named stage."""
        stats = cls.calculate_percentiles(values)
        return StageLatencyStats(
            stage_name=stage_name,
            sample_count=len(values),
            mean_ms=stats["mean"],
            median_ms=stats["median"],
            p95_ms=stats["p95"],
            p99_ms=stats["p99"],
            min_ms=stats["min"],
            max_ms=stats["max"]
        )

    @classmethod
    def calculate_percentiles(cls, values: List[float]) -> Dict[str, float]:
        """
        Calculates mean, median, p95, p99, min, max on a list of floating-point latencies.
        """
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0
            }

        arr = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }

    @classmethod
    def evaluate_latencies(cls, timing_records: List[Dict[str, Any]]) -> LatencyMetricsReport:
        """
        Processes an array of timing dictionaries extracted from RAG runs.
        """
        if not timing_records:
            return LatencyMetricsReport(total_samples=0, summary="No timing records provided.")

        stage_metrics: Dict[str, StageLatencyStats] = {}
        total_samples = len(timing_records)

        for key, display_name in cls.STAGES_TO_PROFILE:
            values = []
            for record in timing_records:
                # Support direct timing dictionary or nested dict
                t_dict = record.get("timings", record)
                if key in t_dict and t_dict[key] is not None:
                    try:
                        values.append(float(t_dict[key]))
                    except (ValueError, TypeError):
                        pass

            stats = cls.calculate_percentiles(values)
            stage_metrics[key] = StageLatencyStats(
                stage_name=display_name,
                sample_count=len(values),
                mean_ms=stats["mean"],
                median_ms=stats["median"],
                p95_ms=stats["p95"],
                p99_ms=stats["p99"],
                min_ms=stats["min"],
                max_ms=stats["max"]
            )

        tot_p95 = stage_metrics.get("total_time_ms", StageLatencyStats("", 0, 0, 0, 0, 0, 0, 0)).p95_ms
        tot_mean = stage_metrics.get("total_time_ms", StageLatencyStats("", 0, 0, 0, 0, 0, 0, 0)).mean_ms

        summary = (
            f"Evaluated {total_samples} RAG queries: "
            f"Mean E2E Latency = {tot_mean:.2f} ms | P95 E2E Latency = {tot_p95:.2f} ms"
        )

        return LatencyMetricsReport(
            total_samples=total_samples,
            stages=stage_metrics,
            summary=summary
        )
