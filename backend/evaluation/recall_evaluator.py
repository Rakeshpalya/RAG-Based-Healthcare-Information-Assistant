"""
Recall Evaluator Module (Phase 2E.1 — Retrieval Recall Evaluation).

Provides quantitative, concept-grounded recall evaluation for retrieval pipelines.
Evaluates:
  - Recall@1
  - Recall@3
  - Recall@5
  - Recall@10

Validates that retrieved chunks actually contain the expected clinical concepts/evidence,
rather than declaring relevance based merely on generic topic keyword overlap.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Union


# ==============================================================================
# Benchmark Data Structures
# ==============================================================================

@dataclass
class BenchmarkQuery:
    """Specification of an evaluation benchmark query with expected concepts."""
    query: str
    expected_concepts: List[str]
    expected_chunk_ids: Optional[List[str]] = None
    expected_document: Optional[str] = None
    topic: Optional[str] = "General"


@dataclass
class QueryRecallResult:
    """Evaluation result for an individual query."""
    query: str
    expected_concepts: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    recall_at_k: Dict[int, float]
    matched_concepts: List[str]
    missing_concepts: List[str]
    concept_coverage: float
    retrieval_status: str
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    retrieved_documents: List[str] = field(default_factory=list)
    similarity_scores: List[float] = field(default_factory=list)
    retrieval_time_ms: float = 0.0


@dataclass
class AggregateRecallMetrics:
    """Aggregated evaluation metrics across a benchmark suite."""
    total_queries: int
    successful_queries: int
    failed_queries: int
    k_values: List[int]
    mean_recall_at_k: Dict[int, float]
    query_results: List[QueryRecallResult]
    average_retrieval_time_ms: float = 0.0


# ==============================================================================
# Canonical Phase 2E.1 Benchmark Dataset
# ==============================================================================

HYPERTENSION_LIFESTYLE_BENCHMARK: List[Dict[str, Any]] = [
    {
        "query": "What lifestyle changes help hypertension?",
        "expected_concepts": [
            "regular physical activity",
            "healthy weight",
            "balanced diet",
            "sodium",
            "tobacco",
            "alcohol",
            "sleep"
        ],
        "expected_document": "synthetic_hypertension_test.pdf",
        "expected_chunk_ids": ["chunk_0"]
    },
    {
        "query": "What lifestyle changes help high blood pressure?",
        "expected_concepts": [
            "regular physical activity",
            "healthy weight",
            "balanced diet",
            "sodium",
            "tobacco",
            "alcohol",
            "sleep"
        ],
        "expected_document": "synthetic_hypertension_test.pdf",
        "expected_chunk_ids": ["chunk_0"]
    },
    {
        "query": "How can lifestyle help control blood pressure?",
        "expected_concepts": [
            "regular physical activity",
            "healthy weight",
            "balanced diet",
            "sodium",
            "tobacco",
            "alcohol",
            "sleep"
        ],
        "expected_document": "synthetic_hypertension_test.pdf",
        "expected_chunk_ids": ["chunk_0"]
    },
    {
        "query": "What non-medication measures help manage hypertension?",
        "expected_concepts": [
            "regular physical activity",
            "healthy weight",
            "balanced diet",
            "sodium",
            "tobacco",
            "alcohol",
            "sleep"
        ],
        "expected_document": "synthetic_hypertension_test.pdf",
        "expected_chunk_ids": ["chunk_0"]
    }
]


# ==============================================================================
# Metric Computation Functions
# ==============================================================================

def verify_concept_in_text(concept: str, text: str) -> bool:
    """
    Determines whether a clinical concept phrase or keyword exists in text.
    Handles case-insensitivity, whitespace normalization, and flexible phrase matches
    (e.g., 'healthy weight' matches 'maintaining a healthy weight',
           'balanced diet' matches 'choosing a balanced diet',
           'regular physical activity' matches 'regular physical activity').
    """
    if not concept or not text:
        return False

    c_norm = concept.strip().lower()
    t_norm = text.lower()

    # Exact normalized substring
    if c_norm in t_norm:
        return True

    # Word boundary match for short single-word concepts (e.g., 'sleep', 'sodium', 'tobacco')
    words = c_norm.split()
    if len(words) == 1:
        pattern = rf"\b{re.escape(c_norm)}\b"
        return bool(re.search(pattern, t_norm))

    # All constituent key tokens present in reasonable proximity
    token_pattern = r".*?".join(re.escape(w) for w in words)
    return bool(re.search(token_pattern, t_norm))


def compute_concept_recall_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    expected_concepts: List[str],
    k: int
) -> float:
    """
    Computes Concept Recall@K:
      (number of expected concepts present in top-K retrieved chunks) / (total expected concepts)

    Args:
        retrieved_chunks: Ranked list of chunk dictionaries from retrieval.
        expected_concepts: List of concept strings that constitute ground-truth evidence.
        k: Rank cutoff (e.g. 1, 3, 5, 10).

    Returns:
        float in range [0.0, 1.0].
    """
    if k <= 0 or not expected_concepts or not retrieved_chunks:
        return 0.0

    top_k_chunks = retrieved_chunks[:k]
    # Concatenate unique texts to prevent duplicate chunk vector inflation
    seen_texts: Set[str] = set()
    combined_texts = []
    for c in top_k_chunks:
        txt = (c.get("text") or "").strip()
        if txt and txt not in seen_texts:
            seen_texts.add(txt)
            combined_texts.append(txt)

    merged_text = "\n".join(combined_texts)
    if not merged_text:
        return 0.0

    matched_count = sum(
        1 for concept in expected_concepts
        if verify_concept_in_text(concept, merged_text)
    )

    return float(matched_count) / float(len(expected_concepts))


def compute_chunk_recall_at_k(
    retrieved_chunk_ids: List[str],
    expected_chunk_ids: List[str],
    k: int
) -> float:
    """
    Computes Chunk Recall@K:
      (number of expected chunks retrieved in top K) / (total expected chunks).

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        expected_chunk_ids: Expected target chunk IDs.
        k: Rank cutoff.

    Returns:
        float in range [0.0, 1.0].
    """
    if k <= 0 or not expected_chunk_ids or not retrieved_chunk_ids:
        return 0.0

    top_k = retrieved_chunk_ids[:k]
    expected_set = set(expected_chunk_ids)
    matched = sum(1 for cid in top_k if cid in expected_set)
    return float(matched) / float(len(expected_set))


# ==============================================================================
# RecallEvaluator Class
# ==============================================================================

class RecallEvaluator:
    """
    Reusable evaluation engine for measuring retrieval recall across different K thresholds
    (Recall@1, Recall@3, Recall@5, Recall@10).
    """

    DEFAULT_K_VALUES: List[int] = [1, 3, 5, 10]

    def __init__(
        self,
        rag_service: Optional[Any] = None,
        k_values: Optional[List[int]] = None,
        default_user_id: Optional[int] = None
    ):
        """
        Initializes the RecallEvaluator.

        Args:
            rag_service: Optional RAGService instance. If None, instantiates RAGService
                         using the shared VectorStoreService singleton (with persisted index).
            k_values: List of K cutoffs to evaluate (defaults to [1, 3, 5, 10]).
            default_user_id: Optional user ID for ownership isolation.
        """
        if rag_service is None:
            from backend.services.vector_store_service import get_vector_store_service
            from backend.rag.rag_service import RAGService
            vs = get_vector_store_service()
            self.rag_service = RAGService(vector_store=vs)
        else:
            self.rag_service = rag_service

        self.k_values = sorted(list(set(k_values or self.DEFAULT_K_VALUES)))
        # Validate K values
        self.k_values = [k for k in self.k_values if k > 0]
        if not self.k_values:
            self.k_values = [1, 3, 5, 10]

        self.default_user_id = default_user_id

    def evaluate_query(
        self,
        query: Union[str, Dict[str, Any], BenchmarkQuery],
        expected_concepts: Optional[List[str]] = None,
        similarity_threshold: Optional[float] = None,
        user_id: Optional[int] = None,
        pre_retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> QueryRecallResult:
        """
        Evaluates retrieval recall for a single query.

        Args:
            query: Query string, dictionary, or BenchmarkQuery object.
            expected_concepts: Optional explicit list of concepts (overrides dict/object concepts).
            similarity_threshold: Optional custom similarity threshold.
            user_id: Optional user ID for ownership isolation (falls back to default_user_id).
            pre_retrieved_chunks: Optional pre-retrieved chunks (avoids calling RAGService if provided).

        Returns:
            QueryRecallResult dataclass with Recall@K and concept match details.
        """
        if isinstance(query, BenchmarkQuery):
            q_str = query.query
            concepts = expected_concepts or query.expected_concepts
        elif isinstance(query, dict):
            q_str = query.get("query") or query.get("question", "")
            concepts = expected_concepts or query.get("expected_concepts", [])
        else:
            q_str = str(query)
            concepts = expected_concepts or []

        active_user_id = user_id if user_id is not None else self.default_user_id
        max_k = max(self.k_values)

        t_start = time.perf_counter()
        if pre_retrieved_chunks is not None:
            retrieved_chunks = pre_retrieved_chunks
            status = "success" if retrieved_chunks else "no_relevant_context"
        else:
            rag_res = self.rag_service.query(
                question=q_str,
                top_k=max_k,
                similarity_threshold=similarity_threshold,
                user_id=active_user_id
            )
            retrieved_chunks = rag_res.get("retrieved_chunks", [])
            status = rag_res.get("retrieval_status", "success")

        t_elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        retrieved_ids = [str(c.get("chunk_id", "")) for c in retrieved_chunks]
        retrieved_docs = [
            str(c.get("metadata", {}).get("filename") or c.get("document_id") or "UNKNOWN")
            for c in retrieved_chunks
        ]
        similarity_scores = [round(float(c.get("similarity_score", 0.0)), 4) for c in retrieved_chunks]

        # Compute Recall@K for each requested K
        recall_at_k: Dict[int, float] = {}
        for k in self.k_values:
            recall_at_k[k] = compute_concept_recall_at_k(retrieved_chunks, concepts, k)

        # Track concept matching across all retrieved chunks
        combined_text = "\n".join((c.get("text") or "") for c in retrieved_chunks)
        matched_concepts = []
        missing_concepts = []
        for concept in concepts:
            if verify_concept_in_text(concept, combined_text):
                matched_concepts.append(concept)
            else:
                missing_concepts.append(concept)

        concept_cov = float(len(matched_concepts)) / float(len(concepts)) if concepts else 0.0

        return QueryRecallResult(
            query=q_str,
            expected_concepts=concepts,
            retrieved_chunks=retrieved_chunks,
            recall_at_k=recall_at_k,
            matched_concepts=matched_concepts,
            missing_concepts=missing_concepts,
            concept_coverage=concept_cov,
            retrieval_status=status,
            retrieved_chunk_ids=retrieved_ids,
            retrieved_documents=retrieved_docs,
            similarity_scores=similarity_scores,
            retrieval_time_ms=t_elapsed_ms
        )

    def evaluate_benchmark(
        self,
        benchmark: Optional[List[Union[Dict[str, Any], BenchmarkQuery]]] = None,
        similarity_threshold: Optional[float] = None,
        user_id: Optional[int] = None
    ) -> AggregateRecallMetrics:
        """
        Evaluates a complete benchmark suite and computes aggregate metrics.

        Args:
            benchmark: List of benchmark records (defaults to HYPERTENSION_LIFESTYLE_BENCHMARK).
            similarity_threshold: Optional threshold override.
            user_id: Optional user ID for ownership isolation.

        Returns:
            AggregateRecallMetrics with mean recall across all queries.
        """
        suite = benchmark or HYPERTENSION_LIFESTYLE_BENCHMARK
        results: List[QueryRecallResult] = []
        success_count = 0
        failed_count = 0

        for item in suite:
            res = self.evaluate_query(
                query=item,
                similarity_threshold=similarity_threshold,
                user_id=user_id
            )
            results.append(res)
            # A query succeeded if retrieval_status was success AND at least one chunk retrieved
            if res.retrieval_status == "success" and len(res.retrieved_chunks) > 0:
                success_count += 1
            else:
                failed_count += 1

        n = len(results)
        if n == 0:
            return AggregateRecallMetrics(
                total_queries=0,
                successful_queries=0,
                failed_queries=0,
                k_values=self.k_values,
                mean_recall_at_k={k: 0.0 for k in self.k_values},
                query_results=[]
            )

        mean_recall = {
            k: round(sum(r.recall_at_k.get(k, 0.0) for r in results) / float(n), 4)
            for k in self.k_values
        }

        avg_time = round(sum(r.retrieval_time_ms for r in results) / float(n), 2)

        return AggregateRecallMetrics(
            total_queries=n,
            successful_queries=success_count,
            failed_queries=failed_count,
            k_values=self.k_values,
            mean_recall_at_k=mean_recall,
            query_results=results,
            average_retrieval_time_ms=avg_time
        )

    def format_query_report(self, result: QueryRecallResult) -> str:
        """
        Formats an individual query result into the standard evaluation report format.
        """
        lines = [
            "Query:",
            result.query,
            ""
        ]

        for k in self.k_values:
            val = result.recall_at_k.get(k, 0.0)
            lines.append(f"Recall@{k}: {val:.4f} ({val * 100:.1f}%)")

        lines.append("")
        lines.append("Retrieved chunks:")
        if result.retrieved_chunks:
            for idx, c in enumerate(result.retrieved_chunks):
                cid = c.get("chunk_id", "UNKNOWN")
                doc = c.get("metadata", {}).get("filename") or c.get("document_id") or "UNKNOWN"
                score = c.get("similarity_score", 0.0)
                lines.append(f"- {cid} (Doc: {doc}, Score: {score:.4f})")
        else:
            lines.append("- (none / no_relevant_context)")

        lines.append("")
        lines.append("Expected concepts:")
        combined_text = "\n".join((c.get("text") or "") for c in result.retrieved_chunks)
        for concept in result.expected_concepts:
            is_present = verify_concept_in_text(concept, combined_text)
            marker = "✓" if is_present else "✗"
            lines.append(f"{marker} {concept}")

        return "\n".join(lines)

    def format_benchmark_report(self, metrics: AggregateRecallMetrics) -> str:
        """
        Produces a comprehensive, clean evaluation report covering all queries
        and overall aggregate metrics.
        """
        header = [
            "Retrieval Recall Evaluation",
            "============================",
            f"Total Queries Evaluated: {metrics.total_queries}",
            f"Successful Retrievals:   {metrics.successful_queries}",
            f"Failed Retrievals:       {metrics.failed_queries}",
            f"Average Latency:         {metrics.average_retrieval_time_ms:.1f} ms",
            ""
        ]

        query_sections = []
        for r in metrics.query_results:
            query_sections.append(self.format_query_report(r))
            query_sections.append("-" * 40)

        summary_table = [
            "",
            "Aggregate Recall Baseline Summary",
            "==================================",
        ]
        for k in metrics.k_values:
            score = metrics.mean_recall_at_k.get(k, 0.0)
            summary_table.append(f"Mean Recall@{k:<2}: {score:.4f} ({score * 100:.1f}%)")

        return "\n".join(header + query_sections + summary_table)
