import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field


def compute_precision_at_k(
    retrieved_chunk_ids: List[str],
    expected_chunk_ids: List[str],
    k: int
) -> float:
    """
    Computes Precision@K: (number of relevant retrieved chunks in top K) / K.

    Precision measures what fraction of the retrieved top-K chunks are truly relevant.
    Notice the denominator is strictly K (or max(1, k)), avoiding artificial metric inflation.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs from vector search.
        expected_chunk_ids: Ground-truth list of relevant chunk IDs.
        k: The rank threshold cutoff (e.g. 1, 3, 5).

    Returns:
        float in range [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    top_k_retrieved = retrieved_chunk_ids[:k]
    if not top_k_retrieved:
        return 0.0

    expected_set: Set[str] = set(expected_chunk_ids)
    relevant_count = sum(1 for cid in top_k_retrieved if cid in expected_set)
    return float(relevant_count) / float(k)


def compute_recall_at_k(
    retrieved_chunk_ids: List[str],
    expected_chunk_ids: List[str],
    k: int
) -> float:
    """
    Computes Recall@K: (number of relevant retrieved chunks in top K) / (number of expected relevant chunks).

    Recall measures what fraction of all expected relevant chunks were successfully retrieved in top K.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs from vector search.
        expected_chunk_ids: Ground-truth list of relevant chunk IDs.
        k: The rank threshold cutoff (e.g. 1, 3, 5).

    Returns:
        float in range [0.0, 1.0]. Returns 1.0 if expected is empty and none retrieved, else 0.0 if expected empty.
    """
    if not expected_chunk_ids:
        return 0.0  # Cannot compute recall on query with no expected ground-truth chunks
    if k <= 0:
        return 0.0

    top_k_retrieved = retrieved_chunk_ids[:k]
    expected_set: Set[str] = set(expected_chunk_ids)
    relevant_count = sum(1 for cid in top_k_retrieved if cid in expected_set)
    return float(relevant_count) / float(len(expected_set))


def compute_mrr_at_k(
    retrieved_chunk_ids: List[str],
    expected_chunk_ids: List[str],
    k: int
) -> float:
    """
    Computes Mean Reciprocal Rank (MRR@K): 1 / rank of first relevant result.

    Rank is 1-indexed. If no relevant result appears within the top K: MRR = 0.0.

    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs from vector search.
        expected_chunk_ids: Ground-truth list of relevant chunk IDs.
        k: The rank threshold cutoff (e.g. 1, 3, 5).

    Returns:
        float in range [0.0, 1.0].
    """
    if k <= 0 or not expected_chunk_ids or not retrieved_chunk_ids:
        return 0.0

    expected_set: Set[str] = set(expected_chunk_ids)
    for rank_idx, cid in enumerate(retrieved_chunk_ids[:k], start=1):
        if cid in expected_set:
            return 1.0 / float(rank_idx)

    return 0.0


@dataclass
class QueryEvaluationResult:
    """Metrics for an individual evaluation query."""
    query_id: str
    question: str
    topic: str
    expected_chunk_ids: List[str]
    retrieved_chunk_ids: List[str]
    similarity_scores: List[float]
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr_at_k: Dict[int, float] = field(default_factory=dict)
    is_first_result_relevant: bool = False
    embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0
    total_retrieval_time_ms: float = 0.0


@dataclass
class AggregateRetrievalMetrics:
    """Summary metrics aggregated across all evaluation queries."""
    total_queries: int
    k_values: List[int]
    mean_precision_at_k: Dict[int, float] = field(default_factory=dict)
    mean_recall_at_k: Dict[int, float] = field(default_factory=dict)
    mean_mrr_at_k: Dict[int, float] = field(default_factory=dict)
    average_embedding_time_ms: float = 0.0
    average_search_time_ms: float = 0.0
    average_total_retrieval_time_ms: float = 0.0
    failed_queries: List[QueryEvaluationResult] = field(default_factory=list)


class RetrievalEvaluator:
    """
    Executes quantitative retrieval evaluations against a RAGService instance.
    Evaluates Precision@K, Recall@K, and MRR for K in [1, 3, 5].
    """

    def __init__(self, rag_service: Optional[Any] = None, k_values: Optional[List[int]] = None):
        if rag_service is None:
            from backend.rag.rag_service import RAGService
            self.rag_service = RAGService()
        else:
            self.rag_service = rag_service
        self.k_values = k_values or [1, 3, 5]

    def evaluate_query(
        self,
        query_record: Dict[str, Any],
        similarity_threshold: Optional[float] = None
    ) -> QueryEvaluationResult:
        """
        Evaluates a single query record against the RAGService.
        """
        query_id = query_record.get("id", "UNKNOWN")
        question = query_record.get("question", "")
        topic = query_record.get("topic", "General")
        expected_chunk_ids = query_record.get("expected_chunk_ids", [])

        # Request maximum K needed
        max_k = max(self.k_values)

        t_start = time.perf_counter()
        rag_response = self.rag_service.query(
            question=question,
            top_k=max_k,
            similarity_threshold=similarity_threshold
        )
        t_end = time.perf_counter()

        retrieved_chunks = rag_response.get("retrieved_chunks", [])
        retrieved_chunk_ids = [c["chunk_id"] for c in retrieved_chunks]
        similarity_scores = [c.get("similarity_score", 0.0) for c in retrieved_chunks]

        # Calculate metrics for each K
        p_at_k = {}
        r_at_k = {}
        mrr_at_k = {}

        for k in self.k_values:
            p_at_k[k] = compute_precision_at_k(retrieved_chunk_ids, expected_chunk_ids, k)
            r_at_k[k] = compute_recall_at_k(retrieved_chunk_ids, expected_chunk_ids, k)
            mrr_at_k[k] = compute_mrr_at_k(retrieved_chunk_ids, expected_chunk_ids, k)

        is_top1_relevant = (
            len(retrieved_chunk_ids) > 0 and
            retrieved_chunk_ids[0] in set(expected_chunk_ids)
        )

        timings = rag_response.get("timings", {})
        emb_time = timings.get("query_embedding_time_ms", 0.0)
        search_time = timings.get("vector_search_time_ms", 0.0)
        tot_time = timings.get("total_retrieval_time_ms", (t_end - t_start) * 1000.0)

        return QueryEvaluationResult(
            query_id=query_id,
            question=question,
            topic=topic,
            expected_chunk_ids=expected_chunk_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            similarity_scores=similarity_scores,
            precision_at_k=p_at_k,
            recall_at_k=r_at_k,
            mrr_at_k=mrr_at_k,
            is_first_result_relevant=is_top1_relevant,
            embedding_time_ms=emb_time,
            search_time_ms=search_time,
            total_retrieval_time_ms=tot_time
        )

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        similarity_threshold: Optional[float] = None
    ) -> AggregateRetrievalMetrics:
        """
        Evaluates a complete dataset of questions and returns aggregate metrics.
        """
        results: List[QueryEvaluationResult] = []
        failed: List[QueryEvaluationResult] = []

        for record in dataset:
            res = self.evaluate_query(record, similarity_threshold=similarity_threshold)
            results.append(res)
            # A query is considered failing if rank 1 was not relevant
            if not res.is_first_result_relevant:
                failed.append(res)

        n = len(results)
        if n == 0:
            return AggregateRetrievalMetrics(total_queries=0, k_values=self.k_values)

        # Compute aggregate averages
        mean_p = {k: sum(r.precision_at_k[k] for r in results) / n for k in self.k_values}
        mean_r = {k: sum(r.recall_at_k[k] for r in results) / n for k in self.k_values}
        mean_mrr = {k: sum(r.mrr_at_k[k] for r in results) / n for k in self.k_values}

        avg_emb = sum(r.embedding_time_ms for r in results) / n
        avg_search = sum(r.search_time_ms for r in results) / n
        avg_total = sum(r.total_retrieval_time_ms for r in results) / n

        return AggregateRetrievalMetrics(
            total_queries=n,
            k_values=self.k_values,
            mean_precision_at_k=mean_p,
            mean_recall_at_k=mean_r,
            mean_mrr_at_k=mean_mrr,
            average_embedding_time_ms=avg_emb,
            average_search_time_ms=avg_search,
            average_total_retrieval_time_ms=avg_total,
            failed_queries=failed
        )

    def run_threshold_experiment(
        self,
        dataset: List[Dict[str, Any]],
        thresholds: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates retrieval performance across multiple similarity thresholds.
        Default: [0.20, 0.25, 0.30, 0.35, 0.40].
        """
        sweep_thresholds = thresholds or [0.20, 0.25, 0.30, 0.35, 0.40]
        experiment_results = []

        for thresh in sweep_thresholds:
            metrics = self.evaluate_dataset(dataset, similarity_threshold=thresh)
            experiment_results.append({
                "threshold": thresh,
                "precision_at_1": metrics.mean_precision_at_k.get(1, 0.0),
                "precision_at_3": metrics.mean_precision_at_k.get(3, 0.0),
                "precision_at_5": metrics.mean_precision_at_k.get(5, 0.0),
                "recall_at_1": metrics.mean_recall_at_k.get(1, 0.0),
                "recall_at_3": metrics.mean_recall_at_k.get(3, 0.0),
                "recall_at_5": metrics.mean_recall_at_k.get(5, 0.0),
                "mrr_at_5": metrics.mean_mrr_at_k.get(5, 0.0),
                "failed_count": len(metrics.failed_queries)
            })

        return experiment_results

    def evaluate_all(
        self,
        test_cases: List[Dict[str, Any]],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Lightweight batch evaluator for pre-retrieved test cases or query records.
        """
        precisions = []
        recalls = []
        mrrs = []
        for case in test_cases:
            exp = case.get("expected_chunk_ids", [])
            ret = case.get("retrieved_chunk_ids")
            if ret is None:
                q = case.get("query") or case.get("question", "")
                rag_res = self.rag_service.query(question=q, top_k=k)
                ret = [c.get("chunk_id", "") for c in rag_res.get("retrieved_chunks", [])]
            p = compute_precision_at_k(ret, exp, k)
            r = compute_recall_at_k(ret, exp, k)
            m = compute_mrr_at_k(ret, exp, k)
            precisions.append(p)
            recalls.append(r)
            mrrs.append(m)

        n = len(test_cases)
        return {
            "total_queries": n,
            "mean_precision": sum(precisions) / n if n > 0 else 0.0,
            "mean_recall": sum(recalls) / n if n > 0 else 0.0,
            "mean_mrr": sum(mrrs) / n if n > 0 else 0.0,
            "k": k
        }


# ==============================================================================
# Phase 5: Hit Rate, Context Sufficiency & Structured Retrieval Evaluator
# ==============================================================================

def compute_hit_rate_at_k(
    retrieved_chunk_ids: List[str],
    expected_chunk_ids: List[str],
    k: int
) -> float:
    """
    Computes Hit Rate@K: 1.0 if at least one expected chunk appears in the top K retrieved chunks, else 0.0.
    If expected_chunk_ids is empty, returns 1.0 if none retrieved in top K, else 0.0.

    Args:
        retrieved_chunk_ids: List of retrieved chunk ID strings.
        expected_chunk_ids: List of ground-truth relevant chunk ID strings.
        k: Rank threshold cutoff.

    Returns:
        float in {0.0, 1.0}.
    """
    if not expected_chunk_ids:
        return 1.0 if not retrieved_chunk_ids[:k] else 0.0
    if k <= 0 or not retrieved_chunk_ids:
        return 0.0

    top_k_set = set(retrieved_chunk_ids[:k])
    return 1.0 if any(cid in top_k_set for cid in expected_chunk_ids) else 0.0


def evaluate_context_sufficiency(
    retrieved_chunks: List[Dict[str, Any]],
    expected_chunk_ids: List[str],
    min_similarity_threshold: float = 0.25
) -> bool:
    """
    Evaluates whether the retrieved context contains sufficient, relevant grounded evidence.

    Returns True if:
    - If expected_chunk_ids is empty: True if no chunks or all chunks below threshold.
    - If expected_chunk_ids is non-empty: at least one expected chunk is present and meets min_similarity_threshold.
    """
    if not expected_chunk_ids:
        # For out-of-domain queries, context is sufficient (no hallucinated context) if no high-similarity chunks returned
        high_scoring = [c for c in retrieved_chunks if float(c.get("similarity_score", 0.0)) >= min_similarity_threshold]
        return len(high_scoring) == 0

    expected_set = set(expected_chunk_ids)
    matching_chunks = [
        c for c in retrieved_chunks
        if (c.get("chunk_id") in expected_set or c.get("document_id") in expected_set)
        and float(c.get("similarity_score", 0.0)) >= min_similarity_threshold
    ]
    return len(matching_chunks) > 0


@dataclass
class StructuredRetrievalEvaluation:
    """Standardized result of a deterministic retrieval evaluation."""
    query: str
    k: int
    retrieved_chunks: List[Dict[str, Any]]
    relevant_chunks: List[str]
    expected_chunk_ids: List[str]
    retrieval_precision: float
    retrieval_recall: float
    hit_rate: float
    mrr: float
    context_sufficiency: bool
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def precision_at_k(self) -> float:
        return self.retrieval_precision

    @property
    def recall_at_k(self) -> float:
        return self.retrieval_recall

    @property
    def hit_rate_at_k(self) -> float:
        return self.hit_rate

    @property
    def mrr_at_k(self) -> float:
        return self.mrr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "k": self.k,
            "retrieved_chunks": [
                {"chunk_id": str(c.get("chunk_id", "")), "similarity_score": round(float(c.get("similarity_score", 0.0)), 4)}
                for c in self.retrieved_chunks[:self.k]
            ],
            "relevant_chunks": self.relevant_chunks,
            "expected_chunk_ids": self.expected_chunk_ids,
            "retrieval_precision": round(self.retrieval_precision, 4),
            "retrieval_recall": round(self.retrieval_recall, 4),
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "context_sufficiency": self.context_sufficiency,
            "passed": self.passed,
            "details": self.details
        }


def evaluate_retrieval(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    expected_chunk_ids: List[str],
    k: int = 5,
    min_similarity_threshold: float = 0.25
) -> StructuredRetrievalEvaluation:
    """
    Executes deterministic evaluation of a single retrieval result against ground-truth expectations.

    Args:
        query: User clinical query.
        retrieved_chunks: List of chunk dictionaries returned by vector search.
        expected_chunk_ids: Ground-truth list of relevant chunk IDs (or document IDs).
        k: Cutoff rank for evaluation (default: 5).
        min_similarity_threshold: Relevance threshold for sufficiency.

    Returns:
        StructuredRetrievalEvaluation instance with all calculated metrics.
    """
    retrieved_ids = [str(c.get("chunk_id", "")) for c in retrieved_chunks]
    expected_ids = [str(cid) for cid in expected_chunk_ids]

    p_at_k = compute_precision_at_k(retrieved_ids, expected_ids, k)
    r_at_k = compute_recall_at_k(retrieved_ids, expected_ids, k)
    hit_rate = compute_hit_rate_at_k(retrieved_ids, expected_ids, k)
    mrr = compute_mrr_at_k(retrieved_ids, expected_ids, k)
    sufficiency = evaluate_context_sufficiency(retrieved_chunks, expected_ids, min_similarity_threshold)

    # Relevant chunks found in top K
    expected_set = set(expected_ids)
    relevant_found = [cid for cid in retrieved_ids[:k] if cid in expected_set]

    # Deterministic pass condition:
    # If expected chunks exist: pass if hit_rate > 0 (or precision > 0)
    # If expected chunks empty (out-of-domain): pass if sufficiency is True (no false positives)
    if expected_ids:
        passed = hit_rate > 0.0 and sufficiency
    else:
        passed = sufficiency

    return StructuredRetrievalEvaluation(
        query=query,
        k=k,
        retrieved_chunks=retrieved_chunks,
        relevant_chunks=relevant_found,
        expected_chunk_ids=expected_ids,
        retrieval_precision=p_at_k,
        retrieval_recall=r_at_k,
        hit_rate=hit_rate,
        mrr=mrr,
        context_sufficiency=sufficiency,
        passed=passed,
        details={
            "total_retrieved": len(retrieved_chunks),
            "top_k_evaluated": min(k, len(retrieved_chunks)),
            "similarity_scores": [round(float(c.get("similarity_score", 0.0)), 4) for c in retrieved_chunks[:k]]
        }
    )
