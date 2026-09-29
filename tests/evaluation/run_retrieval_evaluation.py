"""
Retrieval Evaluation Runner and Metrics Engine for Phase 2D.

Evaluates the RAG retrieval pipeline without LLM/Gemini generation overhead.
Calculates standard information retrieval metrics:
- Precision@1, Precision@3, Precision@5
- Recall@1, Recall@3, Recall@5
- Hit Rate@1, Hit Rate@3, Hit Rate@5
- MRR (Mean Reciprocal Rank)
- Evidence Coverage
- Irrelevant Evidence Rate

Produces overall and per-category metrics, plus structured error analysis.
"""

import os
import sys
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.abspath("."))

from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import get_vector_store_service, VectorStoreService
from tests.evaluation.retrieval_eval_dataset import EVALUATION_DATASET


def matches_target(doc_identifier: str, target_list: List[str]) -> bool:
    """Checks if a retrieved document identifier matches any entry in the target list."""
    if not doc_identifier or not target_list:
        return False
    doc_norm = str(doc_identifier).lower().strip()
    for target in target_list:
        target_norm = str(target).lower().strip()
        if target_norm == doc_norm or target_norm in doc_norm or doc_norm in target_norm:
            return True
    return False


def calculate_query_metrics(
    retrieved_items: List[Dict[str, Any]],
    relevant_docs: List[str],
    irrelevant_docs: List[str],
    is_out_of_scope: bool = False,
    retrieval_status: str = "success"
) -> Dict[str, float]:
    """
    Computes precision, recall, hit rate, MRR, evidence coverage, and irrelevant rate
    for a single retrieval query against ground truth.

    Guarantees:
    - Never divides by zero.
    - Gracefully handles cases where fewer than K results exist.
    - Handles out-of-scope and empty retrieval scenarios correctly.
    """
    retrieved_doc_names = []
    for item in retrieved_items:
        fname = item.get("metadata", {}).get("filename") or item.get("filename")
        did = item.get("document_id")
        retrieved_doc_names.append(str(fname or did or "UNKNOWN"))

    total_retrieved = len(retrieved_doc_names)
    total_relevant = len(relevant_docs)

    # 1. Out-of-scope / Unsupported query handling
    if total_relevant == 0:
        # Expected behavior for irrelevant or out-of-scope query: zero retrieval
        if total_retrieved == 0 or retrieval_status == "no_relevant_context":
            return {
                "precision_1": 1.0,
                "precision_3": 1.0,
                "precision_5": 1.0,
                "recall_1": 1.0,
                "recall_3": 1.0,
                "recall_5": 1.0,
                "hit_rate_1": 1.0,
                "hit_rate_3": 1.0,
                "hit_rate_5": 1.0,
                "mrr": 1.0,
                "evidence_coverage": 1.0,
                "irrelevant_rate": 0.0,
            }
        else:
            # Irrelevant evidence was returned
            return {
                "precision_1": 0.0,
                "precision_3": 0.0,
                "precision_5": 0.0,
                "recall_1": 0.0,
                "recall_3": 0.0,
                "recall_5": 0.0,
                "hit_rate_1": 0.0,
                "hit_rate_3": 0.0,
                "hit_rate_5": 0.0,
                "mrr": 0.0,
                "evidence_coverage": 0.0,
                "irrelevant_rate": 1.0,
            }

    # 2. Precision@K, Recall@K, HitRate@K
    metrics: Dict[str, float] = {}

    for k in (1, 3, 5):
        top_k_items = retrieved_doc_names[:k]
        rel_in_k = sum(1 for d in top_k_items if matches_target(d, relevant_docs))

        # Precision@K: relevant items in top K divided by K
        metrics[f"precision_{k}"] = round(rel_in_k / float(k), 4)

        # Recall@K: relevant items in top K divided by total known relevant documents
        metrics[f"recall_{k}"] = round(min(1.0, rel_in_k / float(total_relevant)), 4) if total_relevant > 0 else 0.0

        # HitRate@K: 1.0 if at least one relevant item is in top K, else 0.0
        metrics[f"hit_rate_{k}"] = 1.0 if rel_in_k > 0 else 0.0

    # 3. Mean Reciprocal Rank (MRR)
    first_rel_rank = 0
    for idx, d in enumerate(retrieved_doc_names, start=1):
        if matches_target(d, relevant_docs):
            first_rel_rank = idx
            break
    metrics["mrr"] = round(1.0 / first_rel_rank, 4) if first_rel_rank > 0 else 0.0

    # 4. Evidence Coverage: distinct relevant documents retrieved / total relevant documents
    distinct_rel_retrieved = set()
    for d in retrieved_doc_names:
        for r in relevant_docs:
            if matches_target(d, [r]):
                distinct_rel_retrieved.add(r)
    metrics["evidence_coverage"] = (
        round(len(distinct_rel_retrieved) / float(total_relevant), 4)
        if total_relevant > 0 else 1.0
    )

    # 5. Irrelevant Evidence Rate: retrieved items matching irrelevant_docs or not relevant / total retrieved
    irrelevant_count = 0
    for d in retrieved_doc_names:
        if matches_target(d, irrelevant_docs) or not matches_target(d, relevant_docs):
            irrelevant_count += 1
    metrics["irrelevant_rate"] = (
        round(irrelevant_count / float(total_retrieved), 4)
        if total_retrieved > 0 else 0.0
    )

    return metrics


class RetrievalEvaluator:
    """
    Evaluation harness for running the benchmark dataset against the RAGService.
    Does not modify vector store or call generative LLM.
    """

    def __init__(self, rag_service: Optional[RAGService] = None):
        if rag_service is not None:
            self.rag_service = rag_service
        else:
            self.rag_service = RAGService(vector_store=get_vector_store_service())

    def evaluate_query(self, eval_case: Dict[str, Any]) -> Dict[str, Any]:
        query = eval_case["query"]
        relevant_docs = eval_case.get("relevant_documents", [])
        irrelevant_docs = eval_case.get("irrelevant_documents", [])
        q_type = eval_case.get("query_type", "general")

        # Execute retrieval query without LLM generation
        result = self.rag_service.query(query, top_k=5)

        retrieved_chunks = result.get("retrieved_chunks", [])
        sources = result.get("sources", [])
        status = result.get("retrieval_status", "success")

        # Compute metrics
        is_out_of_scope = (q_type in ("out_of_scope", "irrelevant") and len(relevant_docs) == 0)
        metrics = calculate_query_metrics(
            retrieved_items=retrieved_chunks,
            relevant_docs=relevant_docs,
            irrelevant_docs=irrelevant_docs,
            is_out_of_scope=is_out_of_scope,
            retrieval_status=status
        )

        retrieved_doc_names = [
            str(c.get("metadata", {}).get("filename") or c.get("document_id") or "UNKNOWN")
            for c in retrieved_chunks
        ]
        scores = [round(float(c.get("similarity_score", 0.0)), 4) for c in retrieved_chunks]

        # Determine pass/fail
        # For relevant queries: pass if hit_rate_5 == 1.0 and MRR > 0
        # For out_of_scope / irrelevant queries: pass if status == 'no_relevant_context' or 0 sources
        if is_out_of_scope:
            passed = (status == "no_relevant_context" or len(retrieved_chunks) == 0)
            failure_reason = "irrelevant_evidence_retrieved" if not passed else None
        else:
            passed = (metrics["hit_rate_5"] == 1.0)
            if not passed:
                if len(retrieved_chunks) == 0:
                    failure_reason = "false_negative_no_chunks_retrieved"
                else:
                    failure_reason = "lexical_or_embedding_mismatch"
            else:
                failure_reason = None

        return {
            "id": eval_case.get("id"),
            "query": query,
            "query_type": q_type,
            "difficulty": eval_case.get("difficulty"),
            "expected_documents": relevant_docs,
            "retrieved_documents": retrieved_doc_names,
            "similarity_scores": scores,
            "retrieval_status": status,
            "number_of_sources": len(sources),
            "metrics": metrics,
            "passed": passed,
            "failure_reason": failure_reason
        }

    def run_benchmark(self, dataset: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        cases = dataset or EVALUATION_DATASET
        case_results = []

        initial_count = self.rag_service.vector_store.count()

        for case in cases:
            res = self.evaluate_query(case)
            case_results.append(res)

        final_count = self.rag_service.vector_store.count()
        assert initial_count == final_count, "Vector store was modified during evaluation!"

        # Aggregate overall metrics
        metric_keys = [
            "precision_1", "precision_3", "precision_5",
            "recall_1", "recall_3", "recall_5",
            "hit_rate_1", "hit_rate_3", "hit_rate_5",
            "mrr", "evidence_coverage", "irrelevant_rate"
        ]

        overall: Dict[str, float] = {}
        for k in metric_keys:
            vals = [r["metrics"][k] for r in case_results]
            overall[k] = round(sum(vals) / float(len(vals)), 4) if vals else 0.0

        # Per-category aggregation
        categories: Dict[str, Dict[str, Any]] = {}
        for r in case_results:
            cat = r["query_type"]
            if cat not in categories:
                categories[cat] = {"cases": [], "count": 0}
            categories[cat]["cases"].append(r)
            categories[cat]["count"] += 1

        per_category: Dict[str, Dict[str, float]] = {}
        for cat, data in categories.items():
            cat_results = data["cases"]
            cat_metrics = {}
            for k in metric_keys:
                vals = [r["metrics"][k] for r in cat_results]
                cat_metrics[k] = round(sum(vals) / float(len(vals)), 4) if vals else 0.0
            cat_metrics["query_count"] = float(len(cat_results))
            cat_metrics["pass_rate"] = round(sum(1 for r in cat_results if r["passed"]) / float(len(cat_results)), 4)
            per_category[cat] = cat_metrics

        failures = [r for r in case_results if not r["passed"]]

        return {
            "total_queries": len(case_results),
            "total_passed": sum(1 for r in case_results if r["passed"]),
            "overall_metrics": overall,
            "per_category_metrics": per_category,
            "failures": failures,
            "case_results": case_results,
            "vector_store_count_before": initial_count,
            "vector_store_count_after": final_count,
        }


def print_evaluation_report(report: Dict[str, Any]):
    print("==================================================================")
    print("=== PHASE 2D: RETRIEVAL EVALUATION & BENCHMARK REPORT ===")
    print("==================================================================")
    print(f"Total Benchmark Queries: {report['total_queries']}")
    print(f"Passed: {report['total_passed']}/{report['total_queries']} ({report['total_passed']/report['total_queries']*100:.1f}%)")
    print(f"Vector Store Count Unchanged: {report['vector_store_count_before']} == {report['vector_store_count_after']}\n")

    print("--- OVERALL RETRIEVAL METRICS ---")
    ov = report["overall_metrics"]
    print(f"  Precision@1: {ov['precision_1']:.4f} | Precision@3: {ov['precision_3']:.4f} | Precision@5: {ov['precision_5']:.4f}")
    print(f"  Recall@1:    {ov['recall_1']:.4f} | Recall@3:    {ov['recall_3']:.4f} | Recall@5:    {ov['recall_5']:.4f}")
    print(f"  Hit Rate@1:  {ov['hit_rate_1']:.4f} | Hit Rate@3:  {ov['hit_rate_3']:.4f} | Hit Rate@5:  {ov['hit_rate_5']:.4f}")
    print(f"  MRR:         {ov['mrr']:.4f}")
    print(f"  Evidence Coverage:       {ov['evidence_coverage']:.4f}")
    print(f"  Irrelevant Evidence Rate: {ov['irrelevant_rate']:.4f}\n")

    print("--- PER-CATEGORY METRICS ---")
    header = f"{'Category':<22} | {'Queries':<7} | {'P@5':<7} | {'R@5':<7} | {'HR@5':<7} | {'MRR':<7} | {'Irrel Rate':<10} | {'Pass Rate':<9}"
    print(header)
    print("-" * len(header))
    for cat, m in sorted(report["per_category_metrics"].items()):
        print(
            f"{cat:<22} | {int(m['query_count']):<7} | {m['precision_5']:<7.4f} | {m['recall_5']:<7.4f} | "
            f"{m['hit_rate_5']:<7.4f} | {m['mrr']:<7.4f} | {m['irrelevant_rate']:<10.4f} | {m['pass_rate']:<9.1%}"
        )

    print("\n--- ERROR ANALYSIS & FAILURES ---")
    failures = report["failures"]
    if not failures:
        print("  Zero failures detected across all benchmark queries!")
    else:
        print(f"  Total failures: {len(failures)}")
        for idx, f in enumerate(failures, start=1):
            print(f"\n[{idx}] Query: \"{f['query']}\"")
            print(f"    Category: {f['query_type']} (Difficulty: {f['difficulty']})")
            print(f"    Expected: {f['expected_documents']}")
            print(f"    Retrieved: {f['retrieved_documents']}")
            print(f"    Scores: {f['similarity_scores']}")
            print(f"    Failure Reason: {f['failure_reason']}")


if __name__ == "__main__":
    evaluator = RetrievalEvaluator()
    rep = evaluator.run_benchmark()
    print_evaluation_report(rep)
