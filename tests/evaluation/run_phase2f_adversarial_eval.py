"""
Phase 2F: Adversarial Benchmark Evaluation & False Positive Analysis Runner.

Executes the 42 adversarial queries against the RAG retrieval pipeline and classifies
any false positives into:
1. Benign extra context
2. Weak semantic match
3. True irrelevant retrieval
4. Safety-critical false positive
"""

import os
import sys
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath("."))

from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import get_vector_store_service
from tests.evaluation.phase2f_adversarial_dataset import PHASE2F_ADVERSARIAL_DATASET
from tests.evaluation.run_retrieval_evaluation import calculate_query_metrics, matches_target


def classify_false_positive(query: str, q_type: str, doc_name: str, score: float, should_reject: bool) -> str:
    """Classifies a false positive retrieval into 4 diagnostic tiers."""
    q_lower = query.lower()
    doc_lower = doc_name.lower()

    if should_reject:
        # If the query was out-of-scope/unsupported and evidence was returned:
        if any(term in q_lower for term in ["chemotherapy", "cancer", "surgery", "dosage", "infant", "meningitis"]):
            return "4. Safety-critical false positive"
        return "3. True irrelevant retrieval"

    # For valid medical queries where secondary documents were retrieved:
    if "hypertension" in q_lower and ("synthetic_hypertension" in doc_lower or "hypertension_summary" in doc_lower or "healthai" in doc_lower):
        return "1. Benign extra context"
    if "diabetes" in q_lower and ("trial_report" in doc_lower or "healthai" in doc_lower):
        return "1. Benign extra context"
    if "asthma" in q_lower and ("asthma" in doc_lower or "healthai" in doc_lower):
        return "1. Benign extra context"

    if score < 0.35:
        return "2. Weak semantic match"

    return "3. True irrelevant retrieval"


def run_adversarial_evaluation():
    print("=" * 75)
    print("=== PHASE 2F: ADVERSARIAL RETRIEVAL EVALUATION & AUDIT ===")
    print("=" * 75)

    vs = get_vector_store_service()
    rag = RAGService(vector_store=vs)

    count_before = vs.count()
    print(f"Vector Store Count (Before): {count_before}")

    results = []
    category_buckets: Dict[str, List[Dict[str, Any]]] = {}

    fp_classifications = {
        "1. Benign extra context": 0,
        "2. Weak semantic match": 0,
        "3. True irrelevant retrieval": 0,
        "4. Safety-critical false positive": 0
    }
    fp_details = []

    for case in PHASE2F_ADVERSARIAL_DATASET:
        qid = case["id"]
        query = case["query"]
        q_type = case["query_type"]
        category = case.get("category", q_type)
        rel_docs = case.get("relevant_documents", [])
        irrel_docs = case.get("irrelevant_documents", [])
        should_reject = case.get("should_reject", False)

        ret_res = rag.query(query, top_k=5)
        status = ret_res.get("retrieval_status", "success")
        retrieved_chunks = ret_res.get("retrieved_chunks", [])
        sources = ret_res.get("sources", [])

        retrieved_doc_names = [
            str(c.get("metadata", {}).get("filename") or c.get("document_id") or "UNKNOWN")
            for c in retrieved_chunks
        ]
        scores = [round(float(c.get("similarity_score", 0.0)), 4) for c in retrieved_chunks]

        # Calculate metrics
        is_out_of_scope = (len(rel_docs) == 0)
        metrics = calculate_query_metrics(
            retrieved_items=retrieved_chunks,
            relevant_docs=rel_docs,
            irrelevant_docs=irrel_docs,
            is_out_of_scope=is_out_of_scope,
            retrieval_status=status
        )

        # Pass / Fail criteria
        if should_reject:
            passed = (status == "no_relevant_context" or len(retrieved_chunks) == 0)
            failure_reason = "failed_to_reject_out_of_scope_query" if not passed else None
        elif is_out_of_scope:
            passed = (status == "no_relevant_context" or len(retrieved_chunks) == 0)
            failure_reason = "irrelevant_evidence_retrieved" if not passed else None
        else:
            passed = (metrics["hit_rate_5"] == 1.0)
            failure_reason = "false_negative_relevant_evidence_missed" if not passed else None

        # Inspect False Positives
        for doc_name, sc in zip(retrieved_doc_names, scores):
            is_relevant_doc = matches_target(doc_name, rel_docs)
            is_explicit_irrel = matches_target(doc_name, irrel_docs)

            if not is_relevant_doc or is_explicit_irrel:
                tier = classify_false_positive(query, q_type, doc_name, sc, should_reject)
                fp_classifications[tier] += 1
                fp_details.append({
                    "query_id": qid,
                    "query": query,
                    "doc": doc_name,
                    "score": sc,
                    "tier": tier
                })

        entry = {
            "id": qid,
            "query": query,
            "category": category,
            "query_type": q_type,
            "relevant_docs": rel_docs,
            "retrieved_docs": retrieved_doc_names,
            "scores": scores,
            "status": status,
            "passed": passed,
            "failure_reason": failure_reason,
            "metrics": metrics
        }
        results.append(entry)

        if category not in category_buckets:
            category_buckets[category] = []
        category_buckets[category].append(entry)

    count_after = vs.count()
    assert count_before == count_after, "Vector store modified during adversarial evaluation!"

    total_queries = len(results)
    total_passed = sum(1 for r in results if r["passed"])

    # Aggregate overall metrics
    metric_keys = [
        "precision_1", "precision_3", "precision_5",
        "recall_1", "recall_3", "recall_5",
        "hit_rate_1", "hit_rate_3", "hit_rate_5",
        "mrr", "evidence_coverage", "irrelevant_rate"
    ]
    overall = {}
    for k in metric_keys:
        vals = [r["metrics"][k] for r in results]
        overall[k] = round(sum(vals) / float(len(vals)), 4) if vals else 0.0

    print(f"\n--- OVERALL ADVERSARIAL BENCHMARK RESULTS ---")
    print(f"Total Adversarial Queries: {total_queries}")
    print(f"Passed: {total_passed}/{total_queries} ({total_passed/total_queries*100:.1f}%)")
    print(f"Precision@1: {overall['precision_1']:.4f} | Precision@5: {overall['precision_5']:.4f}")
    print(f"Recall@1:    {overall['recall_1']:.4f} | Recall@5:    {overall['recall_5']:.4f}")
    print(f"Hit Rate@1:  {overall['hit_rate_1']:.4f} | Hit Rate@5:  {overall['hit_rate_5']:.4f}")
    print(f"MRR:         {overall['mrr']:.4f}")
    print(f"Evidence Coverage:        {overall['evidence_coverage']:.4f}")
    print(f"Irrelevant Evidence Rate: {overall['irrelevant_rate']:.4f}")

    print(f"\n--- PER-CATEGORY BREAKDOWN ---")
    header = f"{'Category':<22} | {'Queries':<7} | {'P@5':<7} | {'R@5':<7} | {'HR@5':<7} | {'MRR':<7} | {'Irrel Rate':<10} | {'Pass Rate':<9}"
    print(header)
    print("-" * len(header))
    for cat, items in sorted(category_buckets.items()):
        cnt = len(items)
        p5 = sum(x["metrics"]["precision_5"] for x in items) / cnt
        r5 = sum(x["metrics"]["recall_5"] for x in items) / cnt
        hr5 = sum(x["metrics"]["hit_rate_5"] for x in items) / cnt
        mrr = sum(x["metrics"]["mrr"] for x in items) / cnt
        irrel = sum(x["metrics"]["irrelevant_rate"] for x in items) / cnt
        passed_cnt = sum(1 for x in items if x["passed"])
        pass_rate = passed_cnt / cnt
        print(f"{cat:<22} | {cnt:<7} | {p5:<7.4f} | {r5:<7.4f} | {hr5:<7.4f} | {mrr:<7.4f} | {irrel:<10.4f} | {pass_rate:<9.1%}")

    print(f"\n--- FALSE POSITIVE / IRRELEVANT EVIDENCE CLASSIFICATION ---")
    total_fps = sum(fp_classifications.values())
    print(f"Total Irrelevant / Unmatched Retrieved Chunks: {total_fps}")
    for tier, count in fp_classifications.items():
        pct = (count / total_fps * 100) if total_fps > 0 else 0.0
        print(f"  {tier}: {count} ({pct:.1f}%)")

    failures = [r for r in results if not r["passed"]]
    print(f"\n--- ADVERSARIAL FAILURES & DEFECT AUDIT ({len(failures)} failures) ---")
    if not failures:
        print("  Zero failures detected across all 42 adversarial queries!")
    else:
        for idx, f in enumerate(failures, start=1):
            print(f"\n[{idx}] Query: \"{f['query']}\"")
            print(f"    Category: {f['category']}")
            print(f"    Expected: {f['relevant_docs']}")
            print(f"    Retrieved: {f['retrieved_docs']}")
            print(f"    Scores: {f['scores']}")
            print(f"    Status: {f['status']}")
            print(f"    Reason: {f['failure_reason']}")

    print("\n" + "=" * 75)
    print(f"Vector Store Count (After): {count_after} (Integrity: 100% Intact)")
    print("=" * 75)

    return {
        "total_queries": total_queries,
        "total_passed": total_passed,
        "overall": overall,
        "category_buckets": category_buckets,
        "fp_classifications": fp_classifications,
        "failures": failures,
        "fp_details": fp_details
    }


if __name__ == "__main__":
    run_adversarial_evaluation()
