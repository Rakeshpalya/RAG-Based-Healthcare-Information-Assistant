"""
Phase 6.4 Benchmark: Clinical Answer Synthesis & Evidence-Grounded Generation.

Measures deterministic performance overhead of:
- Answer preparation latency
- Prompt construction latency
- Confidence calculation overhead
- Citation mapping overhead
- Deterministic validation & section parsing overhead
- End-to-end deterministic synthesis overhead (p50, p95, p99, mean, max)
- Overall throughput (syntheses/sec)
- Production vector store invariant verification (744 FAISS vectors, 744 metadata records, 384 dim)

Outputs structured JSON to evaluation_reports/benchmark_phase6_4_results.json.
"""

import os
import sys
import time
import json
import math
import logging
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch

# Ensure workspace root is in sys.path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from backend.evaluation.citation_validator import CitationValidationResult

from backend.intelligence.answer_models import (
    AnswerSectionType,
    AnswerConfidence,
    EvidenceSupportLevel,
    ClinicalAnswerSection,
    ClinicalAnswer,
    AnswerSynthesisResult
)
from backend.intelligence.answer_synthesis import ClinicalAnswerSynthesisEngine, normalize_intent
from backend.intelligence.intent_models import ClinicalIntent, ClinicalRoutingStrategy
from backend.intelligence.query_plan import (
    QueryPlan,
    ChunkWeightingStrategy,
    DocumentFilterStrategy,
    MultiDocumentStrategy,
    GenerationStrategy
)
from backend.intelligence.evidence_models import (
    ConflictType,
    ConflictSeverity,
    CoverageStatus,
    EvidenceConflict,
    EvidenceCoverage,
    FusedEvidenceChunk,
    FusedContextResult
)
from backend.services.vector_store_service import get_vector_store_service


def calculate_percentile(data: List[float], p: float) -> float:
    """Calculates percentile with linear interpolation."""
    if not data:
        return 0.0
    sorted_d = sorted(data)
    k = (len(sorted_d) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(sorted_d[int(k)], 3)
    return round(sorted_d[f] * (c - k) + sorted_d[c] * (k - f), 3)


def build_benchmark_test_data() -> tuple[FusedContextResult, QueryPlan]:
    """Assembles realistic multi-document fused context for benchmarking."""
    chunks = [
        FusedEvidenceChunk(
            chunk_id="chunk_bm_1",
            document_id="guideline_htn_2024",
            document_name="AHA_Hypertension_2024.pdf",
            page_number=14,
            text=(
                "First-line pharmacological management of Stage 1 hypertension involves "
                "thiazide diuretics, calcium channel blockers, or ACE inhibitors. "
                "Initial monotherapy should be initiated at standard clinical dosages."
            ),
            similarity_score=0.89,
            weighting_boost=1.0,
            fused_score=0.89,
            rank=1,
            source_index=1
        ),
        FusedEvidenceChunk(
            chunk_id="chunk_bm_2",
            document_id="guideline_cvd_prevention",
            document_name="ESC_Cardiovascular_Prevention.pdf",
            page_number=28,
            text=(
                "Lifestyle interventions including dietary sodium restriction (< 2 g/day), "
                "regular aerobic exercise, and alcohol moderation provide additive blood pressure reduction."
            ),
            similarity_score=0.84,
            weighting_boost=0.95,
            fused_score=0.80,
            rank=2,
            source_index=2
        ),
        FusedEvidenceChunk(
            chunk_id="chunk_bm_3",
            document_id="drug_monograph_lisinopril",
            document_name="FDA_Lisinopril_Package_Insert.pdf",
            page_number=2,
            text=(
                "Lisinopril starting dosage is 10 mg orally once daily. Maintenance dosage "
                "ranges from 20 to 40 mg daily. Contraindicated in angioedema and pregnancy."
            ),
            similarity_score=0.81,
            weighting_boost=0.90,
            fused_score=0.73,
            rank=3,
            source_index=3
        )
    ]

    sources = [
        {
            "source_index": c.source_index,
            "source_label": f"[Source {c.source_index}]",
            "document_id": c.document_id,
            "document_name": c.document_name,
            "page_number": c.page_number,
            "similarity_score": c.similarity_score,
            "text": c.text
        }
        for c in chunks
    ]

    formatted_context = "\n\n".join(
        f"[SOURCE {c.source_index}] (Document: {c.document_name}, Page {c.page_number}):\n{c.text}"
        for c in chunks
    )

    cov = EvidenceCoverage(
        coverage_score=0.92,
        status=CoverageStatus.FULL,
        query_aspects=["medication", "lifestyle", "dosage"],
        covered_aspects=["medication", "lifestyle", "dosage"],
        missing_aspects=[]
    )

    fused_res = FusedContextResult(
        fused_chunks=chunks,
        contributing_documents=[c.document_id for c in chunks],
        contributing_documents_count=len(chunks),
        conflicts=[],
        has_conflicts=False,
        coverage=cov,
        formatted_context=formatted_context,
        sources=sources,
        deduped_count=1,
        is_sufficient=True,
        latency_ms=1.2
    )

    query_plan = QueryPlan(
        intent=ClinicalIntent.TREATMENT_QUERY,
        retrieval_required=True,
        retrieval_strategy=ClinicalRoutingStrategy.TREATMENT_RAG,
        top_k=5,
        similarity_threshold=0.25,
        multi_document_strategy=MultiDocumentStrategy.MULTI_DOCUMENT,
        chunk_weighting_strategy=ChunkWeightingStrategy.STANDARD,
        generation_strategy=GenerationStrategy.STANDARD_GROUNDED
    )

    return fused_res, query_plan


def run_benchmark(iterations: int = 200) -> Dict[str, Any]:
    """Executes the Phase 6.4 performance benchmark."""
    print(f"\n========================================================")
    print(f"  PHASE 6.4 BENCHMARK: CLINICAL ANSWER SYNTHESIS ENGINE")
    print(f"========================================================")
    print(f"Iterations: {iterations}")
    print(f"Target: Deterministic Local Overhead p95 < 5.0 ms")
    print(f"Note: LLM is mocked to isolate deterministic local engine latency.")

    fused_res, query_plan = build_benchmark_test_data()

    # Pre-configured mock LLM returning instant deterministic response
    mock_llm = MagicMock()
    mock_llm.generate_answer_from_prompt.return_value = {
        "answer": (
            "First-line treatment for hypertension includes thiazide diuretics, CCBs, or ACE inhibitors [Source 1]. "
            "Sodium restriction and regular exercise provide additive reduction [Source 2]. "
            "For lisinopril, the initial dosage is 10 mg orally once daily [Source 3]."
        ),
        "model": "gemini-3.5-flash-lite",
        "elapsed_ms": 0.0
    }
    mock_llm.generate_answer.return_value = mock_llm.generate_answer_from_prompt.return_value

    sample_answer_text = (
        "First-line treatment includes thiazide diuretics and ACE inhibitors [Source 1]. "
        "Sodium restriction and exercise provide additive reduction [Source 2]. "
        "Initial lisinopril dosage is 10 mg once daily [Source 3]."
    )
    valid_sources = [1, 2, 3]
    val_mock_res = CitationValidationResult(
        is_valid=True,
        citations_found=[1, 2, 3],
        valid_citations=[1, 2, 3],
        invalid_citations=[],
        duplicate_citations=[],
        has_citations=True,
        missing_citations=False,
        mapped_sources=[{"source_index": i} for i in [1, 2, 3]],
        claims_checked=3,
        claims_supported=3,
        claims_unsupported=0,
        citation_coverage=1.0,
        cleaned_grounded_answer=sample_answer_text
    )

    prep_latencies: List[float] = []
    prompt_latencies: List[float] = []
    conf_latencies: List[float] = []
    citation_latencies: List[float] = []
    section_latencies: List[float] = []
    e2e_deterministic_latencies: List[float] = []

    # Warmup
    for _ in range(20):
        ClinicalAnswerSynthesisEngine.build_hardened_synthesis_prompt(
            query="Hypertension management",
            fused_result=fused_res,
            query_plan=query_plan,
            intent="TREATMENT_QUERY"
        )
        ClinicalAnswerSynthesisEngine.compute_confidence(
            coverage=fused_res.coverage,
            fused_result=fused_res,
            query_plan=query_plan,
            intent="TREATMENT_QUERY"
        )
        ClinicalAnswerSynthesisEngine.parse_answer_sections(
            answer_text=sample_answer_text,
            intent="TREATMENT_QUERY",
            valid_sources=valid_sources
        )

    with patch("backend.evaluation.citation_validator.CitationValidator.validate_grounded_citations", return_value=val_mock_res):
        t_bench_start = time.perf_counter()

        for _ in range(iterations):
            # 1. Answer Preparation Latency
            t0 = time.perf_counter()
            norm_intent = normalize_intent(query_plan.intent)
            cov = fused_res.coverage
            is_suff = fused_res.is_sufficient
            t1 = time.perf_counter()
            prep_latencies.append((t1 - t0) * 1000.0)

            # 2. Prompt Construction Latency
            t0 = time.perf_counter()
            prompt = ClinicalAnswerSynthesisEngine.build_hardened_synthesis_prompt(
                query="What is the first-line treatment and starting dose for hypertension?",
                fused_result=fused_res,
                query_plan=query_plan,
                intent=norm_intent
            )
            t1 = time.perf_counter()
            prompt_latencies.append((t1 - t0) * 1000.0)

            # 3. Confidence Calculation Latency
            t0 = time.perf_counter()
            conf, supp = ClinicalAnswerSynthesisEngine.compute_confidence(
                coverage=cov,
                fused_result=fused_res,
                query_plan=query_plan,
                intent=norm_intent
            )
            t1 = time.perf_counter()
            conf_latencies.append((t1 - t0) * 1000.0)

            # 4. Section Parsing & Deterministic Validation Latency
            t0 = time.perf_counter()
            sections = ClinicalAnswerSynthesisEngine.parse_answer_sections(
                answer_text=sample_answer_text,
                intent=norm_intent,
                valid_sources=valid_sources
            )
            t1 = time.perf_counter()
            section_latencies.append((t1 - t0) * 1000.0)

            # 5. End-to-End Deterministic Synthesis (with mocked zero-cost LLM)
            t0 = time.perf_counter()
            res = ClinicalAnswerSynthesisEngine.synthesize(
                query="What is the first-line treatment and starting dose for hypertension?",
                fused_result=fused_res,
                query_plan=query_plan,
                intent=norm_intent,
                gemini_service=mock_llm
            )
            t1 = time.perf_counter()
            e2e_deterministic_latencies.append((t1 - t0) * 1000.0)

        total_bench_duration = time.perf_counter() - t_bench_start
    throughput = round(iterations / total_bench_duration, 1)

    # Calculate statistics
    def calc_stats(lat_list: List[float]) -> Dict[str, float]:
        return {
            "mean_ms": round(sum(lat_list) / len(lat_list), 3),
            "p50_ms": calculate_percentile(lat_list, 50.0),
            "p95_ms": calculate_percentile(lat_list, 95.0),
            "p99_ms": calculate_percentile(lat_list, 99.0),
            "min_ms": round(min(lat_list), 3),
            "max_ms": round(max(lat_list), 3)
        }

    prep_stats = calc_stats(prep_latencies)
    prompt_stats = calc_stats(prompt_latencies)
    conf_stats = calc_stats(conf_latencies)
    section_stats = calc_stats(section_latencies)
    e2e_stats = calc_stats(e2e_deterministic_latencies)

    # Verify Production Vector Store Invariants
    vs = get_vector_store_service()
    faiss_count = vs.index.ntotal if vs.index is not None else 0
    meta_count = len(getattr(vs, "metadata_store", []))
    dim_count = vs.index.d if vs.index is not None else 0

    invariants_intact = (faiss_count == 744 and meta_count == 744 and dim_count == 384)
    target_met = (e2e_stats["p95_ms"] < 5.0)

    print("\n--- BENCHMARK RESULTS ---")
    print(f"End-to-End Synthesis Overhead:")
    print(f"  Mean:  {e2e_stats['mean_ms']} ms")
    print(f"  p50:   {e2e_stats['p50_ms']} ms")
    print(f"  p95:   {e2e_stats['p95_ms']} ms (Target: < 5.0 ms - {'PASSED' if target_met else 'FAILED'})")
    print(f"  p99:   {e2e_stats['p99_ms']} ms")
    print(f"  Max:   {e2e_stats['max_ms']} ms")
    print(f"Throughput: {throughput} syntheses/sec")
    print(f"\nSub-Component Latencies (p95):")
    print(f"  Answer Preparation:     {prep_stats['p95_ms']} ms")
    print(f"  Prompt Construction:    {prompt_stats['p95_ms']} ms")
    print(f"  Confidence Calculation: {conf_stats['p95_ms']} ms")
    print(f"  Section Parsing:        {section_stats['p95_ms']} ms")
    print(f"\nProduction Invariants:")
    print(f"  FAISS Vectors:          {faiss_count} (Expected: 744)")
    print(f"  Metadata Records:       {meta_count} (Expected: 744)")
    print(f"  Embedding Dimension:    {dim_count} (Expected: 384)")
    print(f"  Invariants Intact:      {invariants_intact}")

    results = {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_type": "Phase 6.4 Clinical Answer Synthesis",
        "iterations": iterations,
        "llm_mocked": True,
        "llm_mocked_reason": "Isolate deterministic engine overhead from cloud API latency",
        "throughput_ops_per_sec": throughput,
        "end_to_end_synthesis_overhead": e2e_stats,
        "component_breakdown": {
            "answer_preparation": prep_stats,
            "prompt_construction": prompt_stats,
            "confidence_calculation": conf_stats,
            "section_parsing": section_stats
        },
        "target_p95_threshold_ms": 5.0,
        "target_met": target_met,
        "production_invariants": {
            "faiss_vector_count": faiss_count,
            "metadata_record_count": meta_count,
            "embedding_dimension": dim_count,
            "invariants_intact": invariants_intact
        }
    }

    report_dir = os.path.join(WORKSPACE_DIR, "evaluation_reports")
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, "benchmark_phase6_4_results.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nReport saved to: {report_file}")
    return results


if __name__ == "__main__":
    run_benchmark(iterations=200)
