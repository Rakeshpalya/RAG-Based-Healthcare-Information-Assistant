"""
Phase 6.6 Benchmark: Clinical Grounding Verification & Hallucination Guardrail Engine.

Measures deterministic performance overhead of:
- Clinical entity extraction latency (medications, dosages, negations, directions)
- Directional and negation contradiction detection latency
- Numerical dosage and entity grounding verification latency
- Prescriptive and diagnostic safety sanitization latency
- End-to-end verification overhead (verify_and_guard p50, p95, p99, mean, max)
- Full synthesis integration overhead (Answer synthesis with Phase 6.6 verification)
- Overall throughput (evaluations/sec)
- Production vector store invariant verification (744 FAISS vectors, 744 metadata records, 384 dim)

Outputs structured JSON to evaluation_reports/benchmark_phase6_6_results.json.
"""

import os
import sys
import time
import json
import math
import logging
from typing import List, Dict, Any
from unittest.mock import MagicMock

# Ensure workspace root is in sys.path
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from backend.intelligence.verification_models import (
    GroundingVerificationStatus,
    ClinicalHallucinationType,
    SafetyPostScreenAction,
    ExtractedClinicalEntities,
    ClinicalVerificationClaim,
    ClinicalVerificationResult
)
from backend.intelligence.clinical_verification import ClinicalVerificationEngine
from backend.intelligence.citation_attribution import ClinicalCitationAttributionEngine
from backend.intelligence.answer_synthesis import ClinicalAnswerSynthesisEngine
from backend.intelligence.evidence_models import (
    FusedEvidenceChunk,
    FusedContextResult,
    EvidenceCoverage,
    CoverageStatus
)
from backend.intelligence.query_plan import (
    QueryPlan,
    ChunkWeightingStrategy,
    DocumentFilterStrategy,
    MultiDocumentStrategy,
    GenerationStrategy
)
from backend.intelligence.intent_models import (
    ClinicalIntent,
    ClinicalRoutingStrategy
)
from backend.services.vector_store_service import get_vector_store_service


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_percentile(data: List[float], percentile: float) -> float:
    """Computes exact percentile rank."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(sorted_data[int(k)], 3)
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return round(d0 + d1, 3)


def run_phase6_6_benchmark(iterations: int = 200) -> Dict[str, Any]:
    """Executes the Phase 6.6 performance benchmark."""
    print("=" * 66)
    print("  PHASE 6.6 BENCHMARK: CLINICAL VERIFICATION & GUARDRAIL ENGINE  ")
    print("=" * 66)
    print(f"Iterations: {iterations}")
    print(f"Target: Deterministic Local Overhead p95 < 5.0 ms\n")

    # Representative multi-source clinical synthesis setup
    sources = [
        {
            "source_index": 1,
            "source_number": 1,
            "chunk_id": "chunk_htn_001",
            "document_id": "doc_aha_2024",
            "document_name": "AHA_Hypertension_2024.pdf",
            "page_number": 12,
            "text": "First-line pharmacotherapy for stage 1 hypertension includes thiazide diuretics, calcium channel blockers, and ACE inhibitors. The standard starting dose for amlodipine is 5 mg daily, which reduces systolic blood pressure by 8 to 12 mmHg.",
            "preview_text": "First-line pharmacotherapy for stage 1 hypertension includes thiazide diuretics...",
            "similarity_score": 0.89
        },
        {
            "source_index": 2,
            "source_number": 2,
            "chunk_id": "chunk_htn_002",
            "document_id": "doc_aha_2024",
            "document_name": "AHA_Hypertension_2024.pdf",
            "page_number": 15,
            "text": "ACE inhibitors such as lisinopril are strictly contraindicated during pregnancy due to fetal renal toxicity. For non-pregnant patients, starting dosage is typically 10 mg once daily.",
            "preview_text": "ACE inhibitors such as lisinopril are strictly contraindicated during pregnancy...",
            "similarity_score": 0.84
        }
    ]

    answer_text = (
        "First-line pharmacotherapy for stage 1 hypertension includes calcium channel blockers [Source 1]. "
        "The standard starting dose for amlodipine is 5 mg daily, which effectively reduces systolic blood pressure [Source 1]. "
        "ACE inhibitors such as lisinopril are strictly contraindicated during pregnancy [Source 2]."
    )

    # Pre-generate attribution report for verification input
    attribution_report = ClinicalCitationAttributionEngine.attribute_and_validate(
        answer_text=answer_text,
        retrieved_sources=sources
    )

    # Mock fused context for synthesis testing
    fused_chunks = [
        FusedEvidenceChunk(
            chunk_id=s["chunk_id"],
            document_id=s["document_id"],
            document_name=s["document_name"],
            page_number=s["page_number"],
            text=s["text"],
            similarity_score=s["similarity_score"],
            source_index=s["source_index"],
            weighting_boost=1.0,
            fused_score=s["similarity_score"],
            rank=idx + 1
        )
        for idx, s in enumerate(sources)
    ]

    fused_res = FusedContextResult(
        fused_chunks=fused_chunks,
        contributing_documents=[s["document_name"] for s in sources],
        contributing_documents_count=len(sources),
        conflicts=[],
        has_conflicts=False,
        coverage=EvidenceCoverage(
            coverage_score=0.95,
            status=CoverageStatus.FULL,
            query_aspects=["hypertension", "treatment", "dosage"],
            covered_aspects=["hypertension", "treatment", "dosage"],
            missing_aspects=[]
        ),
        formatted_context="\n\n".join(f"[SOURCE {s['source_index']}]: {s['text']}" for s in sources),
        sources=sources,
        deduped_count=0,
        is_sufficient=True,
        latency_ms=1.1
    )

    from backend.intelligence.intent_models import ClinicalRoutingStrategy
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

    mock_llm = MagicMock()
    mock_llm.generate_answer_from_prompt.return_value = {
        "answer": answer_text,
        "model": "gemini-1.5-flash",
        "api_request_time_ms": 115.0,
        "input_tokens": 300,
        "output_tokens": 120,
        "disclaimer": "Consult a healthcare professional."
    }

    # Latency tracking arrays
    entity_latencies: List[float] = []
    contra_latencies: List[float] = []
    sanitization_latencies: List[float] = []
    pruning_latencies: List[float] = []
    e2e_verification_latencies: List[float] = []
    synthesis_e2e_latencies: List[float] = []

    # Warmup
    for _ in range(25):
        ClinicalVerificationEngine.extract_clinical_entities(answer_text)
        ClinicalVerificationEngine.sanitize_prescriptions_and_diagnoses(answer_text)
        ClinicalVerificationEngine.verify_and_guard(
            answer_text=answer_text,
            retrieved_sources=sources,
            attribution_report=attribution_report
        )

    t_bench_start = time.perf_counter()

    for _ in range(iterations):
        # 1. Entity Extraction Latency
        t0 = time.perf_counter()
        entities = ClinicalVerificationEngine.extract_clinical_entities(answer_text)
        t1 = time.perf_counter()
        entity_latencies.append((t1 - t0) * 1000.0)

        # 2. Contradiction & Negation Detection Latency
        t0 = time.perf_counter()
        ClinicalVerificationEngine.check_directional_contradiction(entities, sources[0]["text"])
        ClinicalVerificationEngine.check_negation_conflict(entities, sources[0]["text"])
        t1 = time.perf_counter()
        contra_latencies.append((t1 - t0) * 1000.0)

        # 3. Prescriptive / Diagnostic Sanitization Latency
        t0 = time.perf_counter()
        ClinicalVerificationEngine.sanitize_prescriptions_and_diagnoses(answer_text)
        t1 = time.perf_counter()
        sanitization_latencies.append((t1 - t0) * 1000.0)

        # 4. End-to-End Verification Overhead (verify_and_guard)
        t0 = time.perf_counter()
        verif_res = ClinicalVerificationEngine.verify_and_guard(
            answer_text=answer_text,
            retrieved_sources=sources,
            attribution_report=attribution_report
        )
        t1 = time.perf_counter()
        e2e_verification_latencies.append((t1 - t0) * 1000.0)

        # 5. Pruning Latency
        t0 = time.perf_counter()
        ClinicalVerificationEngine.prune_ungrounded_content(
            answer_text=answer_text,
            verification_claims=verif_res.claim_verifications
        )
        t1 = time.perf_counter()
        pruning_latencies.append((t1 - t0) * 1000.0)

        # 6. Full Synthesis Engine Integration Overhead
        t0 = time.perf_counter()
        synth_res = ClinicalAnswerSynthesisEngine.synthesize(
            query="What is the first-line treatment and starting dose for hypertension?",
            fused_result=fused_res,
            query_plan=query_plan,
            intent="TREATMENT_QUERY",
            gemini_service=mock_llm
        )
        t1 = time.perf_counter()
        synthesis_e2e_latencies.append((t1 - t0) * 1000.0)

    total_bench_duration = time.perf_counter() - t_bench_start
    throughput = round(iterations / total_bench_duration, 2)

    def calc_stats(lat_list: List[float]) -> Dict[str, float]:
        return {
            "mean_ms": round(sum(lat_list) / len(lat_list), 3),
            "p50_ms": calculate_percentile(lat_list, 50.0),
            "p95_ms": calculate_percentile(lat_list, 95.0),
            "p99_ms": calculate_percentile(lat_list, 99.0),
            "min_ms": round(min(lat_list), 3),
            "max_ms": round(max(lat_list), 3)
        }

    entity_stats = calc_stats(entity_latencies)
    contra_stats = calc_stats(contra_latencies)
    sanit_stats = calc_stats(sanitization_latencies)
    prune_stats = calc_stats(pruning_latencies)
    e2e_verif_stats = calc_stats(e2e_verification_latencies)
    synthesis_e2e_stats = calc_stats(synthesis_e2e_latencies)

    # Verify Production Vector Store Invariants
    vs = get_vector_store_service()
    faiss_count = vs.index.ntotal if vs.index is not None else 0
    meta_count = len(getattr(vs, "metadata_store", []))
    dim_count = vs.index.d if vs.index is not None else 0

    invariants_intact = (faiss_count == 744 and meta_count == 744 and dim_count == 384)
    target_met = (e2e_verif_stats["p95_ms"] < 5.0)

    print("\n--- BENCHMARK RESULTS ---")
    print(f"Clinical Verification Engine Overhead (verify_and_guard):")
    print(f"  Mean:  {e2e_verif_stats['mean_ms']} ms")
    print(f"  p50:   {e2e_verif_stats['p50_ms']} ms")
    print(f"  p95:   {e2e_verif_stats['p95_ms']} ms (Target: < 5.0 ms - {'PASSED' if target_met else 'FAILED'})")
    print(f"  p99:   {e2e_verif_stats['p99_ms']} ms")
    print(f"  Max:   {e2e_verif_stats['max_ms']} ms")
    print(f"Throughput: {throughput} verification cycles/sec")
    print(f"\nSub-Component Latencies (p95):")
    print(f"  Entity Extraction:          {entity_stats['p95_ms']} ms")
    print(f"  Contradiction Checking:     {contra_stats['p95_ms']} ms")
    print(f"  Safety Sanitization:        {sanit_stats['p95_ms']} ms")
    print(f"  Content Pruning:            {prune_stats['p95_ms']} ms")
    print(f"  Full Synthesis Integration: {synthesis_e2e_stats['p95_ms']} ms")
    print(f"\nProduction Invariants:")
    print(f"  FAISS Vectors:              {faiss_count} (Expected: 744)")
    print(f"  Metadata Records:           {meta_count} (Expected: 744)")
    print(f"  Embedding Dimension:        {dim_count} (Expected: 384)")
    print(f"  Invariants Intact:          {invariants_intact}")
    print("=" * 66)

    benchmark_data = {
        "benchmark_type": "Phase 6.6 Clinical Grounding Verification & Hallucination Guardrails",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iterations": iterations,
        "target_p95_ms": 5.0,
        "target_met": target_met,
        "throughput_cycles_per_sec": throughput,
        "verification_engine_stats": e2e_verif_stats,
        "subcomponents": {
            "entity_extraction": entity_stats,
            "contradiction_checking": contra_stats,
            "safety_sanitization": sanit_stats,
            "content_pruning": prune_stats,
            "full_synthesis_integration": synthesis_e2e_stats
        },
        "production_invariants": {
            "faiss_count": faiss_count,
            "expected_faiss": 744,
            "meta_count": meta_count,
            "expected_meta": 744,
            "dimension": dim_count,
            "expected_dim": 384,
            "invariants_intact": invariants_intact
        }
    }

    # Save output
    output_path = os.path.join(WORKSPACE_DIR, "evaluation_reports", "benchmark_phase6_6_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Benchmark results saved to: {output_path}")

    return benchmark_data


if __name__ == "__main__":
    run_phase6_6_benchmark(iterations=200)
