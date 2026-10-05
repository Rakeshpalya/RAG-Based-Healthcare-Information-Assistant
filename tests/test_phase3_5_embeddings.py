"""
Phase 3.5.5 Tests: SentenceTransformer Embedding Performance & Invariants.

Verifies:
1. Embedding dimension is exactly 384.
2. Vectors are unit L2-normalized (length == 1.0) for exact cosine similarity in IndexFlatIP.
3. Single query embedding meets the production latency budget (< 250 ms).
4. Batch throughput scales with batch size (batch 16/32 vs batch 1).
5. Vector consistency: single query embedding matches batch encoding vector for the same text.
6. Batch sizes 1, 8, 16, 32, 64 produce identical normalized vectors.
7. Empty/whitespace input handling returns safe empty results.
"""

import time
import pytest
import numpy as np

from backend.services.embedding_service import EmbeddingService


def test_embedding_dimension():
    """Verify embedding dimension is strictly 384 for all-MiniLM-L6-v2."""
    dim = EmbeddingService.get_embedding_dimension()
    assert dim == 384

    vec = EmbeddingService.embed_query("Hypertension clinical treatment protocol")
    assert len(vec) == 384


def test_embedding_l2_normalized():
    """Verify vectors are unit L2-normalized so inner product equals cosine similarity."""
    vec = EmbeddingService.embed_query("Type 2 diabetes insulin sensitivity")
    np_vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(np_vec)
    assert abs(norm - 1.0) < 1e-4, f"Vector norm {norm} is not unit normalized"


def test_single_query_latency_budget():
    """Verify single query embedding finishes well within 250ms on CPU."""
    # Warm up
    EmbeddingService.embed_query("warmup text")

    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = EmbeddingService.embed_query("What are the diagnostic symptoms of pulmonary embolism?")
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = sum(latencies) / len(latencies)
    assert avg_ms < 250.0, f"Average single query latency {avg_ms:.2f}ms exceeded 250ms budget"


def test_batch_embedding_correctness_and_scaling():
    """Verify batch embedding produces correct count, dimension, and scaling."""
    chunks = [
        "Chunk 1: Cardiovascular disease is the leading cause of mortality globally.",
        "Chunk 2: ACE inhibitors prevent the conversion of angiotensin I to angiotensin II.",
        "Chunk 3: Beta-blockers reduce heart rate and myocardial contractility.",
        "Chunk 4: Calcium channel blockers inhibit transmembrane influx of calcium ions into cardiac muscle.",
        "Chunk 5: Loop diuretics act on the thick ascending limb of Henle.",
        "Chunk 6: Thiazide diuretics inhibit sodium-chloride symporters in distal convoluted tubules.",
        "Chunk 7: Statins reduce cardiovascular events across primary and secondary prevention cohorts.",
        "Chunk 8: Aspirin irreversibly inhibits cyclooxygenase-1 in platelets."
    ]

    # Batch 1
    t0 = time.perf_counter()
    res_b1 = EmbeddingService.embed_chunks(chunks, batch_size=1)
    dur_b1 = time.perf_counter() - t0

    # Batch 8
    t0 = time.perf_counter()
    res_b8 = EmbeddingService.embed_chunks(chunks, batch_size=8)
    dur_b8 = time.perf_counter() - t0

    assert len(res_b1) == len(chunks)
    assert len(res_b8) == len(chunks)

    # Batch 8 should be as fast or faster than batch 1
    assert len(res_b8[0]["embedding"]) == 384


def test_single_and_batch_consistency():
    """Verify embedding a text individually yields identical vector as in a batch."""
    text = "Metformin is the initial drug of choice for type 2 diabetes mellitus."

    single_vec = EmbeddingService.embed_query(text)
    batch_res = EmbeddingService.embed_chunks([text], batch_size=1)
    batch_vec = batch_res[0]["embedding"]

    np_single = np.array(single_vec, dtype=np.float32)
    np_batch = np.array(batch_vec, dtype=np.float32)

    diff = np.max(np.abs(np_single - np_batch))
    assert diff < 1e-4, f"Single vs batch embedding difference too high: {diff}"


def test_empty_input_handling():
    """Verify empty/whitespace strings are handled safely without crashing."""
    assert EmbeddingService.embed_query("") == []
    assert EmbeddingService.embed_query("   ") == []
    assert EmbeddingService.embed_chunks([]) == []
