import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.services.embedding_service import EmbeddingService
from backend.services.chunking_service import TextChunkingService


def test_model_singleton_reuse():
    """Test that EmbeddingService loads the model once and reuses the instance."""
    model1 = EmbeddingService.get_model()
    model2 = EmbeddingService.get_model()
    assert model1 is model2, "Model instance was not reused (Singleton pattern failed)"
    print("[PASS] test_model_singleton_reuse passed.")


def test_embedding_dimensions():
    """Test that embedding dimensions equal 384 for all-MiniLM-L6-v2."""
    dim = EmbeddingService.get_embedding_dimension()
    assert dim == 384, f"Expected 384 dimensions, got {dim}"
    print(f"[PASS] test_embedding_dimensions passed. Dimension: {dim}")


def test_single_query_embedding():
    """Test single query string embedding generation."""
    query = "What are the contraindications for Furosemide in cardiac patients?"
    vector = EmbeddingService.embed_query(query)
    assert len(vector) == 384
    assert isinstance(vector[0], float)
    print("[PASS] test_single_query_embedding passed.")


def test_empty_input_handling():
    """Test empty string and empty list handling."""
    assert EmbeddingService.embed_query("") == []
    assert EmbeddingService.embed_query("   ") == []
    assert EmbeddingService.embed_chunks([]) == []
    print("[PASS] test_empty_input_handling passed.")


def test_deterministic_repeatability():
    """Test that identical text generates identical embedding vectors."""
    text = "Patient Jane Smith diagnosed with Stage II Hypertension."
    vec1 = EmbeddingService.embed_query(text)
    vec2 = EmbeddingService.embed_query(text)
    assert len(vec1) == len(vec2) == 384
    for val1, val2 in zip(vec1, vec2):
        assert abs(val1 - val2) < 1e-6, "Embeddings are not deterministic for identical input"
    print("[PASS] test_deterministic_repeatability passed.")


def test_batch_chunk_embeddings():
    """Test batch chunk embedding generation."""
    chunks = [
        {"chunk_id": "chunk_0", "text": "Patient has acute dyspnea and bilateral edema."},
        {"chunk_id": "chunk_1", "text": "Administered intravenous Furosemide 40mg daily."},
        {"chunk_id": "chunk_2", "text": "Follow-up Cardiology consultation scheduled in 7 days."}
    ]
    embedded_chunks = EmbeddingService.embed_chunks(chunks, batch_size=32)
    assert len(embedded_chunks) == 3
    for chunk in embedded_chunks:
        assert "embedding" in chunk
        assert len(chunk["embedding"]) == 384
        assert chunk["embedding_dim"] == 384
    print("[PASS] test_batch_chunk_embeddings passed.")


def run_embedding_performance_demo():
    """Demonstration of text embedding generation on medical chunks with performance timing."""
    sample_medical_text = (
        "CLINICAL ASSESSMENT & TREATMENT PLAN\n\n"
        "1. DIAGNOSIS: Acute Decompensated Heart Failure (ADHF). Patient presented with severe orthopnea and pedal edema.\n\n"
        "2. INTERVENTION: Administered IV loop diuretics (Furosemide 80mg) achieving 2.5L diuresis in 24 hours.\n\n"
        "3. PHARMACOLOGY: Started Lisinopril 5mg daily for ACE-inhibition and blood pressure regulation.\n\n"
        "4. OUTCOME: Respiratory rate stabilized at 16 breaths/min. Patient discharged with outpatient cardiology follow-up."
    )

    # Chunk the medical text using TextChunkingService
    chunks = TextChunkingService.chunk_text(sample_medical_text, chunk_size=30, chunk_overlap=5)

    print("\n" + "=" * 70)
    print("MEDICAL TEXT EMBEDDING DEMO")
    print("=" * 70)
    print(f"Model Name           : {EmbeddingService.MODEL_NAME}")
    print(f"Embedding Dimension  : {EmbeddingService.get_embedding_dimension()}")
    print(f"Number of Chunks     : {len(chunks)}\n")

    # Measure batch embedding processing time
    start_time = time.perf_counter()
    embedded_chunks = EmbeddingService.embed_chunks(chunks, batch_size=32)
    total_time_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"Total Batch Processing Time: {total_time_ms:.2f} ms")
    print(f"Average Time Per Chunk     : {(total_time_ms / max(1, len(chunks))):.2f} ms\n")

    for idx, item in enumerate(embedded_chunks):
        vec = item["embedding"]
        first_5_vals = [round(val, 4) for val in vec[:5]]
        print(f"--- [{item['chunk_id'].upper()}] ---")
        print(f"Word Count      : {item['word_count']}")
        print(f"Vector Dim      : {item['embedding_dim']}")
        print(f"Vector Preview  : {first_5_vals} ... (showing first 5 of 384 values)")
        print(f"Text Preview    : {item['text'][:80]}...\n")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_model_singleton_reuse()
    test_embedding_dimensions()
    test_single_query_embedding()
    test_empty_input_handling()
    test_deterministic_repeatability()
    test_batch_chunk_embeddings()
    run_embedding_performance_demo()
    print("[SUCCESS] All EmbeddingService unit tests & performance demo passed successfully!")
