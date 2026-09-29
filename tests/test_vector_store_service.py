import os
import sys
import time
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.services.vector_store_service import VectorStoreService
from backend.services.embedding_service import EmbeddingService


SYNTHETIC_MEDICAL_CHUNKS = [
    {
        "chunk_id": "MED_CHUNK_0",
        "document_id": "DOC_CARDIO_001",
        "page_number": 1,
        "text": "Hypertension is a condition in which blood pressure remains chronically elevated, increasing cardiovascular risk.",
        "specialty": "Cardiology"
    },
    {
        "chunk_id": "MED_CHUNK_1",
        "document_id": "DOC_ENDO_001",
        "page_number": 1,
        "text": "Type 2 diabetes affects how the body processes and metabolizes blood glucose, leading to insulin resistance.",
        "specialty": "Endocrinology"
    },
    {
        "chunk_id": "MED_CHUNK_2",
        "document_id": "DOC_PULM_001",
        "page_number": 2,
        "text": "Asthma is a chronic inflammatory condition affecting the airways, resulting in recurrent wheezing, cough, and dyspnea.",
        "specialty": "Pulmonology"
    },
    {
        "chunk_id": "MED_CHUNK_3",
        "document_id": "DOC_PREV_001",
        "page_number": 3,
        "text": "Regular physical activity and dietary modification can support cardiovascular health, lower resting heart rate, and improve lipid profiles.",
        "specialty": "Preventive Cardiology"
    },
    {
        "chunk_id": "MED_CHUNK_4",
        "document_id": "DOC_CARDIO_002",
        "page_number": 1,
        "text": "Myocardial infarction, commonly termed a heart attack, happens when blood flow decreases or stops to a part of the heart muscle.",
        "specialty": "Cardiology"
    },
    {
        "chunk_id": "MED_CHUNK_5",
        "document_id": "DOC_NEPHRO_001",
        "page_number": 4,
        "text": "Chronic kidney disease involves gradual loss of renal function over time, frequently exacerbated by uncontrolled systemic hypertension.",
        "specialty": "Nephrology"
    },
    {
        "chunk_id": "MED_CHUNK_6",
        "document_id": "DOC_PHARM_001",
        "page_number": 5,
        "text": "Furosemide is a potent loop diuretic used in the management of edema and fluid retention associated with congestive heart failure.",
        "specialty": "Pharmacology"
    }
]


def test_vector_store_initialization():
    """1. Test that VectorStoreService initializes with expected defaults."""
    store = VectorStoreService()
    assert store.dimension == 384, f"Expected dimension 384, got {store.dimension}"
    assert store.count() == 0, f"Expected 0 vectors on initialization, got {store.count()}"
    assert len(store.metadata_store) == 0
    print("[PASS] test_vector_store_initialization passed.")


def test_correct_embedding_dimension():
    """2. Test dimension matching and dimension validation error handling."""
    store = VectorStoreService(dimension=384)
    expected_dim = EmbeddingService.get_embedding_dimension()
    assert store.dimension == expected_dim == 384

    # Test that adding vectors with wrong dimension raises ValueError
    wrong_dim_vecs = [[0.1] * 128]  # 128 instead of 384
    try:
        store.add_embeddings(wrong_dim_vecs, [{"text": "Invalid vector test"}])
        assert False, "Should have raised ValueError on dimension mismatch"
    except ValueError:
        pass
    print("[PASS] test_correct_embedding_dimension passed.")


def test_adding_embeddings_and_vector_count():
    """3 & 4. Test adding embeddings and verifying stored vector count."""
    store = VectorStoreService()
    sample_chunks = SYNTHETIC_MEDICAL_CHUNKS[:3]
    added = store.add_chunks(sample_chunks)
    assert added == 3, f"Expected 3 added, got {added}"
    assert store.count() == 3, f"Expected store.count() == 3, got {store.count()}"

    # Add 2 more chunks
    added_more = store.add_chunks(SYNTHETIC_MEDICAL_CHUNKS[3:5])
    assert added_more == 2
    assert store.count() == 5
    print("[PASS] test_adding_embeddings_and_vector_count passed. Total vectors: 5")


def test_metadata_mapping():
    """5. Test 1-to-1 position-to-metadata mapping."""
    store = VectorStoreService()
    store.add_chunks(SYNTHETIC_MEDICAL_CHUNKS)

    assert len(store.metadata_store) == len(SYNTHETIC_MEDICAL_CHUNKS)
    for idx, expected in enumerate(SYNTHETIC_MEDICAL_CHUNKS):
        record = store.metadata_store[idx]
        assert record["vector_id"] == idx
        assert record["chunk_id"] == expected["chunk_id"]
        assert record["text"] == expected["text"]
        assert record["document_id"] == expected["document_id"]
        assert record["page_number"] == expected["page_number"]
        assert record["metadata"]["specialty"] == expected["specialty"]
    print("[PASS] test_metadata_mapping passed. All 7 positions correctly mapped.")


def test_similarity_search_and_result_schema():
    """6. Test similarity search and verify returned result keys."""
    store = VectorStoreService()
    store.add_chunks(SYNTHETIC_MEDICAL_CHUNKS)

    results = store.search_by_text("elevated blood pressure condition", top_k=3)
    assert len(results) > 0
    top_result = results[0]

    # Verify result structure
    assert "chunk_id" in top_result
    assert "text" in top_result
    assert "similarity_score" in top_result
    assert "document_id" in top_result
    assert "page_number" in top_result
    assert "metadata" in top_result
    assert isinstance(top_result["similarity_score"], float)

    # Top result should be the hypertension chunk
    assert top_result["chunk_id"] == "MED_CHUNK_0", f"Expected MED_CHUNK_0, got {top_result['chunk_id']}"
    assert "Hypertension" in top_result["text"]
    print(f"[PASS] test_similarity_search_and_result_schema passed. Top match: {top_result['chunk_id']} (score: {top_result['similarity_score']:.4f})")


def test_top_k_behavior():
    """7. Test top_k limiting and edge cases."""
    store = VectorStoreService()
    store.add_chunks(SYNTHETIC_MEDICAL_CHUNKS)

    # top_k = 1
    res1 = store.search_by_text("cardiovascular", top_k=1)
    assert len(res1) == 1

    # top_k = 4
    res4 = store.search_by_text("cardiovascular", top_k=4)
    assert len(res4) == 4

    # top_k larger than stored count (e.g. 50 when count is 7)
    res_all = store.search_by_text("health", top_k=50)
    assert len(res_all) == 7
    print("[PASS] test_top_k_behavior passed.")


def test_result_ordering():
    """8. Test that results are strictly sorted in descending similarity order."""
    store = VectorStoreService()
    store.add_chunks(SYNTHETIC_MEDICAL_CHUNKS)

    results = store.search_by_text("shortness of breath and respiratory issues", top_k=5)
    assert len(results) == 5

    # Check non-increasing similarity score order
    for i in range(len(results) - 1):
        score_curr = results[i]["similarity_score"]
        score_next = results[i + 1]["similarity_score"]
        assert score_curr >= score_next, f"Ordering violation: {score_curr} < {score_next}"
    print("[PASS] test_result_ordering passed. Scores ordered strictly descending.")


def test_empty_query_handling():
    """9. Test empty query strings, empty embeddings, and empty index."""
    store = VectorStoreService()
    
    # Search on empty store
    assert store.search_by_text("test query") == []
    assert store.search([0.1] * 384) == []

    store.add_chunks(SYNTHETIC_MEDICAL_CHUNKS)
    # Empty query strings
    assert store.search_by_text("") == []
    assert store.search_by_text("   ") == []
    # Empty embedding list
    assert store.search([]) == []
    print("[PASS] test_empty_query_handling passed.")


def test_persistence_save_load_and_reload_search():
    """10, 11, & 12. Test save to disk, load from disk, and search after reload."""
    original_store = VectorStoreService()
    original_store.add_chunks(SYNTHETIC_MEDICAL_CHUNKS)

    query = "blood glucose insulin resistance"
    original_results = original_store.search_by_text(query, top_k=3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 10. Save
        saved_path = original_store.save(tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "index.faiss"))
        assert os.path.exists(os.path.join(tmp_dir, "metadata.json"))

        # 11. Load into a new instance
        reloaded_store = VectorStoreService(storage_dir=tmp_dir)
        success = reloaded_store.load()
        assert success is True
        assert reloaded_store.count() == original_store.count() == len(SYNTHETIC_MEDICAL_CHUNKS)
        assert len(reloaded_store.metadata_store) == len(original_store.metadata_store)

        # 12. Search after reload
        reloaded_results = reloaded_store.search_by_text(query, top_k=3)
        assert len(reloaded_results) == len(original_results)

        for orig, reloaded in zip(original_results, reloaded_results):
            assert orig["chunk_id"] == reloaded["chunk_id"]
            assert abs(orig["similarity_score"] - reloaded["similarity_score"]) < 1e-5
            assert orig["text"] == reloaded["text"]

    print("[PASS] test_persistence_save_load_and_reload_search passed. Identical results across persistence reload.")


def run_semantic_search_demo_and_benchmarks():
    """
    Demonstration of semantic similarity search using FAISS IndexFlatIP
    and performance timing benchmarks.
    """
    print("\n" + "=" * 70)
    print("PHASE 5: FAISS SEMANTIC SIMILARITY SEARCH DEMO")
    print("=" * 70)

    store = VectorStoreService()

    # 1. Measure index insertion time
    t0 = time.perf_counter()
    store.add_chunks(SYNTHETIC_MEDICAL_CHUNKS)
    insertion_time_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Index Type           : FAISS IndexFlatIP (Inner Product / Cosine)")
    print(f"Embedding Model      : sentence-transformers/all-MiniLM-L6-v2")
    print(f"Vector Dimension     : {store.dimension}")
    print(f"Indexed Chunks       : {store.count()}")
    print(f"Index Insertion Time : {insertion_time_ms:.2f} ms\n")

    # 2. Test semantic query
    test_query = "What condition is associated with high blood pressure?"

    # Measure query embedding time
    t_embed_start = time.perf_counter()
    query_vec = EmbeddingService.embed_query(test_query)
    query_embed_time_ms = (time.perf_counter() - t_embed_start) * 1000.0

    # Measure FAISS vector search time
    t_search_start = time.perf_counter()
    results = store.search(query_vec, top_k=3)
    vector_search_time_ms = (time.perf_counter() - t_search_start) * 1000.0

    total_search_time_ms = query_embed_time_ms + vector_search_time_ms

    print(f"Query: \"{test_query}\"")
    print("-" * 70)
    print(f"Query Embedding Time : {query_embed_time_ms:.2f} ms")
    print(f"Vector Search Time   : {vector_search_time_ms:.2f} ms")
    print(f"Total Search Time    : {total_search_time_ms:.2f} ms\n")

    print("Top Results (ranked by cosine similarity):")
    for rank, res in enumerate(results, start=1):
        print(f"\n{rank}. Chunk ID        : {res['chunk_id']}")
        print(f"   Similarity Score: {res['similarity_score']:.4f}")
        print(f"   Specialty       : {res['metadata'].get('specialty', 'N/A')}")
        print(f"   Retrieved Text  : \"{res['text']}\"")

    print("\n" + "-" * 70)
    print("Semantic Search Insight:")
    print("Notice that although the query asked about 'high blood pressure',")
    print("FAISS successfully retrieved MED_CHUNK_0 ('Hypertension') with the highest score,")
    print("demonstrating semantic understanding beyond literal keyword matching.")
    print("=" * 70)

    # 3. Document Accuracy Consideration
    print("\n[NOTE: Accuracy Consideration]")
    print("FAISS performs geometric nearest-neighbor similarity search in vector space.")
    print("It computes mathematical proximity between embeddings, but does NOT verify")
    print("clinical correctness, medical truth, or evidence veracity.")
    print("Clinical accuracy and relevance will be evaluated in subsequent RAG phases using")
    print("Precision@K, Recall@K, MRR, answer faithfulness, and source citation verification.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_vector_store_initialization()
    test_correct_embedding_dimension()
    test_adding_embeddings_and_vector_count()
    test_metadata_mapping()
    test_similarity_search_and_result_schema()
    test_top_k_behavior()
    test_result_ordering()
    test_empty_query_handling()
    test_persistence_save_load_and_reload_search()
    run_semantic_search_demo_and_benchmarks()
    print("[SUCCESS] All VectorStoreService unit tests and benchmarks passed successfully!")
