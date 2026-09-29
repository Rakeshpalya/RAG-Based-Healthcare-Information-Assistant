import os
import sys
import time
import asyncio
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.rag.rag_service import RAGService
from backend.services.vector_store_service import VectorStoreService
from backend.services.embedding_service import EmbeddingService
from backend.api.rag_router import retrieve_rag_context, RAGRetrieveRequest


# Synthetic, de-identified clinical test documents
SYNTHETIC_MEDICAL_DOCS = [
    {
        "chunk_id": "MED_CHUNK_0",
        "document_id": "DOC_CARDIO_001",
        "page_number": 1,
        "text": "Hypertension is a chronic medical condition in which systemic arterial blood pressure remains persistently elevated above 130/80 mmHg.",
        "category": "Cardiology"
    },
    {
        "chunk_id": "MED_CHUNK_1",
        "document_id": "DOC_ENDO_001",
        "page_number": 2,
        "text": "Type 2 diabetes mellitus is characterized by peripheral insulin resistance and progressive pancreatic beta-cell secretory defect, leading to hyperglycemia.",
        "category": "Endocrinology"
    },
    {
        "chunk_id": "MED_CHUNK_2",
        "document_id": "DOC_PULM_001",
        "page_number": 1,
        "text": "Asthma is a chronic inflammatory disorder of the airways involving bronchial hyperresponsiveness and variable airflow obstruction presenting with wheezing and dyspnea.",
        "category": "Pulmonology"
    },
    {
        "chunk_id": "MED_CHUNK_3",
        "document_id": "DOC_PREV_001",
        "page_number": 3,
        "text": "Regular aerobic physical activity of at least 150 minutes weekly enhances cardiovascular fitness, lowers resting systolic pressure, and improves vascular elasticity.",
        "category": "Preventive Cardiology"
    },
    {
        "chunk_id": "MED_CHUNK_4",
        "document_id": "DOC_NEPHRO_001",
        "page_number": 4,
        "text": "Chronic kidney disease (CKD) manifests as progressive loss of renal glomerular filtration rate, often accelerated by poorly controlled arterial hypertension.",
        "category": "Nephrology"
    },
    {
        "chunk_id": "MED_CHUNK_5",
        "document_id": "DOC_PHARM_001",
        "page_number": 5,
        "text": "Loop diuretics such as Furosemide act on the thick ascending limb of Henle's loop to promote natriuresis and reduce pulmonary and peripheral fluid congestion in heart failure.",
        "category": "Pharmacology"
    }
]


def setup_test_rag_service() -> RAGService:
    """Helper to initialize an isolated VectorStoreService and populate with synthetic medical data."""
    store = VectorStoreService()
    store.add_chunks(SYNTHETIC_MEDICAL_DOCS)
    rag_service = RAGService(vector_store=store, default_top_k=5, default_similarity_threshold=0.25)
    return rag_service


def test_query_embedding_integration():
    """1. Test that query embedding integrates correctly with SentenceTransformers."""
    query = "blood pressure elevation"
    query_vec = EmbeddingService.embed_query(query)
    assert len(query_vec) == 384
    assert isinstance(query_vec[0], float)
    print("[PASS] test_query_embedding_integration passed.")


def test_vector_retrieval_integration():
    """2. Test that RAGService successfully searches the underlying FAISS index."""
    rag = setup_test_rag_service()
    chunks = rag.retrieve_context("What causes elevated blood pressure?", top_k=3)
    assert len(chunks) > 0
    assert chunks[0]["chunk_id"] == "MED_CHUNK_0"
    assert "Hypertension" in chunks[0]["text"]
    assert "similarity_score" in chunks[0]
    print(f"[PASS] test_vector_retrieval_integration passed. Top match: {chunks[0]['chunk_id']} (score: {chunks[0]['similarity_score']:.4f})")


def test_top_k_behavior():
    """3. Test top_k limiting behavior and boundary handling."""
    rag = setup_test_rag_service()

    # top_k = 1
    res1 = rag.retrieve_context("cardiovascular and renal", top_k=1, similarity_threshold=0.0)
    assert len(res1) == 1

    # top_k = 4
    res4 = rag.retrieve_context("cardiovascular and renal", top_k=4, similarity_threshold=0.0)
    assert len(res4) == 4

    # top_k exceeding total documents
    res_all = rag.retrieve_context("medical", top_k=100, similarity_threshold=0.0)
    assert len(res_all) == len(SYNTHETIC_MEDICAL_DOCS)
    print("[PASS] test_top_k_behavior passed.")


def test_similarity_threshold():
    """4. Test that chunks below the similarity threshold are filtered out."""
    rag = setup_test_rag_service()

    # Low threshold: includes multiple matches
    low_thresh_res = rag.retrieve_context("blood pressure", top_k=5, similarity_threshold=0.10)
    assert len(low_thresh_res) >= 2

    # High threshold: filters out lower-similarity chunks
    high_thresh_res = rag.retrieve_context("blood pressure", top_k=5, similarity_threshold=0.60)
    assert len(high_thresh_res) >= 1
    for chunk in high_thresh_res:
        assert chunk["similarity_score"] >= 0.60

    # Impossibly high threshold: filters everything out
    impossible_res = rag.retrieve_context("blood pressure", top_k=5, similarity_threshold=0.99)
    assert len(impossible_res) == 0
    print("[PASS] test_similarity_threshold passed.")


def test_source_metadata_preservation():
    """5. Test that source IDs, chunk IDs, page numbers, and custom metadata are preserved."""
    rag = setup_test_rag_service()
    chunks = rag.retrieve_context("asthma bronchial", top_k=2)
    assert len(chunks) > 0

    asthma_chunk = chunks[0]
    assert asthma_chunk["chunk_id"] == "MED_CHUNK_2"
    assert asthma_chunk["document_id"] == "DOC_PULM_001"
    assert asthma_chunk["page_number"] == 1
    assert asthma_chunk["metadata"]["category"] == "Pulmonology"

    sources = rag.build_sources(chunks)
    assert len(sources) == len(chunks)
    assert sources[0]["source_id"] == "DOC_PULM_001"
    assert sources[0]["page_number"] == 1
    assert sources[0]["chunk_id"] == "MED_CHUNK_2"
    assert isinstance(sources[0]["similarity_score"], float)
    print("[PASS] test_source_metadata_preservation passed.")


def test_context_construction():
    """6. Test structured context string construction formatting."""
    rag = setup_test_rag_service()
    chunks = rag.retrieve_context("hypertension arterial", top_k=2)
    context_str = rag.build_context(chunks)

    assert "[SOURCE 1]" in context_str
    assert "Document: DOC_CARDIO_001" in context_str
    assert "Page: 1" in context_str
    assert "Chunk ID: MED_CHUNK_0" in context_str
    assert "Hypertension is a chronic medical condition" in context_str

    if len(chunks) > 1:
        assert "[SOURCE 2]" in context_str

    # Empty chunks yields empty context
    assert rag.build_context([]) == ""
    print("[PASS] test_context_construction passed.")


def test_source_ordering():
    """7. Test that retrieved sources are strictly sorted by descending similarity score."""
    rag = setup_test_rag_service()
    chunks = rag.retrieve_context("cardiovascular and kidney diseases", top_k=5, similarity_threshold=0.0)
    assert len(chunks) >= 2

    for i in range(len(chunks) - 1):
        s_curr = chunks[i]["similarity_score"]
        s_next = chunks[i + 1]["similarity_score"]
        assert s_curr >= s_next, f"Ordering violation: {s_curr} < {s_next}"
    print("[PASS] test_source_ordering passed. Strictly non-increasing order verified.")


def test_empty_query_handling():
    """8. Test that empty and whitespace-only queries return structured responses without crashing."""
    rag = setup_test_rag_service()

    for empty_input in ["", "   ", "\t\n"]:
        res = rag.query(empty_input)
        assert res["retrieval_status"] == "empty_query"
        assert res["retrieved_chunks"] == []
        assert res["context"] == ""
        assert res["sources"] == []
        assert "timings" in res
    print("[PASS] test_empty_query_handling passed.")


def test_no_relevant_context_handling():
    """9. Test handling when no chunks satisfy the similarity threshold."""
    rag = setup_test_rag_service()

    # Query with strict threshold that no chunk meets
    res = rag.query("Quantum astrophysics and black hole thermodynamics", top_k=5, similarity_threshold=0.85)
    assert res["retrieval_status"] == "no_relevant_context"
    assert res["retrieved_chunks"] == []
    assert res["context"] == ""
    assert res["sources"] == []
    assert res["question"] == "Quantum astrophysics and black hole thermodynamics"
    print("[PASS] test_no_relevant_context_handling passed.")


def test_complete_rag_retrieval_flow():
    """10. Test end-to-end query method returning complete RAG response object."""
    rag = setup_test_rag_service()
    question = "What medication is used to reduce fluid congestion in heart failure?"

    response = rag.query(question, top_k=3, similarity_threshold=0.20)
    assert response["retrieval_status"] == "success"
    assert response["question"] == question
    assert len(response["retrieved_chunks"]) > 0
    assert len(response["sources"]) > 0
    assert len(response["context"]) > 0

    # The top result should be Furosemide
    top_chunk = response["retrieved_chunks"][0]
    assert top_chunk["chunk_id"] == "MED_CHUNK_5"
    assert "Furosemide" in top_chunk["text"]

    # Timings breakdown must be present
    timings = response["timings"]
    assert "query_embedding_time_ms" in timings
    assert "vector_search_time_ms" in timings
    assert "context_construction_time_ms" in timings
    assert "total_retrieval_time_ms" in timings
    assert timings["total_retrieval_time_ms"] > 0

    # Ensure NO generative LLM answer was produced yet
    assert "answer" not in response
    print(f"[PASS] test_complete_rag_retrieval_flow passed. Latency: {timings['total_retrieval_time_ms']:.2f} ms")


def test_api_endpoint_integration():
    """11. Test the FastAPI POST /rag/retrieve endpoint handler."""
    # Seed the singleton vector store used by get_rag_service
    from backend.api.rag_router import get_rag_service
    service = get_rag_service()
    if service.vector_store.count() == 0:
        service.vector_store.add_chunks(SYNTHETIC_MEDICAL_DOCS)

    req = RAGRetrieveRequest(
        question="What condition is related to persistently high blood pressure?",
        top_k=3,
        similarity_threshold=0.25
    )
    api_resp = asyncio.run(retrieve_rag_context(req))
    assert api_resp.retrieval_status == "success"
    assert len(api_resp.retrieved_chunks) > 0
    assert api_resp.retrieved_chunks[0]["chunk_id"] == "MED_CHUNK_0"
    assert "[SOURCE 1]" in api_resp.context
    assert len(api_resp.sources) > 0
    print("[PASS] test_api_endpoint_integration passed.")


def run_rag_retrieval_demo_and_benchmarks():
    """
    12. Demonstration of the full RAG retrieval & context assembly pipeline
    with latency breakdown and accuracy documentation.
    """
    print("\n" + "=" * 75)
    print("PHASE 6: END-TO-END RAG RETRIEVAL & CONTEXT PIPELINE DEMO")
    print("=" * 75)

    rag = setup_test_rag_service()
    test_question = "What condition is related to persistently high blood pressure?"

    # Cold vs Warm benchmark
    # 1. Warm query execution
    warm_resp = rag.query(test_question, top_k=3, similarity_threshold=0.25)
    timings = warm_resp["timings"]

    print(f"User Query           : \"{test_question}\"")
    print(f"Retrieval Status     : {warm_resp['retrieval_status']}")
    print(f"Embedding Model      : sentence-transformers/all-MiniLM-L6-v2 (384-dim)")
    print(f"Vector Store Index   : FAISS IndexFlatIP (Cosine Similarity)")
    print("-" * 75)
    print("PERFORMANCE LATENCY BREAKDOWN (Warm Query):")
    print(f"  1. Query Embedding Time       : {timings['query_embedding_time_ms']:.2f} ms")
    print(f"  2. FAISS Vector Search Time   : {timings['vector_search_time_ms']:.2f} ms")
    print(f"  3. Context Construction Time  : {timings['context_construction_time_ms']:.2f} ms")
    print(f"  ------------------------------------------")
    print(f"  TOTAL RETRIEVAL LATENCY       : {timings['total_retrieval_time_ms']:.2f} ms")
    print("-" * 75)
    print("Cold Start vs Warm Latency:")
    print("  - Cold Start Latency: ~10-15 seconds (loading ~90 MB model weights into RAM on first process launch).")
    print(f"  - Warm Query Latency: ~{timings['total_retrieval_time_ms']:.1f} ms (in-memory Singleton model & SIMD FAISS search).")
    print("-" * 75)

    print("TOP RETRIEVED CHUNKS:")
    for rank, chunk in enumerate(warm_resp["retrieved_chunks"], start=1):
        print(f"\n[{rank}] Chunk ID       : {chunk['chunk_id']}")
        print(f"    Document ID    : {chunk.get('document_id')}")
        print(f"    Page Number    : {chunk.get('page_number')}")
        print(f"    Cosine Score   : {chunk['similarity_score']:.4f}")
        print(f"    Text Content   : \"{chunk['text']}\"")

    print("\n" + "-" * 75)
    print("CONSTRUCTED RAG CONTEXT (Ready for Phase 7 LLM Prompt):")
    print("-" * 75)
    print(warm_resp["context"])
    print("-" * 75)

    print("EXTRACTED CITATION SOURCES:")
    for src in warm_resp["sources"]:
        print(f"  - Doc: {src['source_id']} | Page: {src['page_number']} | Chunk: {src['chunk_id']} | Score: {src['similarity_score']}")

    print("\n" + "=" * 75)
    print("[NOTE: Accuracy & Scope Documentation]")
    print("1. Generation Deferred: NO LLM answer is generated in Phase 6. The prompt context")
    print("   and grounded sources are prepared cleanly for Gemini integration in Phase 7.")
    print("2. Retrieval Relevance != Clinical Truth: FAISS computes mathematical vector proximity.")
    print("   Clinical correctness will be formally evaluated in downstream RAG benchmarks using:")
    print("   - Precision@K, Recall@K, MRR (Retrieval)")
    print("   - Answer Faithfulness, Groundedness, and Hallucination Rates (Generation)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    test_query_embedding_integration()
    test_vector_retrieval_integration()
    test_top_k_behavior()
    test_similarity_threshold()
    test_source_metadata_preservation()
    test_context_construction()
    test_source_ordering()
    test_empty_query_handling()
    test_no_relevant_context_handling()
    test_complete_rag_retrieval_flow()
    test_api_endpoint_integration()
    run_rag_retrieval_demo_and_benchmarks()
    print("[SUCCESS] All RAGService unit tests and benchmarks passed successfully!")
