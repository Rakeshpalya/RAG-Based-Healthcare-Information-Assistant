import os
import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np
import faiss

import hashlib
from backend.services.embedding_service import EmbeddingService


class VectorStoreService:
    """
    Local Vector Store Service powered by Meta's FAISS (Facebook AI Similarity Search).
    
    Why FAISS and Cosine Similarity (IndexFlatIP):
    ---------------------------------------------
    1. Cosine similarity between two vectors u and v is defined as:
          cosine_similarity(u, v) = (u . v) / (||u||_2 * ||v||_2)
    2. When vectors are L2-normalized to unit length (||u||_2 = 1.0 and ||v||_2 = 1.0),
       their cosine similarity reduces exactly to their inner product (dot product):
          cosine_similarity(u, v) = u . v
    3. FAISS 'IndexFlatIP' (Inner Product) computes exact dot products with maximum
       SIMD CPU vectorization. By L2-normalizing all vectors before addition and query,
       IndexFlatIP computes exact cosine similarity scores in the range [-1.0, 1.0].
    4. Metadata is maintained in a clean 1-to-1 positional mapping array, linking each
       FAISS vector index position (0, 1, 2, ...) to its corresponding medical chunk
       text, chunk_id, document_id, page_number, and custom metadata payload.
    """

    DEFAULT_DIMENSION = 384
    DEFAULT_STORAGE_DIR = Path("data/vector_store")

    def __init__(
        self,
        dimension: int = DEFAULT_DIMENSION,
        storage_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initializes the VectorStoreService with a FAISS IndexFlatIP index.

        Args:
            dimension: Embedding dimension (default: 384 for all-MiniLM-L6-v2).
            storage_dir: Directory path for saving/loading persisted indices.
        """
        self.dimension = int(dimension)
        self.storage_dir = Path(storage_dir) if storage_dir else self.DEFAULT_STORAGE_DIR
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.metadata_store: List[Dict[str, Any]] = []

    def count(self) -> int:
        """Returns the total number of vectors currently stored in the FAISS index."""
        return int(self.index.ntotal)

    def reset(self) -> None:
        """Resets the FAISS index and clears all associated metadata records."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store.clear()

    def add_embeddings(
        self,
        embeddings: Union[List[List[float]], np.ndarray],
        metadatas: List[Dict[str, Any]]
    ) -> int:
        """
        Adds pre-computed embeddings and their associated metadata to the FAISS index.

        Args:
            embeddings: List of float vectors or 2D NumPy array of shape (N, dimension).
            metadatas: List of metadata dictionaries corresponding 1:1 to each vector.
                       Expected keys (optional/recommended):
                       - chunk_id: str
                       - text: str
                       - document_id: Optional[str]
                       - page_number: Optional[int]
                       - metadata: Optional[Dict[str, Any]]

        Returns:
            The number of newly added vectors.

        Raises:
            ValueError: If embeddings and metadatas lengths do not match or vector dimension is incorrect.
        """
        if embeddings is None or len(embeddings) == 0:
            return 0

        if len(embeddings) != len(metadatas):
            raise ValueError(
                f"Mismatch: Number of embeddings ({len(embeddings)}) "
                f"must equal number of metadata records ({len(metadatas)})."
            )

        # Convert to contiguous float32 NumPy array
        np_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if np_embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be a 2D array, got ndim={np_embeddings.ndim}.")

        if np_embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: Expected vector dimension {self.dimension}, "
                f"got {np_embeddings.shape[1]}."
            )

        # L2-normalize vectors so that Inner Product equals Cosine Similarity
        faiss.normalize_L2(np_embeddings)

        # Add to FAISS index
        start_idx = self.index.ntotal
        self.index.add(np_embeddings)

        # Store metadata mapping (vector position -> metadata record)
        for i, meta in enumerate(metadatas):
            pos = start_idx + i
            record_user_id = meta.get("user_id")
            record_meta = dict(meta.get("metadata", {}))
            if record_user_id is not None and "user_id" not in record_meta:
                record_meta["user_id"] = record_user_id

            record = {
                "vector_id": pos,
                "chunk_id": meta.get("chunk_id", f"chunk_{pos}"),
                "text": meta.get("text", ""),
                "document_id": meta.get("document_id"),
                "page_number": meta.get("page_number"),
                "user_id": record_user_id,
                "metadata": record_meta
            }
            self.metadata_store.append(record)

        return len(metadatas)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Convenience method: takes chunks, embeds them via EmbeddingService if embeddings
        are not already attached, and adds them to the FAISS index.

        Args:
            chunks: List of chunk dictionaries containing at least a 'text' key.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        # Check if embeddings are already present in chunks
        has_embeddings = all("embedding" in chunk and chunk["embedding"] for chunk in chunks)
        if not has_embeddings:
            embedded_chunks = EmbeddingService.embed_chunks(chunks)
        else:
            embedded_chunks = chunks

        embeddings = [chunk["embedding"] for chunk in embedded_chunks]
        metadatas = []
        for chunk in embedded_chunks:
            meta = {
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text", ""),
                "document_id": chunk.get("document_id"),
                "page_number": chunk.get("page_number"),
                "user_id": chunk.get("user_id"),
                "metadata": {k: v for k, v in chunk.items() if k not in ("chunk_id", "text", "embedding", "document_id", "page_number", "user_id")}
            }
            metadatas.append(meta)

        return self.add_embeddings(embeddings, metadatas)

    def search(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int = 5,
        user_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs cosine similarity search against the FAISS index using a query embedding vector,
        with optional user ownership isolation filtering.

        Args:
            query_embedding: 384-dimensional query vector.
            top_k: Maximum number of most similar results to retrieve (default: 5).
            user_id: Optional user ID to isolate retrieval to documents owned by that user.

        Returns:
            List of ranked results sorted from highest similarity to lowest similarity:
            [
                {
                    "chunk_id": str,
                    "text": str,
                    "similarity_score": float,
                    "document_id": Optional[str],
                    "page_number": Optional[int],
                    "user_id": Optional[int],
                    "metadata": Dict[str, Any]
                },
                ...
            ]
        """
        if query_embedding is None or len(query_embedding) == 0:
            return []

        if self.count() == 0 or top_k <= 0:
            return []

        # Prepare 2D query array (1, dimension)
        query_np = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        if query_np.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: Query dimension {query_np.shape[1]} "
                f"does not match index dimension {self.dimension}."
            )

        # L2-normalize query vector for cosine similarity
        faiss.normalize_L2(query_np)

        # Expand candidate pool if user filtering is enabled or to prevent duplicate vector monopolization
        if user_id is not None:
            k = self.count()
        else:
            k = min(self.count(), max(int(top_k) * 15, 300))

        scores, indices = self.index.search(query_np, k)

        seen_candidates: set = set()
        results: List[Dict[str, Any]] = []
        if len(indices) > 0:
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata_store):
                    continue
                record = self.metadata_store[idx]

                # Apply strict user ownership isolation:
                # When user_id is provided, User A cannot access User B's documents.
                if user_id is not None:
                    rec_user_id = record.get("user_id")
                    if rec_user_id is None:
                        rec_user_id = record.get("metadata", {}).get("user_id")
                    if rec_user_id is not None and str(rec_user_id) != str(user_id):
                        continue


                # Ensure candidate pool is not monopolized by identical duplicate chunk vectors
                c_norm = " ".join(record.get("text", "").strip().split())
                c_hash = hashlib.sha256(c_norm.encode("utf-8")).hexdigest()
                cand_key = c_hash if c_hash else (record.get("document_id"), record.get("chunk_id"))
                if cand_key in seen_candidates:
                    continue
                seen_candidates.add(cand_key)

                results.append({
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "similarity_score": float(score),
                    "document_id": record["document_id"],
                    "page_number": record["page_number"],
                    "user_id": record.get("user_id"),
                    "metadata": record["metadata"]
                })
                if len(results) >= int(top_k):
                    break

        return results

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        user_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        End-to-end semantic search: generates query embedding using EmbeddingService
        and retrieves the top-K most similar chunks from the FAISS index with optional user isolation.

        Args:
            query_text: Natural language search query.
            top_k: Maximum number of results to return.
            user_id: Optional user ID for ownership filtering.

        Returns:
            Ranked list of matching chunk records sorted by similarity score.
        """
        if not query_text or not query_text.strip():
            return []

        query_embedding = EmbeddingService.embed_query(query_text.strip())
        if not query_embedding:
            return []

        return self.search(query_embedding, top_k=top_k, user_id=user_id)

    def save(self, directory_path: Optional[Union[str, Path]] = None) -> str:
        """
        Persists the FAISS index and metadata store to local disk.

        Files created:
        - <directory_path>/index.faiss: Binary FAISS index file.
        - <directory_path>/metadata.json: Serialized chunk text and metadata records.

        Args:
            directory_path: Destination directory (defaults to self.storage_dir).

        Returns:
            Absolute path string of the save directory.
        """
        target_dir = Path(directory_path) if directory_path else self.storage_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "index.faiss"
        metadata_file = target_dir / "metadata.json"

        # Save FAISS index
        faiss.write_index(self.index, str(index_file))

        # Save metadata store as JSON
        payload = {
            "dimension": self.dimension,
            "count": len(self.metadata_store),
            "records": self.metadata_store
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return str(target_dir.resolve())

    def load(self, directory_path: Optional[Union[str, Path]] = None) -> bool:
        """
        Loads the FAISS index and metadata store from local disk.

        Args:
            directory_path: Source directory (defaults to self.storage_dir).

        Returns:
            True if loaded successfully, False if files are missing.

        Raises:
            ValueError: If the loaded index and metadata record counts do not match.
        """
        target_dir = Path(directory_path) if directory_path else self.storage_dir
        index_file = target_dir / "index.faiss"
        metadata_file = target_dir / "metadata.json"

        if not index_file.exists() or not metadata_file.exists():
            return False

        # Load FAISS index
        loaded_index = faiss.read_index(str(index_file))

        # Load metadata store
        with open(metadata_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        records = payload.get("records", [])

        # Validate count consistency
        if loaded_index.ntotal != len(records):
            raise ValueError(
                f"Data inconsistency: FAISS index has {loaded_index.ntotal} vectors, "
                f"but metadata store has {len(records)} records."
            )

        self.index = loaded_index
        self.dimension = loaded_index.d
        self.metadata_store = records
        return True


# ==============================================================================
# Shared / Singleton VectorStoreService Accessor
# ==============================================================================

_shared_vector_store: Optional[VectorStoreService] = None


def get_vector_store_service(storage_dir: Optional[Union[str, Path]] = None) -> VectorStoreService:
    """Returns a singleton shared VectorStoreService instance."""
    global _shared_vector_store
    if _shared_vector_store is None:
        _shared_vector_store = VectorStoreService(storage_dir=storage_dir)
        try:
            _shared_vector_store.load()
        except Exception:
            pass
    elif _shared_vector_store.count() == 0:
        try:
            _shared_vector_store.load()
        except Exception:
            pass
    return _shared_vector_store

