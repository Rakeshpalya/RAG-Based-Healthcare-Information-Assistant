import time
from typing import List, Dict, Any, Union, Optional
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Singleton service for generating local dense text embeddings
    using Hugging Face Sentence Transformers (all-MiniLM-L6-v2).
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    _model_instance: Optional[SentenceTransformer] = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """
        Lazy-loads and caches the SentenceTransformer model instance (Singleton).
        Ensures the model is loaded only once into memory.
        """
        if cls._model_instance is None:
            print(f"[INFO] Loading SentenceTransformer model '{cls.MODEL_NAME}' into memory...")
            start_time = time.perf_counter()
            try:
                cls._model_instance = SentenceTransformer(cls.MODEL_NAME, local_files_only=True)
            except Exception:
                cls._model_instance = SentenceTransformer(cls.MODEL_NAME)
            elapsed = time.perf_counter() - start_time
            print(f"[INFO] Model '{cls.MODEL_NAME}' loaded successfully in {elapsed:.3f} seconds.")
        return cls._model_instance

    @classmethod
    def get_embedding_dimension(cls) -> int:
        """
        Returns the vector dimension of the loaded embedding model (384 for all-MiniLM-L6-v2).
        """
        model = cls.get_model()
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()


    @classmethod
    def embed_query(cls, query_text: str) -> List[float]:
        """
        Generates a normalized embedding vector for a single search query string.

        Args:
            query_text: Single search query or text string.

        Returns:
            List of floats representing the normalized 384-dimensional embedding vector.
        """
        if not query_text or not query_text.strip():
            return []

        model = cls.get_model()
        vector = model.encode(query_text.strip(), normalize_embeddings=True)
        return vector.tolist()

    @classmethod
    def embed_chunks(
        cls,
        chunks: List[Union[str, Dict[str, Any]]],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Generates normalized embedding vectors for a list of text chunks using batch processing.

        Args:
            chunks: List of text strings or chunk metadata dictionaries containing a 'text' key.
            batch_size: Number of texts to process per batch (default: 32).

        Returns:
            List of dictionaries containing chunk details, embedding vector, vector dimension, and processing time.
        """
        if not chunks:
            return []

        # Extract text strings and track original objects
        text_list: List[str] = []
        for item in chunks:
            if isinstance(item, str):
                text_list.append(item.strip())
            elif isinstance(item, dict) and "text" in item:
                text_list.append(str(item["text"]).strip())
            else:
                text_list.append("")

        model = cls.get_model()

        start_time = time.perf_counter()
        # Batch encode with normalization for cosine similarity
        raw_vectors = model.encode(
            text_list,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        total_time_ms = (time.perf_counter() - start_time) * 1000.0

        results: List[Dict[str, Any]] = []
        embedding_dim = cls.get_embedding_dimension()

        for idx, item in enumerate(chunks):

            vector = raw_vectors[idx].tolist()

            if isinstance(item, dict):
                chunk_res = dict(item)
                chunk_res["embedding"] = vector
                chunk_res["embedding_dim"] = embedding_dim
            else:
                chunk_res = {
                    "chunk_id": f"chunk_{idx}",
                    "text": item,
                    "embedding": vector,
                    "embedding_dim": embedding_dim
                }
            results.append(chunk_res)

        return results
