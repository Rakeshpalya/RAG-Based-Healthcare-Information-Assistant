import time
import os
import logging
import threading
from typing import List, Dict, Any, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Singleton service for generating local dense text embeddings
    using Hugging Face Sentence Transformers (all-MiniLM-L6-v2).

    Memory Optimization (Render 512MB RAM):
    SentenceTransformer and PyTorch are lazy-loaded only when embedding
    generation (embed_query or embed_chunks) is actually invoked.
    Startup and lightweight health checks query the known embedding dimension
    without allocating ~400+ MB of PyTorch and Transformer weights into RAM.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    _model_instance: Optional[Any] = None
    _lock = threading.Lock()

    @classmethod
    def get_model(cls) -> Any:
        """
        Lazy-loads and caches the SentenceTransformer model instance (Singleton).
        Ensures the model and heavy PyTorch/Transformer dependencies are loaded
        only once into memory and ONLY when an actual embedding generation request arrives.
        Thread-safe double-checked locking prevents concurrent duplicate loading.
        """
        if cls._model_instance is None:
            with cls._lock:
                if cls._model_instance is None:
                    import gc

                    # Optimize PyTorch CPU execution and thread allocations for low-memory environments
                    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
                    os.environ.setdefault("OMP_NUM_THREADS", "1")
                    os.environ.setdefault("MKL_NUM_THREADS", "1")
                    try:
                        import torch
                        try:
                            torch.set_num_threads(1)
                        except Exception:
                            pass
                        if hasattr(torch, "set_num_interop_threads"):
                            try:
                                torch.set_num_interop_threads(1)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    try:
                        from sentence_transformers import SentenceTransformer

                        logger.info("Lazy-loading SentenceTransformer model '%s' into memory...", cls.MODEL_NAME)
                        start_time = time.perf_counter()
                        try:
                            cls._model_instance = SentenceTransformer(cls.MODEL_NAME, local_files_only=True)
                        except Exception:
                            cls._model_instance = SentenceTransformer(cls.MODEL_NAME)
                        elapsed = time.perf_counter() - start_time
                        logger.info("Model '%s' loaded successfully in %.3f seconds.", cls.MODEL_NAME, elapsed)
                        gc.collect()
                    except Exception as exc:
                        cls._model_instance = None
                        logger.error("Failed to initialize embedding model '%s': %s", cls.MODEL_NAME, exc)
                        raise RuntimeError(
                            f"Embedding model initialization failed for '{cls.MODEL_NAME}': {exc}"
                        ) from exc
        return cls._model_instance

    @classmethod
    def is_model_loaded(cls) -> bool:
        """Checks if the SentenceTransformer model is currently loaded in memory."""
        return cls._model_instance is not None

    @classmethod
    def get_embedding_dimension(cls) -> int:
        """
        Returns the vector dimension of the embedding model (384 for all-MiniLM-L6-v2).
        If the model is already loaded, queries the model instance; otherwise returns
        the known static embedding dimension without incurring a 400+ MB model load.
        """
        if cls._model_instance is not None:
            if hasattr(cls._model_instance, "get_embedding_dimension"):
                return cls._model_instance.get_embedding_dimension()
            return cls._model_instance.get_sentence_embedding_dimension()
        return cls.EMBEDDING_DIMENSION


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
