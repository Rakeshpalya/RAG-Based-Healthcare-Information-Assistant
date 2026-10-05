"""
RAG (Retrieval-Augmented Generation) package for AI-Healthcare-Agent.
"""

from backend.rag.rag_service import RAGService
from backend.rag.query_expander import MedicalQueryExpander
from backend.rag.query_expansion import (
    QueryIntent,
    QueryIntentResult,
    normalize_medical_query,
    CANONICAL_LIFESTYLE_EXPANSION_TERMS,
)

__all__ = [
    "RAGService",
    "MedicalQueryExpander",
    "QueryIntent",
    "QueryIntentResult",
    "normalize_medical_query",
    "CANONICAL_LIFESTYLE_EXPANSION_TERMS",
]
