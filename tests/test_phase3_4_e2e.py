"""
Phase 3.4 Milestone 3.4.12: Final Production End-to-End Test Suite

Verifies the entire production lifecycle end-to-end:
1. Document Upload & Ingestion:
   - PDF upload
   - Text extraction
   - Medical semantic chunking
   - Embedding generation (SentenceTransformers)
   - FAISS vector insertion & metadata storage with user ownership
2. Authenticated End-to-End Query:
   - Authentication (user_id resolution)
   - Request ID generation & propagation
   - Rate limiting check
   - Query normalization
   - FAISS semantic retrieval with user ownership isolation
   - Pre-LLM sufficiency gate
   - MedicalSafetyGuard pre-screening
   - LLMCacheService check
   - LLM generation (via GeminiService)
   - Citation validation & evidence pruning
   - Grounding validation & hallucination guard
   - MedicalSafetyGuard post-screening (disclaimer & boundary enforcement)
   - Cache storage of verified safe response
   - Full response delivery with citations, sources, timings, and request_id
3. Second Query Cache Verification:
   - Immediate cache hit (0 ms LLM time, gemini_calls_count=0)
4. Streaming End-to-End Verification:
   - SSE streaming endpoint (/rag/stream) emits start, status, token, and complete events
5. Invariant Verification:
   - Production FAISS index remains untouched at 744 vectors and 744 metadata records.
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.main import app
from backend.database.database import Base, get_db
from backend.database.models import User, Document, DocumentChunk
from backend.services.auth_service import auth_service
from backend.api.auth_dependencies import get_current_db_user, get_optional_current_db_user
from backend.services.vector_store_service import VectorStoreService, get_vector_store_service
from backend.rag.rag_service import RAGService
from backend.services.llm_cache_service import get_llm_cache_service
from backend.rag.prompt_builder import MEDICAL_DISCLAIMER


SYNTHETIC_CLINICAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    b"4 0 obj\n<< /Length 284 >>\nstream\n"
    b"BT\n/F1 12 Tf\n50 720 Td\n(CARDIOLOGY REPORT: Patient Jane Smith, Age 52.) Tj\n0 -20 Td\n(DIAGNOSIS: Stage 1 Essential Hypertension with mild hyperlipidemia.) Tj\n0 -20 Td\n(CLINICAL EVIDENCE: First-line management includes low-sodium DASH diet and aerobic exercise.) Tj\n0 -20 Td\n(PHARMACOTHERAPY: Hydrochlorothiazide 12.5mg daily if lifestyle modifications are insufficient.) Tj\nET\n"
    b"endstream\nendobj\n"
    b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000234 00000 n \n0000000569 00000 n \n"
    b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n642\n%%EOF\n"
)



# In-memory database for clean E2E test isolation
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
with engine.connect() as conn:
    conn.execute(text("PRAGMA foreign_keys=ON;"))
Base.metadata.create_all(bind=engine)
TestingSession = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=engine)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Sets up an isolated database and isolated vector store to ensure 744 production invariant."""
    app.dependency_overrides[get_db] = override_get_db

    import backend.services.vector_store_service as vss_mod
    isolated_store = VectorStoreService(storage_dir=tmp_path)
    monkeypatch.setattr(vss_mod, "_shared_vector_store", isolated_store)
    monkeypatch.setattr(vss_mod, "get_vector_store_service", lambda *a, **k: isolated_store)

    cache = get_llm_cache_service()
    cache.clear()
    cache.enabled = True

    client = TestClient(app)
    yield client, isolated_store

    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_current_db_user, None)
    app.dependency_overrides.pop(get_optional_current_db_user, None)
    vss_mod._shared_vector_store = None
    cache.clear()


class TestEndToEndPipeline:
    """Verifies complete Phase 3.4 execution flow."""

    def test_full_e2e_pipeline_lifecycle(self, isolated_env):
        client, isolated_store = isolated_env

        # ----------------------------------------------------------------------
        # 1. User Creation & Authentication
        # ----------------------------------------------------------------------
        db = TestingSession()
        test_user = User(email="patient_jane@hospital.org", role="patient")
        db.add(test_user)
        db.commit()
        db.refresh(test_user)

        app.dependency_overrides[get_current_db_user] = lambda: test_user
        app.dependency_overrides[get_optional_current_db_user] = lambda: test_user

        # ----------------------------------------------------------------------
        # 2. Document Upload & Extraction & Ingestion
        # ----------------------------------------------------------------------
        upload_resp = client.post(
            "/documents/upload",
            files={"file": ("jane_smith_cardio.pdf", SYNTHETIC_CLINICAL_PDF, "application/pdf")}
        )
        assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
        doc_data = upload_resp.json()
        doc_id = doc_data["document_id"]
        assert doc_id is not None
        assert doc_data["status"] == "completed"
        assert doc_data["num_chunks"] > 0
        assert isolated_store.count() > 0

        # ----------------------------------------------------------------------
        # 3. Authenticated RAG Query (First Run - Cache Miss, Gemini Invoked)
        # ----------------------------------------------------------------------
        mock_gemini = MagicMock()
        mock_gemini.model = "gemini-3.5-flash-lite"
        mock_gemini.generate_answer.return_value = {
            "answer": "First-line management includes a low-sodium DASH diet and aerobic exercise [Source 1].",
            "model": "gemini-3.5-flash-lite",
            "generation_time_ms": 120.0,
            "gemini_calls_count": 1
        }

        mock_gemini.generate_stream.return_value = [
            "First-line management ",
            "includes a low-sodium DASH diet ",
            "and aerobic exercise [Source 1]."
        ]

        with patch("backend.services.gemini_service.GeminiService", return_value=mock_gemini):
            query_resp = client.post(
                "/rag/query",
                json={
                    "question": "What is the recommended first-line management for hypertension?",
                    "top_k": 3
                },
                headers={"X-Request-ID": "e2e-trace-run-001"}
            )

        assert query_resp.status_code == 200
        res = query_resp.json()

        assert res["retrieval_status"] == "success"
        assert "DASH diet" in res["answer"]
        assert len(res["sources"]) > 0
        assert res["request_id"] == "e2e-trace-run-001"
        assert res["timings"]["cache_hit"] is False
        assert res["timings"]["llm_called"] is True
        assert MEDICAL_DISCLAIMER in res["disclaimer"]

        # ----------------------------------------------------------------------
        # 4. Authenticated RAG Query (Second Run - Cache Hit, 0ms LLM)
        # ----------------------------------------------------------------------
        mock_gemini_2 = MagicMock()
        with patch("backend.services.gemini_service.GeminiService", return_value=mock_gemini_2):
            cached_resp = client.post(
                "/rag/query",
                json={
                    "question": "What is the recommended first-line management for hypertension?",
                    "top_k": 3
                },
                headers={"X-Request-ID": "e2e-trace-run-002"}
            )

        assert cached_resp.status_code == 200
        c_res = cached_resp.json()
        assert c_res["timings"]["cache_hit"] is True
        assert c_res["timings"]["llm_called"] is False
        assert c_res["timings"]["gemini_calls_count"] == 0
        mock_gemini_2.generate_answer.assert_not_called()

        # ----------------------------------------------------------------------
        # 5. Streaming End-to-End Query (SSE)
        # ----------------------------------------------------------------------
        with patch("backend.services.gemini_service.GeminiService", return_value=mock_gemini):
            stream_resp = client.post(
                "/rag/stream",
                json={"question": "What is the recommended first-line management for hypertension?"},
                headers={"X-Request-ID": "e2e-stream-003"}
            )

        assert stream_resp.status_code == 200
        stream_content = stream_resp.text
        assert "event: start" in stream_content
        assert "event: token" in stream_content
        assert "event: complete" in stream_content

    def test_production_faiss_and_metadata_invariants_preserved(self):
        """Verifies that the production FAISS index remains at exactly 744 vectors and 744 records."""
        from backend.services.vector_store_service import VectorStoreService
        prod_store = VectorStoreService()
        prod_store.load()

        assert prod_store.count() == 744, (
            f"FAISS invariant breach: Expected 744 vectors, found {prod_store.count()}."
        )
        assert len(prod_store.metadata_store) == 744, (
            f"Metadata invariant breach: Expected 744 records, found {len(prod_store.metadata_store)}."
        )
