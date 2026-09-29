"""
Comprehensive End-to-End Workflow & User Isolation Test Suite (Part 18).

Validates the full healthcare research assistant pipeline:
1. Live /health operational status
2. APIClient backend connectivity helper
3. User 1 authentication & token management
4. Document upload & ingestion pipeline (PDF -> text -> chunks -> embeddings -> FAISS -> PostgreSQL)
5. Document repository listing & metadata verification
6. RAG query with grounded evidence citations
7. Unrelated query with safe no-context fallback
8. Conversation thread creation & message persistence
9. User 1 session logout & unauthenticated endpoint rejection (401)
10. User 2 authentication & cross-user database isolation
11. User 2 cross-user FAISS vector space isolation
"""

import sys
import io
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
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
from backend.database.models import User, Document, DocumentChunk, Conversation, Message
from backend.services.auth_service import auth_service
from backend.api.auth_dependencies import get_current_user, get_optional_current_db_user, get_current_db_user
from backend.services.vector_store_service import VectorStoreService
from backend.rag.rag_service import RAGService
from frontend.api_client import APIClient

SYNTHETIC_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    b"4 0 obj\n<< /Length 250 >>\nstream\n"
    b"BT\n/F1 12 Tf\n50 720 Td\n(CLINICAL REPORT: Patient John Doe, Age 58) Tj\n0 -20 Td\n(DIAGNOSIS: Stage 2 Essential Hypertension) Tj\n0 -20 Td\n(VITALS: Blood Pressure 158/96 mmHg, Heart Rate 78 bpm) Tj\n0 -20 Td\n(TREATMENT PLAN: Prescribed Amlodipine 5mg daily and low-sodium diet) Tj\nET\n"
    b"endstream\nendobj\n"
    b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000234 00000 n \n0000000535 00000 n \n"
    b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n608\n%%EOF\n"
)


# Test database isolation
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
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


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_vector_store(tmp_path, monkeypatch):
    """Isolates vector store storage to temporary directory to prevent modifying production index."""
    from backend.services.vector_store_service import VectorStoreService
    import backend.services.vector_store_service as vss_module
    test_store = VectorStoreService(storage_dir=tmp_path)
    monkeypatch.setattr(vss_module, "_shared_vector_store", test_store)
    monkeypatch.setattr(vss_module, "get_vector_store_service", lambda *a, **k: test_store)
    yield test_store
    vss_module._shared_vector_store = None


def test_complete_e2e_workflow():
    """Executes the full end-to-end multi-tenant workflow."""
    app.dependency_overrides[get_db] = override_get_db

    # 1. Health check
    h_res = client.get("/health")
    assert h_res.status_code == 200
    assert h_res.json()["status"] == "healthy"

    api_cl = APIClient(base_url="http://test-server")
    with patch("requests.get") as mock_get:
        mock_r = MagicMock()
        mock_r.status_code = 200
        mock_r.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_r
        res = api_cl.check_backend_health()
        assert res["connected"] is True
        assert res["status"] == "healthy"

    # 2. Seed User 1 and User 2 in DB
    db = TestingSession()
    u1 = User(id=8801, email="doctor_alice_8801@clinic.org", full_name="Dr. Alice")
    u2 = User(id=8802, email="researcher_bob_8802@lab.org", full_name="Researcher Bob")
    db.add_all([u1, u2])
    db.commit()
    db.refresh(u1)
    db.refresh(u2)
    db.close()

    # Define mock auth user contexts
    mock_supabase_user1 = MagicMock()
    mock_supabase_user1.id = "supa_user_8801"
    mock_supabase_user1.email = "doctor_alice_8801@clinic.org"

    mock_supabase_user2 = MagicMock()
    mock_supabase_user2.id = "supa_user_8802"
    mock_supabase_user2.email = "researcher_bob_8802@lab.org"

    # --- USER 1 WORKFLOW ---
    # Ingest document as User 1
    app.dependency_overrides[get_current_user] = lambda: mock_supabase_user1
    app.dependency_overrides[get_current_db_user] = lambda: u1
    app.dependency_overrides[get_optional_current_db_user] = lambda: u1

    files = {"file": ("hypertension_summary.pdf", SYNTHETIC_PDF_BYTES, "application/pdf")}
    upload_res = client.post("/documents/upload", files=files)
    assert upload_res.status_code == 200
    doc_data = upload_res.json()
    doc1_id = doc_data["document_id"]
    assert doc_data["status"] == "completed"
    assert doc_data["num_chunks"] >= 1
    assert doc_data["user_id"] == 8801

    # User 1 lists documents -> sees doc1
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    user1_docs = list_res.json()
    assert len(user1_docs) == 1
    assert user1_docs[0]["id"] == doc1_id

    # User 1 queries RAG on hypertension
    rag_payload = {
        "question": "What is the patient diagnosis and blood pressure?",
        "top_k": 3,
        "similarity_threshold": 0.10,
    }
    with patch("backend.services.gemini_service.GeminiService.generate_answer") as mock_gemini:
        mock_gemini.return_value = {
            "answer": "Patient is diagnosed with Stage 2 Essential Hypertension and blood pressure is 158/96 mmHg.",
            "model": "gemini-flash-latest",
            "disclaimer": "Informational only.",
            "generation_time_ms": 120.0,
            "status": "success",
        }
        rag_res = client.post("/rag/query", json=rag_payload)
        assert rag_res.status_code == 200
        rag_data = rag_res.json()
        assert rag_data["retrieval_status"] == "success"
        assert len(rag_data["sources"]) >= 1
        assert "Hypertension" in rag_data["answer"]

    # User 1 queries unrelated question -> no_relevant_context
    unrelated_payload = {
        "question": "What is the orbital speed of Neptune?",
        "top_k": 3,
        "similarity_threshold": 0.50,
    }
    unrelated_res = client.post("/rag/query", json=unrelated_payload)
    assert unrelated_res.status_code == 200
    unrelated_data = unrelated_res.json()
    assert unrelated_data["retrieval_status"] == "no_relevant_context"
    assert len(unrelated_data["sources"]) == 0

    # User 1 conversation persistence
    conv_res = client.post("/conversations", json={"title": "Hypertension Consultation"})
    assert conv_res.status_code == 201
    conv_id = conv_res.json()["id"]

    msg1_res = client.post(f"/conversations/{conv_id}/messages", json={"sender": "user", "text": rag_payload["question"]})
    assert msg1_res.status_code == 201
    msg2_res = client.post(f"/conversations/{conv_id}/messages", json={"sender": "assistant", "text": "Hypertension diagnosed.", "agent_type": "RAGService"})
    assert msg2_res.status_code == 201

    msgs_res = client.get(f"/conversations/{conv_id}/messages")
    assert msgs_res.status_code == 200
    assert len(msgs_res.json()) == 2

    # --- LOGOUT & UNAUTHENTICATED REJECTION ---
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_current_db_user, None)
    app.dependency_overrides.pop(get_optional_current_db_user, None)

    # Without token, protected routes must reject
    assert client.get("/auth/me").status_code == 401
    assert client.post("/documents/upload", files=files).status_code == 401
    assert client.get("/documents").status_code == 401

    # --- USER 2 WORKFLOW & ISOLATION ---
    app.dependency_overrides[get_current_user] = lambda: mock_supabase_user2
    app.dependency_overrides[get_current_db_user] = lambda: u2
    app.dependency_overrides[get_optional_current_db_user] = lambda: u2

    # User 2 lists documents -> sees 0 documents!
    u2_docs_res = client.get("/documents")
    assert u2_docs_res.status_code == 200
    assert len(u2_docs_res.json()) == 0

    # User 2 attempts to fetch User 1's document directly -> 403 Forbidden
    assert client.get(f"/documents/{doc1_id}").status_code == 403

    # User 2 attempts to query User 1's conversation -> 403 Forbidden
    assert client.get(f"/conversations/{conv_id}/messages").status_code == 403

    # User 2 queries RAG on hypertension -> FAISS filter prevents retrieval of User 1 chunks!
    u2_rag_res = client.post("/rag/query", json=rag_payload)
    assert u2_rag_res.status_code == 200
    u2_rag_data = u2_rag_res.json()
    assert u2_rag_data["retrieval_status"] == "no_relevant_context"
    assert len(u2_rag_data["sources"]) == 0

    # Cleanup vector store and dependency overrides so subsequent test suites have clean state
    app.dependency_overrides.clear()
    from backend.services.vector_store_service import get_vector_store_service
    get_vector_store_service().reset()
