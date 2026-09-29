"""
Unit and integration tests for Authenticated Document Ingestion Pipeline (Phase 13, Step 1).

Verifies all 18 requirements:
1. Authenticated PDF upload succeeds
2. Unauthenticated upload returns 401
3. Invalid file type rejected (400)
4. Oversized PDF rejected (400)
5. Empty/no-text PDF rejected (400)
6. Document ownership comes from authenticated user
7. Client cannot spoof document owner (403)
8. Document metadata is persisted in database
9. Chunks are persisted in database
10. Embeddings are generated with correct dimension
11. FAISS vectors are created in vector store
12. embedding_index maps correctly to FAISS position
13. Completed document has correct chunk count
14. Processing failure results in safe failed state
15. Different users cannot own/access each other's documents
16. Uploaded document can be retrieved through existing RAG
17. RAG cannot return another user's document (ownership isolation)
18. API response does not expose secrets
"""

import io
import sys
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
from backend.database.models import User, Document, DocumentChunk
from backend.database.repositories import DocumentRepository, ChunkRepository
from backend.api.auth_dependencies import get_current_user
from backend.services.vector_store_service import VectorStoreService, get_vector_store_service
from backend.rag.rag_service import RAGService
from backend.services.ingestion_service import DocumentIngestionService

# Isolated in-memory database for testing ingestion
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
with test_engine.connect() as conn:
    conn.execute(text("PRAGMA foreign_keys=ON;"))
Base.metadata.create_all(bind=test_engine)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


def create_test_pdf_bytes(text_content: str = "Medical Report: Patient has mild hypertension.") -> bytes:
    """Generates a valid minimal raw PDF byte stream containing extractable text."""
    escaped_text = text_content.replace("(", "\\(").replace(")", "\\)")
    content_stream = f"BT /F1 12 Tf 50 750 Td ({escaped_text}) Tj ET"
    stream_len = len(content_stream)
    pdf_content = (
        "%PDF-1.4\n"
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >> endobj\n"
        f"4 0 obj << /Length {stream_len} >> stream\n"
        f"{content_stream}\n"
        "endstream\n"
        "endobj\n"
        "xref\n"
        "0 5\n"
        "0000000000 65535 f \n"
        "0000000009 00000 n \n"
        "0000000058 00000 n \n"
        "0000000115 00000 n \n"
        "0000000272 00000 n \n"
        "trailer << /Size 5 /Root 1 0 R >>\n"
        "startxref\n"
        "350\n"
        "%%EOF\n"
    )
    return pdf_content.encode("latin-1")


@pytest.fixture(autouse=True)
def isolate_vector_store(tmp_path, monkeypatch):
    """Isolates vector store storage to temporary directory to prevent modifying production index."""
    import backend.services.vector_store_service as vss_module
    test_store = VectorStoreService(storage_dir=tmp_path)
    monkeypatch.setattr(vss_module, "_shared_vector_store", test_store)
    monkeypatch.setattr(vss_module, "get_vector_store_service", lambda *a, **k: test_store)
    yield test_store
    vss_module._shared_vector_store = None


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ==============================================================================
# 1. Authenticated PDF upload succeeds
# ==============================================================================
def test_authenticated_pdf_upload_succeeds(client):
    user_a = {"id": 1001, "email": "doctor_a@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user_a

    pdf_bytes = create_test_pdf_bytes("Clinical Trial: Metformin efficacy in Type 2 Diabetes.")
    files = {"file": ("trial_report.pdf", pdf_bytes, "application/pdf")}
    headers = {"Authorization": "Bearer token_a"}

    response = client.post("/documents/upload", files=files, headers=headers)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "completed"
    assert data["filename"] == "trial_report.pdf"
    assert data["document_id"] is not None
    assert data["num_chunks"] > 0
    assert data["pages"] == 1
    assert data["user_id"] is not None


# ==============================================================================
# 2. Unauthenticated upload returns 401
# ==============================================================================
def test_unauthenticated_upload_returns_401(client):
    # Clear user override so real auth runs
    if get_current_user in app.dependency_overrides:
        del app.dependency_overrides[get_current_user]

    pdf_bytes = create_test_pdf_bytes("Unauthenticated upload test")
    files = {"file": ("unauth.pdf", pdf_bytes, "application/pdf")}

    response = client.post("/documents/upload", files=files)
    assert response.status_code == 401
    assert "Authorization" in response.json()["detail"] or "unauthenticated" in response.json()["detail"].lower()


# ==============================================================================
# 3. Invalid file type rejected (400)
# ==============================================================================
def test_invalid_file_type_rejected(client):
    user_a = {"id": 1002, "email": "doctor_b@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user_a

    files = {"file": ("notes.txt", b"Plain text notes", "text/plain")}
    headers = {"Authorization": "Bearer token_b"}

    response = client.post("/documents/upload", files=files, headers=headers)
    assert response.status_code == 400
    assert "Only PDF files (.pdf) are supported" in response.json()["detail"]


# ==============================================================================
# 4. Oversized PDF rejected (400)
# ==============================================================================
def test_oversized_pdf_rejected(client):
    user_a = {"id": 1003, "email": "doctor_c@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user_a

    oversized_bytes = b"%PDF-1.4\n" + b"0" * (11 * 1024 * 1024)
    files = {"file": ("huge.pdf", oversized_bytes, "application/pdf")}
    headers = {"Authorization": "Bearer token_c"}

    response = client.post("/documents/upload", files=files, headers=headers)
    assert response.status_code == 400
    assert "exceeds the maximum allowed limit" in response.json()["detail"]


# ==============================================================================
# 5. Empty / no-text PDF rejected (400)
# ==============================================================================
def test_empty_or_no_text_pdf_rejected(client):
    user_a = {"id": 1004, "email": "doctor_d@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user_a

    # 1. Zero-byte PDF
    files_empty = {"file": ("empty.pdf", b"", "application/pdf")}
    res_empty = client.post("/documents/upload", files=files_empty, headers={"Authorization": "Bearer token_d"})
    assert res_empty.status_code == 400
    assert "empty (0 bytes)" in res_empty.json()["detail"]

    # 2. Textless blank PDF
    blank_pdf = b"%PDF-1.4\n1 0 obj << >> endobj\ntrailer << >>\n%%EOF"
    files_blank = {"file": ("blank.pdf", blank_pdf, "application/pdf")}
    res_blank = client.post("/documents/upload", files=files_blank, headers={"Authorization": "Bearer token_d"})
    assert res_blank.status_code == 400
    assert "No extractable text found" in res_blank.json()["detail"] or "Failed to parse" in res_blank.json()["detail"]


# ==============================================================================
# 6. Document ownership comes from authenticated user
# ==============================================================================
def test_document_ownership_assigned_from_jwt(client):
    user_a = {"id": 1005, "email": "owner_doc@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user_a

    pdf_bytes = create_test_pdf_bytes("Patient cardiology assessment record.")
    files = {"file": ("cardio.pdf", pdf_bytes, "application/pdf")}
    headers = {"Authorization": "Bearer token_owner"}

    res = client.post("/documents/upload", files=files, headers=headers)
    assert res.status_code == 200
    doc_id = res.json()["document_id"]

    # Verify directly from database
    db = TestingSession()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        assert doc is not None
        assert doc.user_id == res.json()["user_id"]
        assert doc.user.email == "owner_doc@clinic.org"
    finally:
        db.close()


# ==============================================================================
# 7. Client cannot spoof document owner (403)
# ==============================================================================
def test_client_cannot_spoof_document_owner(client):
    user_a = {"id": 1006, "email": "doctor_spoof@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user_a

    pdf_bytes = create_test_pdf_bytes("Neurology consultation notes.")
    files = {"file": ("neuro.pdf", pdf_bytes, "application/pdf")}
    data = {"user_id": 9999}  # Attacker attempts to assign to different user
    headers = {"Authorization": "Bearer token_spoof"}

    res = client.post("/documents/upload", files=files, data=data, headers=headers)
    assert res.status_code == 403
    assert "Cannot assign document to another user" in res.json()["detail"]


# ==============================================================================
# 8. Document metadata is persisted in database
# ==============================================================================
def test_document_metadata_persisted(client):
    user_a = {"id": 1007, "email": "persisted@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user_a

    pdf_bytes = create_test_pdf_bytes("Oncology biopsy pathology analysis findings.")
    files = {"file": ("onco_pathology.pdf", pdf_bytes, "application/pdf")}
    headers = {"Authorization": "Bearer token_persisted"}

    res = client.post("/documents/upload", files=files, headers=headers)
    assert res.status_code == 200
    doc_id = res.json()["document_id"]

    db = TestingSession()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        assert doc is not None
        assert doc.filename == "onco_pathology.pdf"
        assert doc.file_size_bytes == len(pdf_bytes)
        assert doc.num_pages == 1
        assert doc.status == "completed"
        assert doc.num_chunks > 0
        assert doc.created_at is not None
    finally:
        db.close()


# ==============================================================================
# 9. Chunks are persisted in database
# ==============================================================================
def test_chunks_persisted_in_database(client):
    user_a = {"id": 1008, "email": "chunks_doc@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user_a

    pdf_bytes = create_test_pdf_bytes("Paragraph 1: Asthma diagnosis.\n\nParagraph 2: Inhaler treatment prescribed.")
    files = {"file": ("asthma.pdf", pdf_bytes, "application/pdf")}
    headers = {"Authorization": "Bearer token_chunks"}

    res = client.post("/documents/upload", files=files, headers=headers)
    assert res.status_code == 200
    doc_id = res.json()["document_id"]

    db = TestingSession()
    try:
        chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).all()
        assert len(chunks) > 0
        for c in chunks:
            assert c.document_id == doc_id
            assert c.chunk_id is not None
            assert len(c.text) > 0
            assert c.token_count > 0
            assert c.embedding_index is not None
    finally:
        db.close()


# ==============================================================================
# 10. Embeddings generated with correct dimension
# ==============================================================================
def test_embeddings_generated_with_correct_dimension(client):
    from backend.services.embedding_service import EmbeddingService
    embedding_service = EmbeddingService()
    test_chunks = [{"chunk_id": "chunk_dim_1", "text": "Medical cardiology assessment."}]
    embedded = embedding_service.embed_chunks(test_chunks)
    assert len(embedded) == 1
    assert "embedding" in embedded[0]
    assert len(embedded[0]["embedding"]) == 384  # all-MiniLM-L6-v2 dimension


# ==============================================================================
# 11. FAISS vectors are created in vector store
# ==============================================================================
def test_faiss_vectors_created_in_vector_store(client):
    user_a = {"id": 1009, "email": "faiss_doc@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user_a

    vector_store = get_vector_store_service()
    initial_count = vector_store.count()

    pdf_bytes = create_test_pdf_bytes("Radiology CT scan shows clear lungs with no acute infiltrates.")
    files = {"file": ("ct_scan.pdf", pdf_bytes, "application/pdf")}
    headers = {"Authorization": "Bearer token_faiss"}

    res = client.post("/documents/upload", files=files, headers=headers)
    assert res.status_code == 200
    num_chunks = res.json()["num_chunks"]

    # FAISS count should increase exactly by num_chunks
    assert vector_store.count() == initial_count + num_chunks


# ==============================================================================
# 12. embedding_index maps correctly to FAISS position
# ==============================================================================
def test_embedding_index_maps_correctly(client):
    user_a = {"id": 1010, "email": "mapping_doc@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user_a

    vector_store = get_vector_store_service()

    pdf_bytes = create_test_pdf_bytes("Dermatology clinic examination of skin lesion.")
    files = {"file": ("derma.pdf", pdf_bytes, "application/pdf")}
    headers = {"Authorization": "Bearer token_map"}

    res = client.post("/documents/upload", files=files, headers=headers)
    assert res.status_code == 200
    doc_id = res.json()["document_id"]

    db = TestingSession()
    try:
        chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).all()
        for c in chunks:
            # Check metadata store record at vector position c.embedding_index
            pos = c.embedding_index
            assert pos < len(vector_store.metadata_store)
            rec = vector_store.metadata_store[pos]
            assert rec["vector_id"] == pos
            assert rec["chunk_id"] == c.chunk_id
            assert rec["document_id"] == str(doc_id)
            assert rec["user_id"] == res.json()["user_id"]
    finally:
        db.close()


# ==============================================================================
# 13. Completed document has correct chunk count
# ==============================================================================
def test_completed_document_chunk_count(client):
    user_a = {"id": 1011, "email": "count_doc@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user_a

    pdf_bytes = create_test_pdf_bytes("Section 1: Initial triage.\n\nSection 2: Secondary observation.")
    files = {"file": ("triage.pdf", pdf_bytes, "application/pdf")}
    headers = {"Authorization": "Bearer token_count"}

    res = client.post("/documents/upload", files=files, headers=headers)
    assert res.status_code == 200
    doc_id = res.json()["document_id"]
    reported_num_chunks = res.json()["num_chunks"]

    db = TestingSession()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).all()
        assert doc.num_chunks == len(chunks)
        assert doc.num_chunks == reported_num_chunks
        assert doc.status == "completed"
    finally:
        db.close()


# ==============================================================================
# 14. Processing failure results in safe failed state
# ==============================================================================
def test_processing_failure_safe_failed_state(client):
    user_a = {"id": 1012, "email": "fail_doc@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user_a

    pdf_bytes = create_test_pdf_bytes("Text that will fail during embedding generation.")
    files = {"file": ("fail_report.pdf", pdf_bytes, "application/pdf")}
    headers = {"Authorization": "Bearer token_fail"}

    with patch("backend.services.embedding_service.EmbeddingService.embed_chunks", side_effect=RuntimeError("GPU OOM")):
        res = client.post("/documents/upload", files=files, headers=headers)
        assert res.status_code == 500
        assert "Embedding generation failed" in res.json()["detail"]


# ==============================================================================
# 15. Different users cannot own/access each other's documents
# ==============================================================================
def test_different_users_cannot_access_each_other_documents(client):
    user_a = {"id": 1013, "email": "doc_alice@clinic.org", "role": "authenticated"}
    user_b = {"id": 1014, "email": "doc_bob@clinic.org", "role": "authenticated"}

    # User Alice uploads document
    app.dependency_overrides[get_current_user] = lambda: user_a
    pdf_bytes = create_test_pdf_bytes("Alice private patient oncology chart.")
    files = {"file": ("alice_chart.pdf", pdf_bytes, "application/pdf")}
    headers_a = {"Authorization": "Bearer token_alice"}

    res_a = client.post("/documents/upload", files=files, headers=headers_a)
    assert res_a.status_code == 200
    doc_id = res_a.json()["document_id"]

    # Bob attempts to access Alice's document -> 403 Forbidden
    app.dependency_overrides[get_current_user] = lambda: user_b
    headers_b = {"Authorization": "Bearer token_bob"}
    res_b = client.get(f"/documents/{doc_id}", headers=headers_b)
    assert res_b.status_code == 403
    assert "Access denied" in res_b.json()["detail"]


# ==============================================================================
# 16. Uploaded document can be retrieved through existing RAG
# ==============================================================================
def test_uploaded_document_retrievable_through_rag(client):
    user_a = {"id": 1015, "email": "doc_owner_rag@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user_a

    pdf_bytes = create_test_pdf_bytes("Pheochromocytoma is a rare neuroendocrine tumor of the adrenal medulla secreting catecholamines.")
    files = {"file": ("pheo_case.pdf", pdf_bytes, "application/pdf")}
    res = client.post("/documents/upload", files=files, headers={"Authorization": "Bearer token_a"})
    assert res.status_code == 200
    user_a_db_id = res.json()["user_id"]

    rag = RAGService(vector_store=get_vector_store_service())
    results_a = rag.query(question="What is Pheochromocytoma?", top_k=5, user_id=user_a_db_id)
    assert results_a["retrieval_status"] == "success"
    assert any("neuroendocrine tumor" in c["text"] for c in results_a["retrieved_chunks"])


# ==============================================================================
# 17. RAG cannot return another user's document (ownership isolation)
# ==============================================================================
def test_rag_cannot_return_another_user_document(client):
    user_a = {"id": 1015, "email": "doc_owner_rag@clinic.org", "role": "authenticated"}
    user_b = {"id": 1016, "email": "doc_intruder_rag@clinic.org", "role": "authenticated"}

    # Ensure document is uploaded by user_a
    app.dependency_overrides[get_current_user] = lambda: user_a
    pdf_bytes = create_test_pdf_bytes("Pheochromocytoma is a rare neuroendocrine tumor of the adrenal medulla secreting catecholamines.")
    files = {"file": ("pheo_case.pdf", pdf_bytes, "application/pdf")}
    res = client.post("/documents/upload", files=files, headers={"Authorization": "Bearer token_a"})
    assert res.status_code == 200

    # Query RAG as User B (isolated user_id): User A's chunks MUST NOT be returned
    rag = RAGService(vector_store=get_vector_store_service())
    results_b = rag.query(question="What is Pheochromocytoma?", top_k=5, user_id=99999)
    assert not any("neuroendocrine tumor" in c.get("text", "") for c in results_b.get("retrieved_chunks", []))


# ==============================================================================
# 18. API response does not expose secrets
# ==============================================================================
def test_api_response_does_not_expose_secrets(client):
    user_a = {"id": 1017, "email": "security_check@clinic.org", "role": "authenticated"}
    app.dependency_overrides[get_current_user] = lambda: user_a

    pdf_bytes = create_test_pdf_bytes("Routine health checkup report.")
    files = {"file": ("security_test.pdf", pdf_bytes, "application/pdf")}
    headers = {"Authorization": "Bearer token_sec"}

    res = client.post("/documents/upload", files=files, headers=headers)
    assert res.status_code == 200
    text_resp = res.text

    # Verify no credentials, passwords, JWT tokens, or raw embeddings are exposed
    assert "postgresql://" not in text_resp
    assert "password" not in text_resp.lower()
    assert "secret" not in text_resp.lower()
    assert "bearer" not in text_resp.lower()
    assert "eyJ" not in text_resp  # JWT header signature
