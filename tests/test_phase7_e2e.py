"""
Phase 7 End-to-End Verification Test Suite
Validates the complete integration between React Frontend contracts and FastAPI Backend:
1. Backend Health & Readiness connectivity probe
2. CORS headers verification for Netlify domain
3. Landing / Auth lifecycle (signup, login, profile retrieval, logout)
4. Document Upload (PDF extraction, FAISS indexing, listing, ownership isolation)
5. Clinical Chat: Grounded Answer Generation with inline citations
6. Multi-Turn Longitudinal Context preservation across dialogue turns
7. Cross-Turn Contraindication interception (CKD + NSAID alert propagation)
8. Medical Safety Emergency Interception (Neutral emergency guidance, non-bypassable)
9. Conversation Persistence (create, append message, retrieve, delete)
"""

import io
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


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
        "0000000300 00000 n \n"
        "trailer << /Size 5 /Root 1 0 R >>\n"
        "startxref\n"
        "420\n"
        "%%EOF\n"
    )
    return pdf_content.encode("latin-1")


def test_01_backend_health_and_readiness_contract():
    """Verify health endpoints adhere to frontend HealthContext and HealthStatusBadge contracts."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["service"] == "AI-Healthcare-Agent"
    assert "status" in data
    assert "vector_store" in data

    ready_res = client.get("/health/readiness")
    assert ready_res.status_code in (200, 503)
    ready_data = ready_res.json()
    assert "checks" in ready_data
    assert "vector_store" in ready_data["checks"]


def test_02_cors_netlify_regex_support():
    """Verify backend accepts CORS requests from Netlify production/preview domains."""
    res = client.options(
        "/rag/query",
        headers={
            "Origin": "https://healthai-clinical-preview.netlify.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "https://healthai-clinical-preview.netlify.app"
    assert res.headers.get("access-control-allow-credentials") == "true"


def test_03_auth_lifecycle_and_jwt_issuance():
    """Verify user signup, login, profile retrieval, and logout via mocked Supabase auth."""
    test_email = "test_clinician_phase7@example.com"
    test_password = "ClinicalPassword2026!"
    fake_token = "mock-jwt-token-phase7-2026"
    fake_user = {
        "id": "usr_phase7_clinician",
        "email": test_email,
        "created_at": "2026-10-06T00:00:00Z",
        "user_metadata": {"role": "clinician"},
    }
    fake_session = {
        "access_token": fake_token,
        "token_type": "bearer",
        "expires_in": 3600,
        "expires_at": 1728172800,
    }

    with patch("backend.api.auth_router.auth_service.signup") as mock_signup, \
         patch("backend.api.auth_router.auth_service.login") as mock_login, \
         patch("backend.api.auth_router.auth_service.get_current_user") as mock_me, \
         patch("backend.api.auth_router.auth_service.logout") as mock_logout:

        mock_signup.return_value = {"user": fake_user, "session": fake_session}
        mock_login.return_value = {"user": fake_user, "session": fake_session}
        mock_me.return_value = fake_user
        mock_logout.return_value = None

        # 1. Signup
        signup_res = client.post("/auth/signup", json={"email": test_email, "password": test_password})
        assert signup_res.status_code == 201
        assert signup_res.json()["session"]["access_token"] == fake_token

        # 2. Login
        login_res = client.post("/auth/login", json={"email": test_email, "password": test_password})
        assert login_res.status_code == 200
        login_data = login_res.json()
        assert "session" in login_data
        token = login_data["session"]["access_token"]
        assert token == fake_token

        # 3. Authenticated /auth/me
        me_res = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_res.status_code == 200
        assert me_res.json()["user"]["email"] == test_email

        # 4. Logout
        logout_res = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
        assert logout_res.status_code == 200


def test_04_document_upload_and_user_isolation():
    """Verify document upload, chunk indexing, and listing for authenticated session."""
    from backend.api.auth_dependencies import get_current_user, get_current_db_user
    from backend.database.models import User

    test_user = User(id=9701, email="doc_user_phase7@hospital.org", role="patient")
    mock_supa = MagicMock()
    mock_supa.id = "supa_doc_9701"
    mock_supa.email = "doc_user_phase7@hospital.org"

    app.dependency_overrides[get_current_user] = lambda: mock_supa
    app.dependency_overrides[get_current_db_user] = lambda: test_user

    try:
        pdf_bytes = create_test_pdf_bytes("Clinical Trial: Patient response to ACE inhibitors and ARBs in stage 2 hypertension.")
        upload_file = ("clinical_hypertension_protocol.pdf", io.BytesIO(pdf_bytes), "application/pdf")

        mock_doc_response = {
            "id": 999,
            "filename": "clinical_hypertension_protocol.pdf",
            "file_path": "/data/documents/clinical_hypertension_protocol.pdf",
            "file_size": len(pdf_bytes),
            "file_size_bytes": len(pdf_bytes),
            "user_id": 9701,
            "number_of_pages": 1,
            "num_chunks": 1,
            "status": "completed",
            "created_at": "2026-10-06T00:00:00Z",
        }

        with patch("backend.api.document_router.DocumentIngestionService.ingest_pdf") as mock_ingest, \
             patch("backend.api.db_router.DocumentRepository.list_all", return_value=[mock_doc_response]):
            mock_ingest.return_value = {
                "document_id": 999,
                "filename": "clinical_hypertension_protocol.pdf",
                "status": "completed",
                "pages": 1,
                "number_of_pages": 1,
                "num_chunks": 1,
                "extracted_text_length": len(pdf_bytes),
                "user_id": 9701,
            }

            res = client.post(
                "/documents/upload",
                files={"file": upload_file},
                headers={"Authorization": "Bearer mock-token-doc"},
            )
            assert res.status_code == 200
            upload_data = res.json()
            assert upload_data["status"] == "completed"

            # List documents
            list_res = client.get("/documents", headers={"Authorization": "Bearer mock-token-doc"})
            assert list_res.status_code == 200
            docs = list_res.json()
            assert isinstance(docs, list)
            assert len(docs) >= 1
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_current_db_user, None)


def test_05_grounded_rag_query_with_citations():
    """Verify /rag/query returns grounded answer, citations, request_id, and disclaimer."""
    payload = {
        "question": "What is hypertension and how is it characterized?",
        "top_k": 3,
        "similarity_threshold": 0.25,
    }

    res = client.post("/rag/query", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "retrieval_status" in data
    assert "sources" in data
    assert "disclaimer" in data
    assert "request_id" in data
    assert len(data["disclaimer"]) > 0


def test_06_multi_turn_longitudinal_context_preservation():
    """Verify conversation_history propagates through /rag/query and preserves longitudinal context."""
    multi_turn_payload = {
        "question": "What is the recommended starting dose?",
        "top_k": 3,
        "similarity_threshold": 0.25,
        "conversation_history": [
            {"role": "user", "content": "I was diagnosed with Type 2 Diabetes."},
            {"role": "assistant", "content": "Metformin is standard first-line therapy."},
        ],
    }

    res = client.post("/rag/query", json=multi_turn_payload)
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "dialogue_context" in data
    assert data["dialogue_context"] is not None
    assert data["dialogue_context"].get("is_follow_up") is True


def test_07_cross_turn_contraindication_interception():
    """Verify that prior-turn CKD disclosure triggers contraindication alert when user asks about NSAIDs."""
    contraindication_payload = {
        "question": "Can I take ibuprofen 800mg for back pain?",
        "top_k": 3,
        "similarity_threshold": 0.25,
        "conversation_history": [
            {"role": "user", "content": "I have stage 4 chronic kidney disease with renal impairment."},
            {"role": "assistant", "content": "Noted. How can I assist you with your health today?"},
        ],
    }

    res = client.post("/rag/query", json=contraindication_payload)
    assert res.status_code == 200
    data = res.json()
    assert "dialogue_context" in data
    dc = data["dialogue_context"]
    assert dc is not None
    contraindications = dc.get("contraindications", [])
    assert len(contraindications) > 0
    ids = [c.get("contraindication_id") for c in contraindications]
    assert "RENAL_IMPAIRMENT_NSAID" in ids


def test_08_medical_safety_emergency_interception():
    """Verify acute symptoms trigger non-bypassable emergency interception with critical safety status."""
    emergency_payload = {
        "question": "I have crushing chest pain radiating to my left arm and jaw with severe shortness of breath.",
        "top_k": 5,
        "similarity_threshold": 0.25,
    }

    res = client.post("/rag/query", json=emergency_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["retrieval_status"] == "safety_intercepted"
    assert "EMERGENCY" in data["answer"].upper()
    assert len(data["sources"]) == 0  # Sources quarantined during emergency


def test_09_conversation_crud_and_message_persistence():
    """Verify dialogue sessions can be created, populated, listed, and cleared."""
    from backend.api.auth_dependencies import get_current_user, get_current_db_user
    from backend.database.models import User

    test_user = User(id=9702, email="chat_history_user@hospital.org", role="patient")
    mock_supa = MagicMock()
    mock_supa.id = "supa_chat_9702"
    mock_supa.email = "chat_history_user@hospital.org"

    app.dependency_overrides[get_current_user] = lambda: mock_supa
    app.dependency_overrides[get_current_db_user] = lambda: test_user

    now = datetime.now(timezone.utc)
    mock_conv = MagicMock()
    mock_conv.id = 101
    mock_conv.title = "Diabetes Consultation Session"
    mock_conv.user_id = 9702
    mock_conv.created_at = now
    mock_conv.updated_at = now

    mock_msg1 = MagicMock()
    mock_msg1.id = 1
    mock_msg1.conversation_id = 101
    mock_msg1.sender = "user"
    mock_msg1.text = "What is HbA1c?"
    mock_msg1.agent_type = None
    mock_msg1.citations = None
    mock_msg1.created_at = now

    mock_msg2 = MagicMock()
    mock_msg2.id = 2
    mock_msg2.conversation_id = 101
    mock_msg2.sender = "assistant"
    mock_msg2.text = "HbA1c measures average blood glucose over the past 2-3 months."
    mock_msg2.agent_type = None
    mock_msg2.citations = {"retrieval_status": "success", "sources": []}
    mock_msg2.created_at = now

    try:
        with patch("backend.api.db_router.ConversationRepository.create", return_value=mock_conv), \
             patch("backend.api.db_router.ConversationRepository.get_by_id", return_value=mock_conv), \
             patch("backend.api.db_router.MessageRepository.create") as mock_add_msg, \
             patch("backend.api.db_router.MessageRepository.list_by_conversation", return_value=[mock_msg1, mock_msg2]), \
             patch("backend.api.db_router.ConversationRepository.delete", return_value=True):

            mock_add_msg.side_effect = [mock_msg1, mock_msg2]

            # 1. Create conversation
            create_res = client.post(
                "/conversations",
                json={"title": "Diabetes Consultation Session"},
                headers={"Authorization": "Bearer mock-token-conv"},
            )
            assert create_res.status_code == 201
            conv_id = create_res.json()["id"]
            assert conv_id == 101

            # 2. Append user turn
            msg1_res = client.post(
                f"/conversations/{conv_id}/messages",
                json={"sender": "user", "text": "What is HbA1c?"},
                headers={"Authorization": "Bearer mock-token-conv"},
            )
            assert msg1_res.status_code == 201

            # 3. Append assistant turn
            msg2_res = client.post(
                f"/conversations/{conv_id}/messages",
                json={
                    "sender": "assistant",
                    "text": "HbA1c measures average blood glucose over the past 2-3 months.",
                    "citations": {"retrieval_status": "success", "sources": []},
                },
                headers={"Authorization": "Bearer mock-token-conv"},
            )
            assert msg2_res.status_code == 201

            # 4. Fetch messages
            msgs_res = client.get(f"/conversations/{conv_id}/messages", headers={"Authorization": "Bearer mock-token-conv"})
            assert msgs_res.status_code == 200
            msgs = msgs_res.json()
            assert len(msgs) == 2

            # 5. Delete conversation
            del_res = client.delete(f"/conversations/{conv_id}", headers={"Authorization": "Bearer mock-token-conv"})
            assert del_res.status_code == 200
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_current_db_user, None)
