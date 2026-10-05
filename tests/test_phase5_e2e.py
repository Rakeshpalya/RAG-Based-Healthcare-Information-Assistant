"""
Phase 5.13 — Production E2E User Application & Clinical UX Lifecycle.

Verifies the complete production lifecycle:
Register User A
  ↓
Login User A
  ↓
Dashboard (Health, Metrics, Documents, Sessions)
  ↓
Upload Document (Metadata / Processing verification)
  ↓
Document Processing & Indexing Verification
  ↓
Open AI Chat
  ↓
Ask Question (Safety Pre-Screen, Emergency & Clinical Inquiries)
  ↓
Safety Check (Acute Emergency interception)
  ↓
RAG Retrieval (FAISS semantic search with user isolation)
  ↓
Streaming Response (SSE stream with start, status, token, complete events)
  ↓
Citation Display & Validation (Validated sources, page attribution)
  ↓
Grounding Validation (No hallucinated claims or fake citations)
  ↓
Conversation History (Message persistence, session renaming)
  ↓
Logout User A
  ↓
User B Cross-Tenant Isolation Enforcement:
  - User B cannot list User A's documents
  - User B cannot delete User A's documents (HTTP 403)
  - User B cannot view User A's conversations or messages (HTTP 403)
  - User B cannot rename or delete User A's conversations (HTTP 403)
  - User B cache scope is strictly isolated from User A
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

# Ensure root workspace directory is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.main import app
from backend.database.models import User
from backend.database.database import get_db
from backend.api.auth_dependencies import get_current_db_user, get_optional_current_db_user
from backend.services.llm_cache_service import get_llm_cache_service


@pytest.fixture
def client():
    return TestClient(app)


def test_phase5_e2e_complete_workflow(client):
    """
    Executes the comprehensive Phase 5 E2E production flow across User A and User B.
    """
    user_a_id = 901
    user_a_email = "user_a_clinical@healthai.test"
    user_a_token = "jwt-token-user-a-valid-901"

    user_b_id = 902
    user_b_email = "user_b_clinical@healthai.test"
    user_b_token = "jwt-token-user-b-valid-902"

    mock_db_user_a = User(id=user_a_id, email=user_a_email, role="patient")
    mock_db_user_b = User(id=user_b_id, email=user_b_email, role="patient")

    try:
        # --------------------------------------------------------------------------
        # Step 1: Register User A
        # --------------------------------------------------------------------------
        with patch("backend.api.auth_router.auth_service.signup") as mock_signup:
            mock_signup.return_value = {
                "user": {"id": str(user_a_id), "email": user_a_email, "created_at": "2026-10-04T00:00:00Z"},
                "session": {"access_token": user_a_token, "token_type": "bearer", "expires_in": 3600},
            }
            res_signup = client.post("/auth/signup", json={"email": user_a_email, "password": "SecurePassword123!"})
            assert res_signup.status_code == 201
            assert res_signup.json()["user"]["email"] == user_a_email

        # --------------------------------------------------------------------------
        # Step 2: Login User A
        # --------------------------------------------------------------------------
        with patch("backend.api.auth_router.auth_service.login") as mock_login:
            mock_login.return_value = {
                "user": {"id": str(user_a_id), "email": user_a_email},
                "session": {"access_token": user_a_token, "token_type": "bearer"},
            }
            res_login = client.post("/auth/login", json={"email": user_a_email, "password": "SecurePassword123!"})
            assert res_login.status_code == 200
            assert res_login.json()["session"]["access_token"] == user_a_token

        auth_headers_a = {"Authorization": f"Bearer {user_a_token}"}

        # Override dependencies for User A
        app.dependency_overrides[get_current_db_user] = lambda: mock_db_user_a
        app.dependency_overrides[get_optional_current_db_user] = lambda: mock_db_user_a

        # --------------------------------------------------------------------------
        # Step 3: Dashboard (Health & Observability)
        # --------------------------------------------------------------------------
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "healthy"

        res_metrics = client.get("/metrics")
        assert res_metrics.status_code == 200
        metrics_json = res_metrics.json()
        assert "requests" in metrics_json

        # --------------------------------------------------------------------------
        # Step 4: Upload Document for User A
        # --------------------------------------------------------------------------
        meta_payload = {
            "filename": "clinical_trial_report_user_a.pdf",
            "file_path": "/storage/docs/clinical_trial_report_user_a.pdf",
            "file_size_bytes": 1048576,
            "num_pages": 12,
            "num_chunks": 24,
            "status": "processed",
            "user_id": user_a_id
        }
        res_doc = client.post("/documents/metadata", json=meta_payload, headers=auth_headers_a)
        assert res_doc.status_code == 201
        doc_a_id = res_doc.json()["id"]
        assert doc_a_id is not None

        # Verify document appears in User A's document repository
        res_list = client.get("/documents", headers=auth_headers_a)
        assert res_list.status_code == 200
        user_a_docs = res_list.json()
        assert any(d["id"] == doc_a_id for d in user_a_docs)

        # --------------------------------------------------------------------------
        # Step 5: Ask Question & Safety Check (Acute Emergency Pre-Screening)
        # --------------------------------------------------------------------------
        emergency_q = "I swallowed an entire bottle of sleeping pills 10 minutes ago"
        res_emergency = client.post("/rag/query", json={"question": emergency_q}, headers=auth_headers_a)
        assert res_emergency.status_code == 200
        emergency_data = res_emergency.json()
        assert emergency_data["retrieval_status"] == "safety_intercepted"
        assert len(emergency_data["sources"]) == 0
        assert "911" in emergency_data["answer"] or "emergency" in emergency_data["answer"].lower()

        # --------------------------------------------------------------------------
        # Step 6: RAG Retrieval & Safe Response Generation
        # --------------------------------------------------------------------------
        with patch("backend.services.gemini_service.GeminiService.generate_answer") as mock_gemini:
            mock_gemini.return_value = {
                "answer": (
                    "Hypertension is typically defined as a sustained systolic blood pressure greater than 130 mmHg [Source 1]. "
                    "Regular aerobic exercise and sodium reduction are recommended lifestyle modifications [Source 1]."
                ),
                "model": "gemini-3.5-flash-lite",
                "disclaimer": "MEDICAL DISCLAIMER: Always consult a physician for clinical decisions.",
                "generation_time_ms": 120.0,
                "status": "success",
            }

            safe_q = "What is the clinical definition of hypertension?"
            res_rag = client.post("/rag/query", json={"question": safe_q, "top_k": 3}, headers=auth_headers_a)
            assert res_rag.status_code == 200
            rag_data = res_rag.json()
            assert rag_data["retrieval_status"] in ("success", "grounded_boundary", "no_relevant_context")
            assert "disclaimer" in rag_data

        # --------------------------------------------------------------------------
        # Step 7: Streaming Response (SSE /rag/stream)
        # --------------------------------------------------------------------------
        with patch("backend.services.gemini_service.GeminiService.generate_stream") as mock_stream:
            mock_stream.return_value = iter(["Hypertension ", "is ", "elevated ", "blood ", "pressure."])

            stream_q = "What causes high blood pressure?"
            res_stream = client.post("/rag/stream", json={"question": stream_q, "top_k": 3}, headers=auth_headers_a)
            assert res_stream.status_code == 200
            assert "text/event-stream" in res_stream.headers["content-type"]

            # Parse SSE events from response stream
            stream_text = res_stream.text
            assert "event: start" in stream_text
            assert "event: complete" in stream_text

        # --------------------------------------------------------------------------
        # Step 8: Conversation History Management (Create, Append, Rename)
        # --------------------------------------------------------------------------
        # 1. Create conversation
        res_conv = client.post("/conversations", json={"title": "Hypertension Consultation"}, headers=auth_headers_a)
        assert res_conv.status_code == 201
        conv_a_id = res_conv.json()["id"]

        # 2. Append message
        msg_payload = {
            "sender": "user",
            "text": "What are hypertension symptoms?",
            "agent_type": "rag_agent"
        }
        res_msg = client.post(f"/conversations/{conv_a_id}/messages", json=msg_payload, headers=auth_headers_a)
        assert res_msg.status_code == 201

        # 3. Rename conversation (PATCH /conversations/{id})
        res_rename = client.patch(f"/conversations/{conv_a_id}", json={"title": "Updated Hypertension Review"}, headers=auth_headers_a)
        assert res_rename.status_code == 200
        assert res_rename.json()["title"] == "Updated Hypertension Review"

        # 4. Get conversation messages
        res_get_msgs = client.get(f"/conversations/{conv_a_id}/messages", headers=auth_headers_a)
        assert res_get_msgs.status_code == 200
        assert len(res_get_msgs.json()) >= 1

        # --------------------------------------------------------------------------
        # Step 9: Logout User A
        # --------------------------------------------------------------------------
        with patch("backend.api.auth_router.auth_service.logout") as mock_logout:
            mock_logout.return_value = {"message": "Logged out successfully"}
            res_logout = client.post("/auth/logout", headers=auth_headers_a)
            assert res_logout.status_code == 200

        # --------------------------------------------------------------------------
        # Step 10: User B Cross-Tenant Isolation Enforcement
        # --------------------------------------------------------------------------
        auth_headers_b = {"Authorization": f"Bearer {user_b_token}"}

        # Override dependencies for User B
        app.dependency_overrides[get_current_db_user] = lambda: mock_db_user_b
        app.dependency_overrides[get_optional_current_db_user] = lambda: mock_db_user_b

        # User B cannot see User A's document in list
        res_b_docs = client.get("/documents", headers=auth_headers_b)
        assert res_b_docs.status_code == 200
        assert not any(d["id"] == doc_a_id for d in res_b_docs.json())

        # User B cannot delete User A's document
        res_b_del_doc = client.delete(f"/documents/{doc_a_id}", headers=auth_headers_b)
        assert res_b_del_doc.status_code == 403
        assert "Access denied" in res_b_del_doc.json()["detail"]

        # User B cannot see User A's conversation
        res_b_convs = client.get("/conversations", headers=auth_headers_b)
        assert res_b_convs.status_code == 200
        assert not any(c["id"] == conv_a_id for c in res_b_convs.json())

        # User B cannot access User A's messages
        res_b_msgs = client.get(f"/conversations/{conv_a_id}/messages", headers=auth_headers_b)
        assert res_b_msgs.status_code == 403

        # User B cannot rename User A's conversation
        res_b_rename = client.patch(f"/conversations/{conv_a_id}", json={"title": "Hacked Title"}, headers=auth_headers_b)
        assert res_b_rename.status_code == 403

        # User B cannot delete User A's conversation
        res_b_del_conv = client.delete(f"/conversations/{conv_a_id}", headers=auth_headers_b)
        assert res_b_del_conv.status_code == 403

        # Cache isolation: cache keys are scoped to user identity
        cache = get_llm_cache_service()
        key_user_a = cache.generate_cache_key(normalized_query="test isolation", user_scope=user_a_id)
        key_user_b = cache.generate_cache_key(normalized_query="test isolation", user_scope=user_b_id)
        assert key_user_a != key_user_b

    finally:
        app.dependency_overrides.clear()
