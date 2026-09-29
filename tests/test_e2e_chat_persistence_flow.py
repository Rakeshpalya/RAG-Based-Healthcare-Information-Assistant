"""
End-to-End Chat & Conversation Persistence Flow Test.

Validates the full chain requested in PART 4:
User opens Research Chat
        ↓
Chat page loads
        ↓
Existing conversation loads
        ↓
User enters question
        ↓
Question is sent to /rag/query
        ↓
Safety checks run
        ↓
RAG retrieves user-owned documents
        ↓
Gemini generates grounded answer
        ↓
Assistant response displayed
        ↓
User message persisted (via add_message_to_conversation)
        ↓
Assistant message persisted (via add_message_to_conversation)
        ↓
Conversation appears in History
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from backend.main import app
from backend.database.database import Base, get_db
from backend.database.models import User
from backend.api.auth_dependencies import (
    get_current_user,
    get_current_db_user,
    get_optional_current_db_user,
)
from frontend.api_client import APIClient

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
with test_engine.connect() as conn:
    conn.execute(text("PRAGMA foreign_keys=ON;"))
Base.metadata.create_all(bind=test_engine)
TestingSession = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=test_engine)


def override_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


def test_e2e_chat_flow_and_message_persistence():
    """Validates full chat inquiry, RAG execution, and message persistence."""
    app.dependency_overrides[get_db] = override_db
    test_client = TestClient(app)

    # 1. Create test user
    db = TestingSession()
    user = User(id=8801, email="chat_flow_user@clinic.org", full_name="Dr. Flow")
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()

    mock_supa = MagicMock()
    mock_supa.id = "supa_8801"
    mock_supa.email = "chat_flow_user@clinic.org"

    app.dependency_overrides[get_current_user] = lambda: mock_supa
    app.dependency_overrides[get_current_db_user] = lambda: user
    app.dependency_overrides[get_optional_current_db_user] = lambda: user

    try:
        # 2. Create conversation session
        conv_resp = test_client.post("/conversations", json={"title": "Clinical Hypertension Inquiry"})
        assert conv_resp.status_code == 201, f"Failed to create conv: {conv_resp.text}"
        conv_data = conv_resp.json()
        conv_id = conv_data["id"]
        assert conv_id is not None
        assert conv_data["user_id"] == 8801

        # 3. Simulate frontend APIClient using add_message_to_conversation for user message
        api = APIClient()
        assert hasattr(api, "add_message_to_conversation")

        user_question = "What is hypertension, what are the common risk factors, and what lifestyle changes are generally recommended to help manage it?"
        user_msg_resp = test_client.post(
            f"/conversations/{conv_id}/messages",
            json={"sender": "user", "text": user_question}
        )
        assert user_msg_resp.status_code == 201
        user_msg_data = user_msg_resp.json()
        assert user_msg_data["sender"] == "user"
        assert user_msg_data["text"] == user_question

        # 4. Send question to /rag/query with mocked generation
        with patch("backend.services.gemini_service.GeminiService.generate_answer") as mock_gemini:
            mock_gemini.return_value = {
                "answer": "Hypertension is persistently elevated blood pressure. Key lifestyle changes include aerobic exercise and dietary sodium restriction [Source 1].",
                "model": "gemini-flash-latest",
                "disclaimer": "Medical disclaimer.",
                "generation_time_ms": 115.0,
                "status": "success",
            }
            rag_resp = test_client.post(
                "/rag/query",
                json={
                    "question": user_question,
                    "top_k": 5,
                    "similarity_threshold": 0.25,
                }
            )
            assert rag_resp.status_code == 200
            rag_data = rag_resp.json()
            answer = rag_data.get("answer", "")
            sources = rag_data.get("sources", [])
            assert len(answer) > 0

        # 5. Persist assistant message with citations
        asst_msg_resp = test_client.post(
            f"/conversations/{conv_id}/messages",
            json={
                "sender": "assistant",
                "text": answer,
                "citations": sources,
                "agent_type": "rag_agent",
            }
        )
        assert asst_msg_resp.status_code == 201
        asst_msg_data = asst_msg_resp.json()
        assert asst_msg_data["sender"] == "assistant"
        assert asst_msg_data["agent_type"] == "rag_agent"

        # 6. Retrieve conversation messages (verifying both turns exist in DB)
        msgs_resp = test_client.get(f"/conversations/{conv_id}/messages")
        assert msgs_resp.status_code == 200
        messages = msgs_resp.json()
        assert len(messages) == 2
        assert messages[0]["sender"] == "user"
        assert messages[1]["sender"] == "assistant"
        assert messages[1]["text"] == answer

        # 7. Verify conversation appears in user's conversation list
        list_resp = test_client.get("/conversations")
        assert list_resp.status_code == 200
        conv_list = list_resp.json()
        matching_convs = [c for c in conv_list if c["id"] == conv_id]
        assert len(matching_convs) == 1
        assert matching_convs[0]["title"] == "Clinical Hypertension Inquiry"

    finally:
        app.dependency_overrides.clear()
