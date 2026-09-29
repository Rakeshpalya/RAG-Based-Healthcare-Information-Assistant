"""
Tests for Conversation Deletion and History Management with Strict User Isolation.

Verifies:
1. Authenticated user can delete their own conversation (DELETE /conversations/{id}).
2. User cannot delete another user's conversation (403 Forbidden).
3. Deleting nonexistent conversation returns 404 Not Found.
4. User can clear all their conversations (DELETE /conversations).
5. Clearing conversations only affects the authenticated user and preserves other users' data.
6. APIClient delete_conversation and clear_conversations methods.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
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
from backend.database.models import User, Conversation, Message
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


@pytest.fixture(autouse=True)
def setup_db():
    app.dependency_overrides[get_db] = override_db
    yield
    app.dependency_overrides.clear()


def test_delete_own_conversation_success():
    """Verify owner can successfully delete their conversation."""
    db = TestingSession()
    user1 = User(id=9101, email="user9101@clinic.org", full_name="User One")
    db.add(user1)
    db.commit()

    conv1 = Conversation(id=5001, user_id=9101, title="User1 Hypertension Thread")
    db.add(conv1)
    db.commit()

    msg1 = Message(conversation_id=5001, sender="user", text="What is hypertension?")
    db.add(msg1)
    db.commit()
    db.close()

    mock_supa = MagicMock()
    mock_supa.id = "supa_9101"
    mock_supa.email = "user9101@clinic.org"

    app.dependency_overrides[get_current_user] = lambda: mock_supa
    app.dependency_overrides[get_current_db_user] = lambda: user1
    app.dependency_overrides[get_optional_current_db_user] = lambda: user1

    client = TestClient(app)
    res = client.delete("/conversations/5001")
    assert res.status_code == 200
    assert res.json()["success"] is True

    # Verify conversation and message deleted from db
    db2 = TestingSession()
    assert db2.query(Conversation).filter(Conversation.id == 5001).first() is None
    assert db2.query(Message).filter(Message.conversation_id == 5001).first() is None
    db2.close()


def test_delete_other_user_conversation_forbidden():
    """Verify user cannot delete another user's conversation (403 Forbidden)."""
    db = TestingSession()
    user1 = User(id=9102, email="user9102@clinic.org", full_name="User Two")
    user2 = User(id=9103, email="user9103@clinic.org", full_name="User Three")
    db.add_all([user1, user2])
    db.commit()

    conv_user2 = Conversation(id=5002, user_id=9103, title="User3 Private Thread")
    db.add(conv_user2)
    db.commit()
    db.close()

    # Authenticate as User 1
    mock_supa = MagicMock()
    mock_supa.id = "supa_9102"
    mock_supa.email = "user9102@clinic.org"

    app.dependency_overrides[get_current_user] = lambda: mock_supa
    app.dependency_overrides[get_current_db_user] = lambda: user1
    app.dependency_overrides[get_optional_current_db_user] = lambda: user1

    client = TestClient(app)
    res = client.delete("/conversations/5002")
    assert res.status_code == 403
    assert "Access denied" in res.json()["detail"]

    # Verify conversation still exists
    db2 = TestingSession()
    assert db2.query(Conversation).filter(Conversation.id == 5002).first() is not None
    db2.close()


def test_delete_nonexistent_conversation_404():
    """Verify deleting a nonexistent conversation returns 404."""
    db = TestingSession()
    user1 = User(id=9104, email="user9104@clinic.org", full_name="User Four")
    db.add(user1)
    db.commit()
    db.close()

    mock_supa = MagicMock()
    mock_supa.id = "supa_9104"
    mock_supa.email = "user9104@clinic.org"

    app.dependency_overrides[get_current_user] = lambda: mock_supa
    app.dependency_overrides[get_current_db_user] = lambda: user1
    app.dependency_overrides[get_optional_current_db_user] = lambda: user1

    client = TestClient(app)
    res = client.delete("/conversations/999999")
    assert res.status_code == 404


def test_clear_conversations_user_isolation():
    """Verify clearing history removes only the current user's conversations."""
    db = TestingSession()
    user_a = User(id=9105, email="user9105@clinic.org", full_name="User A")
    user_b = User(id=9106, email="user9106@clinic.org", full_name="User B")
    db.add_all([user_a, user_b])
    db.commit()

    # User A has 2 conversations
    conv_a1 = Conversation(id=5003, user_id=9105, title="User A Query 1")
    conv_a2 = Conversation(id=5004, user_id=9105, title="User A Query 2")
    # User B has 1 conversation
    conv_b1 = Conversation(id=5005, user_id=9106, title="User B Query 1")
    db.add_all([conv_a1, conv_a2, conv_b1])
    db.commit()
    db.close()

    # Authenticate as User A
    mock_supa = MagicMock()
    mock_supa.id = "supa_9105"
    mock_supa.email = "user9105@clinic.org"

    app.dependency_overrides[get_current_user] = lambda: mock_supa
    app.dependency_overrides[get_current_db_user] = lambda: user_a
    app.dependency_overrides[get_optional_current_db_user] = lambda: user_a

    client = TestClient(app)
    res = client.delete("/conversations")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["deleted_count"] == 2

    # Verify User A's conversations are gone, but User B's conversation remains intact
    db2 = TestingSession()
    assert db2.query(Conversation).filter(Conversation.user_id == 9105).count() == 0
    assert db2.query(Conversation).filter(Conversation.user_id == 9106).count() == 1
    assert db2.query(Conversation).filter(Conversation.id == 5005).first() is not None
    db2.close()


def test_api_client_delete_and_clear():
    """Verify APIClient methods for delete and clear conversations."""
    from unittest.mock import patch

    client = APIClient(base_url="http://127.0.0.1:8000")
    client.set_token("mock_token")

    # Mock delete_conversation
    with patch("requests.delete") as mock_del:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"success": True, "message": "Deleted"}
        mock_del.return_value = mock_resp

        res = client.delete_conversation(123)
        assert res["success"] is True

    # Mock clear_conversations
    with patch("requests.delete") as mock_del:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b'{"success": true, "deleted_count": 3}'
        mock_resp.json.return_value = {"success": True, "deleted_count": 3}
        mock_del.return_value = mock_resp

        res = client.clear_conversations()
        assert res["success"] is True
        assert res["deleted_count"] == 3
