"""
Unit tests for User Isolation and Protected Resources (Phase 12, Step 6).

Verifies the core security rule:
AUTHENTICATED USER ID -> DATABASE user_id -> ONLY THAT USER'S RESOURCES.

12 Dedicated Test Cases:
1. Authenticated user can access own document (GET /documents/{id} -> 200).
2. Authenticated user cannot access another user's document (GET /documents/{id} -> 403).
3. Document listing returns only current user's documents (GET /documents).
4. Document creation assigns authenticated user as owner (POST /documents/metadata).
5. Client cannot spoof document owner (POST /documents/metadata with conflicting user_id -> 403).
6. Authenticated user can access own conversation (GET /conversations/{id}/messages -> 200).
7. Authenticated user cannot access another user's conversation (GET /conversations/{id}/messages -> 403).
8. Conversation listing returns only current user's conversations (GET /conversations).
9. Messages cannot be added to another user's conversation (POST /conversations/{id}/messages -> 403).
10. Client cannot spoof conversation owner (POST /conversations with conflicting user_id -> 403).
11. Unauthenticated access is rejected (401 Unauthorized across protected endpoints).
12. Cross-user access consistently returns 403 Forbidden.

All tests execute against an isolated in-memory test database with zero real Supabase calls.
"""

import sys
from pathlib import Path
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
from backend.database.models import User
from backend.api.auth_dependencies import get_current_user

# Isolated in-memory database for user isolation tests
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
with engine.connect() as conn:
    conn.execute(text("PRAGMA foreign_keys=ON;"))
Base.metadata.create_all(bind=engine)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


client = TestClient(app)

# Active mock user context for dependency override
_current_mock_user = None


def override_user():
    return _current_mock_user


@pytest.fixture(autouse=True)
def setup_isolation_environment():
    """Sets up fresh tables and seeds User A and User B for each test."""
    global _current_mock_user
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    # Seed User A (id=1) and User B (id=2)
    db = TestingSession()
    user_a = User(email="user_a@healthcare.org", role="patient", full_name="User Alpha")
    user_b = User(email="user_b@healthcare.org", role="patient", full_name="User Beta")
    db.add_all([user_a, user_b])
    db.commit()
    db.refresh(user_a)
    db.refresh(user_b)
    db.close()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = override_user
    yield
    _current_mock_user = None
    app.dependency_overrides.clear()


def authenticate_as_user_a():
    global _current_mock_user
    _current_mock_user = {"id": 1, "email": "user_a@healthcare.org"}


def authenticate_as_user_b():
    global _current_mock_user
    _current_mock_user = {"id": 2, "email": "user_b@healthcare.org"}


def unauthenticate():
    global _current_mock_user
    _current_mock_user = None
    if get_current_user in app.dependency_overrides:
        del app.dependency_overrides[get_current_user]


# ==============================================================================
# 1. Authenticated user can access own document
# ==============================================================================

def test_authenticated_user_can_access_own_document():
    """1. Verify authenticated user can access their own document."""
    authenticate_as_user_a()
    doc_res = client.post(
        "/documents/metadata",
        json={
            "filename": "alpha_cardiology.pdf",
            "file_path": "/data/alpha_cardiology.pdf",
            "num_pages": 2,
        },
    )
    assert doc_res.status_code == 201
    doc_id = doc_res.json()["id"]

    get_res = client.get(f"/documents/{doc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == doc_id
    assert get_res.json()["user_id"] == 1
    print("[PASS] test_authenticated_user_can_access_own_document passed.")


# ==============================================================================
# 2. Authenticated user cannot access another user's document
# ==============================================================================

def test_authenticated_user_cannot_access_other_user_document():
    """2. Verify authenticated user cannot access another user's document (403 Forbidden)."""
    authenticate_as_user_a()
    doc_res = client.post(
        "/documents/metadata",
        json={"filename": "alpha_private.pdf", "file_path": "/data/alpha_private.pdf"},
    )
    doc_id = doc_res.json()["id"]

    # User B attempts to access User A's document
    authenticate_as_user_b()
    get_res = client.get(f"/documents/{doc_id}")
    assert get_res.status_code == 403
    assert "Access denied" in get_res.json()["detail"]
    print("[PASS] test_authenticated_user_cannot_access_other_user_document passed.")


# ==============================================================================
# 3. Document listing returns only current user's documents
# ==============================================================================

def test_document_listing_returns_only_current_user_documents():
    """3. Verify document listing is strictly isolated to current user's documents."""
    authenticate_as_user_a()
    client.post("/documents/metadata", json={"filename": "a1.pdf", "file_path": "/a1.pdf"})
    client.post("/documents/metadata", json={"filename": "a2.pdf", "file_path": "/a2.pdf"})

    authenticate_as_user_b()
    client.post("/documents/metadata", json={"filename": "b1.pdf", "file_path": "/b1.pdf"})

    # Check User A listing
    authenticate_as_user_a()
    docs_a = client.get("/documents").json()
    assert len(docs_a) == 2
    assert all(d["user_id"] == 1 for d in docs_a)

    # Check User B listing
    authenticate_as_user_b()
    docs_b = client.get("/documents").json()
    assert len(docs_b) == 1
    assert docs_b[0]["user_id"] == 2
    print("[PASS] test_document_listing_returns_only_current_user_documents passed.")


# ==============================================================================
# 4. Document creation assigns authenticated user as owner
# ==============================================================================

def test_document_creation_assigns_authenticated_user_as_owner():
    """4. Verify document creation automatically assigns authenticated user as owner."""
    authenticate_as_user_a()
    res = client.post(
        "/documents/metadata",
        json={"filename": "doc_owner.pdf", "file_path": "/data/doc_owner.pdf"},
    )
    assert res.status_code == 201
    assert res.json()["user_id"] == 1
    print("[PASS] test_document_creation_assigns_authenticated_user_as_owner passed.")


# ==============================================================================
# 5. Client cannot spoof document owner
# ==============================================================================

def test_client_cannot_spoof_document_owner():
    """5. Verify client cannot specify another user_id during document creation (403 Forbidden)."""
    authenticate_as_user_a()
    res = client.post(
        "/documents/metadata",
        json={"user_id": 2, "filename": "spoof.pdf", "file_path": "/data/spoof.pdf"},
    )
    assert res.status_code == 403
    assert "Cannot assign document to another user" in res.json()["detail"]
    print("[PASS] test_client_cannot_spoof_document_owner passed.")


# ==============================================================================
# 6. Authenticated user can access own conversation
# ==============================================================================

def test_authenticated_user_can_access_own_conversation():
    """6. Verify authenticated user can access their own conversation and its messages."""
    authenticate_as_user_a()
    conv_id = client.post("/conversations", json={"title": "Alpha Chat"}).json()["id"]
    client.post(f"/conversations/{conv_id}/messages", json={"sender": "user", "text": "Hello"})

    msgs_res = client.get(f"/conversations/{conv_id}/messages")
    assert msgs_res.status_code == 200
    assert len(msgs_res.json()) == 1
    assert msgs_res.json()[0]["text"] == "Hello"
    print("[PASS] test_authenticated_user_can_access_own_conversation passed.")


# ==============================================================================
# 7. Authenticated user cannot access another user's conversation
# ==============================================================================

def test_authenticated_user_cannot_access_other_user_conversation():
    """7. Verify authenticated user cannot access another user's conversation (403 Forbidden)."""
    authenticate_as_user_a()
    conv_id = client.post("/conversations", json={"title": "Alpha Secret"}).json()["id"]

    # User B attempts to access User A's conversation messages
    authenticate_as_user_b()
    res = client.get(f"/conversations/{conv_id}/messages")
    assert res.status_code == 403
    assert "Access denied" in res.json()["detail"]
    print("[PASS] test_authenticated_user_cannot_access_other_user_conversation passed.")


# ==============================================================================
# 8. Conversation listing returns only current user's conversations
# ==============================================================================

def test_conversation_listing_returns_only_current_user_conversations():
    """8. Verify conversation listing is strictly isolated to current user."""
    authenticate_as_user_a()
    client.post("/conversations", json={"title": "A1"})
    client.post("/conversations", json={"title": "A2"})

    authenticate_as_user_b()
    client.post("/conversations", json={"title": "B1"})

    # Check User A
    authenticate_as_user_a()
    convs_a = client.get("/conversations").json()
    assert len(convs_a) == 2
    assert all(c["user_id"] == 1 for c in convs_a)

    # Check User B
    authenticate_as_user_b()
    convs_b = client.get("/conversations").json()
    assert len(convs_b) == 1
    assert convs_b[0]["user_id"] == 2
    print("[PASS] test_conversation_listing_returns_only_current_user_conversations passed.")


# ==============================================================================
# 9. Messages cannot be added to another user's conversation
# ==============================================================================

def test_messages_cannot_be_added_to_other_user_conversation():
    """9. Verify user cannot append a message to a conversation owned by someone else."""
    authenticate_as_user_a()
    conv_id = client.post("/conversations", json={"title": "Alpha Isolated"}).json()["id"]

    # User B attempts to append message to User A's conversation
    authenticate_as_user_b()
    res = client.post(
        f"/conversations/{conv_id}/messages",
        json={"sender": "user", "text": "Unauthorized post"},
    )
    assert res.status_code == 403
    assert "Cannot add messages to another user's conversation" in res.json()["detail"]
    print("[PASS] test_messages_cannot_be_added_to_other_user_conversation passed.")


# ==============================================================================
# 10. Client cannot spoof conversation owner
# ==============================================================================

def test_client_cannot_spoof_conversation_owner():
    """10. Verify client cannot specify another user_id during conversation creation (403 Forbidden)."""
    authenticate_as_user_a()
    res = client.post(
        "/conversations",
        json={"user_id": 2, "title": "Spoof Session"},
    )
    assert res.status_code == 403
    assert "Cannot create conversation for another user" in res.json()["detail"]
    print("[PASS] test_client_cannot_spoof_conversation_owner passed.")


# ==============================================================================
# 11. Unauthenticated access is rejected across protected endpoints
# ==============================================================================

def test_unauthenticated_access_is_rejected():
    """11. Verify unauthenticated access across protected endpoints returns 401 Unauthorized."""
    unauthenticate()

    # Documents
    assert client.get("/documents").status_code == 401
    assert client.get("/documents/1").status_code == 401
    assert client.post("/documents/metadata", json={"filename": "test.pdf", "file_path": "/test.pdf"}).status_code == 401

    # Conversations
    assert client.get("/conversations").status_code == 401
    assert client.post("/conversations", json={"title": "Test"}).status_code == 401
    assert client.get("/conversations/1/messages").status_code == 401
    assert client.post("/conversations/1/messages", json={"sender": "user", "text": "msg"}).status_code == 401
    print("[PASS] test_unauthenticated_access_is_rejected passed.")


# ==============================================================================
# 12. Cross-user access consistently returns 403 Forbidden
# ==============================================================================

def test_cross_user_access_consistently_returns_403():
    """12. Verify cross-user read/write actions consistently return 403 Forbidden."""
    authenticate_as_user_a()
    doc_id = client.post("/documents/metadata", json={"filename": "alpha.pdf", "file_path": "/alpha.pdf"}).json()["id"]
    conv_id = client.post("/conversations", json={"title": "Alpha"}).json()["id"]

    authenticate_as_user_b()

    # Document access
    assert client.get(f"/documents/{doc_id}").status_code == 403

    # Conversation message access
    assert client.get(f"/conversations/{conv_id}/messages").status_code == 403

    # Message append attempt
    assert client.post(f"/conversations/{conv_id}/messages", json={"sender": "user", "text": "hack"}).status_code == 403

    # Document ownership spoof attempt
    assert client.post("/documents/metadata", json={"user_id": 1, "filename": "b.pdf", "file_path": "/b.pdf"}).status_code == 403

    # Conversation ownership spoof attempt
    assert client.post("/conversations", json={"user_id": 1, "title": "b"}).status_code == 403
    print("[PASS] test_cross_user_access_consistently_returns_403 passed.")


if __name__ == "__main__":
    print("Running User Isolation & Protected Resources unit tests...")
    test_authenticated_user_can_access_own_document()
    test_authenticated_user_cannot_access_other_user_document()
    test_document_listing_returns_only_current_user_documents()
    test_document_creation_assigns_authenticated_user_as_owner()
    test_client_cannot_spoof_document_owner()
    test_authenticated_user_can_access_own_conversation()
    test_authenticated_user_cannot_access_other_user_conversation()
    test_conversation_listing_returns_only_current_user_conversations()
    test_messages_cannot_be_added_to_other_user_conversation()
    test_client_cannot_spoof_conversation_owner()
    test_unauthenticated_access_is_rejected()
    test_cross_user_access_consistently_returns_403()
    print("All 12 User Isolation unit tests passed successfully!")
