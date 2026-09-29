import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.database.database import Base, get_db

# Isolated in-memory database for testing endpoints
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
with test_engine.connect() as conn:
    conn.execute(text("PRAGMA foreign_keys=ON;"))
Base.metadata.create_all(bind=test_engine)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


from backend.api.auth_dependencies import get_current_user


def override_get_current_user():
    return {"id": 1, "email": "sarah.connor@healthcare.org"}


app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = override_get_current_user
client = TestClient(app)


def _run_user_endpoints():
    # 1. Create User
    payload = {
        "email": "sarah.connor@healthcare.org",
        "role": "patient",
        "full_name": "Sarah Connor"
    }
    response = client.post("/users", json=payload)
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["id"] is not None
    assert data["email"] == "sarah.connor@healthcare.org"
    user_id = data["id"]

    # 2. Get User
    get_res = client.get(f"/users/{user_id}")
    assert get_res.status_code == 200
    assert get_res.json()["full_name"] == "Sarah Connor"

    # 3. Duplicate Email Rejection
    dup_res = client.post("/users", json=payload)
    assert dup_res.status_code == 400
    assert "already exists" in dup_res.json()["detail"]

    # 4. Nonexistent User
    missing_res = client.get("/users/99999")
    assert missing_res.status_code == 404
    print("[PASS] test_user_endpoints passed.")
    return user_id


def _run_document_metadata_endpoints(user_id: int):
    # 1. Create Document
    doc_payload = {
        "user_id": user_id,
        "filename": "discharge_summary.pdf",
        "file_path": "/data/discharge_summary.pdf",
        "file_size_bytes": 524288,
        "num_pages": 3,
        "status": "processed"
    }
    create_res = client.post("/documents/metadata", json=doc_payload)
    assert create_res.status_code == 201, create_res.text
    doc_data = create_res.json()
    doc_id = doc_data["id"]
    assert doc_data["filename"] == "discharge_summary.pdf"
    assert doc_data["user_id"] == user_id

    # 2. List Documents (all & filtered)
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1

    user_filtered = client.get(f"/documents?user_id={user_id}")
    assert user_filtered.status_code == 200
    assert len(user_filtered.json()) == 1

    # 3. Get Document by ID
    get_doc = client.get(f"/documents/{doc_id}")
    assert get_doc.status_code == 200
    assert get_doc.json()["id"] == doc_id

    # 4. Nonexistent Document
    missing = client.get("/documents/88888")
    assert missing.status_code == 404
    print("[PASS] test_document_metadata_endpoints passed.")


def _run_conversation_endpoints(user_id: int):
    # 1. Create Conversation
    conv_payload = {
        "user_id": user_id,
        "title": "Hypertension Medication Query"
    }
    res = client.post("/conversations", json=conv_payload)
    assert res.status_code == 201, res.text
    conv_data = res.json()
    conv_id = conv_data["id"]
    assert conv_data["title"] == "Hypertension Medication Query"

    # 2. List Conversations
    list_res = client.get("/conversations")
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1
    assert any(c["id"] == conv_id for c in list_res.json())

    # 3. Nonexistent user for conversation
    bad_res = client.post("/conversations", json={"user_id": 99999, "title": "Invalid"})
    assert bad_res.status_code == 404
    print("[PASS] test_conversation_endpoints passed.")
    return conv_id


def _run_message_endpoints(conv_id: int):
    # 1. Append User Message
    msg1 = {
        "sender": "user",
        "text": "What are side effects of ACE inhibitors?",
    }
    res1 = client.post(f"/conversations/{conv_id}/messages", json=msg1)
    assert res1.status_code == 201, res1.text
    assert res1.json()["sender"] == "user"

    # 2. Append Assistant Message
    msg2 = {
        "sender": "assistant",
        "text": "Common side effects include dry cough, hyperkalemia, and dizziness.",
        "agent_type": "ExplanationAgent",
        "citations": [{"source": "Cardiology Guidelines 2024", "page": 4}]
    }
    res2 = client.post(f"/conversations/{conv_id}/messages", json=msg2)
    assert res2.status_code == 201, res2.text
    assert res2.json()["agent_type"] == "ExplanationAgent"

    # 3. Get Chronological Messages
    history_res = client.get(f"/conversations/{conv_id}/messages")
    assert history_res.status_code == 200
    history = history_res.json()
    assert len(history) == 2
    assert history[0]["sender"] == "user"
    assert history[1]["sender"] == "assistant"
    assert history[1]["citations"] is not None

    # 4. Nonexistent Conversation
    missing = client.get("/conversations/99999/messages")
    assert missing.status_code == 404
    print("[PASS] test_message_endpoints passed.")


def test_full_database_endpoints_workflow():
    """End-to-end integration test executed by pytest and python runner."""
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    try:
        u_id = _run_user_endpoints()
        _run_document_metadata_endpoints(u_id)
        c_id = _run_conversation_endpoints(u_id)
        _run_message_endpoints(c_id)
        print("[SUCCESS] All database API endpoint integration tests passed successfully!")
    finally:
        app.dependency_overrides.clear()


if __name__ == "__main__":
    test_full_database_endpoints_workflow()

