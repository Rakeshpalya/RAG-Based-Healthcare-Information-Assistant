import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.database import Base
from backend.database.models import User, Document, DocumentChunk, Conversation, Message
from backend.database.schemas import (
    UserCreate,
    DocumentCreate,
    DocumentChunkCreate,
    ConversationCreate,
    MessageCreate,
)
from backend.database.repositories import (
    UserRepository,
    DocumentRepository,
    ChunkRepository,
    ConversationRepository,
    MessageRepository,
)


def get_test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON;"))
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return TestingSession()


def test_user_repository_crud():
    db = get_test_db()
    user_in = UserCreate(email="patient1@healthcare.org", role="patient", full_name="Alice Smith")
    user = UserRepository.create(db, user_in)
    assert user.id is not None
    assert user.email == "patient1@healthcare.org"

    # Get by ID
    fetched = UserRepository.get_by_id(db, user.id)
    assert fetched is not None
    assert fetched.full_name == "Alice Smith"

    # Get by email
    fetched_email = UserRepository.get_by_email(db, "patient1@healthcare.org")
    assert fetched_email is not None
    assert fetched_email.id == user.id

    # List
    all_users = UserRepository.list_all(db)
    assert len(all_users) == 1
    db.close()
    print("[PASS] test_user_repository_crud passed.")


def test_document_repository_crud():
    db = get_test_db()
    user = UserRepository.create(db, UserCreate(email="doc_owner@healthcare.org"))

    doc_in = DocumentCreate(
        user_id=user.id,
        filename="cardiology_report.pdf",
        file_path="/storage/docs/cardiology_report.pdf",
        file_size_bytes=1048576,
        num_pages=4,
        status="uploaded"
    )
    doc = DocumentRepository.create(db, doc_in)
    assert doc.id is not None
    assert doc.status == "uploaded"

    # Update status
    updated = DocumentRepository.update_status(db, doc.id, status="indexed", num_chunks=12)
    assert updated is not None
    assert updated.status == "indexed"
    assert updated.num_chunks == 12

    # List by user
    user_docs = DocumentRepository.list_all(db, user_id=user.id)
    assert len(user_docs) == 1
    assert user_docs[0].filename == "cardiology_report.pdf"
    db.close()
    print("[PASS] test_document_repository_crud passed.")


def test_chunk_repository_bulk_create():
    db = get_test_db()
    doc = DocumentRepository.create(
        db,
        DocumentCreate(filename="lab_results.pdf", file_path="/storage/lab_results.pdf")
    )

    chunks = [
        DocumentChunkCreate(
            chunk_id=f"CHUNK_{i}",
            chunk_index=i,
            text=f"Medical text snippet {i}",
            page_number=1,
            token_count=15,
            embedding_index=i
        )
        for i in range(5)
    ]
    created = ChunkRepository.bulk_create(db, doc.id, chunks)
    assert len(created) == 5

    # Check that parent document num_chunks was updated automatically
    doc_refreshed = DocumentRepository.get_by_id(db, doc.id)
    assert doc_refreshed.num_chunks == 5
    db.close()
    print("[PASS] test_chunk_repository_bulk_create passed.")


def test_chunk_repository_bridge_to_vector_store():
    """Verify that chunk metadata can be queried by FAISS index or chunk_id."""
    db = get_test_db()
    doc = DocumentRepository.create(
        db,
        DocumentCreate(filename="faiss_bridge.pdf", file_path="/storage/faiss_bridge.pdf")
    )

    chunks = [
        DocumentChunkCreate(
            chunk_id="MED_CHUNK_42",
            chunk_index=0,
            text="Hypertension is diagnosed when blood pressure consistently exceeds 130/80 mmHg.",
            page_number=2,
            token_count=18,
            embedding_index=42
        )
    ]
    ChunkRepository.bulk_create(db, doc.id, chunks)

    # Retrieval by chunk_id
    by_id = ChunkRepository.get_by_chunk_id(db, "MED_CHUNK_42")
    assert by_id is not None
    assert by_id.embedding_index == 42
    assert "Hypertension" in by_id.text

    # Retrieval by FAISS embedding index
    by_emb = ChunkRepository.get_by_embedding_index(db, 42)
    assert by_emb is not None
    assert by_emb.chunk_id == "MED_CHUNK_42"
    db.close()
    print("[PASS] test_chunk_repository_bridge_to_vector_store passed.")


def test_conversation_repository_crud():
    db = get_test_db()
    user = UserRepository.create(db, UserCreate(email="conv_user@healthcare.org"))

    conv_in = ConversationCreate(user_id=user.id, title="Blood Pressure Consultation")
    conv = ConversationRepository.create(db, conv_in)
    assert conv.id is not None
    assert conv.title == "Blood Pressure Consultation"

    # List by user
    convs = ConversationRepository.list_by_user(db, user_id=user.id)
    assert len(convs) == 1
    assert convs[0].id == conv.id
    db.close()
    print("[PASS] test_conversation_repository_crud passed.")


def test_message_repository_ordering():
    db = get_test_db()
    conv = ConversationRepository.create(
        db,
        ConversationCreate(title="Ordering Consultation")
    )

    # Add messages in sequence
    m1 = MessageRepository.create(
        db, conv.id,
        MessageCreate(sender="user", text="What are early diabetes symptoms?")
    )
    m2 = MessageRepository.create(
        db, conv.id,
        MessageCreate(
            sender="assistant",
            text="Common symptoms include increased thirst, frequent urination, and fatigue.",
            agent_type="ExplanationAgent",
            citations=[{"source": "CDC Diabetes Factsheet", "chunk_id": "MED_CHUNK_10"}]
        )
    )

    messages = MessageRepository.list_by_conversation(db, conv.id)
    assert len(messages) == 2
    assert messages[0].id == m1.id
    assert messages[0].sender == "user"
    assert messages[1].id == m2.id
    assert messages[1].sender == "assistant"
    assert messages[1].citations is not None
    db.close()
    print("[PASS] test_message_repository_ordering passed.")


def test_cascade_delete_user_to_docs_and_convs():
    db = get_test_db()
    user = UserRepository.create(db, UserCreate(email="delete_me@healthcare.org"))
    doc = DocumentRepository.create(
        db, DocumentCreate(user_id=user.id, filename="user_doc.pdf", file_path="/p/1")
    )
    ChunkRepository.bulk_create(
        db, doc.id,
        [DocumentChunkCreate(chunk_id="C_1", chunk_index=0, text="chunk")]
    )
    conv = ConversationRepository.create(
        db, ConversationCreate(user_id=user.id, title="Chat 1")
    )
    MessageRepository.create(
        db, conv.id, MessageCreate(sender="user", text="Hello")
    )

    # Delete User
    success = UserRepository.delete(db, user.id)
    assert success is True

    # Check cascading deletions
    assert DocumentRepository.get_by_id(db, doc.id) is None
    assert ChunkRepository.get_by_chunk_id(db, "C_1") is None
    assert ConversationRepository.get_by_id(db, conv.id) is None
    assert len(MessageRepository.list_by_conversation(db, conv.id)) == 0
    db.close()
    print("[PASS] test_cascade_delete_user_to_docs_and_convs passed.")


def test_cascade_delete_document_to_chunks():
    db = get_test_db()
    doc = DocumentRepository.create(
        db, DocumentCreate(filename="standalone.pdf", file_path="/p/2")
    )
    ChunkRepository.bulk_create(
        db, doc.id,
        [
            DocumentChunkCreate(chunk_id="ST_1", chunk_index=0, text="t1"),
            DocumentChunkCreate(chunk_id="ST_2", chunk_index=1, text="t2"),
        ]
    )

    # Delete Document
    deleted = DocumentRepository.delete(db, doc.id)
    assert deleted is True

    # Verify chunks are gone
    assert ChunkRepository.get_by_chunk_id(db, "ST_1") is None
    assert ChunkRepository.get_by_chunk_id(db, "ST_2") is None
    db.close()
    print("[PASS] test_cascade_delete_document_to_chunks passed.")


def test_cascade_delete_conversation_to_messages():
    db = get_test_db()
    conv = ConversationRepository.create(db, ConversationCreate(title="Delete Chat"))
    MessageRepository.create(db, conv.id, MessageCreate(sender="user", text="msg 1"))
    MessageRepository.create(db, conv.id, MessageCreate(sender="agent", text="msg 2"))

    # Delete Conversation
    deleted = ConversationRepository.delete(db, conv.id)
    assert deleted is True

    # Verify messages are gone
    messages = MessageRepository.list_by_conversation(db, conv.id)
    assert len(messages) == 0
    db.close()
    print("[PASS] test_cascade_delete_conversation_to_messages passed.")


def test_repository_isolation_unrelated_records():
    db = get_test_db()
    u1 = UserRepository.create(db, UserCreate(email="u1@healthcare.org"))
    u2 = UserRepository.create(db, UserCreate(email="u2@healthcare.org"))

    d1 = DocumentRepository.create(db, DocumentCreate(user_id=u1.id, filename="d1.pdf", file_path="/p/d1"))
    d2 = DocumentRepository.create(db, DocumentCreate(user_id=u2.id, filename="d2.pdf", file_path="/p/d2"))

    # Delete user 1
    UserRepository.delete(db, u1.id)

    # User 2 and Document 2 must remain intact
    assert UserRepository.get_by_id(db, u2.id) is not None
    assert DocumentRepository.get_by_id(db, d2.id) is not None
    assert DocumentRepository.get_by_id(db, d1.id) is None
    db.close()
    print("[PASS] test_repository_isolation_unrelated_records passed.")


if __name__ == "__main__":
    test_user_repository_crud()
    test_document_repository_crud()
    test_chunk_repository_bulk_create()
    test_chunk_repository_bridge_to_vector_store()
    test_conversation_repository_crud()
    test_message_repository_ordering()
    test_cascade_delete_user_to_docs_and_convs()
    test_cascade_delete_document_to_chunks()
    test_cascade_delete_conversation_to_messages()
    test_repository_isolation_unrelated_records()
    print("[SUCCESS] All 10 repository unit tests passed successfully!")
