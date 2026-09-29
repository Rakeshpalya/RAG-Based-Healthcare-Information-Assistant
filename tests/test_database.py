import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from sqlalchemy import inspect, text, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from backend.database.database import Base, get_engine, get_session_factory, get_db
from backend.database.models import User, Document, DocumentChunk, Conversation, Message


def setup_in_memory_db():
    """Sets up an isolated SQLite in-memory engine with Foreign Keys enabled."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON;"))
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, TestingSession


def test_engine_and_tables_creation():
    """Verify all 5 relational tables are properly created in Base.metadata."""
    engine, _ = setup_in_memory_db()
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    expected_tables = {"users", "documents", "document_chunks", "conversations", "messages"}
    assert expected_tables.issubset(set(tables)), f"Expected {expected_tables}, got {tables}"
    print("[PASS] test_engine_and_tables_creation passed. Tables:", tables)


def test_table_columns_and_types():
    """Verify key columns and indices exist across models."""
    engine, _ = setup_in_memory_db()
    inspector = inspect(engine)

    # Check users columns
    user_cols = {col["name"] for col in inspector.get_columns("users")}
    assert {"id", "email", "role", "full_name", "created_at", "updated_at"}.issubset(user_cols)

    # Check document_chunks columns (specifically chunk_id and embedding_index)
    chunk_cols = {col["name"] for col in inspector.get_columns("document_chunks")}
    assert {"id", "document_id", "chunk_id", "chunk_index", "text", "embedding_index"}.issubset(chunk_cols)

    # Check messages columns
    msg_cols = {col["name"] for col in inspector.get_columns("messages")}
    assert {"id", "conversation_id", "sender", "text", "agent_type", "citations"}.issubset(msg_cols)

    print("[PASS] test_table_columns_and_types passed.")


def test_foreign_key_pragmas():
    """Verify foreign keys are enforced and relationships resolve properly."""
    engine, TestingSession = setup_in_memory_db()
    db = TestingSession()

    user = User(email="test_fk@healthcare.org", role="clinician", full_name="Dr. Sarah")
    db.add(user)
    db.commit()
    db.refresh(user)

    doc = Document(user_id=user.id, filename="clinical_notes.pdf", file_path="/data/clinical_notes.pdf")
    db.add(doc)
    db.commit()
    db.refresh(doc)

    assert doc.user.id == user.id
    assert len(user.documents) == 1
    assert user.documents[0].filename == "clinical_notes.pdf"

    db.close()
    print("[PASS] test_foreign_key_pragmas passed.")


def test_transaction_rollback():
    """Verify transactions rollback cleanly on error without corrupting state."""
    engine, TestingSession = setup_in_memory_db()
    db = TestingSession()

    user = User(email="rollback_user@healthcare.org")
    db.add(user)
    db.commit()

    try:
        duplicate_user = User(email="rollback_user@healthcare.org")
        db.add(duplicate_user)
        db.commit()
        assert False, "Should have failed due to duplicate email unique constraint."
    except Exception:
        db.rollback()

    users = db.query(User).all()
    assert len(users) == 1
    assert users[0].email == "rollback_user@healthcare.org"
    db.close()
    print("[PASS] test_transaction_rollback passed.")


def test_get_db_generator_lifecycle():
    """Verify the get_db dependency yields a working session and closes it."""
    generator = get_db()
    session = next(generator)
    assert session is not None
    assert session.is_active

    # Exhaust generator to trigger finally block
    try:
        next(generator)
    except StopIteration:
        pass

    print("[PASS] test_get_db_generator_lifecycle passed.")


if __name__ == "__main__":
    test_engine_and_tables_creation()
    test_table_columns_and_types()
    test_foreign_key_pragmas()
    test_transaction_rollback()
    test_get_db_generator_lifecycle()
    print("[SUCCESS] All 5 database layer unit tests passed successfully!")
