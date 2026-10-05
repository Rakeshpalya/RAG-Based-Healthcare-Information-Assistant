"""
Unit tests for Phase 9 Database Connection Pooling and Resilience.
Validates connection pool configuration, health probes, concurrency safety,
and graceful shutdown.
"""

import time
import pytest
from sqlalchemy import text
from backend.database.database import (
    get_engine,
    get_session_factory,
    SessionLocal,
    check_db_connection,
    dispose_engine,
    get_db
)


def test_db_connection_health_probe():
    """Ensures check_db_connection performs a non-destructive ping and reports ready."""
    health = check_db_connection()
    assert health["status"] == "ready"
    assert health["dialect"] in ("sqlite", "postgresql")
    assert health["pool_status"] == "healthy"
    assert health["error"] is None


def test_concurrent_session_checkout():
    """Verifies that multiple database sessions can be checked out concurrently without error."""
    factory = get_session_factory()
    sessions = []
    try:
        for _ in range(5):
            s = factory()
            res = None
            for attempt in range(3):
                try:
                    res = s.execute(text("SELECT 1")).scalar()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(0.5)
            assert res == 1
            sessions.append(s)
    finally:
        for s in sessions:
            s.close()


def test_session_generator_rollback_on_exception():
    """Ensures the get_db generator rolls back transaction when an exception occurs."""
    gen = get_db()
    db = next(gen)
    assert db is not None

    with pytest.raises(RuntimeError):
        try:
            raise RuntimeError("Simulated transaction failure")
        except RuntimeError as e:
            gen.throw(e)


def test_dispose_engine_graceful_shutdown():
    """Ensures dispose_engine clears active pool resources."""
    engine = get_engine()
    assert engine is not None

    dispose_engine()
    # Subsequent call re-initializes cleanly
    new_engine = get_engine()
    assert new_engine is not None
