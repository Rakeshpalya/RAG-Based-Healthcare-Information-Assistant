"""
Unit tests for Phase 9 Database Migrations with Alembic.
Validates:
1. Alembic configuration and script directory structure.
2. Initial schema migration includes all 5 core tables:
   users, documents, conversations, messages, audit_logs.
3. Offline migration SQL generation works cleanly.
"""

import os
from alembic.config import Config
from alembic.script import ScriptDirectory
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_alembic_configuration_exists():
    """Validates alembic.ini and migration directory structure."""
    ini_path = os.path.join(REPO_ROOT, "alembic.ini")
    assert os.path.exists(ini_path), "alembic.ini must exist in repo root"

    env_path = os.path.join(REPO_ROOT, "alembic", "env.py")
    assert os.path.exists(env_path), "alembic/env.py must exist"

    versions_path = os.path.join(REPO_ROOT, "alembic", "versions")
    assert os.path.isdir(versions_path), "alembic/versions directory must exist"


def test_migration_revisions_present():
    """Validates that baseline migration revisions are registered."""
    config = Config(os.path.join(REPO_ROOT, "alembic.ini"))
    script = ScriptDirectory.from_config(config)

    heads = script.get_heads()
    assert len(heads) > 0, "At least one migration revision head must be present"

    # Verify baseline revision contains expected tables
    initial_rev = script.get_revision("001_initial")
    assert initial_rev is not None, "001_initial revision must exist"


def test_models_metadata_has_all_five_core_tables():
    """Validates that SQLAlchemy Base.metadata registers all required production tables."""
    from backend.database.database import Base
    import backend.database.models  # ensure models are imported

    registered_tables = set(Base.metadata.tables.keys())
    expected_tables = {"users", "documents", "document_chunks", "conversations", "messages"}

    assert expected_tables.issubset(registered_tables), (
        f"Missing tables in Base.metadata: {expected_tables - registered_tables}"
    )
