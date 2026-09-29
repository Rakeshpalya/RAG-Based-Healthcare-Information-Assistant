"""Enable Row Level Security (RLS) on public application tables.

Revision ID: 002_enable_rls
Revises: 001_initial_schema
Create Date: 2026-09-14
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002_enable_rls'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES = ["users", "documents", "document_chunks", "conversations", "messages"]


def upgrade() -> None:
    """
    Enables Row Level Security on public tables when running on PostgreSQL.
    Safely skips on SQLite to preserve local in-memory test compatibility.
    """
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for table in TABLES:
            op.execute(sa.text(f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY;"))


def downgrade() -> None:
    """
    Disables Row Level Security on public tables when running on PostgreSQL.
    """
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for table in TABLES:
            op.execute(sa.text(f"ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY;"))
