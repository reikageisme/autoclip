"""clips.source_path: file gốc của clip đã hạ proxy

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-24 18:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = '0005'
down_revision: str | None = '0004'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('clips') as batch:
        batch.add_column(sa.Column('source_path', sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('clips') as batch:
        batch.drop_column('source_path')
