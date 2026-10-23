"""batch schedules

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-23 16:05:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = '0004'
down_revision: str | None = '0003'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'batch_schedules',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('recipe', sa.String(), nullable=False),
        sa.Column('count', sa.Integer(), nullable=False),
        sa.Column('seed', sa.Integer(), nullable=True),
        sa.Column('options', sa.JSON(), nullable=False),
        sa.Column('run_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('repeat_minutes', sa.Integer(), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        sa.Column('last_run_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('batch_schedules', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_batch_schedules_run_at'), ['run_at'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('batch_schedules', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_batch_schedules_run_at'))
    op.drop_table('batch_schedules')
