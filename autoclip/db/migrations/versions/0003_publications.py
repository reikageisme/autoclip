"""publications

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-23 16:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = '0003'
down_revision: str | None = '0002'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'publications',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('render_id', sa.Integer(), nullable=False),
        sa.Column('platform', sa.String(length=32), nullable=False),
        sa.Column('page_id', sa.String(length=64), nullable=False),
        sa.Column('caption', sa.String(), nullable=True),
        sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('fb_post_id', sa.String(length=128), nullable=True),
        sa.Column('error', sa.String(), nullable=True),
        sa.Column('attempts', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['render_id'], ['renders.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('publications', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_publications_render_id'), ['render_id'], unique=False)
        batch_op.create_index(
            batch_op.f('ix_publications_scheduled_at'), ['scheduled_at'], unique=False
        )
        batch_op.create_index(batch_op.f('ix_publications_status'), ['status'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('publications', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_publications_status'))
        batch_op.drop_index(batch_op.f('ix_publications_scheduled_at'))
        batch_op.drop_index(batch_op.f('ix_publications_render_id'))
    op.drop_table('publications')
