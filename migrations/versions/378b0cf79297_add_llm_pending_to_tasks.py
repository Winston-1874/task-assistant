"""add_llm_pending_to_tasks

Revision ID: 378b0cf79297
Revises: 1fea5b89554e
Create Date: 2026-04-15 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '378b0cf79297'
down_revision: Union[str, None] = '1fea5b89554e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('tasks', sa.Column('llm_pending', sa.Boolean(), nullable=False, server_default=sa.text('0')))


def downgrade() -> None:
    op.drop_column('tasks', 'llm_pending')
