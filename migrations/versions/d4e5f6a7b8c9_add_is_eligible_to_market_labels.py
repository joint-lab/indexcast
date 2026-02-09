"""Add is_eligible to market_labels

Revision ID: d4e5f6a7b8c9
Revises: 151800aa33ee
Create Date: 2026-02-09

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, None] = '151800aa33ee'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add is_eligible column to market_labels, defaulting existing rows to True."""
    with op.batch_alter_table('market_labels', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('is_eligible', sa.Boolean(), nullable=False)
        )


def downgrade() -> None:
    """Remove is_eligible column from market_labels."""
    with op.batch_alter_table('market_labels', schema=None) as batch_op:
        batch_op.drop_column('is_eligible')
