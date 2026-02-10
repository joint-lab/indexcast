"""add eligibility labels

Revision ID: a7f0e32a8395
Revises: 151800aa33ee
Create Date: 2026-02-09 19:55:15.821450

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel



# revision identifiers, used by Alembic.
revision: str = 'a7f0e32a8395'
down_revision: Union[str, None] = '151800aa33ee'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Add is_eligible column to market_labels, defaulting existing rows to True."""
    with op.batch_alter_table('market_labels', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column('is_eligible', sa.Boolean(), nullable=False, server_default=sa.text('true'))
        )


def downgrade() -> None:
    """Remove is_eligible column from market_labels."""
    with op.batch_alter_table('market_labels', schema=None) as batch_op:
        batch_op.drop_column('is_eligible')
