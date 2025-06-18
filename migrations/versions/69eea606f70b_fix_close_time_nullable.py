"""fix close time nullable

Revision ID: 69eea606f70b
Revises: 2e1b70af07e3
Create Date: 2025-06-18 16:34:38.502540

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel



# revision identifiers, used by Alembic.
revision: str = '69eea606f70b'
down_revision: Union[str, None] = '2e1b70af07e3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('markets', schema=None) as batch_op:
        batch_op.alter_column('closed_time', existing_type=sa.DATETIME(), nullable=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('markets', schema=None) as batch_op:
        batch_op.alter_column('closed_time', existing_type=sa.DATETIME(), nullable=False)
    # ### end Alembic commands ###
