"""Adding eligibility label

Revision ID: b64848614e7f
Revises: 0deb26a1d07e
Create Date: 2025-07-23 16:37:53.796830

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel



# revision identifiers, used by Alembic.
revision: str = 'b64848614e7f'
down_revision: Union[str, None] = '0deb26a1d07e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Insert eligibility label."""
    op.execute("""
        INSERT INTO market_label_types (label_name) VALUES ('rule_eligible')
    """)

def downgrade() -> None:
    """Remove eligibility label."""
    op.execute("DELETE FROM market_label_types WHERE label_name = 'rule_eligible'")
