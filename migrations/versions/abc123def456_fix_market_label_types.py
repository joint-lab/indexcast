"""fix market_label_types ordering and remove rule_eligible

Revision ID: abc123def456
Revises: b5318e39aee1
Create Date: 2025-11-17 11:06:59
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "abc123def456"
down_revision: Union[str, None] = "b5318e39aee1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Clear out existing rows
    op.execute("DELETE FROM market_label_types;")

    # Insert the desired rows fresh
    op.execute("INSERT INTO market_label_types (id, label_name) VALUES (1, 'h5n1');")
    op.execute("INSERT INTO market_label_types (id, label_name) VALUES (2, 'ai_frontier_milestone_12mo');")
    op.execute("INSERT INTO market_label_types (id, label_name) VALUES (3, 'annual_war_deaths_exceed_average');")
    op.execute("INSERT INTO market_label_types (id, label_name) VALUES (4, 'next_national_election_democratic_party');")


def downgrade():
    # Clear out current rows
    op.execute("DELETE FROM market_label_types;")

    # Restore the old state
    op.execute("INSERT INTO market_label_types (id, label_name) VALUES (1, 'h5n1');")
    op.execute("INSERT INTO market_label_types (id, label_name) VALUES (2, 'rule_eligible');")
    op.execute("INSERT INTO market_label_types (id, label_name) VALUES (3, 'next_national_election_democratic_party');")
    op.execute("INSERT INTO market_label_types (id, label_name) VALUES (4, 'annual_war_deaths_exceed_average');")
    op.execute("INSERT INTO market_label_types (id, label_name) VALUES (5, 'ai_frontier_milestone_12mo');")
