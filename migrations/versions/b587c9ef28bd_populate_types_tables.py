"""populate types tables

Revision ID: b587c9ef28bd
Revises: f2a895a51c71
Create Date: 2025-06-17 13:56:52.972310

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel



# revision identifiers, used by Alembic.
revision: str = 'b587c9ef28bd'
down_revision: Union[str, None] = 'f2a895a51c71'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def upgrade() -> None:
    """Insert initial label and score types."""
    # Insert into market_label_types
    op.execute("""
        INSERT INTO market_label_types (label_name) VALUES ('h5n1')
    """)
    # Insert into market_relevance_score_types
    op.execute("""
        INSERT INTO market_relevance_score_types (score_name) VALUES
        ('volume_total'),
        ('volume_24h'),
        ('volume_144h'),
        ('num_traders'),
        ('num_comments'),
        ('temporal_relevance'),
        ('geographical_relevance'),
        ('index_question_relevance')
    """)

def downgrade() -> None:
    """Remove initial label and score types."""
    op.execute("DELETE FROM market_label_types WHERE label_name = 'h5n1'")
    op.execute("""
        DELETE FROM market_relevance_score_types WHERE score_name IN (
            'volume_total',
            'volume_24h',
            'volume_144h',
            'num_traders',
            'num_comments',
            'temporal_relevance',
            'geographical_relevance',
            'index_question_relevance'
        )
    """)
