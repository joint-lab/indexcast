"""remove_market_relevance_score_types_table

Revision ID: bb941224b09f
Revises: 307761332748
Create Date: 2025-10-22 14:33:37.505035

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel



# revision identifiers, used by Alembic.
revision: str = 'bb941224b09f'
down_revision: Union[str, None] = '307761332748'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Remove FK from market_relevance_scores and drop market_relevance_score_types table."""
    # In SQLite, we can't just drop a foreign key. We need to recreate the table.
    # Use batch mode to handle this properly
    
    # Step 1: Create a temporary table with the correct schema (no FK to market_relevance_score_types)
    op.create_table(
        'market_relevance_scores_new',
        sa.Column('market_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('score_type_id', sa.Integer(), nullable=False),
        sa.Column('score_value', sa.Float(), nullable=False),
        sa.Column('scores', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('chain_of_thoughts', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.ForeignKeyConstraint(['market_id'], ['markets.id'], ),
        sa.PrimaryKeyConstraint('market_id', 'score_type_id')
    )
    
    # Step 2: Copy data from old table to new table
    connection = op.get_bind()
    connection.execute(sa.text("""
        INSERT INTO market_relevance_scores_new 
        (market_id, score_type_id, score_value, scores, chain_of_thoughts)
        SELECT market_id, score_type_id, score_value, scores, chain_of_thoughts
        FROM market_relevance_scores
    """))
    
    # Step 3: Drop old table
    op.drop_table('market_relevance_scores')
    
    # Step 4: Rename new table to original name
    op.rename_table('market_relevance_scores_new', 'market_relevance_scores')
    
    # Step 5: Drop the market_relevance_score_types table
    op.drop_table('market_relevance_score_types')


def downgrade() -> None:
    """Recreate market_relevance_score_types table and FK."""
    # Recreate market_relevance_score_types table
    op.create_table(
        'market_relevance_score_types',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('score_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('score_name')
    )
    
    # Re-populate the table with data
    connection = op.get_bind()
    score_types = [
        (1, 'volume_total'),
        (2, 'volume_24h'),
        (3, 'volume_144h'),
        (4, 'num_traders'),
        (5, 'num_comments'),
        (6, 'temporal_relevance'),
        (7, 'geographical_relevance'),
        (8, 'index_question_relevance'),
    ]
    
    for score_id, score_name in score_types:
        connection.execute(
            sa.text("""
                INSERT INTO market_relevance_score_types (id, score_name)
                VALUES (:id, :score_name)
            """),
            {"id": score_id, "score_name": score_name}
        )
    
    # Recreate market_relevance_scores with the FK
    op.create_table(
        'market_relevance_scores_new',
        sa.Column('market_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('score_type_id', sa.Integer(), nullable=False),
        sa.Column('score_value', sa.Float(), nullable=False),
        sa.Column('scores', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column('chain_of_thoughts', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.ForeignKeyConstraint(['market_id'], ['markets.id'], ),
        sa.ForeignKeyConstraint(['score_type_id'], ['market_relevance_score_types.id'], ),
        sa.PrimaryKeyConstraint('market_id', 'score_type_id')
    )
    
    # Copy data back
    connection = op.get_bind()
    connection.execute(sa.text("""
        INSERT INTO market_relevance_scores_new 
        (market_id, score_type_id, score_value, scores, chain_of_thoughts)
        SELECT market_id, score_type_id, score_value, scores, chain_of_thoughts
        FROM market_relevance_scores
    """))
    
    op.drop_table('market_relevance_scores')
    op.rename_table('market_relevance_scores_new', 'market_relevance_scores')
