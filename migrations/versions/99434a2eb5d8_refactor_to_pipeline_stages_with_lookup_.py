"""refactor to pipeline stages with lookup table

Revision ID: 99434a2eb5d8
Revises: 45360a7086da
Create Date: 2025-10-22 13:27:11.616558

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel



# revision identifiers, used by Alembic.
revision: str = '99434a2eb5d8'
down_revision: Union[str, None] = '45360a7086da'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Create pipeline stages lookup table
    op.execute("DROP TABLE IF EXISTS pipeline_stages")
    op.create_table('pipeline_stages',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('stage_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('description', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
                    sa.PrimaryKeyConstraint('id'),
                    sa.UniqueConstraint('stage_name')
                    )

    # 2. Populate the pipeline stages table
    connection = op.get_bind()
    stages = [
        (1, 'classified', 'Market has been classified'),
        (2, 'full_market', 'Full market data has been retrieved'),
        (3, 'market_data_relevances_recorded', 'Market data relevances have been recorded'),
        (4, 'index_question_relevance_scored', 'Index question relevance has been scored'),
        (5, 'rule_eligibility', 'Rule eligibility has been determined'),
    ]

    for stage_id, stage_name, description in stages:
        connection.execute(sa.text("""
            INSERT INTO pipeline_stages (id, stage_name, description)
            VALUES (:id, :stage_name, :description)
        """), {"id": stage_id, "stage_name": stage_name, "description": description})

    # 3. Create new events table with foreign key to pipeline_stages
    op.create_table('market_pipeline_events',
                    sa.Column('market_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('stage_id', sa.Integer(), nullable=False),
                    sa.Column('completed_at', sa.DateTime(), nullable=False),
                    sa.ForeignKeyConstraint(['market_id'], ['markets.id'], ),
                    sa.ForeignKeyConstraint(['stage_id'], ['pipeline_stages.id'], ),
                    sa.PrimaryKeyConstraint('market_id', 'stage_id')
                    )

    # 4. Migrate data from old table to new table
    result = connection.execute(sa.text("""
        SELECT market_id, classified_at, full_market_at, 
               market_data_relevances_recorded_at, temp_relevance_scored_at,
               geo_relevance_scored_at, index_question_relevance_scored_at,
               rule_eligibility_at
        FROM market_updates
    """))

    for row in result:
        market_id = row[0]
        stage_mappings = [
            (1, row[1]),  # CLASSIFIED
            (2, row[2]),  # FULL_MARKET
            (3, row[3]),  # MARKET_DATA_RELEVANCES_RECORDED
            (4, row[4]),  # TEMP_RELEVANCE_SCORED
            (5, row[5]),  # GEO_RELEVANCE_SCORED
            (6, row[6]),  # INDEX_QUESTION_RELEVANCE_SCORED
            (7, row[7]),  # RULE_ELIGIBILITY
        ]

        for stage_id, timestamp in stage_mappings:
            if timestamp is not None:
                connection.execute(sa.text("""
                    INSERT INTO market_pipeline_events (market_id, stage_id, completed_at)
                    VALUES (:market_id, :stage_id, :completed_at)
                """), {"market_id": market_id, "stage_id": stage_id, "completed_at": timestamp})

    # 5. Drop old table
    op.drop_table('market_updates')

    # 6. Handle market_relevance_scores changes (remove FK to non-existent table)
    with op.batch_alter_table(
            'market_relevance_scores',
            schema=None,
            recreate='always'
    ) as batch_op:
        batch_op.add_column(sa.Column('chain_of_thoughts', sqlmodel.sql.sqltypes.AutoString(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Recreate old market_updates table
    op.create_table('market_updates',
                    sa.Column('market_id', sa.VARCHAR(), nullable=False),
                    sa.Column('classified_at', sa.DATETIME(), nullable=True),
                    sa.Column('full_market_at', sa.DATETIME(), nullable=True),
                    sa.Column('market_data_relevances_recorded_at', sa.DATETIME(), nullable=True),
                    sa.Column('temp_relevance_scored_at', sa.DATETIME(), nullable=True),
                    sa.Column('geo_relevance_scored_at', sa.DATETIME(), nullable=True),
                    sa.Column('index_question_relevance_scored_at', sa.DATETIME(), nullable=True),
                    sa.Column('rule_eligibility_at', sa.DATETIME(), nullable=True),
                    sa.ForeignKeyConstraint(['market_id'], ['markets.id'], ),
                    sa.PrimaryKeyConstraint('market_id')
                    )

    # Migrate data back
    connection = op.get_bind()
    result = connection.execute(sa.text("SELECT DISTINCT market_id FROM market_pipeline_events"))

    for row in result:
        market_id = row[0]
        events = connection.execute(sa.text("""
            SELECT stage_id, completed_at 
            FROM market_pipeline_events 
            WHERE market_id = :market_id
        """), {"market_id": market_id})

        columns = {
            'classified_at': None,
            'full_market_at': None,
            'market_data_relevances_recorded_at': None,
            'temp_relevance_scored_at': None,
            'geo_relevance_scored_at': None,
            'index_question_relevance_scored_at': None,
            'rule_eligibility_at': None,
        }

        stage_to_column = {
            1: 'classified_at',
            2: 'full_market_at',
            3: 'market_data_relevances_recorded_at',
            4: 'temp_relevance_scored_at',
            5: 'geo_relevance_scored_at',
            6: 'index_question_relevance_scored_at',
            7: 'rule_eligibility_at',
        }

        for event_row in events:
            stage_id, completed_at = event_row
            if stage_id in stage_to_column:
                columns[stage_to_column[stage_id]] = completed_at

        connection.execute(sa.text("""
            INSERT INTO market_updates 
            (market_id, classified_at, full_market_at, market_data_relevances_recorded_at,
             temp_relevance_scored_at, geo_relevance_scored_at, index_question_relevance_scored_at,
             rule_eligibility_at)
            VALUES (:market_id, :classified_at, :full_market_at, :market_data_relevances_recorded_at,
                    :temp_relevance_scored_at, :geo_relevance_scored_at, 
                    :index_question_relevance_scored_at, :rule_eligibility_at)
        """), {"market_id": market_id, **columns})

    # Remove chain_of_thoughts column and drop new tables
    with op.batch_alter_table('market_relevance_scores', schema=None, recreate='always') as batch_op:
        batch_op.drop_column('chain_of_thoughts')

    op.drop_table('market_pipeline_events')
    op.drop_table('pipeline_stages')