"""Make label_id part of composite key for relevance scores table

Revision ID: 151800aa33ee
Revises: 71f4f82e4d42
Create Date: 2025-12-12 09:55:50.616708

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel



# revision identifiers, used by Alembic.
revision: str = '151800aa33ee'
down_revision: Union[str, None] = '71f4f82e4d42'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table(
        "market_relevance_scores",
        recreate="always"
    ) as batch_op:

        # Make label_id NOT NULL
        batch_op.alter_column(
            "label_id",
            existing_type=sa.Integer(),
            nullable=False
        )

        # Recreate FK
        batch_op.create_foreign_key(
            "fk_market_relevance_scores_label_id",
            "market_label_types",
            ["label_id"],
            ["id"]
        )

        # Define the new composite PK
        batch_op.create_primary_key(
            "pk_market_relevance_scores",
            ["market_id", "label_id", "score_type_id"]
        )


def downgrade() -> None:
    with op.batch_alter_table(
        "market_relevance_scores",
        recreate="always"
    ) as batch_op:

        batch_op.drop_constraint(
            "fk_market_relevance_scores_label_id",
            type_="foreignkey"
        )

        batch_op.alter_column(
            "label_id",
            existing_type=sa.Integer(),
            nullable=True
        )

        batch_op.create_primary_key(
            "pk_market_relevance_scores",
            ["market_id", "score_type_id"]
        )
