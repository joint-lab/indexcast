"""editing dump table

Revision ID: c9cbecb6336a
Revises: abc123def456
Create Date: 2025-11-17 11:41:17.272850

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel



# revision identifiers, used by Alembic.
revision: str = 'c9cbecb6336a'
down_revision: Union[str, None] = 'abc123def456'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema (SQLite-safe)."""

    # 0. Drop leftover index from old table (SQLite keeps global indexes)
    op.execute("DROP INDEX IF EXISTS ix_labels_info_dump_type")

    # 1. Create new table with the desired schema
    op.create_table(
        "labels_info_dump_new",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("market_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("type", sa.Enum("initial", "final", name="labeltype"), nullable=False),
        sa.Column("output", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("dumped_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["market_id"], ["markets.id"]),
    )

    # 2. Recreate index on `type`
    op.create_index(
        "ix_labels_info_dump_type",
        "labels_info_dump_new",
        ["type"],
        unique=False
    )
    # 3. Drop old table
    op.drop_table("labels_info_dump")

    # 4. Rename new → old
    op.rename_table("labels_info_dump_new", "labels_info_dump")


def downgrade() -> None:
    """Downgrade schema (SQLite-safe)."""

    op.execute("DROP INDEX IF EXISTS ix_labels_info_dump_type")

    op.create_table(
        "labels_info_dump_old",
        sa.Column("market_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("type", sa.Enum("initial", "final", name="labeltype"), nullable=False),
        sa.Column("output", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(["market_id"], ["markets.id"]),
        sa.PrimaryKeyConstraint("market_id"),
    )

    op.create_index(
        "ix_labels_info_dump_type",
        "labels_info_dump_old",
        ["type"],
        unique=False,
    )

    op.drop_table("labels_info_dump")
    op.rename_table("labels_info_dump_old", "labels_info_dump")