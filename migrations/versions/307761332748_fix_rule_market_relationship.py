"""fix rule market relationship

Revision ID: 307761332748
Revises: 99434a2eb5d8
Create Date: 2025-10-22 14:15:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '307761332748'
down_revision = '99434a2eb5d8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """This migration is now a no-op as the FK already exists from 99434a2eb5d8"""
    pass


def downgrade() -> None:
    """No-op downgrade"""
    pass
