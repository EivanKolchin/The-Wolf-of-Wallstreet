"""add highest_price_seen to trades

Revision ID: 29a1b2c3d4e5
Revises: 
Create Date: 2026-04-18 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '29a1b2c3d4e5'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add the new column to the existing 'trades' table
    op.add_column('trades', sa.Column('highest_price_seen', sa.Float(), nullable=True))


def downgrade() -> None:
    # Remove the column from the 'trades' table
    op.drop_column('trades', 'highest_price_seen')
