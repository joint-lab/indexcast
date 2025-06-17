"""
Markets database models.

Authors: 
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""
from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class Market(SQLModel, table=True):
    """Prediction market model."""

    __tablename__ = "markets"

    # identifiers
    id: str = Field(primary_key=True)
    creator_id: str
    url: str
    # content
    question: str
    description: str | None = None
    # trading summaries
    probability: float | None = None
    volume: float | None = Field(default=0.0)
    is_resolved: bool = Field(default=False)
    resolution: str | None = None
    # timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    closed_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
