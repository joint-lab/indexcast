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

    id: str = Field(primary_key=True)
    creator_id: str
    created_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    url: str
    question: str
    description: str | None = None
    close_time: datetime | None = None
    probability: float | None = None
    volume: float | None = Field(default=0.0)
    is_resolved: bool = Field(default=False)
    resolution: str | None = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))
