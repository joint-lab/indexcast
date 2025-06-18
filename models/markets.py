"""
Markets database models.

Authors: 
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""
from datetime import UTC, datetime

from sqlmodel import Field, Relationship, SQLModel


# API data
class Market(SQLModel, table=True):
    """Prediction market model."""

    __tablename__ = "markets"

    # identifiers
    id: str = Field(primary_key=True)
    creator_id: str
    creator_username: str
    creator_name: str
    url: str

    # content
    question: str
    description: str | None = None

    # trading summaries
    probability: float | None = None  # multiple choice markets don't have a probability
    volume: float = Field(default=0.0)
    volume_24h: float = Field(default=0.0)
    unique_bettor_count: int = Field(default=0)
    is_resolved: bool = Field(default=False)
    resolution: str | None = None

    # financials
    total_liquidity: float = Field(default=0.0)
    outcome_type: str = Field(default=None)
    mechanism: str = Field(default=None)

    # timestamps
    created_time: datetime
    last_updated_time: datetime
    closed_time: datetime | None = None  # posts and bountied questions don't have a close time
    resolution_time: datetime | None = None

    # internal timestamps
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    # Relationships
    labels: list["MarketLabel"] = Relationship(back_populates="market")
    scores: list["MarketRelevanceScore"] = Relationship(back_populates="market")

# Market classifications
class MarketLabelType(SQLModel, table=True):
    """All possible label types for markets."""

    __tablename__ = "market_label_types"

    id: int = Field(default=None, primary_key=True)
    label_name: str = Field(unique=True)  # renamed from tag_name

    # Relationship back to markets that use this label
    market_labels: list["MarketLabel"] = Relationship(back_populates="label_type")

class MarketLabel(SQLModel, table=True):
    """Junction table linking markets to their labels."""

    __tablename__ = "market_labels"

    market_id: str = Field(primary_key=True, foreign_key="markets.id")
    label_type_id: int = Field(primary_key=True, foreign_key="market_label_types.id")

    # Relationships
    market: "Market" = Relationship(back_populates="labels")
    label_type: MarketLabelType = Relationship(back_populates="market_labels")


# Market ranking
class MarketRelevanceScoreType(SQLModel, table=True):
    """Types of market relevance scores."""

    __tablename__ = "market_relevance_score_types"

    id: int = Field(default=None, primary_key=True)
    score_name: str = Field(unique=True)

    # Relationship back to market scores
    market_scores: list["MarketRelevanceScore"] = Relationship(back_populates="score_type")

class MarketRelevanceScore(SQLModel, table=True):
    """Relevance scores assigned to markets by the ranker."""

    __tablename__ = "market_relevance_scores"

    market_id: str = Field(primary_key=True, foreign_key="markets.id")
    score_type_id: int = Field(primary_key=True, foreign_key="market_relevance_score_types.id")
    score_value: float

    # Relationships
    market: "Market" = Relationship(back_populates="scores")
    score_type: MarketRelevanceScoreType = Relationship(back_populates="market_scores")
