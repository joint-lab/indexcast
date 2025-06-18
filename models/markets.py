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
    url: str
    # content
    question: str
    description: str | None = None
    # trading summaries
    probability: float | None = None
    volume: float | None = Field(default=0.0)
    is_resolved: bool
    resolution: str | None = None
    # timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    closed_at: datetime | None = None
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    # Relationships
    comments: list["MarketComment"] = Relationship(back_populates="market")
    labels: list["MarketLabel"] = Relationship(back_populates="market")
    scores: list["MarketRelevanceScore"] = Relationship(back_populates="market")
    bets: list["MarketBet"] = Relationship(back_populates="market")

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

# API data 
class MarketComment(SQLModel, table=True):
    """Prediction market comment model."""

    __tablename__ = "market_comments"

    # identifiers
    id: str = Field(primary_key=True)
    market_id: str = Field(foreign_key="markets.id")

    # contract info
    contract_id: str | None = None
    contract_question: str | None = None
    contract_slug: str | None = None

    # content
    comment_type: str
    content: str | None = None

    # commentor info
    user_avatar_url: str | None = None
    user_id: str
    user_name: str
    user_username: str
    commentor_position_answer_id: str
    commentor_position_outcome: str
    commentor_position_shares: str
    commentor_position_prob: str | None = None

    # other info
    is_api: bool
    reply_to_comment_id: str | None = None
    visibility: str

    # time stamp
    created_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Relationships
    market: "Market" = Relationship(back_populates="comments")

# API data
class MarketBet(SQLModel, table=True):
    """Prediction market bet model."""

    __tablename__ = "market_bets"

    id: str = Field(primary_key=True)
    user_id: str
    contract_id: str
    outcome: str
    amount: float
    order_amount: float
    shares: float
    prob_before: float
    prob_after: float
    limit_prob: float | None = None
    is_filled: bool
    is_cancelled: bool
    created_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    loan_amount: float | None = None

    # fees
    platform_fee: float | None = None
    liquidity_fee: float | None = None
    creator_fee: float | None = None

    fills: list["MarketFill"] = Relationship(back_populates="bet")
    market: "Market" = Relationship(back_populates="bets")


# API data
class MarketFill(SQLModel, table=True):
    """Prediction market fill model."""

    __tablename__ = "market_fills"

    id: str = Field(primary_key=True)
    bet_id: str = Field(foreign_key="market_bets.id")
    matched_bet_id: str | None = None
    amount: float
    shares: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    bet: "MarketBet" = Relationship(back_populates="fills")
