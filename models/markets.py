"""
Markets database models.

Authors: 
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""
from datetime import UTC, datetime

from sqlmodel import Field, Relationship, SQLModel


class MarketRuleLink(SQLModel, table=True):
    """Table linking markets and rules."""

    __tablename__ = "market_rule_links"

    market_id: str = Field(foreign_key="markets.id", primary_key=True)
    rule_id: int = Field(foreign_key="market_rules.id", primary_key=True)

    market: "Market" = Relationship()
    rule: "MarketRule" = Relationship()

class IndexRuleLink(SQLModel, table=True):
    """Table linking indicies and rules."""

    __tablename__ = "index_rule_links"

    index_id: int = Field(foreign_key="index.id", primary_key=True)
    rule_id: int = Field(foreign_key="market_rules.id", primary_key=True)

    index: "Index" = Relationship()
    rule: "MarketRule" = Relationship()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# API data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    description: str | None = None #json
    text_rep: str | None = None

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
    comments: list["MarketComment"] = Relationship(back_populates="market")
    labels: list["MarketLabel"] = Relationship(back_populates="market")
    updates: list["MarketUpdate"] = Relationship(back_populates="market")
    scores: list["MarketRelevanceScore"] = Relationship(back_populates="market")
    bets: list["MarketBet"] = Relationship(back_populates="market")
    rules: list["MarketRule"] = Relationship(
        back_populates="markets",
        link_model=MarketRuleLink
    )

class MarketComment(SQLModel, table=True):
    """Prediction market comment model."""

    __tablename__ = "market_comments"

    # identifiers
    id: str = Field(primary_key=True)
    market_id: str = Field(foreign_key="markets.id")
    user_id: str
    reply_to_comment_id: str | None = None

    # content
    comment_type: str #json
    is_api: bool
    content: str

    # other info
    visibility: str
    hidden: bool = False  # defaults to False, omitted in API response if False

    # timestamps
    created_time: datetime
    hidden_time: datetime | None = None
    edited_ime: datetime | None = None
    
    # internal timestamps
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    # Relationships
    market: "Market" = Relationship(back_populates="comments")

class MarketBet(SQLModel, table=True):
    """Prediction market bet model."""

    __tablename__ = "market_bets"

    # identifiers
    id: str = Field(primary_key=True)
    contract_id: str = Field(foreign_key="markets.id")
    user_id: str
    bet_group_id: str | None = None

    # bet info
    outcome: str
    amount: float
    order_amount: float
    loan_amount: float = 0
    shares: float
    fills: str | None = None  # fills json

    # probabilities
    prob_before: float
    prob_after: float
    limit_prob: float | None = None

    # status
    visibility: str
    is_api: bool = False
    is_filled: bool = False
    is_cancelled: bool = False
    is_redemption: bool = False

    created_time: datetime
    updated_time: datetime | None = None

    # fees
    platform_fee: float = 0
    liquidity_fee: float = 0
    creator_fee: float = 0

    # internal timestamps
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    market: "Market" = Relationship(back_populates="bets")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Market classifications
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Market ranking
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    scores: str | None = None
    chain_of_thoughts: str | None = None

    # Relationships
    market: "Market" = Relationship(back_populates="scores")
    score_type: MarketRelevanceScoreType = Relationship(back_populates="market_scores")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Market updates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MarketUpdate(SQLModel, table=True):
    """Junction table linking markets to last time they underwent parts of the pipeline."""

    __tablename__ = "market_updates"

    market_id: str = Field(primary_key=True, foreign_key="markets.id")
    classified_at: datetime | None = None
    full_market_at: datetime | None = None
    market_data_relevances_recorded_at: datetime | None = None
    temp_relevance_scored_at: datetime | None = None
    geo_relevance_scored_at: datetime | None = None
    index_question_relevance_scored_at: datetime | None = None
    rule_eligibility_at: datetime | None = None

    # Relationships
    market: "Market" = Relationship(back_populates="updates")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Market rules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class MarketRule(SQLModel, table=True):
    """Rule generated using eligible markets."""

    __tablename__ = "market_rules"

    id: int = Field(primary_key=True)
    rule: str  # JSON string representing logical conditions
    readable_rule: str  # human-readable description
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    strength_weight: float | None = None
    relevance_weight: float | None = None
    chain_of_thoughts: str
    # the scores that were averaged to get the two weights
    strength_scores: str | None = None
    relevance_scores: str | None = None
    # chain of thoughts for the weights
    strength_chain: str | None = None
    relevance_chain: str | None = None
    # batch id
    batch_id: int | None = None

    # Relationship
    markets: list["Market"] = Relationship(
        back_populates="rules",
        link_model=MarketRuleLink
    )

    indices: list["Index"] = Relationship(
        back_populates="rules", link_model=IndexRuleLink
    )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Indexcast index
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Index(SQLModel, table=True):
    """Monte Carlo simulation results for H5N1 outbreak probability."""

    __tablename__ = "index"

    id: int = Field(primary_key=True)
    index_probability: float = Field(description="Calculated probability of H5N1 outbreak")
    json_representation: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    rules: list["MarketRule"] = Relationship(
        back_populates="indices", link_model=IndexRuleLink
    )


