"""
Market assets.

This module contains Dagster assets for fetching, processing, and analyzing
prediction market data from the Manifold API. It handles market ingestion,
labeling, scoring, rule generation, and probability calculations.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""
import json
import re
import time
from datetime import UTC, datetime, timedelta

import dagster as dg
import numpy as np
import pandas as pd
import requests
from pydantic import TypeAdapter
from scipy.stats import norm
from sqlalchemy import delete, func
from sqlmodel import Session, select

from ml.clients import get_client
from ml.dspy_market_scorer import DSPyMarketScorer
from ml.lm_initial_labeler import get_initial_label, get_initial_labeling_prompt
from ml.ranker import IndexInformation, get_prompt, get_relevance
from ml.rules import (
    Formula,
    OperatorNode,
    PromptInformation,
    VariableNode,
    extract_literals_from_formula,
    get_rules,
    get_rules_prompt,
    get_weight,
    stringify_formula,
)
from models.markets import (
    Index,
    IndexQuestion,
    IndexRuleLink,
    LabelInfo,
    LabelType,
    Market,
    MarketBet,
    MarketComment,
    MarketLabel,
    MarketLabelType,
    MarketPipelineEvent,
    MarketRelevanceScore,
    MarketRelevanceScoreType,
    MarketRule,
    MarketRuleLink,
    PipelineStageType,
)
from pipelines.resources.db import locked_session

# =============================================================================
# Helper Functions - Timestamp Conversion
# =============================================================================


def _safe_fromtimestamp(ms: int) -> datetime | None:
    """
    Safely convert milliseconds since epoch to datetime.

    Handles edge cases where timestamps are out of the valid datetime range
    or cause overflow errors.

    Args:
        ms: Milliseconds since Unix epoch.

    Returns:
        A timezone-aware datetime object, or None if conversion fails.

    """
    try:
        dt = datetime.fromtimestamp(ms / 1000, UTC)
        if 1 <= dt.year <= 9999:
            return dt
        return None
    except (OverflowError, OSError, ValueError):
        return None


# =============================================================================
# Helper Functions - Data Preparation
# =============================================================================


def _prepare_market(market_data: dict) -> Market:
    """
    Prepare a Market object from raw Manifold API data.

    Converts raw JSON response from the Manifold API into a Market SQLModel
    object suitable for database insertion.

    Args:
        market_data: Raw dictionary from Manifold API response.

    Returns:
        A Market object ready for database insertion.

    """
    # Convert timestamps using safe conversion
    created_time = _safe_fromtimestamp(market_data["createdTime"])
    closed_time = (
        _safe_fromtimestamp(market_data["closeTime"])
        if market_data.get("closeTime") else None
    )
    resolution_time = (
        _safe_fromtimestamp(market_data["resolutionTime"])
        if market_data.get("resolutionTime") else None
    )
    last_updated_timed = (
        _safe_fromtimestamp(market_data["lastUpdatedTime"])
        if market_data.get("lastUpdatedTime") else created_time
    )

    return Market(
        id=market_data["id"],
        creator_id=market_data.get("creatorId", ""),
        creator_username=market_data.get("creatorUsername", ""),
        creator_name=market_data.get("creatorName", ""),
        url=market_data["url"],
        question=market_data["question"],
        description=market_data.get("description"),
        probability=market_data.get("probability"),
        volume=market_data.get("volume", 0.0),
        volume_24h=market_data.get("volume24Hours", 0.0),
        unique_bettor_count=market_data.get("uniqueBettorCount", 0),
        is_resolved=market_data.get("isResolved", False),
        resolution=market_data.get("resolution"),
        total_liquidity=market_data.get("totalLiquidity", 0.0),
        outcome_type=market_data.get("outcomeType"),
        mechanism=market_data.get("mechanism"),
        created_time=created_time,
        last_updated_time=last_updated_timed,
        closed_time=closed_time,
        resolution_time=resolution_time,
        updated_at=datetime.now(UTC),
    )


def _prepare_bet(bet_data: dict) -> MarketBet:
    """
    Prepare a MarketBet object from raw Manifold API data.

    Converts raw JSON response from the Manifold API into a MarketBet SQLModel
    object suitable for database insertion.

    Args:
        bet_data: Raw dictionary from Manifold API response.

    Returns:
        A MarketBet object ready for database insertion.

    """
    created_time = _safe_fromtimestamp(bet_data["createdTime"])
    updated_time = (
        _safe_fromtimestamp(bet_data["updatedTime"])
        if bet_data.get("updatedTime") else None
    )

    return MarketBet(
        # Identifiers
        id=bet_data["id"],
        contract_id=bet_data["contractId"],
        user_id=bet_data["userId"],
        bet_group_id=bet_data.get("betGroupId"),

        # Bet info
        outcome=bet_data["outcome"],
        amount=bet_data["amount"],
        order_amount=bet_data.get("orderAmount", 0.0),
        loan_amount=bet_data.get("loanAmount", 0.0),
        shares=bet_data["shares"],
        fills=json.dumps(bet_data.get("fills")),

        # Probabilities
        prob_before=bet_data["probBefore"],
        prob_after=bet_data["probAfter"],
        limit_prob=bet_data.get("limitProb"),

        # Status flags
        visibility=bet_data.get("visibility", ""),
        is_api=bet_data.get("isApi", False),
        is_filled=bet_data.get("isFilled", False),
        is_cancelled=bet_data.get("isCancelled", False),
        is_redemption=bet_data.get("isRedemption", False),

        # Timestamps
        created_time=created_time,
        updated_time=updated_time,

        # Fee breakdown
        platform_fee=bet_data.get("fees", {}).get("platformFee", 0.0),
        liquidity_fee=bet_data.get("fees", {}).get("liquidityFee", 0.0),
        creator_fee=bet_data.get("fees", {}).get("creatorFee", 0.0),

        # Internal tracking
        updated_at=datetime.now(UTC),
    )


def _prepare_comment(comment_data: dict) -> MarketComment:
    """
    Prepare a MarketComment object from raw Manifold API data.

    Converts raw JSON response from the Manifold API into a MarketComment
    SQLModel object suitable for database insertion.

    Args:
        comment_data: Raw dictionary from Manifold API response.

    Returns:
        A MarketComment object ready for database insertion.

    """
    created_time = _safe_fromtimestamp(comment_data["createdTime"])
    hidden_time = (
        _safe_fromtimestamp(comment_data["hiddenTime"])
        if comment_data.get("hiddenTime") else None
    )
    edited_time = (
        _safe_fromtimestamp(comment_data["editedTime"])
        if comment_data.get("editedTime") else None
    )

    return MarketComment(
        # Identifiers
        id=comment_data["id"],
        market_id=comment_data.get("contractId"),
        user_id=comment_data["userId"],
        reply_to_comment_id=comment_data.get("replyToCommentId"),

        # Content
        comment_type=comment_data["commentType"],
        is_api=comment_data.get("isApi", False),
        content=json.dumps(
            comment_data.get("content")
            or ""
        ),

        # Visibility
        visibility=comment_data["visibility"],
        hidden=comment_data.get("hidden", False),

        # Timestamps
        created_time=created_time,
        hidden_time=hidden_time,
        edited_time=edited_time,

        # Internal tracking
        updated_at=datetime.now(UTC),
    )


# =============================================================================
# Helper Functions - Market Metrics
# =============================================================================


def get_volume(session: Session, market_id: str, query_time: datetime) -> float:
    """
    Retrieve the trading volume for a market since a given time.

    Calculates the sum of absolute bet amounts for a market within the
    specified time window.

    Args:
        session: SQLAlchemy session used to query the database.
        market_id: Unique identifier for the market to query.
        query_time: Start timestamp for the query window.

    Returns:
        The total trading volume since query_time, or 0.0 if no bets found.

    """
    now = datetime.now(UTC)
    result = session.exec(
        select(func.sum(func.abs(MarketBet.amount)))
        .where(MarketBet.contract_id == market_id)
        .where(MarketBet.created_time <= now)
        .where(MarketBet.created_time >= query_time)
    ).one_or_none()
    return result or 0.0


# =============================================================================
# Helper Functions - Text Processing
# =============================================================================


def fetch_text_from_url(
    url: str,
    retries: int = 3,
    timeout: tuple = (5, 10),
    backoff: float = 2
) -> str:
    """
    Fetch LLM-friendly text from a URL using Jina AI's reader service.

    Uses r.jina.ai to convert web pages into clean, readable text suitable
    for language model processing. Implements exponential backoff for retries.

    Args:
        url: Target webpage URL (e.g., "https://example.com").
        retries: Number of retry attempts on timeout.
        timeout: Tuple of (connect_timeout, read_timeout) in seconds.
        backoff: Base seconds to wait between retries (exponential).

    Returns:
        The extracted text content, or an error message string on failure.

    """
    endpoint = f"https://r.jina.ai/{url}"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(endpoint, timeout=timeout)
            resp.raise_for_status()
            return resp.text

        except requests.exceptions.ReadTimeout:
            if attempt < retries:
                time.sleep(backoff * (2 ** (attempt - 1)))
            else:
                return "Error: read timeout after retries"
        except requests.exceptions.HTTPError:
            return f"HTTP error {resp.status_code}"
        except Exception as e:
            return f"Error: {e}"

    return "Error: failed after multiple retries"


def extract_urls(text: str) -> list[str]:
    """
    Extract all HTTP/HTTPS URLs from a text string.

    Uses regex pattern matching to find URL strings in the given text.

    Args:
        text: Input text to search for URLs.

    Returns:
        A list of URL strings found in the text.

    """
    url_pattern = r'(https?://[^\s]+)'
    return re.findall(url_pattern, text)


def get_text_rep(market: Market) -> str:
    """
    Create a text representation of a market for LLM processing.

    Builds a structured text representation including the market title,
    description, and optionally fetched content from URLs in the description.

    Args:
        market: Market object to create text representation for.

    Returns:
        A formatted string containing the market's text representation.

    """
    title = market.question
    # Limit description length to avoid token overflow
    description = market.description[:4875]

    # Fetch content from URLs in the description for additional context
    urls = extract_urls(market.description)
    url_text = ""
    if len(urls) > 0:
        for url in urls:
            url_text += fetch_text_from_url(url)

    # Build text representation with or without URL content
    if len(url_text) > 0 and "error" not in url_text[:20].lower():
        trimmed = url_text[:4875]
        text_rep = f"""<Title>{title}</Title>
        <Description>{description}</Description> 
        <Url Text>{trimmed}</Url Text> """
    else:
        text_rep = f"""<Title>{title}</Title>
        <Description>{description}</Description>"""
    return text_rep


# =============================================================================
# Dagster Assets - Market Ingestion
# =============================================================================


@dg.asset(
    required_resource_keys={"database_engine", "manifold_client"},
    description="Manifold markets table.",
)
def manifold_markets(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Fetch latest markets from the Manifold API and populate the Markets table.

    This asset fetches all markets from the Manifold API, checks if they already exist
    in the database, and inserts new markets. It also respects the Manifold API rate limits
    and handles pagination. Stops as soon as it encounters markets that already exists.
    """
    new_markets = []
    before = None
    total_fetched = 0
    batch_num = 0

    # Get Manifold API client from resources
    manifold_client = context.resources.manifold_client

    # Fetch markets in batches until we hit existing markets
    with Session(context.resources.database_engine) as session:
        stop_fetching = False
        while not stop_fetching:
            batch_num += 1
            batch = manifold_client.markets(limit=1000, before=before)
            batch_size = len(batch) if batch else 0
            total_fetched += batch_size
            context.log.debug(
                f"Fetched batch {batch_num}: {batch_size} markets "
                f"(total fetched: {total_fetched})"
            )

            if not batch:
                break

            for i, m in enumerate(batch):
                # Stop if we encounter an existing market
                if session.get(Market, m["id"]):
                    context.log.debug(
                        f"Market {m['id']} already exists in DB. Stopping fetch."
                    )
                    stop_fetching = True
                    break

                market = _prepare_market(m)
                session.add(market)
                new_markets.append(market)

                if (i + 1) % 100 == 0:
                    context.log.debug(
                        f"Processed {i + 1} markets in current batch."
                    )

            session.commit()
            context.log.debug(
                f"Committed batch {batch_num} to DB. "
                f"New markets so far: {len(new_markets)}."
            )

            if stop_fetching or len(batch) < 1000:
                break
            before = batch[-1]["id"]

    context.log.info(
        f"Inserted {len(new_markets)} new markets from Manifold "
        f"(fetched {total_fetched} total)"
    )

    return dg.MaterializeResult(
        metadata={
            "num_markets_inserted": dg.MetadataValue.int(len(new_markets)),
        }
    )


# =============================================================================
# Dagster Assets - Market Classification
# =============================================================================


@dg.asset(
    deps=[manifold_markets],
    required_resource_keys={"database_engine"},
    description="Market labels table."
)
def market_labels(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Classify markets and assign relevance labels using LLM-based scoring.

    This asset processes unclassified markets through a two-stage labeling pipeline:
    1. Initial labeling using a prompt-based classifier to identify potentially relevant markets
    2. DSPy-based scoring to determine final relevance for each index question

    Markets are labeled with index question IDs if they pass both stages, and
    relevance scores are stored for further processing.
    """
    initial_prompt = get_initial_labeling_prompt()
    dspy_scorer = DSPyMarketScorer()
    now = datetime.now(UTC)
    client = get_client()

    with Session(context.resources.database_engine) as session:
        # Find markets needing classification
        subquery = select(MarketPipelineEvent.market_id).where(
            MarketPipelineEvent.stage_id == PipelineStageType.CLASSIFIED
        )
        markets_to_process = session.exec(
            select(Market).where(Market.id.not_in(subquery))
        ).all()
        context.log.info(
            f"Found {len(markets_to_process)} markets needing classification."
        )

        results = []

        # Load index question mapping
        index_questions = session.exec(
            select(IndexQuestion)
        ).all()

        index_question_map = {
            iq.id: iq.question for iq in index_questions
        }
        context.log.info(
            f"Loaded {len(index_question_map)} index questions from DB"
        )

        # Process each market
        for market in markets_to_process:
            context.log.debug(f"Processing market {market.id}")

            # Skip non-binary or resolved markets
            if market.outcome_type != "BINARY" or market.resolution is not None:
                context.log.info(
                    f"Skipping market {market.id}: invalid type/resolved."
                )
                continue

            # Initial labeling
            initial_response = get_initial_label(
                prompt=initial_prompt,
                market_question=market.question,
                client=client,
            )

            # Store initial label info for debugging/auditing
            session.add(LabelInfo(
                market_id=market.id,
                type=LabelType.initial,
                output=json.dumps({
                    "labels": initial_response.labels,
                    "reasoning": initial_response.reasoning
                }),
            ))
            labels = initial_response.labels
            context.log.info(f"Initial labels for {market.id}: {labels}")

            # Skip DSPy if none-of-the-above
            if labels == [-1]:
                results.append((market.id, [], []))
                continue

            market_label_rows = []

            # DSPy scoring for each relevant label
            for label_id in labels:
                if label_id not in index_question_map:
                    context.log.warning(
                        f"Unknown label {label_id} for market {market.id}"
                    )
                    continue

                index_question = index_question_map[label_id]
                dspy_out = dspy_scorer.predict(index_question, market.question)

                # Store DSPy output for auditing
                session.add(LabelInfo(
                    market_id=market.id,
                    type=LabelType.final,
                    output=json.dumps({
                        "rationale": dspy_out["rationale"],
                        "label": dspy_out["label"],
                        "score": dspy_out["score"]
                    }),
                ))

                is_relevant = dspy_out["label"].strip() == "1"

                if is_relevant:
                    # Add market label
                    session.add(
                        MarketLabel(
                            market_id=market.id,
                            label_type_id=label_id,
                        )
                    )
                    # Add relevance score
                    session.add(
                        MarketRelevanceScore(
                            market_id=market.id,
                            score_type_id=MarketRelevanceScoreType.INDEX_QUESTION_RELEVANCE,
                            score_value=dspy_out["score"],
                            chain_of_thoughts=dspy_out["rationale"],
                            label_id=label_id,
                        )
                    )
                    market_label_rows.append(label_id)

            results.append((market.id, market_label_rows))

        # Mark all processed markets as classified
        for market in markets_to_process:
            session.add(MarketPipelineEvent(
                market_id=market.id,
                stage_id=PipelineStageType.CLASSIFIED,
                completed_at=now
            ))

        session.commit()

    return dg.MaterializeResult(
        metadata={
            "num_markets_processed": dg.MetadataValue.int(len(markets_to_process)),
            "num_markets_labeled": dg.MetadataValue.int(
                sum(len(r[1]) for r in results)
            ),
        }
    )


# =============================================================================
# Dagster Assets - Full Market Data
# =============================================================================


@dg.asset(
    deps=[market_labels],
    required_resource_keys={"database_engine", "manifold_client"},
    description="Full Market table (bets, comments, description).",
)
def manifold_full_markets(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Fetch complete market data including bets, comments, and descriptions.

    This asset enriches labeled markets with full data from the Manifold API:
    - Fetches and stores all bets with pagination handling
    - Replaces existing comments with fresh data
    - Updates market descriptions and generates text representations

    The asset respects API rate limits and handles pagination for large datasets.
    Bets are upserted (unfilled bets replaced, new bets inserted) while comments
    are fully replaced on each run.
    """
    manifold_client = context.resources.manifold_client

    # Get all labeled market IDs
    with Session(context.resources.database_engine) as session:
        all_market_ids = session.exec(
            select(MarketLabel.market_id)
            .join(MarketLabelType, MarketLabel.label_type_id == MarketLabelType.id)
        ).all()
        unique_market_ids = set(all_market_ids)

    context.log.info(f"Found {len(unique_market_ids)} labeled markets.")

    total_comments_inserted = 0
    total_bets_updated = 0

    with Session(context.resources.database_engine) as session:
        # Pre-fetch existing bets for efficiency
        existing_bets_rows = session.exec(
            select(MarketBet.id, MarketBet.contract_id, MarketBet.is_filled)
            .where(MarketBet.contract_id.in_(unique_market_ids))
        ).all()

        # Bucket bets by market ID for quick lookup
        bets_by_market = {}
        for bid, mid, filled in existing_bets_rows:
            bets_by_market.setdefault(mid, []).append((bid, filled))

        # Process each market
        for m in unique_market_ids:
            # Fetch full market data from API
            full_market = manifold_client.full_market(m)
            description_json = json.dumps(full_market.get("description"))

            temp_market = Market(id=m, description=description_json)
            text_rep = get_text_rep(temp_market)

            # Update market with description and text representation
            market = session.exec(
                select(Market).where(Market.id == m)
            ).first()
            if market:
                market.description = description_json
                market.text_rep = text_rep

            # Fetch and replace comments
            all_comments = []
            page = 0

            while True:
                batch = manifold_client.comments(m, limit=1000, page=page)
                if not batch:
                    break
                all_comments.extend(batch)
                if len(batch) < 1000:
                    break
                page += 1

            # Delete existing comments and insert fresh data
            session.exec(
                delete(MarketComment).where(MarketComment.market_id == m)
            )
            objs = [_prepare_comment(c) for c in all_comments]
            session.bulk_save_objects(objs)

            total_comments_inserted += len(all_comments)
            context.log.info(f"Market {m}: found {len(all_comments)} comments.")

            # Fetch bets with pagination
            all_bets = []
            before = None

            while True:
                batch = manifold_client.bets(m, limit=1000, before=before)
                if not batch:
                    break
                all_bets.extend(batch)
                if len(batch) < 1000:
                    break
                before = batch[-1]["id"]

            # Get existing bet IDs for this market
            existing = bets_by_market.get(m, [])
            existing_ids = {bid for (bid, filled) in existing}
            unfilled_ids = [bid for (bid, filled) in existing if not filled]

            # Delete unfilled bets (they may have been updated)
            if unfilled_ids:
                session.exec(
                    delete(MarketBet).where(MarketBet.id.in_(unfilled_ids))
                )

            # Insert only new bets
            to_insert = [
                _prepare_bet(b)
                for b in all_bets
                if b["id"] not in existing_ids
            ]
            session.bulk_save_objects(to_insert)
            total_bets_updated += len(all_bets)

            # Track pipeline event
            existing_event = session.exec(
                select(MarketPipelineEvent).where(
                    (MarketPipelineEvent.market_id == m) &
                    (MarketPipelineEvent.stage_id == PipelineStageType.FULL_MARKET)
                )
            ).first()

            if not existing_event:
                session.add(
                    MarketPipelineEvent(
                        market_id=m,
                        stage_id=PipelineStageType.FULL_MARKET,
                        completed_at=datetime.now(UTC)
                    )
                )

        session.commit()

    return dg.MaterializeResult(
        metadata={
            "num_markets_processed": dg.MetadataValue.int(len(unique_market_ids)),
            "total_comments_inserted": dg.MetadataValue.int(total_comments_inserted),
            "total_bets_updated": dg.MetadataValue.int(total_bets_updated),
        }
    )


# =============================================================================
# Dagster Assets - Relevance Scoring
# =============================================================================


@dg.asset(
    deps=[manifold_full_markets],
    required_resource_keys={"database_engine"},
    description="Relevance summary scores for the labeled markets.",
)
def relevance_summary_statistics(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Calculate and store relevance summary statistics for labeled markets.

    This asset computes market activity metrics that indicate relevance and
    reliability:
    - volume_total: Total trading volume
    - volume_24h: Trading volume in the last 24 hours
    - volume_144h: Trading volume in the last 144 hours (6 days)
    - num_traders: Number of unique traders
    - num_comments: Number of comments on the market

    These scores are stored per market-label pair and used for eligibility
    filtering in downstream assets.
    """
    # Get all labeled market-label pairs
    with Session(context.resources.database_engine) as session:
        labeled_markets = session.exec(
            select(MarketLabel.market_id, MarketLabel.label_type_id)
        ).all()

    context.log.info(f"Found {len(labeled_markets)} labeled market/label pairs.")

    # Build score type name-to-ID mapping
    score_type_map = {s.relevance_score: s.value for s in MarketRelevanceScoreType}
    context.log.info(f"Found {len(score_type_map)} relevance label types.")

    with locked_session(context.resources.database_engine) as session:
        for market_id, label_id in labeled_markets:
            # Define which score types to replace
            types_to_delete = [
                "volume_24h", "num_comments", "volume_total",
                "volume_144h", "num_traders"
            ]
            type_ids = [
                score_type_map[name]
                for name in types_to_delete
                if name in score_type_map
            ]

            existing_event = session.exec(
                select(MarketPipelineEvent).where(
                    (MarketPipelineEvent.market_id == market_id) &
                    (MarketPipelineEvent.stage_id ==
                     PipelineStageType.MARKET_DATA_RELEVANCES_RECORDED)
                )
            ).first()

            if not existing_event:
                session.add(
                    MarketPipelineEvent(
                        market_id=market_id,
                        stage_id=PipelineStageType.MARKET_DATA_RELEVANCES_RECORDED,
                        completed_at=datetime.now(UTC)
                    )
                )

            # Delete existing scores of these types for this market-label pair
            session.exec(
                delete(MarketRelevanceScore)
                .where(MarketRelevanceScore.market_id == market_id)
                .where(MarketRelevanceScore.label_id == label_id)
                .where(MarketRelevanceScore.score_type_id.in_(type_ids))
            )

            # Calculate metrics
            volume_total = get_volume(session, market_id, datetime.min)
            volume_24h = get_volume(
                session, market_id, datetime.now(UTC) - timedelta(hours=24)
            )
            volume_144h = get_volume(
                session, market_id, datetime.now(UTC) - timedelta(hours=144)
            )

            num_traders = session.exec(
                select(func.count(func.distinct(MarketBet.user_id)))
                .where(MarketBet.contract_id == market_id)
            ).one_or_none() or 0

            num_comments = session.exec(
                select(func.count())
                .where(MarketComment.market_id == market_id)
            ).one_or_none() or 0

            # Insert new scores
            to_insert = []
            for name, val in [
                ("volume_total", volume_total),
                ("volume_24h", volume_24h),
                ("volume_144h", volume_144h),
                ("num_traders", num_traders),
                ("num_comments", num_comments)
            ]:
                type_id = score_type_map.get(name)
                if type_id is None:
                    raise ValueError(f"Unknown score type: {name}")

                to_insert.append(
                    MarketRelevanceScore(
                        market_id=market_id,
                        label_id=label_id,
                        score_type_id=type_id,
                        score_value=val,
                    )
                )

            session.add_all(to_insert)
            context.log.info(f"Scored market: {market_id}.")
            session.commit()

        return dg.MaterializeResult(
            metadata={
                "num_label_pairs_processed": dg.MetadataValue.int(len(labeled_markets))
            }
        )


# =============================================================================
# Dagster Assets - Eligibility Filtering
# =============================================================================


@dg.asset(
    deps=[relevance_summary_statistics],
    required_resource_keys={"database_engine"},
    description="Update eligibility status of index labels per market based on metrics."
)
def market_rule_eligibility_labels(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Filter market labels based on eligibility criteria.

    This asset updates the is_eligible flag on market-label associations based
    on minimum quality thresholds:
    - Must be a BINARY outcome type
    - Must have volume >= 200
    - Must have >= 11 unique traders
    - Must have question relevance score >= 0.6

    Labels that don't meet these criteria are marked as ineligible. Labels that
    were previously ineligible but now meet criteria are restored to eligible.
    """
    # Index labels to check for eligibility
    index_labels = [
        "h5n1",
        "next_national_election_democratic_party",
        "annual_war_deaths_exceed_average",
        "ai_frontier_milestone_12mo",
    ]

    with Session(context.resources.database_engine) as session:
        # Fetch index label types
        label_types = {
            label.label_name: label
            for label in session.exec(
                select(MarketLabelType).where(
                    MarketLabelType.label_name.in_(index_labels)
                )
            ).all()
        }
        index_label_type_ids = [lt.id for lt in label_types.values()]

        # Fetch all market labels for these index types
        market_labels = session.exec(
            select(MarketLabel).where(
                MarketLabel.label_type_id.in_(index_label_type_ids)
            )
        ).all()

    market_ids = list(set(label.market_id for label in market_labels))

    # Score type IDs for eligibility checks
    question_score_type_id = MarketRelevanceScoreType.INDEX_QUESTION_RELEVANCE.value
    volume_score_type_id = MarketRelevanceScoreType.VOLUME_TOTAL.value
    traders_score_type_id = MarketRelevanceScoreType.NUM_TRADERS.value
    score_type_ids_set = {
        question_score_type_id, volume_score_type_id, traders_score_type_id
    }

    # Fetch all relevant scores
    with Session(context.resources.database_engine) as session:
        all_scores = session.exec(
            select(MarketRelevanceScore)
            .where(
                (MarketRelevanceScore.market_id.in_(market_ids)) &
                (MarketRelevanceScore.score_type_id.in_(score_type_ids_set))
            )
        ).all()

    # Build score lookup for efficient access
    score_lookup = {
        (score.market_id, score.score_type_id): score.score_value
        for score in all_scores
    }

    # Eligibility thresholds
    volume_threshold = 200
    traders_threshold = 11
    num_labels_marked_ineligible = 0
    num_labels_restored = 0

    with Session(context.resources.database_engine) as session:
        for label in market_labels:
            m_id = label.market_id
            question_score = score_lookup.get((m_id, question_score_type_id))
            volume_score = score_lookup.get((m_id, volume_score_type_id))
            trades_score = score_lookup.get((m_id, traders_score_type_id))

            # Fetch market outcome type
            market = session.exec(
                select(Market).where(Market.id == m_id)
            ).first()

            # Check eligibility criteria
            is_eligible = True
            if market.outcome_type != "BINARY":
                is_eligible = False
            elif volume_score is None or volume_score < volume_threshold:
                is_eligible = False
            elif trades_score is None or trades_score < traders_threshold:
                is_eligible = False
            elif question_score is None or question_score < 0.6:
                is_eligible = False

            # Merge detached label into current session and update eligibility
            attached_label = session.merge(label)
            if not is_eligible and attached_label.is_eligible:
                attached_label.is_eligible = False
                num_labels_marked_ineligible += 1
            elif is_eligible and not attached_label.is_eligible:
                attached_label.is_eligible = True
                num_labels_restored += 1

        session.commit()

    return dg.MaterializeResult(
        metadata={
            "num_labels_checked": dg.MetadataValue.int(len(market_labels)),
            "num_labels_marked_ineligible": dg.MetadataValue.int(
                num_labels_marked_ineligible
            ),
            "num_labels_restored": dg.MetadataValue.int(num_labels_restored),
        }
    )


# =============================================================================
# Dagster Assets - Rule Generation
# =============================================================================


@dg.asset(
    deps=[market_rule_eligibility_labels],
    required_resource_keys={"database_engine"},
    description="Generate 30 logical rules per index using LLM."
)
def index_rules(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Generate logical rules for each index question using LLM.

    This asset creates Boolean logical rules that combine multiple prediction
    markets to infer the probability of index events. For each index question:
    1. Fetches all eligible markets labeled for that index
    2. Generates 30 logical rules using an LLM
    3. Scores each rule for strength and relevance
    4. Stores rules and links to constituent markets

    Rules are stored with chain-of-thought reasoning and scored using separate
    strength and relevance prompts.
    """
    with Session(context.resources.database_engine) as session:
        # Load all index questions
        index_questions = session.exec(select(IndexQuestion)).all()

        if not index_questions:
            context.log.warning("No index questions found.")
            return dg.MaterializeResult()

        client = get_client()
        total_rules = 0
        total_markets_used = 0
        batch_number = 0

        # Process each index question
        for index_q in index_questions:
            index_id = index_q.id
            context.log.info(f"Processing index: {index_q.question}")

            # Determine batch number for this index
            previous_batch = session.exec(
                select(func.max(MarketRule.batch_id))
                .where(MarketRule.index_id == index_id)
            ).one_or_none()
            batch_number = (previous_batch or 0) + 1
            context.log.info(
                f"Assigned batch number {batch_number} for index_id={index_id}"
            )

            # Get all eligible markets for this index
            eligible_markets = session.exec(
                select(Market)
                .join(MarketLabel, Market.id == MarketLabel.market_id)
                .where(MarketLabel.label_type_id == index_id)
                .where(MarketLabel.is_eligible.is_(True))
            ).all()

            if not eligible_markets:
                context.log.info(
                    f"No markets labeled for index {index_q.question}"
                )
                continue

            context.log.info(f"Found {len(eligible_markets)} markets for index.")

            # Build market data for prompt
            market_data = {
                m.id: {
                    "question": m.question,
                    "description": m.description or "",
                    "text_rep": m.text_rep,
                }
                for m in eligible_markets
            }

            prompt_data = PromptInformation(
                date=datetime.now(UTC),
                overall_index_question=index_q.question,
                max_num_of_rules=30,
            )
            prompt = get_rules_prompt("rule_gen_prompt.j2", prompt_data, market_data)
            valid_market_ids = set(market_data.keys())

            # Generate rules using LLM
            logical_rules = get_rules(
                prompt=prompt,
                max_num_rules=30,
                client=client,
                allowed_market_ids=valid_market_ids,
            )
            context.log.info(f"Generated {len(logical_rules)} rules.")

            used_markets = set()

            # Score and store each rule
            for rule_obj in logical_rules:
                rule_json = rule_obj.rule.model_dump_json()
                readable = stringify_formula(rule_obj.rule)

                index_info = IndexInformation(
                    todays_date=datetime.now(UTC),
                    overall_index_question=index_q.question,
                )

                # Calculate strength score
                s_prompt = get_prompt("rule_strength_score_prompt.j2", index_info)
                s_reasonings, s_scores, s_avg = get_weight(
                    s_prompt, readable, client
                )

                # Get markets in this rule and their metrics
                rule_markets = extract_literals_from_formula(rule_obj.rule)
                market_metrics = {}

                for mid in rule_markets:
                    used_markets.add(mid)
                    scores = session.exec(
                        select(MarketRelevanceScore)
                        .where(MarketRelevanceScore.market_id == mid)
                    ).all()
                    market_metrics[mid] = {
                        sc.score_type.relevance_score: sc.score_value
                        for sc in scores
                    }

                # Build metrics text for relevance scoring
                metrics_text = f"Rule: {readable}\n\nMarket Metrics:\n"
                for mid, metrics in market_metrics.items():
                    m = session.exec(
                        select(Market).where(Market.id == mid)
                    ).first()
                    metrics_text += f"\nMarket {mid}: {m.question}\n"
                    for k, v in metrics.items():
                        metrics_text += f"  {k}: {v}\n"

                # Calculate relevance score
                r_prompt = get_prompt("rule_relevance_score_prompt.j2", index_info)
                r_reasonings, r_scores, r_avg = get_relevance(
                    r_prompt, metrics_text, client
                )

                # Store the rule
                new_rule = MarketRule(
                    rule=rule_json,
                    readable_rule=readable,
                    created_at=datetime.now(UTC),
                    chain_of_thoughts=rule_obj.reasoning,
                    strength_weight=s_avg,
                    relevance_weight=r_avg,
                    strength_chain=str(s_reasonings),
                    relevance_chain=str(r_reasonings),
                    strength_scores=str(s_scores),
                    relevance_scores=str(r_scores),
                    batch_id=batch_number,
                    index_id=index_id,
                )
                session.add(new_rule)
                session.flush()

                # Link rule to its constituent markets
                for mid in rule_markets:
                    session.add(MarketRuleLink(market_id=mid, rule_id=new_rule.id))

            session.commit()

            context.log.info(
                f"Stored {len(logical_rules)} rules for index '{index_q.question}'."
            )

            total_rules += len(logical_rules)
            total_markets_used += len(used_markets)

        return dg.MaterializeResult(
            metadata={
                "num_rules": dg.MetadataValue.int(total_rules),
                "num_markets_used": dg.MetadataValue.int(total_markets_used),
                "num_indices": dg.MetadataValue.int(len(index_questions)),
                "batch_id": dg.MetadataValue.int(batch_number),
            }
        )


# =============================================================================
# Helper Functions - Simulation
# =============================================================================


def get_market_prob_series(
    session: Session,
    market_id: str,
    start_date: datetime,
    end_date: datetime
) -> pd.Series:
    """
    Retrieve the probability time series for a market.

    Constructs an hourly probability time series from bet data, forward-filling
    gaps. Handles resolved markets by returning constant series.

    Args:
        session: SQLAlchemy session for database queries.
        market_id: Unique identifier for the market.
        start_date: Start of the time window.
        end_date: End of the time window.

    Returns:
        A pandas Series indexed by hourly timestamps with probability values.
        Returns -1.0 for time points with no data.

    """
    resolution = session.exec(
        select(Market.resolution).where(Market.id == market_id)
    ).first()
    time_index = pd.date_range(start=start_date, end=end_date, freq="1h")

    # Handle resolved markets with constant probability
    if resolution == "YES":
        return pd.Series(1.0, index=time_index)
    elif resolution == "NO":
        return pd.Series(0.0, index=time_index)

    # Fetch bet history
    bets = session.exec(
        select(MarketBet.created_time, MarketBet.prob_after)
        .where(MarketBet.contract_id == market_id)
        .where(MarketBet.created_time <= end_date)
        .order_by(MarketBet.created_time)
    ).all()

    if not bets:
        return pd.Series(-1.0, index=time_index)

    # Build time-indexed probability series
    time_index = pd.date_range(
        start=start_date, end=end_date, freq="1h", tz="UTC"
    )
    df_bets = pd.DataFrame(bets, columns=["created_time", "prob_after"])
    df_bets["created_time"] = pd.to_datetime(df_bets["created_time"], utc=True)
    df_bets = df_bets.drop_duplicates(
        subset="created_time", keep="last"
    ).set_index("created_time")

    # Forward-fill probabilities to hourly grid
    series = df_bets["prob_after"].reindex(time_index, method="ffill").fillna(-1.0)
    return series


def eval_formula(
    formula: Formula,
    world_row: np.ndarray,
    market_index_map: dict[str, int]
) -> bool:
    """
    Recursively evaluate a Boolean formula against a simulation row.

    Evaluates a parsed Boolean formula tree by looking up variable values
    in the simulation row and applying logical operators.

    Args:
        formula: A parsed Boolean formula (VariableNode or OperatorNode).
        world_row: A binary array representing one simulation outcome.
        market_index_map: Mapping from variable names to column indices.

    Returns:
        The Boolean result of evaluating the formula.

    Raises:
        ValueError: If an unknown operator is encountered.
        TypeError: If the formula node type is invalid.

    """
    # Base case: variable node - look up value in simulation row
    if isinstance(formula, VariableNode):
        idx = market_index_map.get(formula.var)
        return bool(world_row[idx])

    # Recursive case: operator node - evaluate children and apply operator
    elif isinstance(formula, OperatorNode):
        args = [
            eval_formula(arg, world_row, market_index_map)
            for arg in formula.arguments
        ]
        op = formula.node_type

        if op == "and":
            return all(args)
        elif op == "or":
            return any(args)
        elif op == "not":
            return not args[0]
        elif op == "xor":
            return sum(args) % 2 == 1
        elif op == "nand":
            return not all(args)
        elif op == "nor":
            return not any(args)
        else:
            raise ValueError(f"Unknown operator: {op}")

    else:
        raise TypeError(f"Invalid formula node: {formula}")


# =============================================================================
# Dagster Assets - Index Probability Calculation
# =============================================================================


@dg.asset(
    deps=[index_rules],
    required_resource_keys={"database_engine"},
    description="Monte Carlo simulation for index probability using Gaussian copula."
)
def index_value(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Calculate index probabilities using Monte Carlo simulation.

    This asset runs a Gaussian copula-based Monte Carlo simulation to estimate
    the probability of index events. For each index question:
    1. Loads the latest batch of rules for that index
    2. Builds probability time series for all constituent markets
    3. Computes correlation structure using Gaussian copula
    4. Simulates 10,000 worlds and evaluates rules
    5. Calculates weighted risk index based on rule satisfaction

    The simulation accounts for market correlations and uses rule weights
    (strength and relevance) to compute a final probability estimate.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    with (Session(context.resources.database_engine) as session):
        index_questions = session.exec(select(IndexQuestion)).all()

        if not index_questions:
            context.log.error("No index questions found.")
            return dg.MaterializeResult(metadata={})

        total_indices = 0
        index_probabilities = {}

        # Process each index question
        for index_q in index_questions:
            context.log.info(
                f"Running simulation for index question: {index_q.question}"
            )

            # Get all rules for this index
            rules_for_index = session.exec(
                select(MarketRule).where(MarketRule.index_id == index_q.id)
            ).all()

            if not rules_for_index:
                context.log.warning(
                    f"No rules found for index question {index_q.id}"
                )
                continue

            # Find the most recent batch
            latest_batch = session.exec(
                select(MarketRule.batch_id)
                .where(MarketRule.index_id == index_q.id)
                .order_by(MarketRule.created_at.desc())
                .limit(1)
            ).first()

            if latest_batch is None:
                context.log.warning(
                    f"No batch found for index question {index_q.id}"
                )
                continue

            # Filter to only rules in the latest batch
            rules = [r for r in rules_for_index if r.batch_id == latest_batch]

            if not rules:
                context.log.warning(
                    f"No rules in latest batch for index question {index_q.id}"
                )
                continue

            # Extract rule metadata
            rule_ids = [r.id for r in rules]
            rule_strength_weights = [r.strength_weight for r in rules]
            rule_relevance_weights = [r.relevance_weight for r in rules]
            rule_avg_weights = [
                (s + r) / 2
                for s, r in zip(
                    rule_strength_weights, rule_relevance_weights, strict=False
                )
            ]

            # Get all markets linked to these rules
            market_links_by_rule = {}
            all_market_ids = set()
            for rule in rules:
                market_ids = list(session.exec(
                    select(MarketRuleLink.market_id)
                    .where(MarketRuleLink.rule_id == rule.id)
                ))
                market_links_by_rule[rule.id] = market_ids
                all_market_ids.update(market_ids)

            unique_literals = list(all_market_ids)

            # Build probability time series
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=60)
            time_index = pd.date_range(
                start=start_date, end=end_date, freq='1h'
            )

            df_probs = pd.DataFrame(
                index=time_index, columns=unique_literals, dtype=float
            )

            for market_id in unique_literals:
                df_probs[market_id] = get_market_prob_series(
                    session, market_id, start_date, end_date
                )
                context.log.info(
                    f"Loaded probability series for market {market_id}"
                )

            # Compute Gaussian copula correlation
            # Replace missing values and transform to latent Gaussian space
            df_probs_clean = df_probs.replace(-1, np.nan)
            latent_vars = df_probs_clean.apply(norm.ppf)

            # Compute correlation matrix (minimum 10 overlapping points)
            corr_df = latent_vars.corr(min_periods=10).fillna(0)
            corr_matrix = corr_df.values

            # Get current thresholds from most recent probabilities
            thresholds_dict = {}
            for market_id in unique_literals:
                series = df_probs[market_id].dropna()
                thresholds_dict[market_id] = series.iloc[-1]

            # Prepare for simulation
            dim = len(unique_literals)
            thresholds = np.array([
                norm.ppf(np.clip(thresholds_dict.get(lit), 1e-6, 1 - 1e-6))
                for lit in unique_literals
            ])

            # Ensure correlation matrix is positive semi-definite
            eps = 1e-3
            eigvals, eigvecs = np.linalg.eigh(corr_matrix)
            eigvals_clipped = np.clip(eigvals, eps, None)
            corr_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

            # Run Monte Carlo simulation
            # Sample 10,000 correlated normal vectors
            z_samples = np.random.multivariate_normal(
                mean=np.zeros(dim), cov=corr_psd, size=10000
            )
            # Convert to binary outcomes based on thresholds
            bool_samples = (z_samples < thresholds).astype(bool)

            # Normalize rule weights (guard against zero sum)
            weights = np.array(rule_avg_weights, dtype=float)
            weight_sum = weights.sum()

            if weight_sum == 0:
                context.log.warning(
                    f"All rule weights are zero for index question {index_q.id}. "
                    "Falling back to uniform weights."
                )
                norm_weights = np.ones_like(weights) / len(weights)
            else:
                norm_weights = weights / weight_sum

            # Evaluate rules across simulations
            market_index_map = {
                name: idx for idx, name in enumerate(unique_literals)
            }
            n_simulations = bool_samples.shape[0]
            rule_indicator_matrix = np.zeros((n_simulations, len(rules)))

            for i, rule in enumerate(rules):
                try:
                    raw_rule = rule.rule
                    raw_rule = json.loads(raw_rule)
                    adapter = TypeAdapter(Formula)
                    formula = adapter.validate_python(raw_rule)
                except Exception as e:
                    context.log.warning(f"Failed to parse rule {rule.id}: {e}")
                    continue

                for j in range(n_simulations):
                    world_row = bool_samples[j]
                    try:
                        rule_result = eval_formula(
                            formula, world_row, market_index_map
                        )
                        rule_indicator_matrix[j, i] = float(rule_result)
                    except Exception as e:
                        context.log.warning(
                            f"Error evaluating rule {rule.id} on simulation {j}: {e}"
                        )

            # Calculate weighted risk index
            scores = rule_indicator_matrix @ norm_weights
            risk_index = scores.mean()

            # Build JSON summary
            rules_json = []
            for rule in rules:
                rules_json.append({
                    "id": rule.id,
                    "readable_rule": rule.readable_rule,
                    "rule": rule.rule,
                    "strength_weight": rule.strength_weight,
                    "relevance_weight": rule.relevance_weight,
                    "avg_weight": (
                        (rule.strength_weight + rule.relevance_weight) / 2
                    ),
                    "chain_of_thoughts": rule.chain_of_thoughts,
                    "strength_scores": json.loads(rule.strength_scores or "[]"),
                    "relevance_scores": json.loads(rule.relevance_scores or "[]"),
                    "strength_chain": rule.strength_chain,
                    "relevance_chain": rule.relevance_chain,
                    "market_ids": [m.id for m in rule.markets]
                })

            json_rep = {
                "index_probability": risk_index,
                "market_ids_in_rules": list(unique_literals),
                "rules": rules_json
            }

            # Store index result
            new_index = Index(
                index_probability=float(risk_index),
                index_question_id=str(index_q.id),
                created_at=datetime.now(UTC),
                json_representation=json.dumps(json_rep)
            )
            session.add(new_index)
            session.flush()
            context.log.info("Added index")

            # Link index to its constituent rules
            for rule_id in rule_ids:
                rule_obj = session.exec(
                    select(MarketRule).where(MarketRule.id == rule_id)
                ).first()

                if rule_obj:
                    new_link = IndexRuleLink(
                        rule_id=rule_obj.id,
                        index_id=new_index.id,
                    )
                    session.add(new_link)
                else:
                    context.log.warning(f"Rule not found for ID: {rule_id}")

            index_probabilities[index_q.question] = float(risk_index)
            total_indices += 1
            session.commit()

        return dg.MaterializeResult(
            metadata={
                "num_indices_generated": dg.MetadataValue.int(total_indices),
                "index_probabilities": dg.MetadataValue.json(index_probabilities),
            }
        )
