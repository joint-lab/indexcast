"""
Market assets.

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


def _safe_fromtimestamp(ms: int) -> datetime | None:
    """Safely convert milliseconds since epoch to datetime, or return None if out of range."""
    try:
        dt = datetime.fromtimestamp(ms / 1000, UTC)
        if 1 <= dt.year <= 9999:
            return dt
        return None
    except (OverflowError, OSError, ValueError):
        return None


def _prepare_market(market_data: dict) -> Market:
    """Prepare a Market object from raw API data."""
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
    """Prepare a MarketBet object from raw API data."""
    created_time = _safe_fromtimestamp(bet_data["createdTime"])
    updated_time = (
        _safe_fromtimestamp(bet_data["updatedTime"])
        if bet_data.get("updatedTime") else None
    )

    return MarketBet(
        # identifiers
        id=bet_data["id"],
        contract_id=bet_data["contractId"],
        user_id=bet_data["userId"],
        bet_group_id=bet_data.get("betGroupId"),

        # bet info
        outcome=bet_data["outcome"],
        amount=bet_data["amount"],
        order_amount=bet_data.get("orderAmount",0.0),
        loan_amount=bet_data.get("loanAmount", 0.0),
        shares=bet_data["shares"],
        fills=json.dumps(bet_data.get("fills")),

        # probabilities
        prob_before=bet_data["probBefore"],
        prob_after=bet_data["probAfter"],
        limit_prob=bet_data.get("limitProb"),

        # status
        visibility=bet_data.get("visibility", ""),
        is_api=bet_data.get("isApi", False),
        is_filled=bet_data.get("isFilled", False),
        is_cancelled=bet_data.get("isCancelled", False),
        is_redemption=bet_data.get("isRedemption", False),

        created_time=created_time,
        updated_time=updated_time,

        # fees
        platform_fee=bet_data.get("fees", {}).get("platformFee", 0.0),
        liquidity_fee=bet_data.get("fees", {}).get("liquidityFee", 0.0),
        creator_fee=bet_data.get("fees", {}).get("creatorFee", 0.0),

        # internal timestamp
        updated_at=datetime.now(UTC),
    )

def _prepare_comment(comment_data: dict) -> MarketComment:
    """Prepare a MarketComment object from raw API data."""
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
        # identifiers
        id=comment_data["id"],
        market_id=comment_data.get("contractId"),
        user_id=comment_data["userId"],
        reply_to_comment_id=comment_data.get("replyToCommentId"),

        # content
        comment_type=comment_data["commentType"],
        is_api=comment_data.get("isApi", False),
        content=json.dumps(
            comment_data.get("content")
            or ""
        ),
        # other info
        visibility=comment_data["visibility"],
        hidden=comment_data.get("hidden", False),

        # timestamps
        created_time=created_time,
        hidden_time=hidden_time,
        edited_time=edited_time,

        # internal timestamp
        updated_at=datetime.now(UTC),
    )


def get_volume(session: Session, market_id: str, query_time: datetime) -> float:
    """
    Retrieve the volume for a given market since the given query time.

    Args:
        session (Session): SQLAlchemy session used to query the database.
        market_id (str): Unique identifier for the market to query.
        query_time (datetime): Timestamp representing the time from which to query.

    Returns:
        float: The `volume` since the given query time.

    """
    now = datetime.now(UTC)
    result = session.exec(select(func.sum(func.abs(MarketBet.amount)))
                          .where(MarketBet.contract_id == market_id).where(
                          MarketBet.created_time <= now)
                          .where(MarketBet.created_time >= query_time)).one_or_none()
    return result or 0.0

def fetch_text_from_url(
    url: str,
    retries: int = 3,
    timeout: tuple = (5, 10),  # (connect_timeout, read_timeout)
    backoff: float = 2,
    rate_limit_pause: float = 0
) -> str:
    """
    Fetch LLM-friendly text from r.jina.ai with API key authorization.

    - url: target webpage (e.g. "https://example.com")
    - retries: number of retry attempts on timeout
    - timeout: (connect, read) timeouts in seconds
    - backoff: base seconds to wait (exponential backoff)
    - rate_limit_pause: optional pause after successful fetch (seconds)
    """
    endpoint = f"https://r.jina.ai/{url}"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(endpoint, timeout=timeout)
            resp.raise_for_status()
            if rate_limit_pause:
                time.sleep(rate_limit_pause)
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
    Find all HTTP/HTTPS URLs in the given text.

    Returns a list of matching URL strings.
    """
    # Regex pattern to match most http(s) URLs
    url_pattern = r'(https?://[^\s]+)'
    return re.findall(url_pattern, text)

def get_text_rep(market: Market) -> str:
    """
    Create a text representation of the given market.

    Returns a string representation of the given market.
    """
    # Extract the title and description
    title = market.question
    # make sure the description is not too long
    description = market.description[:4875]

    # if there are urls in the description, might need them to add context
    urls = extract_urls(market.description)
    url_text = ""
    if len(urls) > 0:
        for url in urls:
            url_text += fetch_text_from_url(url)

    # build text rep
    # make sure there were urls and fetching the contents did not result in an error
    if len(url_text) > 0 and "error" not in url_text[:20].lower():
        # trim for now to make sure not too many tokens
        trimmed = url_text[:4875]
        text_rep = f"""<Title>{title}</Title>
        <Description>{description}</Description> 
        <Url Text>{trimmed}</Url Text> """
    else:
        text_rep = f"""<Title>{title}</Title>
        <Description>{description}</Description>"""
    return text_rep

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

    # Manifold API client
    manifold_client = context.resources.manifold_client

    # Fetch markets in batches
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
                if session.get(Market, m["id"]):  # Stop if market already exists
                    context.log.debug(f"Market {m['id']} already exists in DB. Stopping fetch.")
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
        f"Inserted {len(new_markets)} new markets from Manifold (fetched {total_fetched} total)"
    )
    return dg.MaterializeResult(
        metadata={
            "num_markets_inserted": dg.MetadataValue.int(len(new_markets)),
        }
    )

@dg.asset(
    deps=[manifold_markets],
    required_resource_keys={"database_engine"},
    description="Market labels table."
)
def market_labels(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """Update market labels when materialized."""
    initial_prompt = get_initial_labeling_prompt()
    dspy_scorer = DSPyMarketScorer()
    now = datetime.now(UTC)
    client = get_client()

    with Session(context.resources.database_engine) as session:
        # Only process markets that haven't completed the CLASSIFIED stage yet
        subquery = select(MarketPipelineEvent.market_id).where(
            MarketPipelineEvent.stage_id == PipelineStageType.CLASSIFIED
        )
        markets_to_process = session.exec(
            select(Market).where(Market.id.not_in(subquery))
        ).all()
        context.log.info(f"Found {len(markets_to_process)} markets needing classification.")
        results = []

        # Label ID map
        index_questions = session.exec(
            select(IndexQuestion)
        ).all()

        index_question_map = {
            iq.id: iq.question for iq in index_questions
        }
        context.log.info(f"Loaded {len(index_question_map)} index questions from DB")

        for market in markets_to_process:
            context.log.debug(f"Processing market {market.id}")
            # Must be binary & unresolved
            if market.outcome_type != "BINARY" or market.resolution is not None:
                context.log.info(f"Skipping market {market.id}: invalid type/resolved.")
                continue

            # initial labeler
            initial_response = get_initial_label(
                prompt=initial_prompt,
                market_question=market.question,
                client=client,
            )

            # Store the dump info
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

            # If none-of-the-above, skip DSPy entirely
            if labels == [-1]:
                results.append((market.id, [], []))  # No labels or scores
                continue

            market_label_rows = []
            relevance_rows = []

            # For each relevant label, run DSPy
            for label_id in labels:
                if label_id not in index_question_map:
                    context.log.warning(f"Unknown label {label_id} for market {market.id}")
                    continue

                index_question = index_question_map[label_id]
                dspy_out = dspy_scorer.predict(index_question, market.question)
                # store data from dspy
                session.add(LabelInfo(
                    market_id=market.id,
                    type=LabelType.final,
                    output=json.dumps({
                        "rationale": dspy_out["rationale"],
                        "label": dspy_out["label"],
                        "score": dspy_out["score"]
                    }),
                ))

                # DSPy output format:
                #   dspy_out = { rationale, label, score }
                is_relevant = dspy_out["label"].strip() == "1"

                if is_relevant:
                    # add labels
                    session.add(
                        MarketLabel(
                            market_id=market.id,
                            label_type_id=label_id,
                        )
                    )
                    # add scores
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

                relevance_rows.append({
                    "label_id": label_id,
                    "dspy_rationale": dspy_out["rationale"],
                    "dspy_score": dspy_out["score"],
                    "is_relevant": is_relevant
                })

            results.append((market.id, market_label_rows, relevance_rows))
        for market in markets_to_process:
            # Mark pipeline stage complete
            session.add(MarketPipelineEvent(
                market_id=market.id,
                stage_id=PipelineStageType.CLASSIFIED,
                completed_at=now
            ))
        session.commit()

    return dg.MaterializeResult(
        metadata={
            "num_markets_processed": dg.MetadataValue.int(len(markets_to_process)),
            "num_markets_labeled": dg.MetadataValue.int(sum(len(r[1]) for r in results)),
        }
    )

@dg.asset(
    deps=[market_labels],
    required_resource_keys={"database_engine", "manifold_client"},
    description="Full Market table (bets, comments, description).",
)
def manifold_full_markets(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Fetch the latest bets and comments from the Manifold API and populates the market_bets table.

    This asset looks at the labels table, fetches all bets, comments and the descriptions
    from the Manifold API for those markets. It inserts new bets and overwites all comments and
    the description fresh each time.It also respects the Manifold API rate limits and
    handles pagination.
    """
    # Manifold API client
    manifold_client = context.resources.manifold_client

    # get list of market ids that have been labeled
    with Session(context.resources.database_engine) as session:
        # Get all market IDs from all labels
        all_market_ids = session.exec(
            select(MarketLabel.market_id)
            .join(MarketLabelType, MarketLabel.label_type_id == MarketLabelType.id)
        ).all()

        # Convert to a unique set
        unique_market_ids = set(all_market_ids)

    context.log.info(f"Found {len(unique_market_ids)} labeled markets.")

    total_comments_inserted = 0
    total_bets_updated = 0

    with Session(context.resources.database_engine) as session:
        existing_bets_rows = session.exec(
            select(MarketBet.id, MarketBet.contract_id, MarketBet.is_filled)
            .where(MarketBet.contract_id.in_(unique_market_ids))
        ).all()

        # bucket by market_id
        bets_by_market = {}
        for bid, mid, filled in existing_bets_rows:
            bets_by_market.setdefault(mid, []).append((bid, filled))

        # main loop
        for m in unique_market_ids:

            # full market data
            full_market = manifold_client.full_market(m)

            # description
            description_json = json.dumps(full_market.get("description"))

            # Build a stub Market-like object for get_text_rep
            temp_market = Market(id=m, description=description_json)
            text_rep = get_text_rep(temp_market)

            # Market description + text_rep
            market = session.exec(
                select(Market).where(Market.id == m)
            ).first()
            if market:
                market.description = description_json
                market.text_rep = text_rep

            # FETCH & REPLACE COMMENTS
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

            # delete existing comments
            session.exec(
                delete(MarketComment).where(MarketComment.market_id == m)
            )

            # insert new comments
            objs = [_prepare_comment(c) for c in all_comments]
            session.bulk_save_objects(objs)

            total_comments_inserted += len(all_comments)
            context.log.info(f"Market {m}: found {len(all_comments)} comments.")

            # FETCH BETS
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

            # existing bets for this market
            existing = bets_by_market.get(m, [])
            existing_ids = {bid for (bid, filled) in existing}
            unfilled_ids = [bid for (bid, filled) in existing if not filled]

            # delete unfilled bets
            if unfilled_ids:
                session.exec(
                    delete(MarketBet).where(MarketBet.id.in_(unfilled_ids))
                )

            # prepare new bets
            to_insert = [
                _prepare_bet(b)
                for b in all_bets
                if b["id"] not in existing_ids
            ]

            session.bulk_save_objects(to_insert)
            total_bets_updated += len(all_bets)

            # TRACK PIPELINE EVENT
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

        # commit everything for all markets
        session.commit()

    return dg.MaterializeResult(
        metadata={
            "num_markets_processed": dg.MetadataValue.int(len(unique_market_ids)),
            "total_comments_inserted": dg.MetadataValue.int(total_comments_inserted),
            "total_bets_updated": dg.MetadataValue.int(total_bets_updated),
        }
    )


@dg.asset(
    deps=[manifold_full_markets],
    required_resource_keys={"database_engine"},
    description="Relevance summary scores for the labeled markets.",
)
def relevance_summary_statistics(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Use bets and comments tables to populate relevance_scores table.

    This asset looks at the labels table and for all labeled markets, it generates 5 relevance
    scores and stores them in the database.
    """
    # get list of market ids that have been labeled
    with Session(context.resources.database_engine) as session:
        labeled_markets = session.exec(
            select(MarketLabel.market_id, MarketLabel.label_type_id)
        ).all()

    context.log.info(f"Found {len(labeled_markets)} labeled market/label pairs.")

    # Load all score‐type rows once and build a name id map
    score_type_map = {s.relevance_score: s.value for s in MarketRelevanceScoreType}
    context.log.info(f"Found {len(score_type_map)} relevance label types.")

    with locked_session(context.resources.database_engine) as session:
        for market_id, label_id in labeled_markets:
            # delete selected score types
            types_to_delete = ["volume_24h", "num_comments", "volume_total",
                               "volume_144h", "num_traders"]
            type_ids = [score_type_map[name] for name in types_to_delete if name in score_type_map]
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
            session.exec(
                delete(MarketRelevanceScore)
                .where(MarketRelevanceScore.market_id == market_id)
                .where(MarketRelevanceScore.label_id == label_id)
                .where(MarketRelevanceScore.score_type_id.in_(type_ids))
            )

            # volume_total
            volume_total = get_volume(session, market_id, datetime.min)

            # volume_24h
            volume_24h = get_volume(session, market_id, datetime.now(UTC) - timedelta(hours=24))

            # volume_144h
            volume_144h = get_volume(session, market_id, datetime.now(UTC) - timedelta(hours=144))

            # num_traders
            num_traders = session.exec(select(func.count(func.distinct(MarketBet.user_id)))
                .where(MarketBet.contract_id == market_id)).one_or_none() or 0

            # num_comments
            num_comments = session.exec(select(func.count())
                .where(MarketComment.market_id == market_id)).one_or_none() or 0

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


@dg.asset(
    deps=[relevance_summary_statistics],
    required_resource_keys={"database_engine"},
    description="Remove non-eligible index labels per market based on metrics."
)
def market_rule_eligibility_labels(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """Update market labels: remove index labels for which a market is not eligible."""
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
                select(MarketLabelType).where(MarketLabelType.label_name.in_(index_labels))
            ).all()
        }
        index_label_type_ids = [lt.id for lt in label_types.values()]

        # Fetch all market labels for these index types
        market_labels = session.exec(
            select(MarketLabel).where(MarketLabel.label_type_id.in_(index_label_type_ids))
        ).all()

    market_ids = list(set(label.market_id for label in market_labels))

    # Score types
    question_score_type_id = MarketRelevanceScoreType.INDEX_QUESTION_RELEVANCE.value
    volume_score_type_id = MarketRelevanceScoreType.VOLUME_TOTAL.value
    traders_score_type_id = MarketRelevanceScoreType.NUM_TRADERS.value

    score_type_ids_set = {question_score_type_id, volume_score_type_id, traders_score_type_id}

    with Session(context.resources.database_engine) as session:
        all_scores = session.exec(
            select(MarketRelevanceScore)
            .where(
                (MarketRelevanceScore.market_id.in_(market_ids)) &
                (MarketRelevanceScore.score_type_id.in_(score_type_ids_set))
            )
        ).all()

    # Create a lookup for scores
    score_lookup = {
        (score.market_id, score.score_type_id): score.score_value
        for score in all_scores
    }

    volume_threshold = 200
    traders_threshold = 11
    num_labels_removed = 0

    with Session(context.resources.database_engine) as session:
        for label in market_labels:
            m_id = label.market_id
            question_score = score_lookup.get((m_id, question_score_type_id))
            volume_score = score_lookup.get((m_id, volume_score_type_id))
            trades_score = score_lookup.get((m_id, traders_score_type_id))

            # Fetch market outcome type
            market = session.exec(select(Market).where(Market.id == m_id)).first()

            # Determine eligibility for this label
            is_eligible = True
            if market.outcome_type != "BINARY":
                is_eligible = False
            elif volume_score is None or volume_score < volume_threshold:
                is_eligible = False
            elif trades_score is None or trades_score < traders_threshold:
                is_eligible = False
            elif question_score is None or question_score < 0.6:
                is_eligible = False

            if not is_eligible:
                session.delete(label)
                num_labels_removed += 1

        session.commit()

    return dg.MaterializeResult(
        metadata={
            "num_labels_checked": dg.MetadataValue.int(len(market_labels)),
            "num_labels_removed": dg.MetadataValue.int(num_labels_removed),
        }
    )

@dg.asset(
    deps=[market_rule_eligibility_labels],
    required_resource_keys={"database_engine"},
    description="Generate 30 logical rules per index using LLM."
)
def index_rules(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """Run Monte Carlo simulation to calculate H5N1 outbreak probability."""
    with Session(context.resources.database_engine) as session:
        # Load all index questions
        index_questions = session.exec(select(IndexQuestion)).all()

        if not index_questions:
            context.log.warning("No index questions found.")
            return dg.MaterializeResult()

        client = get_client()
        total_rules = 0
        total_markets_used = 0

        # PROCESS EACH INDEX
        for index_q in index_questions:
            index_id = index_q.id
            context.log.info(f"Processing index: {index_q.question}")

            # batch number
            previous_batch = session.exec(
                select(func.max(MarketRule.batch_id))
                .where(MarketRule.index_id == index_id)
            ).one_or_none()

            batch_number = (previous_batch or 0) + 1
            context.log.info(f"Assigned batch number {batch_number} for index_id={index_id}")

            # Get all markets labeled for this index
            eligible_markets = session.exec(
                select(Market)
                .join(MarketLabel, Market.id == MarketLabel.market_id)
                .where(MarketLabel.label_type_id == index_id)
            ).all()

            if not eligible_markets:
                context.log.info(f"No markets labeled for index {index_q.question}")
                continue

            context.log.info(f"Found {len(eligible_markets)} markets for index.")

            # Build market dict for prompt
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
                num_of_rules=30,
            )

            prompt = get_rules_prompt("rule_gen_prompt.j2", prompt_data, market_data)

            valid_market_ids = set(market_data.keys())

            # LLM: generate 30 rules
            logical_rules = get_rules(
                prompt=prompt,
                num_rules=30,
                client=client,
                allowed_market_ids=valid_market_ids,
            )
            context.log.info(f"Generated {len(logical_rules)} rules.")

            used_markets = set()

            # Score and store rules
            for rule_obj in logical_rules:
                rule_json = rule_obj.rule.model_dump_json()
                readable = stringify_formula(rule_obj.rule)

                index_info = IndexInformation(
                    todays_date=datetime.now(UTC),
                    overall_index_question=index_q.question,
                )

                # ---- Strength score ----
                s_prompt = get_prompt("rule_strength_score_prompt.j2", index_info)
                s_reasonings, s_scores, s_avg = get_weight(
                    s_prompt, readable, client
                )

                # ---- Relevance score ----
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

                metrics_text = f"Rule: {readable}\n\nMarket Metrics:\n"
                for mid, metrics in market_metrics.items():
                    m = session.exec(select(Market).where(Market.id == mid)).first()
                    metrics_text += f"\nMarket {mid}: {m.question}\n"
                    for k, v in metrics.items():
                        metrics_text += f"  {k}: {v}\n"

                r_prompt = get_prompt("rule_relevance_score_prompt.j2", index_info)
                r_reasonings, r_scores, r_avg = get_relevance(
                    r_prompt, metrics_text, client
                )

                # ---- Store rule ----
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

                # ---- Link rule to markets ----
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



def get_market_prob_series(session: Session, market_id: str,
                           start_date: datetime, end_date: datetime) -> pd.Series:
    """
    Retrieve the full probability time series for a market between start_date and end_date.

    Returns a pandas Series indexed by timestamp (hourly).
    """
    resolution = session.exec(select(Market.resolution).where(Market.id == market_id)).first()
    time_index = pd.date_range(start=start_date, end=end_date, freq="1h")

    if resolution == "YES":
        return pd.Series(1.0, index=time_index)
    elif resolution == "NO":
        return pd.Series(0.0, index=time_index)

    bets = session.exec(
        select(MarketBet.created_time, MarketBet.prob_after)
        .where(MarketBet.contract_id == market_id)
        .where(MarketBet.created_time <= end_date)
        .order_by(MarketBet.created_time)
    ).all()

    if not bets:
        return pd.Series(-1.0, index=time_index)

    time_index = pd.date_range(start=start_date, end=end_date, freq="1h", tz="UTC")

    df_bets = pd.DataFrame(bets, columns=["created_time", "prob_after"])
    df_bets["created_time"] = pd.to_datetime(df_bets["created_time"], utc=True)
    df_bets = df_bets.drop_duplicates(subset="created_time", keep="last").set_index("created_time")

    series = df_bets["prob_after"].reindex(time_index, method="ffill").fillna(-1.0)
    return series

def eval_formula(formula: Formula, world_row: np.ndarray, market_index_map: dict[str, int]) -> bool:
    """
    Recursively evaluates a Boolean formula against a binary simulation row.

    Args:
        formula (Formula): A parsed Boolean formula (Var or Op).
        world_row (np.ndarray): A binary array representing a simulation.
        market_index_map (dict[str, int]): Mapping from var names to column indices in world_row.

    Returns:
        bool: The result of evaluating the formula under the current simulation.

    """
    # Base case: if the formula is a variable node, look up its value in the world row
    if isinstance(formula, VariableNode):
        idx = market_index_map.get(formula.var)
        return bool(world_row[idx])  # Convert to bool in case it's a float or int

    # Recursive case: formula is a logical operation (AND, OR, NOT, etc.)
    elif isinstance(formula, OperatorNode):
        # Evaluate all child formulas recursively
        args = [eval_formula(arg, world_row, market_index_map) for arg in formula.arguments]
        op = formula.node_type

        # Apply the Boolean operator to the evaluated arguments
        if op == "and":
            return all(args)               # True if all subformulas are True
        elif op == "or":
            return any(args)              # True if any subformula is True
        elif op == "not":
            return not args[0]            # True if the only child is False
        elif op == "xor":
            return sum(args) % 2 == 1     # True if exactly one argument is True
        elif op == "nand":
            return not all(args)          # True if not all are True (negated AND)
        elif op == "nor":
            return not any(args)          # True if all are False (negated OR)
        else:
            raise ValueError(f"Unknown operator: {op}") # should never happen if model is validated

    # if input is not a Var or Op, raise an error
    else:
        raise TypeError(f"Invalid formula node: {formula}")


@dg.asset(
    deps=[index_rules],
    required_resource_keys={"database_engine"},
    description="Monte Carlo simulation for H5N1 outbreak probability using Gaussian copula."
)
def index_value(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """Run Monte Carlo simulation to calculate H5N1 outbreak probability."""
    # set seed
    np.random.seed(42)
    with (Session(context.resources.database_engine) as session):
        index_questions = session.exec(select(IndexQuestion)).all()
        if not index_questions:
            context.log.error("No index questions found.")
            return dg.MaterializeResult(metadata={})

        total_indices = 0
        index_probabilities = {}

        for index_q in index_questions:
            context.log.info(f"Running simulation for index question: {index_q.question}")

            # get all rules tied to this index question
            rules_for_index = session.exec(
                select(MarketRule).where(MarketRule.index_id == index_q.id)
            ).all()

            if not rules_for_index:
                context.log.warning(f"No rules found for index question {index_q.id}")
                continue

            # find the most recent batch among those rules
            latest_batch = session.exec(
                select(MarketRule.batch_id)
                .where(MarketRule.index_id == index_q.id)
                .order_by(MarketRule.created_at.desc())
                .limit(1)
            ).first()

            if latest_batch is None:
                context.log.warning(f"No batch found for index question {index_q.id}")
                continue

            # filter rules to only those in that batch
            rules = [r for r in rules_for_index if r.batch_id == latest_batch]

            if not rules:
                context.log.warning(f"No rules in latest batch for index question {index_q.id}")
                continue

            # proceed with simulation using only these rules
            rule_ids = [r.id for r in rules]
            rule_strength_weights = [r.strength_weight for r in rules]
            rule_relevance_weights = [r.relevance_weight for r in rules]
            rule_avg_weights = [
                (s + r) / 2 for s, r in zip(rule_strength_weights,
                                            rule_relevance_weights, strict=False)
            ]

            # Get market links and unique markets from rules
            market_links_by_rule = {}
            all_market_ids = set()
            for rule in rules:
                market_ids = list(session.exec(
                    select(MarketRuleLink.market_id).where(MarketRuleLink.rule_id == rule.id)
                ))
                market_links_by_rule[rule.id] = market_ids
                all_market_ids.update(market_ids)

            unique_literals = list(all_market_ids)

            # Create index of timestamps
            end_date = datetime.now(UTC)
            # use up to 60 days worth of data
            start_date = end_date - timedelta(days=60)
            time_index = pd.date_range(start=start_date, end=end_date, freq='1h')

            # Initialize empty DataFrame
            df_probs = pd.DataFrame(index=time_index, columns=unique_literals, dtype=float)

            # Fill DataFrame using market lookup function
            for market_id in unique_literals:
                df_probs[market_id] = get_market_prob_series(session, market_id,
                                                             start_date, end_date)
                context.log.info(f"Loaded probability series for market {market_id}")

            # Gaussian copula correlation matrix
            # all probs that were not found (ie, that were -1 and NaN)
            df_probs_clean = df_probs.replace(-1, np.nan)
            # transform probabilities into a latent Gaussian space
            # inverse CDF (quantile function) of the standard normal distribution
            latent_vars = df_probs_clean.apply(norm.ppf)
            # the Pearson correlation matrix
            # min_periods=10 so we only compute correlation if there are at least 10
            # overlapping time points — otherwise it returns NaN.
            # NaNs in the correlation matrix are replaced with 0,
            # assuming independence between those pairs
            corr_df = latent_vars.corr(min_periods=10).fillna(0)
            corr_matrix = corr_df.values

            # thresholds are equal to the current market prob (most recent prob_after)
            thresholds_dict = {}
            for market_id in unique_literals:
                series = df_probs[market_id].dropna()
                thresholds_dict[market_id] = series.iloc[-1]

            # sample worlds using copula
            dim = len(unique_literals)
            thresholds = np.array([
                norm.ppf(np.clip(thresholds_dict.get(lit), 1e-6, 1 - 1e-6))
                for lit in unique_literals
            ])

            # make sure the matrix is PSD
            # to sample from a multivariate normal with np.random.multivariate_normal,
            # we need PSD covariance matrix
            eps = 1e-3
            # corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
            eigvals, eigvecs = np.linalg.eigh(corr_matrix)
            # Eigenvalues must be ≥ 0 for the matrix to be PSD
            eigvals_clipped = np.clip(eigvals, eps, None)
            # corr_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
            corr_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

            # sample 10000 "worlds"
            z_samples = np.random.multivariate_normal(mean=np.zeros(dim), cov=corr_psd, size=10000)
            # make markets true or false based on thresholds
            bool_samples = (z_samples < thresholds).astype(bool)

            # normalize rule weights (guard against division by zero)
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

            # set up for evaluating the rules
            market_index_map = {name: idx for idx, name in enumerate(unique_literals)}
            n_simulations = bool_samples.shape[0]
            rule_indicator_matrix = np.zeros((n_simulations, len(rules)))

            # for rule, turn it back into a formula and evaluate it for each world
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
                        rule_result = eval_formula(formula, world_row, market_index_map)
                        rule_indicator_matrix[j, i] = float(rule_result)
                    except Exception as e:
                        context.log.warning(f"Error evaluating rule {rule.id} on simulation {j}: "
                                            f"{e}")

            # use the weights to score each world for how risky it is
            scores = rule_indicator_matrix @ norm_weights
            risk_index = scores.mean()

            # build a json summary of the rules in the sim
            rules_json = []
            for rule in rules:
                rules_json.append({
                    "id": rule.id,
                    "readable_rule": rule.readable_rule,
                    "rule": rule.rule,
                    "strength_weight": rule.strength_weight,
                    "relevance_weight": rule.relevance_weight,
                    "avg_weight": ((rule.strength_weight) + (rule.relevance_weight)) / 2,
                    "chain_of_thoughts": rule.chain_of_thoughts,
                    "strength_scores": json.loads(rule.strength_scores or "[]"),
                    "relevance_scores": json.loads(rule.relevance_scores or "[]"),
                    "strength_chain": rule.strength_chain,
                    "relevance_chain": rule.relevance_chain,
                    "market_ids": [m.id for m in rule.markets]
                })

            # json rep of the simulation
            json_rep = {
                "index_probability": risk_index,
                "market_ids_in_rules": list(unique_literals),
                "rules": rules_json
            }

            # add to the DB
            new_index = Index(
                index_probability=float(risk_index),
                index_question_id=str(index_q.id),
                created_at=datetime.now(UTC),
                json_representation=json.dumps(json_rep)
            )
            session.add(new_index)
            session.flush()
            context.log.info("Added index")
            for rule_id in rule_ids:
                # The rule contains actual rules IDs, not text representations
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