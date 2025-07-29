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
import requests
from sqlalchemy import delete, func
from sqlmodel import Session, select

from ml.classification import H5N1Classifier, RuleEligibilityClassifier
from ml.clients import get_client
from ml.ranker import DiseaseInformation, get_prompt, get_relevance
from ml.rules import PromptInformation, RuleNode, get_rules, get_rules_prompt, get_weight
from models.markets import (
    Market,
    MarketBet,
    MarketComment,
    MarketLabel,
    MarketLabelType,
    MarketRelevanceScore,
    MarketRelevanceScoreType,
    MarketRule,
    MarketRuleLink,
    MarketUpdate,
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
        content=json.dumps(comment_data["content"]),

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


def get_market_prob_at_time(session: Session, market_id: str, query_time: datetime) -> float:
    """
    Retrieve the most recent probability for a given market at or before the specified query time.

    Args:
        session (Session): SQLAlchemy session used to query the database.
        market_id (str): Unique identifier for the market to query.
        query_time (datetime): Timestamp representing the cutoff for bet history.

    Returns:
        float or None: The `prob_after` value from the latest bet before or at query_time.
            Returns None if no such bet exists.

    """
    bet = session.exec(
        select(MarketBet)
        .where(MarketBet.contract_id == market_id)
        .where(MarketBet.created_time <= query_time)
        .order_by(MarketBet.created_time.desc())
        .limit(1)
    ).first()
    return bet.prob_after if bet else None


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

def stringify_node(node: RuleNode, session: Session):
    """Get a string representation of the given rule."""
    if node.type == 'literal':

        market = session.exec(
            select(Market).where(Market.id == node.name.strip('"').strip("'"))
        ).first()

        return market.question if market else f"[Unknown: {node.name}]"

    elif node.type in ['and', 'or']:
        children = [stringify_node(child, session) for child in node.children]
        joined = f" {node.type.upper()} ".join(f"({child})" for child in children)
        return joined
    elif node.type == 'not':
        child_str = stringify_node(node.child, session)
        return f"NOT ({child_str})"

    else:
        return "[Unknown node type]"


def extract_literals_from_rule(rule: RuleNode) -> list[str]:
    """
    Extract all literal names from a logical rule, stripping surrounding quotes.

    Args:
        rule: A LogicalRule instance containing the rule tree

    Returns:
        List of cleaned literal names found in the rule

    """
    literals = []

    def traverse(node: RuleNode):
        if node.type == "literal":
            # Strip leading/trailing quotes (single or double)
            cleaned_name = node.name.strip('"').strip("'")
            literals.append(cleaned_name)
        elif node.type in ("and", "or"):
            for child in node.children:
                traverse(child)
        elif node.type == "not":
            traverse(node.child)

    traverse(rule.rule if hasattr(rule, "rule") else rule)
    return literals


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
    # Only process markets that are new or updated since last classification
    with Session(context.resources.database_engine) as session:
        markets_to_process = session.exec(
            select(Market)
            .outerjoin(MarketUpdate, Market.id == MarketUpdate.market_id)
            .where(
                # Never classified or market updated since last classification
                (MarketUpdate.classified_at.is_(None)) |
                (Market.updated_at > MarketUpdate.classified_at)
            )
        ).all()

        num_markets_processed = len(markets_to_process)
        context.log.info(
            f"Found {num_markets_processed} markets needing classification."
        )

    # Apply classification logic
    classification_results = []
    h5n1_classifier = H5N1Classifier()
    for m in markets_to_process:
        context.log.debug(f"Processing market: {m.id} - {m.question}")
        classification_results.append(
            (m.id, h5n1_classifier.predict(m))
        )
    num_markets_h5n1 = sum(1 for _, is_h5n1 in classification_results if is_h5n1)

    # Obtain id of h5n1 label type
    with Session(context.resources.database_engine) as session:
        result = session.exec(
            select(MarketLabelType).where(MarketLabelType.label_name == "h5n1")
        ).first()
        h5n1_label_type_id = result.id
        context.log.debug(f"Found H5N1 label type ID: {h5n1_label_type_id}")

    # Update classification results in the database
    current_time = datetime.now(UTC)
    with Session(context.resources.database_engine) as session:
        for (market_id, is_h5n1) in classification_results:
            # Update market labels
            existing_label = session.exec(
                select(MarketLabel).where(
                    (MarketLabel.market_id == market_id) &
                    (MarketLabel.label_type_id == h5n1_label_type_id)
                )
            ).first()

            if is_h5n1 and not existing_label:
                # Add label if classified as h5n1 and label does not exist
                market_label = MarketLabel(
                    market_id=market_id,
                    label_type_id=h5n1_label_type_id
                )
                session.add(market_label)
            elif not is_h5n1 and existing_label:
                # Remove label if not classified as h5n1 and label exists
                session.delete(existing_label)

            # Track this market was classified
            existing_update = session.exec(
                select(MarketUpdate).where(MarketUpdate.market_id == market_id)
            ).first()

            if existing_update:
                # Update classified_at field
                existing_update.classified_at = current_time
            else:
                # Create new MarketUpdate with classified_at timestamp
                market_update = MarketUpdate(
                    market_id=market_id,
                    classified_at=current_time
                )
                session.add(market_update)

            session.commit()


    return dg.MaterializeResult(
        metadata={
            "num_markets_processed": dg.MetadataValue.int(num_markets_processed),
            "num_markets_h5n1": dg.MetadataValue.int(num_markets_h5n1),
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

    # get list of market ids that have been labeled H5N1
    with Session(context.resources.database_engine) as session:
        h5n1_market_ids = session.exec(
            select(MarketLabel.market_id)
            .join(MarketLabelType, MarketLabel.label_type_id == MarketLabelType.id)
            .where(MarketLabelType.label_name == "h5n1")
        ).all()
    context.log.info(f"Found {len(h5n1_market_ids)} h5n1 labeled markets.")
    total_comments_inserted = 0
    total_bets_updated = 0

    for m in h5n1_market_ids:
        # get description and text rep of the market
        full_market = manifold_client.full_market(m)
        with Session(context.resources.database_engine) as session:
            market = session.exec(
                select(Market).where(Market.id == m)
            ).first()
            market.description = json.dumps(full_market.get("description"))

            text_rep = get_text_rep(market)
            market.text_rep = text_rep
            session.commit()

        # comments
        all_comments = []
        before = None
        while True:
            batch = manifold_client.comments(m, limit=1000, before=before)
            if not batch:
                break
            all_comments.extend(batch)
            if len(batch) < 1000:
                break
            before = batch[-1]["id"]

        # delete old and bulk-insert fresh
        with Session(context.resources.database_engine) as session:
            # delete existing comments
            session.exec(
                delete(MarketComment)
                .where(MarketComment.market_id == m)
            )
            # prepare & bulk-save new comments
            objs = [_prepare_comment(cdata) for cdata in all_comments]
            session.bulk_save_objects(objs)

            # commit both delete + insert together
            session.commit()
        total_comments_inserted += len(all_comments)
        context.log.info(
            f"Market {m}: found {len(all_comments)} comments."
        )


        # bets: we want all bets that were unfilled, or are new
        all_bets= []
        before = None
        while True:
            batch = manifold_client.bets(m, limit=1000, before=before)
            if not batch:
                break
            all_bets.extend(batch)
            if len(batch) < 1000:
                break
            before = batch[-1]["id"]

        with Session(context.resources.database_engine) as session:
            # get (id, is_filled) for existing bets
            rows = session.exec(
                select(MarketBet.id, MarketBet.is_filled)
                .where(MarketBet.contract_id == m)
            ).all()
            existing_ids = {row[0] for row in rows}
            unfilled_ids = [row[0] for row in rows if not row[1]]

            # delete only the unfilled bets
            if unfilled_ids:
                session.exec(
                    delete(MarketBet)
                    .where(MarketBet.id.in_(unfilled_ids))
                )

            # get the bets that aren't in the db
            to_insert = [
                _prepare_bet(b)
                for b in all_bets
                if b["id"] not in existing_ids
            ]

            # bulk‐insert the new + formerly-unfilled bets
            session.add_all(to_insert)
            session.commit()

        total_bets_updated += len(all_bets)

        with Session(context.resources.database_engine) as session:
            # update the time for full market at
            row = session.get(MarketUpdate, m)
            row.full_market_at = datetime.now(UTC)
            session.commit()

        context.log.info(
            f"Market {m}: updated {len(all_bets)} bets."
        )

    return dg.MaterializeResult(
        metadata={
            "num_markets_processed": dg.MetadataValue.int(len(h5n1_market_ids)),
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
        market_ids = session.exec(select(MarketLabel.market_id)).all()
    context.log.info(f"Found {len(market_ids)} labeled markets.")

    # Load all score‐type rows once and build a name id map
    with Session(context.resources.database_engine) as session:
        score_type_rows = session.exec(select(MarketRelevanceScoreType)).all()
    score_type_map = { row.score_name: row.id for row in score_type_rows }
    context.log.info(f"Found {len(score_type_map)} relevance label types.")

    with locked_session(context.resources.database_engine) as session:
        for market_id in market_ids:
            # delete selected score types
            types_to_delete = ["volume_24h", "num_comments", "volume_total",
                               "volume_144h", "num_traders"]
            type_ids = [score_type_map[name] for name in types_to_delete if name in score_type_map]
            row = session.get(MarketUpdate, market_id)
            row.market_data_relevances_recorded_at = datetime.now(UTC)
            session.exec(
                delete(MarketRelevanceScore)
                .where(MarketRelevanceScore.market_id == market_id)
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
                .where(MarketBet.contract_id == market_id)).one_or_none() | 0

            # num_comments
            num_comments = session.exec(select(func.count())
                .where(MarketComment.market_id == market_id)).one_or_none() | 0

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
                        score_type_id=type_id,
                        score_value=val,
                    )
                )

            session.add_all(to_insert)
            context.log.info(f"Scored market: {market_id}.")

            session.commit()

        return dg.MaterializeResult(
            metadata={
                "num_markets_processed": dg.MetadataValue.int(len(market_ids))
            }
        )


@dg.asset(
    deps=[manifold_full_markets],
    required_resource_keys={"database_engine"},
    description="Temporal relevance scores for the labeled markets.",
)
def relevance_temporal(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Use client.py prompting to populate temporal relevance scores.

    This asset looks at the labels table and for all labeled markets,
    it generates temporal relevance scores and stores them in the database.
    """
    # get list of market ids that have been labeled
    with Session(context.resources.database_engine) as session:
        h5n1_result = session.exec(
            select(MarketLabelType).where(MarketLabelType.label_name == "h5n1")
        ).first()
        market_ids = session.exec(
            select(MarketLabel.market_id).where(
                MarketLabel.label_type_id == h5n1_result.id
            )
        ).all()
        label_for_temp = session.exec(
            select(MarketRelevanceScoreType.id).where(
                MarketRelevanceScoreType.score_name == "temporal_relevance"
            )
        ).first()

    context.log.info(f"Found {len(market_ids)} labeled markets.")

    scores_to_add = []
    market_ids_to_delete = []

    for market_id in market_ids:
        with Session(context.resources.database_engine) as session:
            market_label = session.exec(
                select(MarketLabel).where(MarketLabel.market_id == market_id)
            ).first()
            label_name = market_label.label_type.label_name

            market = session.exec(
                select(Market).where(Market.id == market_id)
            ).first()

        # use the disease information class from the ranker file for structured info
        # for prompting
        # Get the Market label name
        todays_date = datetime.now(UTC)
        disease_info = DiseaseInformation(
            disease=label_name,
            date=todays_date,
            overall_index_question="Will there be a massive H5N1 outbreak in the next 12 months?"
        )
        prompt_temp = get_prompt("temporal_relevance_prompt.j2", disease_info)
        client = get_client()
        reasonings, scores, average_score = get_relevance(prompt_temp, market.text_rep, client)
        # Collect for batch write
        scores_to_add.append(MarketRelevanceScore(
            market_id=market_id,
            score_type_id=label_for_temp,
            score_value=average_score,
            chain_of_thoughts=str(reasonings),
            scores=str(scores),
        ))
        market_ids_to_delete.append(market_id)
        context.log.info(f"Temporal relevance computed for market: {market_id}.")

    with locked_session(context.resources.database_engine) as session:
        for m in market_ids_to_delete:
            row = session.get(MarketUpdate, m)
            row.temp_relevance_scored_at = datetime.now(UTC)

        session.exec(
            delete(MarketRelevanceScore)
            .where(MarketRelevanceScore.market_id.in_(market_ids_to_delete))
            .where(MarketRelevanceScore.score_type_id == label_for_temp)
        )
        session.add_all(scores_to_add)
        session.commit()

    context.log.info("All temporal relevance scores updated in DB.")

    return dg.MaterializeResult(
        metadata={"num_markets_processed": dg.MetadataValue.int(len(market_ids))}
    )


@dg.asset(
    deps=[manifold_full_markets],
    required_resource_keys={"database_engine"},
    description="Geographical relevance scores for the labeled markets.",
)
def relevance_geographical(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Use client.py prompting to populate geographical relevance scores.

    This asset looks at the labels table and for all labeled markets,
    it generates geographical relevance scores and stores them in the database.
    """
    # get list of market ids that have been labeled
    with Session(context.resources.database_engine) as session:
        h5n1_result = session.exec(
            select(MarketLabelType).where(MarketLabelType.label_name == "h5n1")
        ).first()
        market_ids = session.exec(
            select(MarketLabel.market_id).where(
                MarketLabel.label_type_id == h5n1_result.id
            )
        ).all()
        label_for_geo = session.exec(
            select(MarketRelevanceScoreType.id).where(
                MarketRelevanceScoreType.score_name == "geographical_relevance"
            )
        ).first()

    context.log.info(f"Found {len(market_ids)} labeled markets.")

    scores_to_add = []
    market_ids_to_delete = []

    for market_id in market_ids:
        with Session(context.resources.database_engine) as session:
            market_label = session.exec(
                select(MarketLabel).where(MarketLabel.market_id == market_id)
            ).first()
            label_name = market_label.label_type.label_name

            market = session.exec(
                select(Market).where(Market.id == market_id)
            ).first()

        # use the disease information class from the ranker file for structured info
        # for prompting
        # Get the Market label name
        todays_date = datetime.now(UTC)
        disease_info = DiseaseInformation(
            disease=label_name,
            date=todays_date,
            overall_index_question="Will there be a massive H5N1 outbreak in the next 12 months?"
        )
        prompt_temp = get_prompt("geographic_relevance_prompt.j2", disease_info)
        client = get_client()
        reasonings, scores, average_score = get_relevance(prompt_temp, market.text_rep, client)

        # Collect for batch write
        scores_to_add.append(MarketRelevanceScore(
            market_id=market_id,
            score_type_id=label_for_geo,
            score_value=average_score,
            chain_of_thoughts=str(reasonings),
            scores=str(scores),
        ))
        market_ids_to_delete.append(market_id)
        context.log.info(f"Geographical relevance computed for market: {market_id}.")

    with locked_session(context.resources.database_engine) as session:
        for m in market_ids_to_delete:
            row = session.get(MarketUpdate, m)
            row.geo_relevance_scored_at = datetime.now(UTC)

        session.exec(
            delete(MarketRelevanceScore)
            .where(MarketRelevanceScore.market_id.in_(market_ids_to_delete))
            .where(MarketRelevanceScore.score_type_id == label_for_geo)
        )
        session.add_all(scores_to_add)
        session.commit()

    context.log.info("All geographical relevance scores updated in DB.")

    return dg.MaterializeResult(
        metadata={"num_markets_processed": dg.MetadataValue.int(len(market_ids))}
    )


@dg.asset(
    deps=[manifold_full_markets],
    required_resource_keys={"database_engine"},
    description="Index question relevance scores for the labeled markets.",
)
def relevance_index_question(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """
    Use client.py prompting to populate index question relevance scores.

    This asset looks at the labels table and for all labeled markets,
    it generates index question relevance scores and stores them in the database.
    """
    # get list of market ids that have been labeled
    with Session(context.resources.database_engine) as session:
        h5n1_result = session.exec(
            select(MarketLabelType).where(MarketLabelType.label_name == "h5n1")
        ).first()
        market_ids = session.exec(
            select(MarketLabel.market_id).where(
                MarketLabel.label_type_id == h5n1_result.id
            )
        ).all()
        label_for_index = session.exec(
            select(MarketRelevanceScoreType.id).where(
                MarketRelevanceScoreType.score_name == "index_question_relevance"
            )
        ).first()

    context.log.info(f"Found {len(market_ids)} labeled markets.")

    scores_to_add = []
    market_ids_to_delete = []

    for market_id in market_ids:
        with Session(context.resources.database_engine) as session:
            market_label = session.exec(
                select(MarketLabel).where(MarketLabel.market_id == market_id)
            ).first()
            label_name = market_label.label_type.label_name

            market = session.exec(
                select(Market).where(Market.id == market_id)
            ).first()

        # use the disease information class from the ranker file for structured info
        # for prompting
        # Get the Market label name
        todays_date = datetime.now(UTC)
        disease_info = DiseaseInformation(
            disease=label_name,
            date=todays_date,
            overall_index_question="Will there be a massive H5N1 outbreak in the next 12 months?"
        )
        prompt_temp = get_prompt("index_question_relevance_prompt.j2", disease_info)
        client = get_client()
        reasonings, scores, average_score = get_relevance(prompt_temp, market.text_rep, client)
        # Collect for batch write
        scores_to_add.append(MarketRelevanceScore(
            market_id=market_id,
            score_type_id=label_for_index,
            score_value=average_score,
            chain_of_thoughts=str(reasonings),
            scores=str(scores),
        ))
        market_ids_to_delete.append(market_id)
        context.log.info(f"Index question relevance computed for market: {market_id}.")


    with locked_session(context.resources.database_engine) as session:
        for m in market_ids_to_delete:
            row = session.get(MarketUpdate, m)
            row.index_question_relevance_scored_at = datetime.now(UTC)

        session.exec(
            delete(MarketRelevanceScore)
            .where(MarketRelevanceScore.market_id.in_(market_ids_to_delete))
            .where(MarketRelevanceScore.score_type_id == label_for_index)
        )
        session.add_all(scores_to_add)
        session.commit()

    context.log.info("All index question relevance scores updated in DB.")

    return dg.MaterializeResult(
        metadata={"num_markets_processed": dg.MetadataValue.int(len(market_ids))}
    )



@dg.asset(
    deps=[relevance_index_question, relevance_geographical, relevance_temporal],
    required_resource_keys={"database_engine"},
    description="Market eligibility labels."
)
def market_rule_eligibility_labels(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """Generate new market rule eligibility labels when materialized."""
    # Obtain id of h5n1 label type and eligible label type
    with Session(context.resources.database_engine) as session:
        h5n1_result = session.exec(
            select(MarketLabelType).where(MarketLabelType.label_name == "h5n1")
        ).first()
        rule_eligible_result = session.exec(
            select(MarketLabelType).where(MarketLabelType.label_name == "rule_eligible")
        ).first()

        h5n1_label_type_id = h5n1_result.id
        rule_eligible_label_type_id = rule_eligible_result.id

        # Only process markets that are labeled h5n1
        markets_to_process = session.exec(
            select(Market)
            .outerjoin(MarketLabel, Market.id == MarketLabel.market_id)
            .where(
                MarketLabel.label_type_id == h5n1_label_type_id
            )
        ).all()

        num_markets_processed = len(markets_to_process)
        context.log.info(
            f"Found {num_markets_processed} h5n1 markets needing eligibility classification."
        )

        score_type_ids = {}
        score_names = ["temporal_relevance", "geographical_relevance",
                       "index_question_relevance", "volume_total"]

        for score_name in score_names:
            result = session.exec(
                select(MarketRelevanceScoreType)
                .where(MarketRelevanceScoreType.score_name == score_name)
            ).first()

            if result:
                score_type_ids[score_name] = result.id
            else:
                context.log.warning(f"Missing score type for {score_name}")

    temporal_score_type_id = score_type_ids["temporal_relevance"]
    geo_score_type_id = score_type_ids["geographical_relevance"]
    question_score_type_id = score_type_ids["index_question_relevance"]
    volume_score_type_id = score_type_ids["volume_total"]

    # Fetch all relevance scores for these markets
    market_ids = [m.id for m in markets_to_process]
    score_type_ids_set = {
        temporal_score_type_id,
        geo_score_type_id,
        question_score_type_id,
        volume_score_type_id,
    }

    with Session(context.resources.database_engine) as session:
        all_scores = session.exec(
            select(MarketRelevanceScore)
            .where(
                (MarketRelevanceScore.market_id.in_(market_ids)) &
                (MarketRelevanceScore.score_type_id.in_(score_type_ids_set))
            )
        ).all()

    # Organize as a lookup table: (market_id, score_type_id) -> score_value
    score_lookup = {
        (score.market_id, score.score_type_id): score.score_value
        for score in all_scores
    }

    # Apply classification logic
    classification_results = []
    eligibility_classifier = RuleEligibilityClassifier()
    # this threshold is determined using calibration city
    # for manifold volume under 400 mana is not calibrated enough
    volume_threshold = 400
    num_skipped = 0
    for m in markets_to_process:
        with Session(context.resources.database_engine) as session:
            row = session.get(MarketUpdate, m.id)
            row.rule_eligibility_at = datetime.now(UTC)
            session.commit()

        temp_score = score_lookup.get((m.id, temporal_score_type_id))
        geo_score = score_lookup.get((m.id, geo_score_type_id))
        question_score = score_lookup.get((m.id, question_score_type_id))
        volume_score = score_lookup.get((m.id, volume_score_type_id))

        # Check binary and volume threshold first
        if m.outcome_type != "BINARY":
            context.log.info(f"Market {m.id} is NOT BINARY; marking as not eligible.")
            classification_results.append((m.id, False))
            continue

        if volume_score is None:
            context.log.warning(f"Missing volume score for market {m.id}; skipping classification.")
            num_skipped += 1
            continue

        if volume_score < volume_threshold:
            context.log.info(f"Market {m.id} has low volume ({volume_score}); not eligible.")
            classification_results.append((m.id, False))
            continue

        # Skip if relevance scores are missing
        if None in (temp_score, geo_score, question_score):
            context.log.warning(f"Missing score(s) for market {m.id}; skipping classification.")
            num_skipped += 1
            continue


        prediction = eligibility_classifier.predict(temp_score, geo_score, question_score)
        classification_results.append((m.id, prediction))

    num_markets_eligible = sum(1 for _, is_eligible in classification_results if is_eligible)

    # Update classification results in the database
    with Session(context.resources.database_engine) as session:
        for (market_id, is_eligible) in classification_results:
            # Update market labels
            existing_label = session.exec(
                select(MarketLabel).where(
                    (MarketLabel.market_id == market_id) &
                    (MarketLabel.label_type_id == rule_eligible_label_type_id)
                )
            ).first()

            if is_eligible and not existing_label:
                # Add label if classified as eligible and label does not exist
                market_label = MarketLabel(
                    market_id=market_id,
                    label_type_id=rule_eligible_label_type_id
                )
                session.add(market_label)
            elif not is_eligible and existing_label:
                # Remove label if not classified as eligible and label exists
                session.delete(existing_label)

        session.commit()

    return dg.MaterializeResult(
        metadata={
            "num_markets_processed": dg.MetadataValue.int(num_markets_processed),
            "num_markets_eligible": dg.MetadataValue.int(num_markets_eligible),
            "num_markets_skipped": dg.MetadataValue.int(num_skipped),
        }
    )


@dg.asset(
    deps=[market_rule_eligibility_labels],
    required_resource_keys={"database_engine"},
    description="Rules from eligible markets."
)
def index_rules(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """Asset to generate and store logical market rules using LLM."""
    # Load markets that are eligible
    with Session(context.resources.database_engine) as session:
        rule_eligible_result = session.exec(
            select(MarketLabelType).where(MarketLabelType.label_name == "rule_eligible")
        ).first()
        if not rule_eligible_result:
            context.log.warning("No rule_eligible label type found.")
            return dg.MaterializeResult(metadata={})

        rule_eligible_label_type_id = rule_eligible_result.id

        markets_to_process = session.exec(
            select(Market)
            .join(MarketLabel, Market.id == MarketLabel.market_id)
            .where(MarketLabel.label_type_id == rule_eligible_label_type_id)
        ).all()

        if not markets_to_process:
            context.log.warning("No eligible markets found.")
            return dg.MaterializeResult(metadata={})

        # Build structured market data for prompt
        market_data = {}
        for market in markets_to_process:
            market_data[market.id] = {
                "question": market.question,
                "description": market.description or "",
                "text_rep": market.text_rep
            }
        
        # Prepare prompt with proper context
        prompt_data = PromptInformation(
            disease="H5N1",
            date=datetime.now(UTC),
            overall_index_question="Will there be a massive H5N1 outbreak in the next 12 months?",
            num_of_rules=30
        )
        
        context.log.info(f"Generating rules for {len(market_data)} eligible markets")
        prompt = get_rules_prompt(
            "rule_gen_prompt.j2", 
            prompt_data, 
            market_data
        )

        # Generate rules using LLM with validation and chunking
        client = get_client()
        valid_market_ids = set(market_data.keys())
        
        try:
            logical_rules = get_rules(prompt, valid_market_ids, client)
            context.log.info(f"Successfully generated {len(logical_rules)} rules")
        except Exception as e:
            context.log.error(f"Rule generation failed: {e}")
            # Return empty result rather than crashing the entire pipeline
            return dg.MaterializeResult(
                metadata={
                    "num_rules": dg.MetadataValue.int(0),
                    "num_markets_used": dg.MetadataValue.int(0),
                    "error": dg.MetadataValue.text(str(e))
                }
            )

        used_markets = []

        for rule_obj in logical_rules:
            rule_json = rule_obj.rule.model_dump_json()
            readable = stringify_node(rule_obj.rule, session)

            # Get disease information for prompts
            disease_info = DiseaseInformation(
                disease="H5N1",
                date=datetime.now(UTC),
                overall_index_question=
                "Will there be a massive H5N1 outbreak in the next 12 months?"
            )

            # Get strength score using rule strength prompt
            strength_prompt = get_prompt("rule_strength_score_prompt.j2", disease_info)
            client = get_client()
            strength_reasonings, strength_scores, strength_avg = get_weight(
                strength_prompt, readable, client
            )

            # Get relevance score - need to collect market metrics for this rule
            rule_market_ids = extract_literals_from_rule(rule_obj.rule)
            
            # Collect all 8 metrics for markets in this rule
            market_metrics = {}
            for market_id in rule_market_ids:
                # Get all relevance scores for this market
                market_scores = session.exec(
                    select(MarketRelevanceScore)
                    .where(MarketRelevanceScore.market_id == market_id)
                ).all()
                
                # Organize scores by type
                scores_dict = {}
                for score in market_scores:
                    score_type_name = score.score_type.score_name
                    scores_dict[score_type_name] = score.score_value
                
                market_metrics[market_id] = scores_dict

            # Format market metrics for the relevance prompt
            metrics_text = f"Rule: {readable}\n\nMarket Metrics:\n"
            for market_id, metrics in market_metrics.items():
                market_obj = session.exec(select(Market).where(Market.id == market_id)).first()
                market_question = market_obj.question if market_obj else "Unknown market"
                metrics_text += f"\nMarket ID: {market_id}\nQuestion: {market_question}\n"
                for metric_name, value in metrics.items():
                    metrics_text += f"  {metric_name}: {value}\n"

            # Get relevance score using rule relevance prompt
            relevance_prompt = get_prompt("rule_relevance_score_prompt.j2", disease_info)
            relevance_reasonings, relevance_scores, relevance_avg = get_relevance(
                relevance_prompt, metrics_text, client
            )

            new_rule = MarketRule(
                rule=rule_json,
                readable_rule=readable,
                created_at=datetime.now(UTC),
                chain_of_thoughts=rule_obj.reasoning,
                strength_weight=strength_avg,
                relevance_weight=relevance_avg,
                strength_chain=str(strength_reasonings),
                relevance_chain=str(relevance_reasonings),
                strength_scores=str(strength_scores),
                relevance_scores=str(relevance_scores),
            )
            session.add(new_rule)
            session.flush()  # Ensures new_rule.id is available
            context.log.info(f"Successfully scored new rule: {new_rule}")

            for market_id in rule_market_ids:
                if market_id not in used_markets:
                    used_markets.append(market_id)

                # The rule contains actual market IDs, not text representations
                market_obj = session.exec(
                    select(Market).where(Market.id == market_id)
                ).first()

                if market_obj:
                    new_link = MarketRuleLink(
                        market_id=market_obj.id,
                        rule_id=new_rule.id,
                    )
                    session.add(new_link)
                else:
                    context.log.warning(f"Market not found for ID: {market_id}")

        session.commit()

    context.log.info(f"Generated and stored {len(logical_rules)} market rules.")
    return dg.MaterializeResult(
        metadata={
            "num_rules": dg.MetadataValue.int(len(logical_rules)),
            "num_markets_used": dg.MetadataValue.int(len(used_markets)),
        }
    )

