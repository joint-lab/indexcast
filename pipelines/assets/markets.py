"""
Market assets.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""
from datetime import UTC, datetime

import dagster as dg
from sqlalchemy import delete
from sqlmodel import Session, select
import json

from ml.classification import H5N1Classifier
from models.markets import (
    Market,
    MarketBet,
    MarketComment,
    MarketLabel,
    MarketLabelType,
    MarketUpdate,
)


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
    fills = json.dumps(bet_data.get("fills"))

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
        fills=fills,

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
                    # reranked_at will use default value (None)
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
    Fetch latest bets from the Manifold API and populate the market_bets table.

    This asset looks at the labels table, fetches all bets from the Manifold API for those
    markets, checks if they already exist in the database, and inserts new bets.
    It also respects the Manifold API rate limits and handles pagination.
    """
    # Manifold API client
    manifold_client = context.resources.manifold_client

    # get list of market ids that have been labeled
    with Session(context.resources.database_engine) as session:
        market_ids = session.exec(select(MarketLabel.market_id)).all()
    context.log.info(
        f"Found {len(market_ids)} labeled markets."
    )
    total_comments_inserted = 0
    total_bets_updated = 0

    for m in market_ids:
        # get description
        full_market = manifold_client.full_market(m)
        with Session(context.resources.database_engine) as session:
            market = session.exec(
                select(Market).where(Market.id == m)
            ).first()
            market.description = json.dumps(full_market.get("description"))
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
        context.log.info(
            f"Market {m}: updated {len(all_bets)} bets."
        )

    return dg.MaterializeResult(
        metadata={
            "num_markets_processed": dg.MetadataValue.int(len(market_ids)),
            "total_comments_inserted": dg.MetadataValue.int(total_comments_inserted),
            "total_bets_updated": dg.MetadataValue.int(total_bets_updated),
        }
    )