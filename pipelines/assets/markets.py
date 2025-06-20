"""
Market assets.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""
from datetime import UTC, datetime

import dagster as dg
from sqlmodel import Session, select

from ml.classification import H5N1Classifier
from models.markets import Market, MarketLabel, MarketLabelType


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
    # Select all markets from the database
    with Session(context.resources.database_engine) as session:
        markets = session.exec(select(Market)).all()
        num_markets_processed = len(markets)
        context.log.info(
            f"Fetched {num_markets_processed} markets from the database."
        )

    # Apply classification logic
    classification_results = []
    h5n1_classifier = H5N1Classifier()
    for m in markets:
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
    with Session(context.resources.database_engine) as session:
        for (market_id, is_h5n1) in classification_results:
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
            session.commit()

    return dg.MaterializeResult(
        metadata={
            "num_markets_processed": dg.MetadataValue.int(num_markets_processed),
            "num_markets_h5n1": dg.MetadataValue.int(num_markets_h5n1),
        }
    )
