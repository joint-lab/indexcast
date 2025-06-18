"""
Market assets.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""
import dagster as dg
from sqlmodel import Session, select

from ml.classification import h5n1_classifier
from models.markets import Market, MarketLabel, MarketLabelType


@dg.asset(
    required_resource_keys={"database", "manifold_api"}
)
def markets(context: dg.AssetExecutionContext):
    """Fetch markets from the Manifold API and populate the Markets table in the DB."""
    markets_data = context.resources.manifold_api_resource.fetch_markets()  # Stub: returns []
    # TODO: Insert markets_data into the Markets table using db
    context.log.info(f"Fetched {len(markets_data)} markets (stub)")
    return markets_data

@dg.asset(
    deps=[markets],
    required_resource_keys={"database"},
    description="Market labels table."
)
def market_labels(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """Update market labels when materialized."""
    with context.resources.database() as engine:
        # Select all markets from the database
        with Session(engine) as session:
            markets = session.exec(select(Market)).all()
            num_markets_processed = len(markets)
            context.log.info(
                f"Fetched {len(num_markets_processed)} markets from the database."
            )

        # Apply classification logic
        classification_results = []
        for m in markets:
            context.log.debug(f"Processing market: {m.id} - {m.question}")
            classification_results.append(
                (m.id, h5n1_classifier(m))
            )
        num_markets_h5n1 = sum([1 for _, is_h5n1 in classification_results if is_h5n1])

        # Obtain id of h5n1 label type
        with Session(engine) as session:
            result = session.exec(
                select(MarketLabelType).where(MarketLabelType.label_name == "h5n1")
            ).first()
            h5n1_label_type_id = result.id
            context.log.debug(f"Found H5N1 label type ID: {h5n1_label_type_id}")

        # Update classification results in the database
        with Session(engine) as session:
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
