"""
Dagster job and schedule for updating markets and market labels.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""
import dagster as dg

from pipelines.assets.market_labels import market_labels
from pipelines.assets.markets import markets
from pipelines.resources.api import manifold_api_resource
from pipelines.resources.db import sqlite_db_resource


@dg.job(resource_defs={
    "sqlite_db_resource": sqlite_db_resource,
    "manifold_api_resource": manifold_api_resource,
})
def update_markets_job():
    """Job to update markets and market labels."""
    labels = market_labels(markets())
    # do something with labels, e.g., store them in the database
    return labels

# Schedule: run hourly
update_markets_schedule = dg.ScheduleDefinition(
    job=update_markets_job,
    cron_schedule="0 * * * *",  # every hour
)
