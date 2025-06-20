"""
Dagster pipelines definitions.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""
import dagster as dg

from pipelines.assets import markets
from pipelines.jobs.markets import update_markets_job, update_markets_schedule
from pipelines.resources.api import manifold_api_resource
from pipelines.resources.db import sqlite_db_resource

markets_assets = dg.load_assets_from_modules([markets])

# Collect all jobs from update_markets module that end with '_job'
jobs = [getattr(markets, name) for name in dir(markets) if name.endswith('_job')]

defs = dg.Definitions(
    assets=[*markets_assets],
    resources={
        "database_engine": sqlite_db_resource,
        "manifold_client": manifold_api_resource
    },
    jobs=[update_markets_job],
    schedules=[update_markets_schedule]
)
