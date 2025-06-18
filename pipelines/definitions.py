"""
Dagster pipelines definitions.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""
import dagster as dg

from pipelines.assets import markets
from pipelines.resources.api import manifold_api_resource
from pipelines.resources.db import sqlite_db_resource

markets_assets = dg.load_assets_from_modules([markets])
market_job = dg.define_asset_job(
            name="update_markets_job",
            selection="manifold_markets+"
)

defs = dg.Definitions(
    assets=[*markets_assets],
    resources={
        "database_engine": sqlite_db_resource,
        "manifold_client": manifold_api_resource
    },
    jobs=[market_job]
)
