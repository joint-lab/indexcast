"""
Dagster job and schedule for updating markets and market labels.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""
import dagster as dg

# Use the Definitions object to get the job and resources
update_markets_job = dg.define_asset_job(
    name="update_markets_job",
    selection="manifold_markets+"
)

update_markets_schedule = dg.ScheduleDefinition(
    job=update_markets_job,
    cron_schedule="0 * * * *",  # every hour
    execution_timezone="UTC",
)
