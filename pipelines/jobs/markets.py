"""
Dagster job and schedule for updating markets and market labels.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""
import dagster as dg


def should_execute_markets_update(context):
    """
    Check if we should execute the markets update.

    Skip if there are any active runs of the same job.
    """
    instance = context.instance
    active_runs = instance.get_runs(
        filters=dg.RunsFilter(
            job_name="update_markets_job",
            statuses=[
                dg.DagsterRunStatus.STARTED,
                dg.DagsterRunStatus.STARTING,
                dg.DagsterRunStatus.QUEUED,
            ]
        ),
        limit=1
    )

    if active_runs:
        return dg.SkipReason(
            f"Skipping markets update - found {len(active_runs)} active run(s)"
        )

    return True

update_markets_job = dg.define_asset_job(  # pylint: disable=E1111
    name="update_markets_job",
    selection="manifold_markets+"
)

update_markets_schedule = dg.ScheduleDefinition(
    job=update_markets_job,
    cron_schedule="0 * * * *",  # every hour
    execution_timezone="UTC",
    should_execute=should_execute_markets_update,
)
