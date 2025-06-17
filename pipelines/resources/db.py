"""
Database resources for Dagster pipelines.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ernold@uvm.edu>
"""
import os

import dagster as dg
import sqlmodel


@dg.resource
def sqlite_db_resource(context: dg.ResourceContext) -> sqlmodel.engine.Engine:
    """
    Create a resource for connecting to the SQLite database.
    
    Note: The database path is read from the environment variable `SQLITE_DB_PATH`.
    If the variable is not set, it defaults to 'indexcast.db'.
    """
    # Get database path from the environment variable
    db_path = os.getenv('SQLITE_DB_PATH', 'indexcast.db')

    # Connect to the SQLite DB
    engine = sqlmodel.create_engine(f'sqlite:///{db_path}')
    return engine
