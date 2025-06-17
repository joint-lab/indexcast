# Migrations

Alembic migrations are used to manage changes to the database schema over time.
It is configured to track changes in the a sqlite database, `indexcast.db`, stored in the root of the project.


# Usage

From the *root* of the project, you can run the following commands:
```bash
alembic revision --autogenerate -m "Your description of the changes"
alembic upgrade head  # Apply the latest migration
```
