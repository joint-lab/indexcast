# Migrations

Alembic migrations are used to manage changes to the database schema over time.
It is configured to track changes in the a sqlite database, `indexcast.db`, stored in the root of the project.


# Usage

To upgrade the database schema to the latest version (including a fresh database creation), run the following command from the *root* of the project:

```bash
alembic upgrade head
```


To auto-generate a migrations once changes are made to the model, run the following commands (again, from the *root* of the project):

```bash
alembic revision --autogenerate -m "Your description of the changes"
alembic upgrade head  # Apply the latest migration
```

