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

To create a manual migration, first generate an empty file with proper references by running:

```bash
alembic revision -m "Your description of the changes"
```

Then, edit the generated file in the `migrations/versions` directory to add the changes. Then run:

```bash
alembic upgrade head  # Apply the latest migration
``` 