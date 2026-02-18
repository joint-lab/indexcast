# IndexCast backend

IndexCast takes predictions from different prediction markets about the same world event and creates one number---an index---showing the overall trend. 
We focus on health emergencies and major global events.

# Project overview

* `migrations`: Alembic migrations for the database.
* `ml`: Machine learning models for classification, reranking, index creation, etc.
* `models`: SQL models for the database.
* `pipelines`: Dagster pipelines for data scrapping and processing.
* `notebooks`: Python notebooks used for testing some methodology of this project.

# Getting started

```bash
git@github.com:joint-lab/indexcast-backend.git
cd indexcast-backend
uv sync
```

To setup environment variables, copy `example_configuration.sh` to `configuration.sh` and fill in the values, then export them:
```bash
cp example_configuration.sh configuration.sh
nano configuration.sh
source configuration.sh
```

# Acknowledgments

Work on this package was supported by the National Science Foundation under Award No. 2242829.
