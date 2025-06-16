# IndexCast backend

IndexCast takes predictions from different prediction markets about the same world event and creates one number---an index---showing the overall trend. 
We focus on health emergencies and major global events.

# Project overview

* `migrations`: Alembic migrations for the database.
* `ml`: Machine learning models for classification, reranking, index creation, etc.
* `models`: SQL models for the database.
* `pipelines`: Dagster pipelines for data scrapping and processing.

# Getting started

```bash
git@github.com:joint-lab/indexcast-backend.git
cd indexcast-backend
uv sync
```
