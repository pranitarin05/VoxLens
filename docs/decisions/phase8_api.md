# Phase 8: serving API

- Models are loaded once in the FastAPI lifespan; requests only call them.
- `api_requests` gets one row per scored text (best effort; a DB outage degrades
  `/health` but does not fail prediction).
- Stats use ground-truth `reviews.label`; the `date_trunc` unit is whitelisted
  and every filter is a psycopg parameter.
- `/v1/ask` returns 501 until Phase 9.
- Measured single-request latency (including spaCy aspects): p50 222.977 ms,
  p95 239.986 ms, p99 248.929 ms; 0/200 errors. See `results/api_load_test.md`.
- CI fix: pgvector service + `DATABASE_URL`, lazy MLflow imports, and skips for
  tests that need torch/transformers/spaCy. Clean `.[dev]` run: 62 passed.
