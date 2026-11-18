-- 0001_init_schema.sql
-- Full VoC schema (PRD 7.1). Idempotent: safe to re-run (IF NOT EXISTS).
-- Products first, then reviews (FK), then later-phase tables.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS products (
  product_id TEXT PRIMARY KEY,
  category   TEXT NOT NULL,
  title      TEXT
);

CREATE TABLE IF NOT EXISTS reviews (
  review_id   TEXT PRIMARY KEY,
  product_id  TEXT NOT NULL REFERENCES products(product_id),
  rating      SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
  label       TEXT NOT NULL CHECK (label IN ('negative','neutral','positive')),
  title       TEXT,
  text        TEXT NOT NULL,
  clean_text  TEXT NOT NULL,
  reviewed_at TIMESTAMPTZ,
  split       TEXT NOT NULL CHECK (split IN ('train','val','test')),
  ingested_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_reviews_product_time ON reviews(product_id, reviewed_at);
CREATE INDEX IF NOT EXISTS idx_reviews_split ON reviews(split);

CREATE TABLE IF NOT EXISTS predictions (
  id BIGSERIAL PRIMARY KEY,
  review_id TEXT REFERENCES reviews(review_id),
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  label TEXT NOT NULL,
  p_negative REAL, p_neutral REAL, p_positive REAL,
  scored_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (review_id, model_name, model_version)
);

CREATE TABLE IF NOT EXISTS aspects (
  id BIGSERIAL PRIMARY KEY,
  review_id TEXT REFERENCES reviews(review_id),
  aspect TEXT NOT NULL,
  sentiment TEXT NOT NULL,
  span_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_aspects_aspect ON aspects(aspect, sentiment);

CREATE TABLE IF NOT EXISTS review_embeddings (
  review_id TEXT PRIMARY KEY REFERENCES reviews(review_id),
  embedding vector(384)
);
-- HNSW index is created AFTER bulk embedding load (Phase 9), not here.

CREATE TABLE IF NOT EXISTS model_runs (
  id BIGSERIAL PRIMARY KEY,
  mlflow_run_id TEXT, model_name TEXT, dataset TEXT,
  accuracy REAL, macro_f1 REAL, p50_ms REAL, p95_ms REAL,
  cost_per_1k_usd REAL, created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS drift_reports (
  id BIGSERIAL PRIMARY KEY,
  window_start TIMESTAMPTZ, window_end TIMESTAMPTZ,
  drift_detected BOOLEAN, details JSONB, report_path TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS api_requests (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ DEFAULT now(),
  endpoint TEXT, model_name TEXT, latency_ms REAL, label TEXT, text_len INT
);

-- Schema version bookkeeping for the migration runner.
CREATE TABLE IF NOT EXISTS schema_migrations (
  filename TEXT PRIMARY KEY,
  applied_at TIMESTAMPTZ DEFAULT now()
);
