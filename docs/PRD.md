# Product Requirements Document: Voice-of-Customer (VoC) Intelligence Platform

Version 1.0. Status: approved for build.

## 1. Summary

The VoC Intelligence Platform ingests large volumes of customer reviews, classifies sentiment, extracts product aspects with aspect-level sentiment, stores everything in PostgreSQL, serves predictions through a FastAPI service, answers natural-language business questions through a retrieval-augmented LLM layer, tracks experiments in MLflow, is orchestrated by Airflow, and monitors itself for data drift. A Streamlit dashboard presents the results. Version 1 runs entirely locally with Docker Compose. Cloud deployment is out of scope for v1 but the design must not block it.

The project exists to demonstrate the full ML lifecycle (data analysis, model development, deployment-ready serving, integration, monitoring) for an AI/NLP engineering role. Every result shown must be reproducible from the repo.

## 2. Goals and non-goals

### Goals
- G1: A working pipeline from raw reviews (at least 1 million rows) to served predictions.
- G2: A fair, reproducible comparison of four sentiment approaches on accuracy, macro-F1, latency and cost.
- G3: Aspect-level insight: which product aspects drive negative sentiment.
- G4: Natural-language Q&A over the reviews with measured answer quality and hallucination rate.
- G5: A complete local MLOps loop: experiment tracking, model registry, orchestration, drift monitoring.
- G6: One-command reproducibility with tests and continuous integration.

### Non-goals for v1
- Cloud deployment (planned for a later stage).
- Authentication or multi-tenancy.
- Intent or urgency classification (dropped to keep scope realistic).
- Streaming ingestion (Kafka and similar).
- Fine-tuning of large language models.
- UI polish beyond a clean Streamlit app.

## 3. Personas

- P1 Business analyst: asks questions such as 'why did delivery complaints rise?', explores trends and aspects.
- P2 ML engineer: compares models, inspects experiments, checks drift.
- P3 Reviewer (interviewer or recruiter): must understand the value in two minutes from the README, the results table and a short demo.

## 4. Data

### 4.1 Source
Amazon Reviews 2023 (McAuley Lab). Choose 2 to 3 categories so the total is at least 1 million reviews. Fallback: Yelp Open Dataset. The agent must read the dataset card for the current file layout, loading method and licence before writing the downloader, and must not assume them.

### 4.2 Fields used
Review id (generate a stable id if missing), product id, rating (1 to 5), title, text, timestamp, category.

### 4.3 Labels
3-class sentiment derived from rating: 1 and 2 stars = negative, 3 stars = neutral, 4 and 5 stars = positive. The mapping lives in configs/labels.yaml. Neutral is known to be the noisiest class and this must be noted in the analysis.

### 4.4 Cleaning rules
Strip HTML tags, normalise whitespace, drop rows with empty text or fewer than 10 characters, drop exact duplicates on (product id, text), keep both the original text and a clean_text column. Light cleaning only for transformers (they need natural text); heavier normalisation (lowercasing, stopword handling) only inside the TF-IDF pipeline.

### 4.5 Splits
Deterministic split by hash of review id: 80% train, 10% validation, 10% test. Duplicates are removed before splitting to prevent leakage. The split is stored in a split column and never recomputed differently.

### 4.6 Fixed evaluation files
- train_200k: stratified sample of up to 200,000 training reviews (used to train all supervised models).
- val_20k: up to 20,000 validation reviews (model selection and tuning).
- test_50k: up to 50,000 test reviews (final self-hosted model evaluation).
- llm_test_1500: stratified sample of 1,500 reviews from the test set. All four models are scored on this subset for the headline comparison table (apples to apples, controls LLM cost). The three self-hosted models are additionally scored on test_50k for a secondary table.

### 4.7 Class imbalance
Expect positive reviews to dominate. Use class weights in training. Headline metric is macro-F1 and accuracy is always reported next to it.

## 5. Functional requirements

- FR-1 Ingestion. A CLI downloads raw review files for the chosen categories into data/raw/. Acceptance: re-running skips files already present and prints row counts per category.
- FR-2 Cleaning (Pandas reference). Applies section 4.4 and writes parquet files to data/processed/. Acceptance: unit tests on a 10-row fixture cover HTML, empty text, duplicates and label mapping.
- FR-3 Scale processing. The same cleaning implemented in PySpark (local mode) or Dask. Acceptance: output on a 10,000-row sample equals the Pandas output exactly (parity test). A benchmark script records wall time and peak memory at 100k rows, 1M rows and full size into results/benchmarks/processing.csv.
- FR-4 Database. PostgreSQL 16 with the pgvector extension, versioned SQL migration files, bulk loading with COPY, and at least 8 analytical SQL queries in sql/analytics/. Acceptance: row counts in Postgres equal the parquet counts, and every query file starts with a comment stating the business question it answers.
- FR-5 Four models. (a) TF-IDF with Logistic Regression, (b) fine-tuned DistilBERT, (c) fine-tuned BERT-base, (d) LLM zero-shot through an API. All perform 3-class sentiment classification.
- FR-6 Evaluation harness. One shared module used by every model that computes accuracy, macro-F1, per-class precision, recall and F1, and the confusion matrix; measures latency (p50, p95, p99; single request and batch of 32); and computes cost per 1,000 predictions and monthly cost at 1 million predictions. It writes results/model_comparison.csv and results/model_comparison.md. Every run is logged to MLflow.
- FR-7 Model registry. The winning self-hosted model is registered in the MLflow Model Registry with the alias production, chosen by the rule in section 8.7.
- FR-8 Aspect extraction. spaCy-based extraction of aspect terms mapped to a fixed taxonomy defined in configs/aspects.yaml (about 15 aspects such as price, quality, delivery, packaging, battery, durability, size and fit, customer service, ease of use, performance, design, sound, screen, value, setup), with sentence-level sentiment per aspect. Acceptance: evaluated on a hand-labelled gold set of 100 sentences with precision and recall reported.
- FR-9 Serving API. FastAPI service per section 9.
- FR-10 RAG question answering. Per section 10.
- FR-11 Orchestration. Airflow DAG per section 11.
- FR-12 Monitoring. Drift detection per section 12.
- FR-13 Dashboard. Streamlit app per section 13.
- FR-14 Operations. Docker Compose one-command start, a Makefile, tests, linting and a GitHub Actions CI workflow.

## 6. Architecture

```
raw files -> clean (Pandas reference / Spark or Dask at scale) -> parquet -> PostgreSQL (reviews)
parquet train/val/test files -> training (scikit-learn, Hugging Face) -> MLflow (runs + registry)
PostgreSQL + production model -> batch scoring -> predictions, aspects tables
PostgreSQL reviews -> sentence embeddings -> pgvector
FastAPI: /predict, /predict/batch, /ask, /stats/*     <- Streamlit dashboard
Airflow DAG: ingest -> clean -> load -> score -> aspects -> embed -> drift report
Evidently: reference vs current batch -> drift report (HTML + drift_reports table)
Docker Compose services: postgres (pgvector image), mlflow, api, dashboard, airflow (scheduler and webserver).
Batch jobs (cleaning, scoring, training scripts) run with docker compose run or locally in the virtual environment.
```

## 7. Data model and ML specification

### 7.1 SQL schema (implemented as versioned files in sql/migrations/)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE products (
  product_id TEXT PRIMARY KEY,
  category   TEXT NOT NULL,
  title      TEXT
);

CREATE TABLE reviews (
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
CREATE INDEX idx_reviews_product_time ON reviews(product_id, reviewed_at);
CREATE INDEX idx_reviews_split ON reviews(split);

CREATE TABLE predictions (
  id BIGSERIAL PRIMARY KEY,
  review_id TEXT REFERENCES reviews(review_id),
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  label TEXT NOT NULL,
  p_negative REAL, p_neutral REAL, p_positive REAL,
  scored_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (review_id, model_name, model_version)
);

CREATE TABLE aspects (
  id BIGSERIAL PRIMARY KEY,
  review_id TEXT REFERENCES reviews(review_id),
  aspect TEXT NOT NULL,
  sentiment TEXT NOT NULL,
  span_text TEXT
);
CREATE INDEX idx_aspects_aspect ON aspects(aspect, sentiment);

CREATE TABLE review_embeddings (
  review_id TEXT PRIMARY KEY REFERENCES reviews(review_id),
  embedding vector(384)
);
-- create the HNSW index AFTER bulk loading embeddings:
-- CREATE INDEX ON review_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE model_runs (
  id BIGSERIAL PRIMARY KEY,
  mlflow_run_id TEXT, model_name TEXT, dataset TEXT,
  accuracy REAL, macro_f1 REAL, p50_ms REAL, p95_ms REAL,
  cost_per_1k_usd REAL, created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE drift_reports (
  id BIGSERIAL PRIMARY KEY,
  window_start TIMESTAMPTZ, window_end TIMESTAMPTZ,
  drift_detected BOOLEAN, details JSONB, report_path TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE api_requests (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ DEFAULT now(),
  endpoint TEXT, model_name TEXT, latency_ms REAL, label TEXT, text_len INT
);
```

### 7.2 Common rules for all models
Seed 42. Same splits and same label mapping. Library versions logged to MLflow. Macro-F1 is the headline metric, accuracy always reported beside it.

### 7.3 TF-IDF + Logistic Regression
TfidfVectorizer with word n-grams (1,2), min_df=5, sublinear_tf=True, a capped vocabulary (about 500k features), then LogisticRegression with class_weight='balanced'. Tune C over {0.1, 1, 10} on the validation set only.

### 7.4 Transformers
distilbert-base-uncased and bert-base-uncased. Max length 128 tokens, learning rate 2e-5, batch size 32 (16 if memory is short, with gradient accumulation), 2 epochs, 6% warmup, weight decay 0.01, class-weighted cross-entropy, mixed precision on GPU. Evaluate on the validation set every epoch and keep the checkpoint with the best validation macro-F1. Save model and tokenizer.

### 7.5 LLM zero-shot
A provider-agnostic LLMClient (provider and model name come from environment variables). The prompt defines the three labels, asks for exactly one word as output, uses temperature 0 and a very small output token limit. Requests are rate-limited, retried with backoff, and cached on disk keyed by a hash of (model, prompt) so reruns cost nothing. Unparseable outputs are counted as a separate parse_failure_rate and treated as errors. Cost is computed from the token usage returned by the API multiplied by prices in configs/pricing.yaml. A hard spend cap (MAX_LLM_SPEND_USD) aborts the run when exceeded.

### 7.6 Latency protocol
50 warm-up calls, then 1,000 timed single-request calls, then batch size 32. Self-hosted models are timed on CPU (and on GPU separately if available). Timing covers tokenisation, forward pass and post-processing but not network. Report p50, p95, p99. Record the hardware (CPU model, cores, RAM, torch thread count). For the LLM, time 200 end-to-end API calls and state clearly that network time is included.

### 7.7 Cost protocol
Training cost = GPU hours multiplied by the hourly rate in configs/pricing.yaml (free Colab = 0 dollars, but record the hours). Self-hosted inference cost per 1,000 predictions = (assumed instance price per hour divided by measured predictions per hour) multiplied by 1,000, using the batch throughput measured in 7.6; the assumed instance price is a stated, configurable assumption. LLM cost per 1,000 predictions = measured tokens multiplied by prices. Monthly cost at 1 million predictions = cost per 1,000 multiplied by 1,000.

### 7.8 Production model selection rule
Among self-hosted models whose p95 single-request CPU latency is 300 ms or lower, pick the highest validation macro-F1. If two models are within 1 point of macro-F1, pick the cheaper one. Register it with alias production. The LLM is never used for bulk scoring; the analysis must explain why using the cost numbers.

## 8. Aspect extraction specification

Use spaCy (en_core_web_sm) for sentence splitting, noun chunks and dependency parsing. Map candidate terms to the taxonomy in configs/aspects.yaml using lemma and synonym lists plus PhraseMatcher. Sentence-level sentiment per aspect comes from the production sentiment model applied to the sentence containing the aspect. Store results in the aspects table. The gold set eval/aspect_gold.jsonl has 100 sentences labelled by hand by the project owner (aspect and sentiment). Report precision, recall and F1 for aspect detection and accuracy for aspect sentiment.

## 9. API specification (FastAPI, Pydantic v2)

- GET /health: status, loaded model name and version, database connectivity.
- GET /v1/models: loaded models and versions.
- POST /v1/predict: body has text (1 to 5000 characters) and optional model (tfidf, distilbert, bert; default is the production model). Response has label, probabilities, aspects (list of aspect, sentiment, span), model_name, model_version, latency_ms.
- POST /v1/predict/batch: body has texts (list of at most 64). Returns a list of the same objects.
- GET /v1/stats/sentiment-trend: query parameters category, product_id, from, to, granularity (day, week, month).
- GET /v1/stats/top-aspects: query parameters sentiment, category, limit.
- POST /v1/ask: body has question and optional filters. Response has answer, sources (review id, snippet, score), abstained (boolean), route (rag or sql), latency_ms.

Requirements: models are loaded once at startup, never per request; every request gets a request id; logs are structured JSON; invalid input returns 422; every /predict call is recorded in api_requests; all SQL is parameterised.

## 10. RAG specification

- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions) over a stratified sample of 200,000 reviews, stored in pgvector.
- Router: questions that ask for counts, trends or comparisons ('how many', 'trend', 'increase', 'compare') are answered from whitelisted, parameterised SQL templates and the LLM only phrases the result. All other questions use retrieval. The system must never execute SQL written by the LLM. Reason: an LLM cannot count over a handful of retrieved snippets, and executing generated SQL is a security risk.
- Retrieval: embed the question, vector search top 20 with optional filters (category, product, date range, sentiment), keep the best 8.
- Generation: the LLM answers only from the numbered snippets, cites review ids, and must say that the evidence is insufficient when the snippets do not answer the question (abstain).
- Evaluation set eval/rag_gold.jsonl: at least 60 questions: 30 qualitative (with reference key points written by the project owner), 20 analytical (ground truth computed by SQL), 10 unanswerable (the correct behaviour is to abstain).
- Metrics: retrieval hit@8 on the qualitative set; numeric correctness on the analytical set; faithfulness (every claim supported by the cited snippets, scored by an LLM judge and manually verified on 20 answers); abstention accuracy; hallucination rate (share of answers with at least one unsupported claim); latency; cost per question. Written to results/rag_eval.md.

## 11. Orchestration specification (Airflow)

DAG voc_daily (daily schedule, manually triggerable). Tasks in order: check_new_data, clean, load_postgres, score_new_reviews (uses the production model from the registry), extract_aspects, embed_new_reviews, drift_report. Every task is idempotent (re-running never duplicates rows: use ON CONFLICT DO NOTHING or upserts), retries twice with a delay, and logs row counts. New data is simulated by a script that drops batches into data/incoming/, including one deliberately drifted batch for the demo.

## 12. Monitoring specification

Use Evidently (the agent must read the documentation of the installed version because its API changes between versions). Compare a reference window (validation set predictions) with the current batch on: text length, word count, share of non-ASCII characters, predicted label distribution, and confidence distribution. Thresholds are configurable. Each run writes an HTML report to results/drift/ and a row to drift_reports. The deliberately drifted batch must trigger drift_detected = true, and a normal batch must not.

## 13. Dashboard specification (Streamlit)

Pages: (1) Overview with KPIs and sentiment trend; (2) Aspect explorer with filters for category, product and date; (3) Ask, a chat box showing sources and abstentions; (4) Model comparison, rendered from results/model_comparison.csv with charts of F1 against latency and cost; (5) Monitoring with the latest drift status and report link; (6) Live demo, where a typed review is scored by all three self-hosted models side by side with their latencies.

## 14. Non-functional requirements

- Reproducibility: pinned dependencies, fixed seeds, make up starts the stack, make smoke runs a tiny end-to-end test on 1,000 reviews in under 5 minutes.
- Performance targets (to be measured and reported honestly, not forced): TF-IDF p95 under 20 ms per request; production transformer p95 under 300 ms per request on CPU.
- Security: no secrets in git, parameterised SQL, input length limits, no LLM-generated SQL execution.
- Quality: type hints on public functions, ruff clean, pytest coverage of at least 70% on core modules.
- Cost guard: every LLM call path honours MAX_LLM_SPEND_USD.

## 15. Repository structure

```
voc-platform/
  AGENTS.md  README.md  Makefile  docker-compose.yml  pyproject.toml  .env.example  .gitignore
  configs/      pricing.yaml  labels.yaml  aspects.yaml  training.yaml
  data/         raw/  processed/  incoming/            (gitignored)
  docs/         PRD.md  architecture.md  decisions/
  sql/          migrations/  analytics/
  src/voc/
    data/         ingest.py  clean.py  clean_spark.py  splits.py
    db/           load.py  queries.py
    models/       tfidf.py  transformer.py  llm_zero_shot.py  registry.py  import_run.py
    evaluation/   metrics.py  latency.py  cost.py  compare.py
    nlp/          aspects.py
    api/          main.py  schemas.py  deps.py
    rag/          embed.py  retrieve.py  answer.py  router.py
    monitoring/   drift.py
    dashboard/    app.py  pages/
  airflow/dags/
  eval/         rag_gold.jsonl  aspect_gold.jsonl
  results/      benchmarks/  drift/  model_comparison.csv  rag_eval.md
  tests/
  notebooks/    eda.ipynb  colab_train.ipynb
  .github/workflows/ci.yml
```

## 16. Tech stack and version policy

Python 3.11; pandas, pyarrow, pyspark (or dask), scikit-learn, nltk, spaCy; PyTorch, transformers, datasets, sentence-transformers; MLflow; Apache Airflow (installed with the official constraints file for the chosen version); FastAPI, uvicorn, Pydantic v2, SQLAlchemy or psycopg, pgvector client; Streamlit; Evidently; pytest, ruff.

Version policy: before installing anything, check the current stable release and its compatibility with Python 3.11 and the other libraries; pin exact versions in pyproject.toml or a lock file; never write a library call from memory when the installed library's documentation can be read. Airflow and Evidently in particular change their APIs between versions.

## 17. Testing strategy

- Unit: cleaning rules, label mapping, split determinism, metric functions, cost calculator, aspect matcher.
- Integration: API against a test database, bulk loader against a small parquet file.
- Parity: Pandas versus Spark or Dask output on a 10k sample.
- Smoke: make smoke end to end on 1,000 rows.
- CI: lint, unit and smoke tests with the LLM mocked and no GPU required.

## 18. Definition of done (checklist)

- [x] Phase 0 scaffolding and CI green
- [x] Phase 1 data ingestion, cleaning, splits, EDA
- [ ] Phase 2 scale processing and benchmark table
- [ ] Phase 3 Postgres loaded and 8 analytics queries
- [ ] Phase 4 evaluation harness, TF-IDF baseline, MLflow
- [ ] Phase 5 DistilBERT and BERT trained and imported
- [ ] Phase 6 LLM zero-shot and full comparison table
- [ ] Phase 7 aspect extraction with gold-set scores
- [ ] Phase 8 FastAPI service with tests
- [ ] Phase 9 RAG with evaluation report
- [ ] Phase 10 Airflow DAG and drift monitoring
- [ ] Phase 11 Streamlit dashboard
- [ ] Phase 12 README, architecture diagram, demo video

## 19. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Not enough RAM for the full stack | Start Airflow only in Phase 10; use Dask instead of Spark; sample data |
| Transformer training too slow | Colab or Kaggle GPU, 200k training cap, max length 128 |
| LLM costs run away | Disk cache, 1,500-review subset, hard spend cap, provider spend limit |
| Dependency conflicts (Airflow, Evidently) | Official constraints file, separate environments or containers, pinned versions |
| Label noise in the neutral class | Report per-class F1 and discuss it openly |
| Agent invents numbers or APIs | AGENTS.md rules, results only from scripts, manual verification each phase |
