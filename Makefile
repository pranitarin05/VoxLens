# VoxLens Makefile — works on Windows (GNU Make + Git bash) and Linux CI.
# One-word commands are what a reviewer runs (PRD G6 / Phase 0 step 5).

ifeq ($(OS),Windows_NT)
SHELL := C:/Program Files/Git/bin/bash.exe
VENV_PY := .venv/Scripts/python.exe
CREATE_VENV := py -3.11 -m venv .venv
else
VENV_PY := .venv/bin/python
CREATE_VENV := python3.11 -m venv .venv
endif

.PHONY: setup setup-ml lint test up down smoke logs ingest clean splits eda dask parity bench db-migrate db-load db-counts db-analytics eval-tfidf compare mlflow-ui train-distilbert eval-distilbert train-bert smoke-transformer train-chunk

## Install core+dev dependencies into .venv and create .env from example (first run only)
setup:
	$(CREATE_VENV)
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -e ".[dev]"
	@test -f .env || cp .env.example .env
	@echo "setup OK — next: make lint && make test"

## Install the ml extras (torch/transformers/mlflow/spacy/groq) — used from Phase 4 on
setup-ml: setup
	$(VENV_PY) -m pip install -e ".[ml]"

## Lint (ruff check + format check)
lint:
	$(VENV_PY) -m ruff check src tests
	$(VENV_PY) -m ruff format --check src tests

## Unit tests (no GPU, no LLM, no Docker needed in Phase 0)
test:
	$(VENV_PY) -m pytest

## Start Docker services (stage-appropriate: only postgres until Phase 4+)
up:
	docker compose up -d

## Stop Docker services (keeps volumes)
down:
	docker compose down

## Show container status
logs:
	docker compose ps

## Smoke check. Phase 0: unit tests only.
## Phase 12 replaces this with the full 1,000-row end-to-end run (<5 min, LLM mocked).
smoke:
	@echo "[smoke] Phase 0 placeholder — running unit tests; full e2e arrives in Phase 12"
	$(VENV_PY) -m pytest

## Phase 1: download raw category files (skips what exists) and print row counts
ingest:
	$(VENV_PY) -m voc.data.ingest

## Phase 1: Pandas reference cleaner → data/processed/reviews_clean.parquet
clean:
	$(VENV_PY) -m voc.data.clean

## Phase 1: split column + four fixed eval files (seed 42)
splits:
	$(VENV_PY) -m voc.data.splits

## Phase 1: execute notebooks/eda.ipynb in place
eda:
	$(VENV_PY) -c "import nbformat; from nbclient import NotebookClient; p='notebooks/eda.ipynb'; nb=nbformat.read(p, as_version=4); NotebookClient(nb, timeout=600, kernel_name='python3').execute(); nbformat.write(nb, p); print('executed', p)"

## Phase 2: Dask cleaner → data/processed/reviews_clean_dask.parquet
dask:
	$(VENV_PY) -m voc.data.clean_spark

## Phase 2: FR-3 exact parity on a 10k sample (fails if not identical)
parity:
	$(VENV_PY) -m voc.data.clean_spark parity

## Phase 2: one small-bench point. Usage: make bench SIZE=100000 ENGINE=dask
## SIZE is 100000 | 1000000 | full ; ENGINE is dask | pandas
SIZE ?= 100000
ENGINE ?= dask
bench:
	$(VENV_PY) -m voc.data.clean_spark bench --size $(SIZE) --engine $(ENGINE)

## Phase 3: apply sql/migrations (idempotent)
db-migrate:
	$(VENV_PY) -m voc.db.load migrate

## Phase 3: COPY-load data/processed/reviews.parquet into products + reviews
db-load:
	$(VENV_PY) -m voc.db.load load

## Phase 3: print products/reviews row counts
db-counts:
	$(VENV_PY) -m voc.db.load counts

## Phase 3: run all sql/analytics/*.sql once
db-analytics:
	$(VENV_PY) -m voc.db.load analytics

## Phase 4: train TF-IDF (tune C on val) + evaluate on llm_test_1500 + test_50k
eval-tfidf:
	$(VENV_PY) -m voc.models.tfidf train
	$(VENV_PY) -m voc.models.tfidf evaluate

## Phase 4: rebuild results/model_comparison.md from the CSV
compare:
	$(VENV_PY) -m voc.evaluation.compare

## Phase 4: open the MLflow UI (sqlite store)
mlflow-ui:
	$(VENV_PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

## Phase 5: fine-tune DistilBERT on GPU (200k/20k, 2 epochs)
train-distilbert:
	$(VENV_PY) -m voc.models.transformer train --model-name distilbert-base-uncased --epochs 2 --batch-size 8 --grad-accum 4

## Phase 5: score the DistilBERT checkpoint on llm_test_1500 + test_50k
eval-distilbert:
	$(VENV_PY) -m voc.models.transformer evaluate --model-dir models/distilbert-base-uncased

## Phase 5: fine-tune BERT-base on GPU (batch 4 + grad_accum 8 for 4 GB)
train-bert:
	$(VENV_PY) -m voc.models.transformer train --model-name bert-base-uncased --epochs 2 --batch-size 4 --grad-accum 8

## Phase 5: fast GPU smoke (2k/1 epoch) to verify the pipeline
smoke-transformer:
	$(VENV_PY) -m voc.models.transformer train --model-name distilbert-base-uncased --sample 2000 --val-sample 2000 --epochs 1 --batch-size 8 --grad-accum 4 --output-dir models/smoke-distilbert

## Phase 5: ONE small training chunk (resumable). Usage: make train-chunk STEPS=250
STEPS ?= 250
RESUME ?= 0
train-chunk:
	$(VENV_PY) -m voc.models.transformer train --model-name distilbert-base-uncased --sample 20000 --val-sample 5000 --epochs 1 --batch-size 4 --grad-accum 8 --max-steps $(STEPS) --output-dir models/distilbert-chunk $(if $(filter 1,$(RESUME)),--resume,)

## Phase 6: dry-run the LLM zero-shot pipeline (mock, no API calls, no results)
eval-llm-dry:
	$(VENV_PY) -m voc.models.llm_zero_shot evaluate --dataset llm_test_1500 --dry-run

## Phase 6: live zero-shot on the 900-row subset (cached, hard MAX_LLM_SPEND_USD cap)
eval-llm:
	$(VENV_PY) -m voc.models.llm_zero_shot evaluate --dataset llm_test_900

## Phase 7: extract aspects + sentence sentiment (5k reviews, CPU)
aspects-extract:
	$(VENV_PY) -m voc.nlp.aspects extract --limit 5000 --out data/processed/aspects_sample.parquet

## Phase 7: load aspects parquet into Postgres (idempotent on the span key)
aspects-load:
	$(VENV_PY) -m voc.nlp.aspects load --path data/processed/aspects_sample.parquet

## Phase 7: draft the 100-sentence gold file (owner must review the labels)
aspects-gold:
	$(VENV_PY) -m voc.nlp.aspects draft-gold --mentions-path data/processed/aspects_sample.parquet

## Phase 7: score aspect detection + sentiment against the gold file
aspects-eval:
	$(VENV_PY) -m voc.nlp.aspects evaluate

## Phase 8: run the API locally (models load once at startup)
api-run:
	$(VENV_PY) -m uvicorn voc.api.main:app --host 127.0.0.1 --port $(if $(API_PORT),$(API_PORT),8000)

## Phase 8: API contract/validation tests
api-test:
	$(VENV_PY) -m pytest -q tests/test_phase8_api.py

## Phase 8: measured single-request latency against a running API
api-load:
	$(VENV_PY) scripts/load_test_api.py --n 200

.PHONY: rag-embed rag-gold rag-eval rag-test

## Phase 9: sample + embed 200k reviews into pgvector and build the HNSW index
rag-embed:
	$(VENV_PY) -m voc.rag.embed build

## Phase 9: (re)build the 60-question gold set from the database
rag-gold:
	$(VENV_PY) scripts/build_rag_gold.py

## Phase 9: evaluate RAG/SQL answers and write results/rag_eval.{md,json}
rag-eval:
	$(VENV_PY) -m voc.rag.evaluate

## Phase 9: RAG unit + DB integration tests
rag-test:
	$(VENV_PY) -m pytest -q tests/test_phase9_rag.py
