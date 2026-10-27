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

.PHONY: setup setup-ml lint test up down smoke logs ingest clean splits eda

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
