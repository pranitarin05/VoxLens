# VoxLens — Voice-of-Customer Intelligence Platform

Local MLOps demo: Amazon Reviews 2023 → cleaning → PostgreSQL/pgvector → four sentiment models compared → aspect extraction → FastAPI → RAG Q&A → Airflow + drift → Streamlit.

Spec: [`docs/PRD.md`](docs/PRD.md). Flow: [`flow.md`](flow.md). Decisions: [`decision.md`](decision.md).

## Status (measured, not aspirational)

| Phase | State | Evidence |
|---|---|---|
| 0 Scaffolding + CI | done | `make lint` / `make test` green; GitHub Actions green |
| 1 Data (ingest/clean/splits/EDA) | done | 3,202,467 raw → 3,024,584 cleaned; 80/10/10; four eval files; `notebooks/eda.ipynb` executed |
| 2 Scale (Dask + parity + bench) | done | 10k exact parity (9,727 rows); 12 benchmark points in `results/benchmarks/processing.csv` |
| 3–12 | not started | — |

Row counts above come from `make ingest` / `make clean` / `make splits` runs on 2026-10-27 (Amazon_Fashion + All_Beauty). Benchmark wall times and peak RSS are in the CSV (measured on this machine, not estimates).

## Quickstart

```bash
# Python 3.11, GNU make, Docker Desktop
make setup          # venv + pinned deps + .env from example
make lint && make test
make up             # Postgres (pgvector) only until Phase 4+
make ingest         # download raw jsonl.gz (skips existing)
make clean          # pandas reference cleaner → data/processed/
make splits         # hash split + train_200k / val_20k / test_50k / llm_test_1500
make eda            # execute notebooks/eda.ipynb
```

Secrets live only in gitignored `.env` (see `.env.example` for names). Never commit `.env` or data files.

## Dataset

Amazon Reviews 2023 (McAuley Lab). Categories: **Amazon_Fashion** + **All_Beauty** (~3.2M reviews). The dataset card declares no licence field; used for research/portfolio with attribution to McAuley Lab — see `decision.md` D10.

## Layout

```
src/voc/     data, db, models, evaluation, nlp, api, rag, monitoring, dashboard
configs/     labels, pricing, aspects, training (seed 42)
sql/         migrations + analytics queries
tests/       pytest (unit; no GPU/LLM required)
results/     benchmarks, model comparison, drift reports
notebooks/   eda.ipynb
```

## Commands

| Command | Does |
|---|---|
| `make setup` | create `.venv`, install `.[dev]` |
| `make setup-ml` | also install torch/transformers/mlflow/spacy/groq |
| `make lint` / `make test` | ruff + pytest |
| `make up` / `make down` / `make logs` | Docker Compose services |
| `make ingest` `clean` `splits` `eda` | Phase 1 pipeline |

## Licence

Code: MIT (see `pyproject.toml`). Dataset: no explicit licence on the card (D10).
