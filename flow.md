# flow.md â€” How the VoxLens machine works

Companion to `decision.md` (why) and `docs/PRD.md` (spec).
Read this first when returning to the project.

---

## 1. End-to-end data flow

```mermaid
flowchart LR
  subgraph Ingestion [Phase 1 - Ingestion]
    A1[Amazon Reviews 2023\njsonl.gz per category] -->|CLI download\ndata/raw/| A2[Raw rows]
  end

  subgraph Cleaning [Phase 1-2 - Cleaning]
    A2 --> B1[Pandas reference cleaner]
    A2 --> B2[Dask cleaner - parity must match]
    B1 --> C1[parquet\ndata/processed/]
    B2 --> C1
    C1 --> C2[Fixed eval files\ntrain_200k / val_20k / test_50k / llm_test_1500]
  end

  subgraph Storage [Phase 3 - PostgreSQL + pgvector]
    C1 -->|COPY bulk load| D1[(reviews\nproducts)]
    C1 -->|200k stratified sample| D2[(review_embeddings\nvector 384)]
  end

  subgraph Training [Phases 4-6 - Models]
    C2 --> E1[TF-IDF + LogReg]
    C2 --> E2[DistilBERT]
    C2 --> E3[BERT-base]
    C2 --> E4[LLM zero-shot - Groq\n1500 subset only]
    E1 & E2 & E3 & E4 -->|shared harness| F1[MLflow runs]
    E1 & E2 & E3 & E4 --> F2[results/model_comparison.csv]
    F1 -->|alias production\nrule 8.7| G1[Winning self-hosted model]
  end

  subgraph Scoring [Phase 7 - Batch scoring]
    G1 -->|analysis_200k| H1[(predictions)]
    G1 -->|spaCy + taxonomy| H2[(aspects)]
  end

  subgraph Serving [Phase 8-9 - API]
    G1 --> I1[FastAPI models/production/\nloaded once at startup]
    D1 --> I2[stats endpoints - parameterised SQL]
    D2 --> I3[retrieve top20 keep 8]
    H1 & H2 & I1 & I2 & I3 --> J1[POST /v1/ask\nrouter: SQL templates OR RAG]
    I1 --> J2[POST /v1/predict\n+ batch]
  end

  subgraph Dashboard [Phase 11 - Streamlit]
    J1 & J2 & I2 --> K1[6 pages - all numbers\nfrom API or results CSV]
  end

  subgraph Orchestration [Phase 10 - Airflow + Evidently]
    L1[voc_daily DAG] -->|idempotent tasks| M1[clean -> load -> score\n-> aspects -> embed -> drift]
    M1 --> N1[Evidently report\nresults/drift/ + drift_reports]
  end
```

Plain-language version:

1. **Ingest** raw category files into `data/raw/` (re-runnable, skips what exists).
2. **Clean** with Pandas (reference) and Dask (scale) â†’ identical parquet in `data/processed/` â†’ dedupe â†’ deterministic hash split â†’ four fixed evaluation files.
3. **Load** parquet into Postgres with COPY (products first, then reviews); embeddings into pgvector later.
4. **Train/compare** four sentiment models through one shared harness â†’ MLflow runs â†’ comparison table â†’ production alias.
5. **Score** a 200k analysis sample with the production model â†’ `predictions` + `aspects` tables (idempotent).
6. **Serve** via FastAPI (predict, stats, ask) with models loaded once at startup.
7. **Answer** questions via router: counts/trends â†’ whitelisted SQL templates; everything else â†’ pgvector retrieval + LLM generation with citations and abstention.
8. **Orchestrate** daily with Airflow; **monitor** drift with Evidently (reference vs current batch).
9. **Show** everything in Streamlit, which only talks to the API and to `results/` files â€” no hard-coded numbers.

---

## 2. Phase map (input â†’ output â†’ verify)

| Phase | Input | Output | You verify by |
|---|---|---|---|
| 0 Scaffolding | empty repo | folder tree, pinned pyproject, ruff/pytest, Postgres-only compose, Makefile, CI | `make setup && make lint && make test`; `docker compose ps` healthy; CI green |
| 1 Data | dataset card read, category choice | `data/raw/`, `data/processed/*.parquet`, 4 fixed eval files, `notebooks/eda.ipynb` | tests pass; row counts shrink per stage; no review id in two splits |
| 2 Scale | Phase 1 parquet | Dask cleaner, parity test, `results/benchmarks/processing.csv` | parity = exact match; CSV numbers plausible on re-run |
| 3 DB | Phase 1 parquet | migrations, COPY loader, 8 analytics queries | row counts equal parquet; each query run by hand once |
| 4 Harness + TF-IDF | eval files | `metrics/latency/cost.py`, dummy + TF-IDF in MLflow, first comparison rows | metric tests hand-computed; TF-IDF beats dummy; MLflow run visible |
| 5 Transformers | train_200k, val_20k | trained DistilBERT + BERT, `import_run.py`, harness rows | CUDA verified first; local smoke completes; latency measured locally |
| 6 LLM + table | llm_test_900 (free-tier 1000/day), Groq keys | `llm_zero_shot.py`, final comparison + charts, production alias | dry-run approved first; spend â‰¤ cap; registry shows alias |
| 7 Aspects | configs/aspects.yaml, 100 labels (draft awaiting owner) | `aspects.py`, `analysis_200k`, batch score into Postgres | spot-check 20 sentences; double-run creates no dupes |
| 8 API | production model export | FastAPI service, tests, Dockerfile, load test | /docs works; empty string â†’ 422; load-test p95 noted |
| 9 RAG | analysis_200k embeddings, your 60 questions | retrieve/router/answer, `/v1/ask`, `results/rag_eval.md` | 5 hand-asked questions cite real reviews; unanswerable â†’ abstain |
| 10 Airflow + drift | incoming simulator | `voc_daily` DAG, Evidently drift module, 2-case test | DAG green in UI twice without dupes; normal=false, drifted=true |
| 11 Dashboard | API + results files | 6 Streamlit pages, dashboard service | click every page; stop API â†’ friendly error |
| 12 Hardening | everything | README, `make smoke`, coverage, tag v1.0 | fresh clone follows only README; every README number traced to source |

---

## 3. Service topology (when Docker starts)

Containers are **staged** â€” one suspect per failure (PRD risk table: start Airflow only in Phase 10).

| Phase | Compose services running | Host venv runs |
|---|---|---|
| 0â€“3 | `postgres` (pgvector/pg16, healthcheck) | everything else |
| 4 | `postgres`, `mlflow` (persistent volume) | training scripts |
| 5â€“7 | `postgres`, `mlflow` | training, scoring |
| 8 | `postgres`, `mlflow`, `api` | tests, load test |
| 9 | + embeddings loaded into `postgres` | embed job |
| 10 | + `airflow` (webserver + scheduler, separate DB in Postgres) | simulator |
| 11 | + `dashboard` | â€” |
| 12 | `make up` starts the whole stack | `make smoke` runs e2e |

Batch jobs (clean, score, train, embed) are **never** containers-by-default: they run via `docker compose run` or directly in the virtual environment.

---

## 4. Request paths

### `POST /v1/predict` (and `/predict/batch`)
```
request id assigned â†’ Pydantic validation (422 if invalid, text â‰¤ 5000 chars)
  â†’ model from models/production/ (loaded ONCE at startup, never per request)
  â†’ optional model override: tfidf | distilbert | bert
  â†’ label + probabilities + aspects (from same span extraction)
  â†’ write api_requests (endpoint, model, latency_ms, label, text_len)
  â†’ structured JSON log with request id
```

### `POST /v1/ask`
```
question + optional filters
  â†’ ROUTER:
      analytical keywords (how many | trend | increase | compare ...)
        â†’ whitelisted, parameterised SQL templates ONLY
          (the LLM never writes SQL; generated SQL is never executed)
      everything else
        â†’ embed question (all-MiniLM-L6-v2)
        â†’ pgvector top 20 (filters: category/product/date/sentiment)
        â†’ keep best 8 snippets
        â†’ LLM answers ONLY from numbered snippets, cites review ids
        â†’ insufficient evidence â†’ abstained=true
  â†’ answer, sources[], route (rag|sql), abstained, latency_ms
  â†’ every LLM call honours MAX_LLM_SPEND_USD + disk cache
```

---

## 5. Fixed evaluation files (never regenerated differently)

| File | Size | Built in | Used by |
|---|---|---|---|
| `train_200k` | â‰¤200k stratified from train | Phase 1, seed 42 | all supervised training |
| `val_20k` | â‰¤20k from val | Phase 1 | tuning (C sweep, best-checkpoint) |
| `test_50k` | â‰¤50k from test | Phase 1 | final self-hosted evaluation |
| `llm_test_1500` | 1.5k stratified from test | Phase 1 | headline 4-model table (cost control) |

Split rule: `split = f(hash(review_id))` â†’ 80/10/10, computed **after** dedupe, stored in a column, never recomputed.

---

## 6. Make targets (the reviewer's interface)

| Target | Does |
|---|---|
| `make setup` | Python 3.11 venv + pinned install (core first; ml group when needed) |
| `make lint` | ruff check + format --check |
| `make test` | pytest (unit; no GPU, no LLM, no Docker beyond Postgres) |
| `make up` | `docker compose up -d` (stage-appropriate services) |
| `make down` | `docker compose down` (keeps volumes) |
| `make smoke` | end-to-end on 1,000 reviews in <5 min, LLM mocked |

---

## 7. Golden rules while flowing through this repo

1. Numbers come only from commands run or files a script produced.
2. Library APIs come only from the installed version's docs/source.
3. One phase at a time; lint + tests before any "done".
4. Pipelines are idempotent â€” re-run never duplicates rows.
5. LLM paths: disk cache + `MAX_LLM_SPEND_USD` always.
6. Decisions â†’ `decision.md`; flow changes â†’ this file.
