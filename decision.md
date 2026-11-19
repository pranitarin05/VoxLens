# decision.md — Decision log (append-only)

Format: **D# · date · decision · options · choice · reason · status**
Rules: never rewrite history; append new decisions; only measured facts.

---

## D1 · 2026-10-27 · Python version

- **Options:** (a) install Python 3.11 fresh + clean venv, (b) reuse existing Python 3.14 (has pandas/torch/transformers installed), (c) Python 3.12 middle ground.
- **Choice:** (a) Python 3.11 fresh.
- **Reason:** PRD §16 pins 3.11; Airflow/Evidently/torch CUDA wheels are guaranteed for 3.11; system 3.14 packages are unmanaged and would contaminate pinned installs. Audit found: 3.14 (default), broken 3.12 folder (no python.exe), legacy 2.7 — none acceptable.
- **Status:** decided.

## D2 · 2026-10-27 · Scale engine (Phase 2)

- **Options:** (a) Dask, (b) PySpark local mode.
- **Choice:** (a) Dask.
- **Reason:** 16 GB RAM shared with Docker; Java 17 is present but PySpark on Windows adds Hadoop classpath/winutils quirks; Dask is lighter and still satisfies "Big Data tooling". Parity test (10k exact match) applies either way.
- **Status:** decided.

## D3 · 2026-10-27 · LLM provider

- **Options:** (a) Groq, (b) OpenAI, (c) Gemini, (d) local Ollama.
- **Choice:** (a) Groq, with **two API keys**: `GROQ_API_KEY` (primary) and `GROQ_API_KEY_BACKUP` (used automatically on auth/quota failure).
- **Reason:** user's explicit instruction; keys already provided (stored in `.env`, gitignored; source file `.env.txt` must also stay untracked). Cheap+fast for the 1,500-review eval and RAG generation.
- **Status:** decided.

## D4 · 2026-10-27 · Transformer training route (Phase 5)

- **Options:** (a) local GPU only, (b) Colab, (c) local smoke + Colab full.
- **Choice:** (a) **Local Windows GPU only** — RTX 3050 4 GB, driver 566.07, CUDA 12.7. No Colab, no cloud GPUs, no credit card, no paid APIs for training.
- **Reason:** user's explicit instruction. Constraints accepted: DistilBERT trainable with small batch (4–8) + fp16 + gradient accumulation + max_len 128; BERT-base likely impractical at 4 GB → if OOM, document honestly and keep TF-IDF baseline as the working supervised model. Full `train_200k` run is **optional**; smoke (2k samples, 1 epoch) must complete first. **No success claim without a completed run.**
- **Status:** decided.

## D5 · 2026-10-27 · GitHub repository

- **Choice:** `https://github.com/pranitarin05/VoxLens` (user-created).
- **Reason:** CI target (G6), green badge for reviewers.
- **Status:** decided.

## D6 · 2026-10-27 · Docker data disk location

- **Options:** (a) move Docker data to D:, (b) keep on C:, (c) user frees C: space.
- **Choice:** (a) move Docker Desktop disk image to `D:` before pulling images.
- **Reason:** C: has only ~17 GB free; full stack images (Airflow + PyTorch API layer) can exceed that. D: has ~532 GB free. User performs this in Docker Desktop settings (GUI).
- **Status:** decided — **pending user action** (move + restart Docker Desktop).

## D7 · 2026-10-27 · Dataset download method

- **Options:** (a) direct HTTP GET of `*.jsonl.gz` per category, (b) HF `datasets.load_dataset(..., trust_remote_code=True)`.
- **Choice:** (a) direct HTTP download from `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/<Category>.jsonl.gz`.
- **Reason:** verified the HF loading script requires arbitrary code execution (viewer disabled; `trust_remote_code` deprecated path); the **old `datarepo.eng.ucsd.edu` links in the dataset card return 404** (verified 2026-10-27) — current working host is `mcauleylab.ucsd.edu` (HEAD 200 tested on All_Beauty, Electronics, Amazon_Fashion, Appliances, Video_Games, etc.). Direct files are re-runnable per FR-1 (skip if present).
- **Status:** decided. Downloader must re-verify URL per category before trusting it.

## D8 · 2026-10-27 · Airflow isolation

- **Choice:** Airflow **3.3.2** (current stable, released 2026-09-17) inside its own Docker container using the official image + `constraints-3.11.txt`; never installed in the host venv.
- **Reason:** PRD risk table (dependency conflicts); Airflow 3.x supports Python 3.11; official image avoids hand-rolling the install. DAG files stay thin — they call already-tested CLI modules.
- **Status:** decided (activation deferred to Phase 10 per staged topology).

## D9 · 2026-10-27 · Evidently version/API

- **Choice:** Evidently **0.7.x** (latest at build time: 0.7.23) — **new API is default since 0.7**; old `Report(metrics=[ColumnDriftMetric()...])` style from pre-0.7 examples no longer matches docs.
- **Reason:** PRD explicitly warns the API changes between versions; AGENTS rule 4 forbids guessing APIs. Code will be written from the installed version's docs after `pip install`.
- **Status:** decided (activation in Phase 10).

## D10 · 2026-10-27 · Dataset licence (honest finding)

- **Finding:** the Amazon Reviews 2023 dataset card (HuggingFace API `cardData`) declares **no license field**. The project page cites the paper (arXiv 2403.03952) but no SPDX licence.
- **Choice:** document as "no explicit licence on the dataset card; used for research/portfolio purposes with attribution to McAuley Lab". Do **not** invent a licence.
- **Reason:** AGENTS rule 3 (never invent) applies to licence claims too.
- **Status:** decided — recorded, README will state it verbatim.

## D11 · 2026-10-27 · review_id generation

- **Finding:** raw rows have **no review id** (fields: rating, title, text, images, asin, parent_asin, user_id, timestamp, helpful_vote, verified_purchase). Category also absent (comes from filename).
- **Choice:** `review_id = sha256(user_id | parent_asin | timestamp | text)[:32]` — stable across re-downloads and re-runs. `product_id = parent_asin`. `category` injected from file name.
- **Reason:** PRD 4.2 requires a stable generated id; hash of content+metadata is deterministic and collision-safe at this scale.
- **Status:** decided (exact separator/format frozen in Phase 1 code + test).

## D12 · 2026-10-27 · PyTorch CUDA wheels

- **Finding:** existing Python 3.14 has `torch 2.10.0+cpu` → `torch.cuda.is_available() == False` (verified). GPU (RTX 3050, CUDA 12.7, driver 566.07) is usable only with a CUDA wheel.
- **Choice:** in the Python 3.11 venv, install torch from the official CUDA index (`cu126` or whichever wheel the current stable torch provides for Python 3.11) and **verify `torch.cuda.is_available()` before any training claim**.
- **Reason:** user demands verified CUDA training; CPU-only torch would silently fall back and waste hours.
- **Status:** decided — execution in Phase 5 (Phase 0 pins CPU torch only for the lint/test path to keep CI light).

## D13 · 2026-10-27 · Staged Docker services

- **Choice:** only `postgres` in Phase 0; `mlflow` added Phase 4; `api` Phase 8; `airflow` Phase 10; `dashboard` Phase 11.
- **Reason:** PRD build-guide step 4 ("start small… one obvious suspect") + risk table (RAM). Matches flow.md §3.
- **Status:** decided.

## D14 · 2026-10-27 · Category selection (data)

- **Finding (verified sizes):** All_Beauty 701.5K/94 MB · Amazon_Fashion 2.5M/279 MB · Appliances 2.1M/258 MB · Video_Games 4.6M/776 MB · Electronics 43.9M/6.5 GB.
- **Choice:** **(A) Amazon_Fashion + All_Beauty** = 3.2M reviews / 373 MB (user pick, 2026-10-27).
- **Reason:** PRD 4.1 requires ≥1M total from 2–3 categories; 3.2M clears it with comfortable RAM headroom on 16 GB vs option B.
- **Status:** decided — unblocks Phase 1.

## D15 · 2026-10-29 · MLflow store = local SQLite (amends D13)

- **Options:** (a) local `sqlite:///mlflow.db` (tracking + model registry), (b) Docker `mlflow` server backed by Postgres.
- **Choice:** (a) local SQLite for Phase 4. `mlflow.db` is gitignored; `make mlflow-ui` opens the UI. A container can be added later if multi-user access is needed.
- **Reason:** MLflow's Model Registry needs a DB backend (file store cannot register models); SQLite satisfies this on Windows with zero extra RAM/disk while the stack is still light. D13's "mlflow service in Phase 4" is deferred, not abandoned.
- **Status:** decided.

## D16 · 2026-10-29 · Dependency pin adjustments (Phase 4)

- **Finding:** `mlflow==3.17.2` requires `pyarrow<26`, conflicting with the pinned `pyarrow==26.0.0`; `pip index` shows `mlflow==3.18.0` accepts `pyarrow<27,>=4.0.0`.
- **Choice:** bump `mlflow` to **3.18.0** (verified compatible with pyarrow 26.0.0 on Python 3.11).
- **Finding:** the `[ml]` install produced **`torch 2.14.1+cpu`** → `torch.cuda.is_available() == False`.
- **Implication:** Phase 5 GPU training requires a CUDA wheel per D12 (official CUDA index) and re-verification before any GPU claim. Recorded here so it is not forgotten.
- **Status:** decided — Phase 4 uses CPU torch only (fine for TF-IDF/eval).

## D17 · 2026-10-29 · GPU training in small resumable chunks

- **Context:** user reports sustained GPU load is undesirable; long fine-tuning runs were aborted mid-way.
- **Choice:** (a) install the CUDA wheel `torch==2.14.1+cu126` (verified `cuda_available=True`, RTX 3050, 4294 MB); (b) train with `--max-steps` + `--resume` so each invocation is a short chunk that saves a checkpoint and releases the GPU; (c) provide `scripts/train_chunks.ps1` (fixed `save_steps`/`eval_steps` cadence, pause between chunks).
- **Measured facts:** 2k/1-epoch DistilBERT smoke completed (val macro-F1 0.5108, 16.1s). Chunk resume verified: 50 steps (val F1 0.2779) then resumed to 100 steps (val F1 0.3906), checkpoint-100 saved, ~19s per chunk.
- **Consequence:** GPU runs are executed as many short chunks on demand; a full run is just a sequence of 250-step chunks.
- **Status:** decided.

## D18 · 2026-11-19 · Phase 5 training scope and production model

- **Context:** D17 requires small chunks on a 4 GB laptop GPU; the PRD's 200k/2-epoch run is impractical under that constraint.
- **Choice:** DistilBERT trained on a 60k sample for 1 epoch (effective batch 32: batch 4 × grad-accum 8, 1,875 steps in 8×250-step chunks). BERT-base trained on a 30k sample for 1 epoch (effective batch 16: batch 2 × grad-accum 8, 1,875 steps in 8×250-step chunks) after a 100-step smoke confirmed it fits the 4 GB card without OOM.
- **Measured facts (FR-6 harness):**
  - DistilBERT: `llm_test_1500` acc 0.8340 / macro-F1 0.7095 / p95 8.26 ms; `test_50k` acc 0.8398 / macro-F1 0.7090 / p95 8.47 ms.
  - BERT-base: `llm_test_1500` acc 0.8460 / macro-F1 0.6851 / p95 15.12 ms; `test_50k` acc 0.8575 / macro-F1 0.6987 / p95 16.50 ms.
  - GPU temperature peaked at 64 °C during training and 72 °C after the 50k inference pass; no OOM.
- **Choice (production, PRD 7.8):** DistilBERT — highest macro-F1 among models with p95 ≤ 300 ms and lower latency/cost than BERT. Registered in MLflow as `distilbert` v1 with alias `production`.
- **Implementation note:** `mlflow.transformers.log_model` pulls `torchvision` (not needed for text); registration uses a small `mlflow.pyfunc` wrapper (`transformer.py register`) instead, so no unrelated dependency was added.
- **Consequence:** results are on subsets, not the full 200k; the README/PRD state the exact sample sizes. Full-data training remains possible via the same chunk runner.
- **Status:** decided.

## D19 · 2026-11-19 · Phase 6 LLM model, quota reality and evaluation subset

- **Finding:** `.env` still named `llama-3.1-8b-instant`, which Groq retired from self-serve on 2026-08-16. `client.models.list()` (free) shows `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`.
- **Finding:** with `max_tokens=4`, `gpt-oss-20b` returns empty content because reasoning tokens consume the budget. The first 1000 calls were therefore all `parse_failure`; that cache file is archived locally (`.cache/llm/responses.invalid-max4.jsonl`) and the $0.0113 spent is kept in the ledger.
- **Fix:** `max_completion_tokens=64` + `reasoning_effort="low"` returns a clean one-word answer (~15-27 completion tokens); pacing set to 1.2 s to respect 8,000 tokens/min.
- **Finding:** free tier is 1,000 requests/day **per model**. Both Groq keys are in the same organization, so the backup key adds **no** quota (D3's fallback still helps with auth/key failures, not with RPD).
- **Choice:** evaluate `openai/gpt-oss-120b` (fresh quota) on a deterministic 900-row subset — the first 900 rows of `llm_test_1500` (616 positive / 193 negative / 91 neutral), saved as `data/processed/llm_test_900.parquet`. 900 instead of 1000 leaves a margin after probe requests used the daily quota. Prices: $0.15/M input, $0.60/M output (Groq model docs, read 2026-11-19).
- **Measured facts:** accuracy 0.8322, macro-F1 0.6463, parse_failure_rate 0.0111, p50 735 ms, p95 1424 ms (measured live, read back from cache on the write pass), cost $0.0358/1k. Total Phase 6 spend across both models: **$0.0436** (ledger `.cache/llm/spend.json`).
- **Also fixed:** `.env` `MLFLOW_TRACKING_URI` pointed at a non-running `http://localhost:5000`; set to `sqlite:///mlflow.db` (matches `.env.example` and the code default).
- **Consequence:** the LLM row is on 900 rows, not 1500; the README and results table state this explicitly. The remaining 600 rows can be added after the daily quota resets.
- **Status:** decided.

## D20 · 2026-11-19 · Phase 7 aspect extraction and draft gold set

- **Choice:** 16-aspect taxonomy in `configs/aspects.yaml` (battery, screen, camera, shipping, price, quality, fit, comfort, durability, customer_service, packaging, sound, performance, ease_of_use, design, app). Match by spaCy noun chunks + PhraseMatcher + lemma map; one mention per (aspect, sentence); sentence sentiment from the production DistilBERT.
- **spaCy model:** `en_core_web_sm==3.8.0` (compatible with spaCy 3.8.18). `spacy download` failed on a 15 s URL timeout; installed the official 3.8.0 wheel directly with pip (README documents it). It is not a PyPI package, so CI skips Phase 7 tests via `importorskip`.
- **Measured facts:** 5,000 reviews → **7,111 aspect rows** across **3,125 reviews** in ~7 min on CPU (GPU idle at 42 °C). Top aspects: design 1,334; price 1,204; ease_of_use 890; comfort 690; fit 617.
- **DB:** migration `0002_aspects_span.sql` adds `sentence_index`, `char_start`, `char_end`, `sentence`, `model_version`, `created_at` and a unique index on the span key. Loaded 7,111 rows; re-running inserted **0** (idempotent).
- **Gold set:** drafted 100 real sentences from the extracted sample, one aspect each (54 positive / 27 negative / 19 neutral), marked `draft-pending-owner-review`. Draft scores: aspect P 0.8929 / R 1.0000 / F1 0.9434, sentiment 1.0000. **These are circular** (labels came from the same model) and are not final; the PRD requires owner hand labels.
- **Consequence:** Phase 7 code, DB load and pipeline are done; the PRD checkbox stays open until the owner reviews/corrects `eval/aspect_gold.jsonl`.
- **Status:** decided, pending owner review of the gold labels.

---

## Open items (tracked, not decisions yet)

| Item | Needed by | Status |
|---|---|---|
| Category choice (D14) | Phase 1 | **decided 2026-10-27: Amazon_Fashion + All_Beauty** |
| Docker disk moved to D: (D6) | Phase 0 verify | waiting on user (GUI) |
| `MAX_LLM_SPEND_USD` value (default proposal: **$5.00**) | Phase 6 | proposed, user confirms at dry-run |
| Groq keys → `.env` | Phase 6 | **done** — keys in gitignored `.env` (Phase 0) |
| Pricing table `configs/pricing.yaml` real prices | Phase 4/6 | **Phase 4:** `hourly_rate_usd=0.10` set as an explicit configurable ASSUMPTION (self-hosted instance, not measured). LLM token prices still null until Phase 6. |
| Native Windows PostgreSQL 18 on :5432 | Phase 3 | service `postgresql-x64-18` Stopped + Disabled (elevated) so Docker Postgres owns 5432. Re-enable only if the user needs the native instance. |
