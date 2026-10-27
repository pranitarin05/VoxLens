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
- **Choice:** **pending user pick** — options offered: (A) Amazon_Fashion + All_Beauty = 3.2M reviews / 373 MB, (B) + Appliances = 5.3M / 631 MB, (C) user's own 2–3.
- **Reason:** PRD 4.1 requires ≥1M total from 2–3 categories; PRD Phase 1 requires waiting for user's choice.
- **Status:** **open — blocks Phase 1 only** (Phase 0 can proceed).

---

## Open items (tracked, not decisions yet)

| Item | Needed by | Status |
|---|---|---|
| Category choice (D14) | Phase 1 | waiting on user |
| Docker disk moved to D: (D6) | Phase 0 verify | waiting on user (GUI) |
| `MAX_LLM_SPEND_USD` value (default proposal: **$5.00**) | Phase 6 | proposed, user confirms at dry-run |
| Groq keys → `.env` | Phase 6 | keys received in `.env.txt`; will be copied to gitignored `.env` in Phase 0 |
| Pricing table `configs/pricing.yaml` real prices | Phase 4/6 | placeholder first, filled with dated real prices before any cost claim |
