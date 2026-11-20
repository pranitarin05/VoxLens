# Phase 9 - RAG question answering

Scope: PRD section 10 (RAG pipeline: sample, embeddings, retrieval, grounded
answer, abstention, analytical router) and the `/v1/ask` endpoint.

## Pipeline

1. **Sample** (`voc.rag.embed`): 200,000 reviews from `data/processed/reviews.parquet`,
   stratified by `(category, label)` with proportional quotas via the
   largest-remainder method (seed 42, deterministic md5 row order inside each
   stratum). Quotas are capped by stratum size.
2. **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions,
   `normalize_embeddings=True`, CPU), inserted into `review_embeddings`
   (`VECTOR(384)`). The load is idempotent: existing ids are skipped,
   duplicates hit `ON CONFLICT DO NOTHING`, and the HNSW index is created after
   the bulk insert.
3. **Retrieval** (`voc.rag.retrieve`): cosine HNSW top-20, keep the best 8.
   Filters (`category`, `product_id`, `sentiment`, `start`, `end`) are
   whitelisted WHERE fragments with psycopg parameters only.
4. **Routing** (`voc.rag.router`): analytical questions (`how many`, `count`,
   `trend`, `compare`, `top`, `most`, `which product`, ...) use whitelisted SQL
   templates; everything else goes to RAG. Filters come from the database
   (categories, aspects) or validated literals (sentiment, 4-digit year) and
   always travel as parameters. The LLM never writes SQL.
5. **Answers** (`voc.rag.answer`): SQL facts are phrased by the LLM and must
   not invent numbers; RAG answers must be a JSON object with `answer`,
   `citations` and `abstained`, may only cite 1..8 and must abstain when the
   snippets are insufficient. Invalid output becomes an explicit abstention with
   `parse_failed=true`.
6. **API** (`/v1/ask`): returns answer, sources (id, snippet, score, label,
   category), `abstained`, `route`, latency and cost. LLM/spend errors and DB
   outages map to `503`; `API_RAG=0` returns `501`.

## Measured facts

| Item | Command | Result |
| --- | --- | --- |
| Sample + embeddings | `python -m voc.rag.embed build` | 200,000 rows, dim 384, 0 nulls |
| HNSW build | same run | `idx_review_embeddings_hnsw` created in 269 s |
| Model load (startup) | first query | 26.8 s CPU |
| Query embed + retrieve | `retrieve(keep=8)` | 0.084 s |
| Retrieval quality | `python -m voc.rag.evaluate` | hit@8 96.7% (30 questions) |
| Analytical correctness | same | 20/20 facts equal to independent SQL |
| Abstention | same | 8/10 unanswerable questions abstained |
| Faithfulness | LLM judge | 93.1% of 29 judged answers |
| Live cached `/v1/ask` | SQL question | ~1.0 s wall (count query scans 3.0 M rows) |
| Live uncached `/v1/ask` | RAG question | 2.4 s wall / 2.06 s service time, $0.00018 |

Evaluation details, per-question tables and the spend ledger are in
`results/rag_eval.md` / `results/rag_eval.json`.

## Limitations (honest)

- The 30 qualitative reference key points, the LLM-judge faithfulness labels
  and the unanswerable set were drafted by the agent and are marked
  `draft-pending-owner-review`; the PRD still requires the owner's manual spot
  check of 20 answers.
- hit@8 is a substring proxy (a gold key point in a retrieved snippet), not a
  human relevance grade.
- Two unanswerable questions were answered instead of abstained (lifetime
  warranty, subscription refill); 8/10 is reported, not rounded up.
- One judge call failed on a provider connection error and is excluded from the
  faithfulness denominator (29 judged of 30).
- Counts by category are seq scans today; an index on
  `(category, label, reviewed_at)` would help if `/v1/ask` latency matters.
- The 200k sample is the retrieval corpus; the SQL router still answers from
  the full 3.02 M reviews.
