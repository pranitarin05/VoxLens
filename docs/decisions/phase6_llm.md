# Phase 6: LLM zero-shot

- Model: `openai/gpt-oss-120b` (Groq), temperature 0, `max_completion_tokens=64`, `reasoning_effort="low"`, one-word prompt.
- Subset: deterministic first 900 rows of `llm_test_1500` (`data/processed/llm_test_900.parquet`, 616 positive / 193 negative / 91 neutral). The free tier allows 1,000 requests/day/model and both keys share one organization, so 1500 rows were not possible in one day.
- Prices: $0.15/M input, $0.60/M output (Groq model docs, read 2026-11-19).
- Measured: accuracy 0.8322, macro-F1 0.6463, parse_failure_rate 0.0111, p50 735 ms, p95 1424 ms, $0.0358 per 1k calls.
- Spend: $0.0436 total, hard cap `MAX_LLM_SPEND_USD` never exceeded; responses cached in `.cache/llm/responses.jsonl` (reruns are free).
- Note: `gpt-oss-20b` with `max_tokens=4` returned empty content (reasoning consumed the limit); the client now uses `max_completion_tokens` and a low reasoning effort.
