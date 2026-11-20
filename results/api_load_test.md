# API load test (Phase 8)

Measured on 2026-11-19 with `scripts/load_test_api.py`, against a local uvicorn
process serving the production DistilBERT plus live aspect extraction.

- Endpoint: `POST /v1/predict` (single request, sequential httpx client)
- Payload: one review (~80 characters), production model `distilbert-60k-v1`
- Warm-up: 20 requests, then 200 timed requests
- Hardware: AMD Ryzen 7 5800HS (8 cores / 16 threads), 15.4 GB RAM, CPU inference
- Host: Windows, Python 3.11.9, torch 2.14.1

| Metric | Measured |
|---|---|
| Errors | 0 / 200 |
| p50 | 222.977 ms |
| p95 | 239.986 ms |
| p99 | 248.929 ms |
| Throughput | 4.46 req/s |

Target from PRD section 14: production transformer p95 **under 300 ms** on CPU
for a single request. This run meets it (239.986 ms).

Caveats (stated honestly):

- The timing includes spaCy aspect extraction, DistilBERT sentence scoring,
  tokenisation and the best-effort `api_requests` insert. Model-only latency is
  much lower: Phase 5 measured DistilBERT p95 **8.47 ms** on `test_50k`.
- The request is latency-bound by spaCy; batch requests amortise the model work
  but still run spaCy per text.
- The test is sequential from one client; it is not a concurrency benchmark.

The same run added **220 rows** to `api_requests` (1 pre-existing row + 220 =
221), confirming every predict call is recorded.
