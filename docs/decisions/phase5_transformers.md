# Phase 5: transformer scope and production choice

- DistilBERT: 60k sample, 1 epoch, effective batch 32 (4 x grad-accum 8), 1,875 steps.
- BERT-base: 30k sample, 1 epoch, effective batch 16 (2 x grad-accum 8), 1,875 steps.
- Both runs were split into 250-step resumable GPU chunks; peak GPU temperature 64 C (72 C after the 50k inference pass); no OOM.

Measured with the shared FR-6 harness (see `results/model_comparison.csv`):

| model | dataset | accuracy | macro-F1 | p95 (ms) |
|---|---|---|---|---|
| distilbert-60k | llm_test_1500 | 0.8340 | 0.7095 | 8.26 |
| distilbert-60k | test_50k | 0.8398 | 0.7090 | 8.47 |
| bert-30k | llm_test_1500 | 0.8460 | 0.6851 | 15.12 |
| bert-30k | test_50k | 0.8575 | 0.6987 | 16.50 |

Production (PRD 7.8): DistilBERT, registered as `distilbert` v1 with alias `production`.
Registration uses a `mlflow.pyfunc` wrapper because `mlflow.transformers.log_model` requires `torchvision`, which text classification does not need.
