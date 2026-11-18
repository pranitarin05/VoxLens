# Model comparison (measured)

Headline metric is macro-F1; accuracy is shown beside it. Latency is
single-request p50/p95 (ms). Cost uses the stated instance-price
assumption in configs/pricing.yaml (edit it to your provider).

| model_name | model_version | dataset | n | accuracy | macro_f1 | single_p50_ms | single_p95_ms | cost_per_1k_usd | monthly_cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tfidf | tfidf-logreg-v1 | llm_test_1500 | 1500 | 0.8120 | 0.6837 | 0.62 | 0.77 | 0.000002 | 0.001519 |
| tfidf | tfidf-logreg-v1 | test_50k | 50000 | 0.8211 | 0.6979 | 0.62 | 0.77 | 0.000002 | 0.001515 |

## Per-class metrics

| model_name | model_version | dataset | precision_negative | recall_negative | f1_negative | precision_neutral | recall_neutral | f1_neutral | precision_positive | recall_positive | f1_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tfidf | tfidf-logreg-v1 | llm_test_1500 | 0.7568 | 0.7442 | 0.7504 | 0.3008 | 0.5347 | 0.3850 | 0.9673 | 0.8692 | 0.9156 |
| tfidf | tfidf-logreg-v1 | test_50k | 0.7547 | 0.7654 | 0.7601 | 0.3335 | 0.5559 | 0.4169 | 0.9649 | 0.8732 | 0.9168 |
