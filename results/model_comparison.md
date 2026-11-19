# Model comparison (measured)

Headline metric is macro-F1; accuracy is shown beside it. Latency is
single-request p50/p95 (ms). Cost uses the stated instance-price
assumption in configs/pricing.yaml (edit it to your provider).

| model_name | model_version | dataset | n | accuracy | macro_f1 | single_p50_ms | single_p95_ms | cost_per_1k_usd | monthly_cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bert-30k | bert-30k | llm_test_1500 | 1500 | 0.8460 | 0.6851 | 9.50 | 15.12 | 0.000228 | 0.228428 |
| distilbert-60k | distilbert-60k | llm_test_1500 | 1500 | 0.8340 | 0.7095 | 4.78 | 8.26 | 0.000117 | 0.116627 |
| tfidf | tfidf-logreg-v1 | llm_test_1500 | 1500 | 0.8120 | 0.6837 | 0.62 | 0.77 | 0.000002 | 0.001519 |
| bert-30k | bert-30k | test_50k | 50000 | 0.8575 | 0.6987 | 9.73 | 16.50 | 0.000237 | 0.237119 |
| distilbert-60k | distilbert-60k | test_50k | 50000 | 0.8398 | 0.7090 | 4.98 | 8.47 | 0.000121 | 0.120750 |
| tfidf | tfidf-logreg-v1 | test_50k | 50000 | 0.8211 | 0.6979 | 0.62 | 0.77 | 0.000002 | 0.001515 |

## Per-class metrics

| model_name | model_version | dataset | precision_negative | recall_negative | f1_negative | precision_neutral | recall_neutral | f1_neutral | precision_positive | recall_positive | f1_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bert-30k | bert-30k | llm_test_1500 | 0.7463 | 0.8306 | 0.7862 | 0.3622 | 0.3194 | 0.3395 | 0.9374 | 0.9223 | 0.9298 |
| distilbert-60k | distilbert-60k | llm_test_1500 | 0.7360 | 0.7874 | 0.7608 | 0.3750 | 0.5417 | 0.4432 | 0.9649 | 0.8872 | 0.9244 |
| tfidf | tfidf-logreg-v1 | llm_test_1500 | 0.7568 | 0.7442 | 0.7504 | 0.3008 | 0.5347 | 0.3850 | 0.9673 | 0.8692 | 0.9156 |
| bert-30k | bert-30k | test_50k | 0.7438 | 0.8563 | 0.7961 | 0.4051 | 0.3269 | 0.3618 | 0.9461 | 0.9303 | 0.9381 |
| distilbert-60k | distilbert-60k | test_50k | 0.7572 | 0.8131 | 0.7841 | 0.3596 | 0.4907 | 0.4150 | 0.9633 | 0.8950 | 0.9279 |
| tfidf | tfidf-logreg-v1 | test_50k | 0.7547 | 0.7654 | 0.7601 | 0.3335 | 0.5559 | 0.4169 | 0.9649 | 0.8732 | 0.9168 |
