"""PRD 7.6: warmup + timed single/batch latency protocol.

Measures end-to-end ``predict_fn`` time (tokenisation + forward + post-process).
No network. Reports p50/p95/p99 for single requests and a batch of 32.
"""

from __future__ import annotations

import itertools
import statistics
import time
from collections.abc import Callable, Sequence

import numpy as np

PredictFn = Callable[[list[str]], Sequence[str]]


def _percentiles(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
    }


def measure_latency(
    predict_fn: PredictFn,
    texts: Sequence[str],
    *,
    warmup: int = 50,
    n_single: int = 1000,
    batch_size: int = 32,
    n_batches: int = 31,
) -> dict[str, float | int]:
    """Run the latency protocol and return p50/p95/p99 for single + batch calls.

    ``n_batches=31`` timed batches of ``batch_size`` gives ~1000 predictions,
    matching the single-request sample size.
    """
    if not texts:
        raise ValueError("texts must not be empty")

    pool = list(texts)
    cycle = itertools.cycle(pool)

    for _ in range(warmup):
        predict_fn([next(cycle)])

    single_samples: list[float] = []
    for _ in range(n_single):
        text = next(cycle)
        start = time.perf_counter()
        predict_fn([text])
        single_samples.append((time.perf_counter() - start) * 1000.0)

    batch_samples: list[float] = []
    for _ in range(n_batches):
        batch = [next(cycle) for _ in range(batch_size)]
        start = time.perf_counter()
        predict_fn(batch)
        batch_samples.append((time.perf_counter() - start) * 1000.0)

    single = _percentiles(single_samples)
    batch = _percentiles(batch_samples)
    total_batch_s = sum(batch_samples) / 1000.0
    throughput = (n_batches * batch_size) / total_batch_s if total_batch_s else 0.0

    return {
        "warmup": warmup,
        "n_single": n_single,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "single_p50_ms": single["p50_ms"],
        "single_p95_ms": single["p95_ms"],
        "single_p99_ms": single["p99_ms"],
        "single_mean_ms": single["mean_ms"],
        "batch_p50_ms": batch["p50_ms"],
        "batch_p95_ms": batch["p95_ms"],
        "batch_p99_ms": batch["p99_ms"],
        "batch_mean_ms": batch["mean_ms"],
        "batch_throughput_per_s": float(throughput),
        "single_p50_median_check": statistics.median(single_samples),
    }
