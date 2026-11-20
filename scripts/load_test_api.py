"""Phase 8 load test: measure p50/p95/p99 against a running API (no invented numbers)."""

from __future__ import annotations

import argparse
import json
import time

import httpx

TEXT = "The battery life is great but shipping was slow and the box arrived damaged."


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    payload = {"text": TEXT}
    if args.model:
        payload["model"] = args.model

    latencies: list[float] = []
    errors = 0
    with httpx.Client(base_url=args.url, timeout=60) as client:
        for _ in range(args.warmup):
            client.post("/v1/predict", json=payload).raise_for_status()
        started = time.perf_counter()
        for index in range(args.n):
            body = {**payload, "text": f"{TEXT} [{index}]"}
            begin = time.perf_counter()
            response = client.post("/v1/predict", json=body)
            elapsed = (time.perf_counter() - begin) * 1000
            if response.status_code == 200:
                latencies.append(elapsed)
            else:
                errors += 1
        wall = time.perf_counter() - started

    if not latencies:
        raise SystemExit("no successful requests; is the API running?")
    print(
        json.dumps(
            {
                "url": args.url,
                "model": args.model or "production",
                "n": args.n,
                "warmup": args.warmup,
                "errors": errors,
                "p50_ms": round(percentile(latencies, 50), 3),
                "p95_ms": round(percentile(latencies, 95), 3),
                "p99_ms": round(percentile(latencies, 99), 3),
                "throughput_rps": round(len(latencies) / wall, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
