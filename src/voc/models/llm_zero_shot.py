"""PRD 7.5: provider-agnostic LLMClient with cache, retries and spend cap.

Design notes (all configurable, nothing hidden):
- Prompt asks for exactly one of the three labels, temperature 0 and a tiny
  output limit. The model name comes from ``GROQ_MODEL`` (see .env.example).
- Every request is cached on disk under a hash of (model, prompt), so re-runs
  cost nothing.
- The primary key is tried first; the backup key is used on auth/permission
  failures or sustained rate limits (D3).
- Cost is computed from the token usage returned by the API multiplied by the
  list prices in configs/pricing.yaml. ``MAX_LLM_SPEND_USD`` is a hard cap:
  the client aborts before/after a call when the accumulated spend would cross
  it. Spend persists across runs in ``.cache/llm/spend.json``.
- Unparseable outputs are recorded as ``parse_failure`` and counted in
  ``parse_failure_rate``; they count as wrong predictions.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv

from voc.evaluation.compare import upsert_comparison_row
from voc.evaluation.cost import llm_cost_per_1k, load_pricing, monthly_cost_at_1m
from voc.evaluation.metrics import LABELS, compute_metrics
from voc.models.registry import configure

PARSE_FAILURE = "parse_failure"
SYSTEM_PROMPT = (
    "You are a strict sentiment classifier. Reply with exactly one word: "
    "negative, neutral, or positive."
)
USER_TEMPLATE = "Review:\n{text}\n\nLabel:"
MAX_CHARS = 2000

CACHE_DIR = Path(".cache/llm")
CACHE_PATH = CACHE_DIR / "responses.jsonl"
SPEND_PATH = CACHE_DIR / "spend.json"
# Bump when prompt/generation settings change so stale cache entries are ignored.
CACHE_VERSION = "v2-gpt-oss-low-64"

app = typer.Typer(add_completion=False, help="LLM zero-shot sentiment (Groq).")


class SpendCapExceeded(RuntimeError):
    """Raised when MAX_LLM_SPEND_USD would be exceeded."""


class LLMError(RuntimeError):
    """Raised when the provider cannot return a usable response."""


@dataclass(frozen=True)
class Prediction:
    label: str
    raw: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    cached: bool


Transport = Callable[[str, str], tuple[str, int, int]]


def build_prompt(text: str) -> str:
    """Deterministic user prompt; truncation keeps cost bounded."""
    return USER_TEMPLATE.format(text=str(text)[:MAX_CHARS])


def cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{CACHE_VERSION}|{model}|{prompt}".encode()).hexdigest()


def _header(exc: Exception, name: str) -> str | None:
    """Read a response header (case-insensitive) from an SDK exception."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    for key, value in headers.items():
        if key.lower() == name:
            return str(value)
    return None


def parse_label(raw: str) -> str:
    """Return one of LABELS or PARSE_FAILURE (tolerant of punctuation)."""
    for token in re.split(r"[^a-z]+", (raw or "").strip().lower()):
        if token in LABELS:
            return token
    return PARSE_FAILURE


def _price_pair(pricing: dict[str, Any]) -> tuple[float | None, float | None]:
    return (
        pricing.get("llm_input_per_mtok_usd"),  # type: ignore[return-value]
        pricing.get("llm_output_per_mtok_usd"),  # type: ignore[return-value]
    )


def usage_cost_usd(input_tokens: int, output_tokens: int, pricing: dict[str, Any]) -> float:
    """Call-level cost from measured tokens and configured list prices."""
    price_in, price_out = _price_pair(pricing)
    if price_in is None or price_out is None:
        raise LLMError("LLM prices are unset in configs/pricing.yaml")
    return (input_tokens / 1_000_000.0) * float(price_in) + (output_tokens / 1_000_000.0) * float(
        price_out
    )


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return entries
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            entries[record["key"]] = record
    return entries


def _load_spend(path: Path) -> float:
    if not path.exists():
        return 0.0
    return float(json.loads(path.read_text(encoding="utf-8")).get("spent_usd", 0.0))


def _save_spend(path: Path, spent: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"spent_usd": spent}), encoding="utf-8")


def mock_transport(prompt: str, api_key: str) -> tuple[str, int, int]:
    """Deterministic lexicon stand-in for dry runs/tests (not a real model)."""
    text = prompt.lower()
    negative = ("broke", "broken", "terrible", "awful", "hate", "waste", "poor", "bad")
    positive = ("love", "great", "perfect", "excellent", "amazing", "best", "recommend")
    if any(word in text for word in negative):
        raw = "negative"
    elif any(word in text for word in positive):
        raw = "positive"
    else:
        raw = "neutral"
    return raw, max(1, len(prompt) // 4), 1


class LLMClient:
    """Cache-first chat client with retries, key fallback and a spend cap."""

    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str = "groq",
        max_spend_usd: float | None = None,
        cache_path: Path = CACHE_PATH,
        spend_path: Path = SPEND_PATH,
        transport: Transport | None = None,
        min_interval_s: float = 1.2,
        max_retries: int = 4,
        temperature: float = 0.0,
        max_completion_tokens: int = 64,
        reasoning_effort: str = "low",
        pricing: dict[str, Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        load_dotenv(override=False)
        self.pricing = pricing or load_pricing()
        self.model = model or os.getenv("GROQ_MODEL") or self.pricing.get("llm_model")
        if not self.model:
            raise LLMError("no LLM model: set GROQ_MODEL or pass --model")
        cap = max_spend_usd if max_spend_usd is not None else os.getenv("MAX_LLM_SPEND_USD")
        if cap is None:
            raise LLMError("MAX_LLM_SPEND_USD is not set; refusing to call a paid API")
        self.max_spend_usd = float(cap)
        self.provider = provider
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.cache_path = cache_path
        self.spend_path = spend_path
        self.sleep = sleep
        self.transport = transport
        self.keys = [
            key
            for key in (
                os.getenv("GROQ_API_KEY"),
                os.getenv("GROQ_API_KEY_BACKUP"),
            )
            if key
        ]
        if provider == "groq" and self.transport is None and not self.keys:
            raise LLMError("GROQ_API_KEY is not set")
        self.cache = _load_cache(cache_path)
        self.spent_usd = _load_spend(spend_path)
        self.run_spend_usd = 0.0
        self.live_calls = 0
        self.cache_hits = 0
        self._last_call = 0.0
        self._key_index = 0

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval_s:
            self.sleep(self.min_interval_s - elapsed)
        self._last_call = time.monotonic()

    def _call_once(self, prompt: str, api_key: str) -> tuple[str, int, int, float]:
        start = time.perf_counter()
        if self.transport is not None:
            raw, tokens_in, tokens_out = self.transport(prompt, api_key)
        else:
            from groq import Groq

            client = Groq(api_key=api_key, max_retries=0)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                reasoning_effort=self.reasoning_effort,
            )
            raw = response.choices[0].message.content or ""
            tokens_in = int(response.usage.prompt_tokens)
            tokens_out = int(response.usage.completion_tokens)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return raw, tokens_in, tokens_out, latency_ms

    def _call(self, prompt: str) -> tuple[str, int, int, float]:
        if self.transport is not None:
            if self.spent_usd >= self.max_spend_usd:
                raise SpendCapExceeded(
                    f"spend ${self.spent_usd:.4f} >= cap ${self.max_spend_usd:.2f}"
                )
            self._respect_rate_limit()
            key = self.keys[0] if self.keys else "mock"
            return self._call_once(prompt, key)

        from groq import (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            InternalServerError,
            PermissionDeniedError,
            RateLimitError,
        )

        retryable = (APIConnectionError, APITimeoutError, InternalServerError)
        auth = (AuthenticationError, PermissionDeniedError)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if self.spent_usd >= self.max_spend_usd:
                raise SpendCapExceeded(
                    f"spend ${self.spent_usd:.4f} >= cap ${self.max_spend_usd:.2f}"
                )
            self._respect_rate_limit()
            key = self.keys[self._key_index] if self.keys else ""
            try:
                return self._call_once(prompt, key)
            except auth as exc:
                if self._key_index + 1 < len(self.keys):
                    self._key_index += 1
                    last_error = exc
                    continue
                raise LLMError(f"authentication failed: {exc}") from exc
            except RateLimitError as exc:
                last_error = exc
                remaining = _header(exc, "x-ratelimit-remaining-requests")
                if remaining == "0" and len(self.keys) > 1 and attempt < self.max_retries:
                    self._key_index = (self._key_index + 1) % len(self.keys)
                    continue
                if attempt >= self.max_retries:
                    break
                delay = min(30.0, 1.5 * (2**attempt)) + random.uniform(0, 0.3)
                self.sleep(delay)
            except retryable as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = min(30.0, 1.5 * (2**attempt)) + random.uniform(0, 0.3)
                self.sleep(delay)
        raise LLMError(f"provider failed after retries: {last_error}")

    def predict(self, text: str) -> Prediction:
        prompt = build_prompt(text)
        key = cache_key(str(self.model), prompt)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return Prediction(
                label=cached["label"],
                raw=cached["raw"],
                input_tokens=int(cached["input_tokens"]),
                output_tokens=int(cached["output_tokens"]),
                latency_ms=float(cached["latency_ms"]),
                cost_usd=0.0,
                cached=True,
            )

        raw, tokens_in, tokens_out, latency_ms = self._call(prompt)
        label = parse_label(raw)
        cost = usage_cost_usd(tokens_in, tokens_out, self.pricing)
        self.spent_usd += cost
        self.run_spend_usd += cost
        self.live_calls += 1
        _save_spend(self.spend_path, self.spent_usd)

        record = {
            "key": key,
            "model": str(self.model),
            "raw": raw,
            "label": label,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "latency_ms": latency_ms,
            "cost_usd": cost,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.cache[key] = record
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if self.spent_usd > self.max_spend_usd:
            raise SpendCapExceeded(
                f"spend ${self.spent_usd:.4f} exceeded cap ${self.max_spend_usd:.2f}"
            )
        return Prediction(
            label=label,
            raw=raw,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost,
            cached=False,
        )


def evaluate_dataset(client: LLMClient, dataset: str, limit: int | None = None) -> dict[str, Any]:
    """Run zero-shot over an eval parquet; returns metrics + measured usage."""
    path = Path("data/processed") / f"{dataset}.parquet"
    frame = pd.read_parquet(path, columns=["text", "label"])
    if limit is not None:
        frame = frame.head(limit)

    predictions: list[Prediction] = []
    for i, text in enumerate(frame["text"].tolist(), start=1):
        predictions.append(client.predict(text))
        if i % 100 == 0:
            print(f"{dataset}: {i}/{len(frame)}")

    y_pred = [p.label for p in predictions]
    metrics = compute_metrics(frame["label"].tolist(), y_pred)
    live_latencies = [p.latency_ms for p in predictions if not p.cached]
    latency_source = "live"
    if not live_latencies:
        live_latencies = [p.latency_ms for p in predictions]
        latency_source = "cache"
    live_tokens = [p for p in predictions if not p.cached]
    avg_in = float(np.mean([p.input_tokens for p in predictions]))
    avg_out = float(np.mean([p.output_tokens for p in predictions]))
    price_in, price_out = _price_pair(client.pricing)
    cost_per_1k = llm_cost_per_1k(avg_in, avg_out, price_in, price_out)

    return {
        "metrics": metrics,
        "n": len(predictions),
        "parse_failure_rate": float(np.mean([p.label == PARSE_FAILURE for p in predictions])),
        "cache_hits": client.cache_hits,
        "live_calls": client.live_calls,
        "avg_input_tokens": avg_in,
        "avg_output_tokens": avg_out,
        "p50_ms": float(np.percentile(live_latencies, 50)) if live_latencies else None,
        "p95_ms": float(np.percentile(live_latencies, 95)) if live_latencies else None,
        "p99_ms": float(np.percentile(live_latencies, 99)) if live_latencies else None,
        "cost_per_1k_usd": cost_per_1k,
        "monthly_cost_usd": monthly_cost_at_1m(cost_per_1k),
        "run_spend_usd": client.run_spend_usd,
        "cumulative_spend_usd": client.spent_usd,
        "latency_source": latency_source,
        "live_latency_n": len(live_latencies),
        "avg_live_input_tokens": float(np.mean([p.input_tokens for p in live_tokens]))
        if live_tokens
        else None,
    }


@app.callback()
def _root() -> None:
    """LLM zero-shot sentiment (Groq)."""


@app.command()
def evaluate(
    dataset: Annotated[str, typer.Option()] = "llm_test_1500",
    limit: Annotated[int | None, typer.Option(help="only first N rows")] = None,
    model: Annotated[str | None, typer.Option(help="provider model id")] = None,
    dry_run: Annotated[bool, typer.Option(help="mock transport, no API spend")] = False,
    write: Annotated[bool, typer.Option("--write/--no-write")] = True,
) -> None:
    """Score the LLM zero-shot model and upsert its comparison row."""
    if dry_run:
        client = LLMClient(
            model=model or "mock-dry-run",
            provider="mock",
            max_spend_usd=1.0,
            transport=mock_transport,
            cache_path=Path(".cache/llm/dry_responses.jsonl"),
            spend_path=Path(".cache/llm/dry_spend.json"),
        )
    else:
        client = LLMClient(model=model)

    result = evaluate_dataset(client, dataset, limit)
    metrics = result["metrics"]
    print(
        f"{dataset}: acc={metrics['accuracy']:.4f} macroF1={metrics['macro_f1']:.4f} "
        f"parse_failure_rate={result['parse_failure_rate']:.4f} "
        f"spend=${result['run_spend_usd']:.6f} "
        f"cache_hits={result['cache_hits']} live_calls={result['live_calls']}"
    )
    if dry_run:
        print("dry run: no comparison row written, no MLflow run")
        return
    if not write:
        print("no comparison row written (--no-write)")
        return

    row: dict[str, Any] = {
        "model_name": "llm-zero-shot",
        "model_version": str(client.model),
        "dataset": dataset,
        "n": result["n"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "single_p50_ms": result["p50_ms"],
        "single_p95_ms": result["p95_ms"],
        "single_p99_ms": result["p99_ms"],
        "batch_p50_ms": None,
        "batch_throughput_per_s": None,
        "preds_per_hour": None,
        "assumed_usd_per_hour": None,
        "cost_per_1k_usd": result["cost_per_1k_usd"],
        "monthly_cost_usd": result["monthly_cost_usd"],
        "avg_input_tokens": result["avg_input_tokens"],
        "avg_output_tokens": result["avg_output_tokens"],
        "parse_failure_rate": result["parse_failure_rate"],
        "run_spend_usd": result["run_spend_usd"],
        "cumulative_spend_usd": result["cumulative_spend_usd"],
        "latency_source": result["latency_source"],
        "live_latency_n": result["live_latency_n"],
    }
    for label, cls in metrics["per_class"].items():  # type: ignore[union-attr]
        row[f"precision_{label}"] = cls["precision"]
        row[f"recall_{label}"] = cls["recall"]
        row[f"f1_{label}"] = cls["f1"]

    configure()
    import mlflow

    with mlflow.start_run(run_name=f"llm-zero-shot-{dataset}") as run:
        mlflow.log_params(
            {
                "model": str(client.model),
                "dataset": dataset,
                "n": result["n"],
                "temperature": client.temperature,
                "max_completion_tokens": client.max_completion_tokens,
                "reasoning_effort": client.reasoning_effort,
                "parse_failure_rate": result["parse_failure_rate"],
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "parse_failure_rate": result["parse_failure_rate"],
                "avg_input_tokens": result["avg_input_tokens"],
                "avg_output_tokens": result["avg_output_tokens"],
                "run_spend_usd": result["run_spend_usd"],
                **(
                    {"cost_per_1k_usd": result["cost_per_1k_usd"]}
                    if result["cost_per_1k_usd"] is not None
                    else {}
                ),
            }
        )
        row["run_id"] = run.info.run_id
    upsert_comparison_row(row)
    print(f"wrote comparison row; MLflow run {row['run_id']}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
