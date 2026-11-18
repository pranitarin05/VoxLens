"""FR-6: run one model through the shared harness and build the comparison files.

Writes results/model_comparison.csv and results/model_comparison.md. Rows are
upserted by (model_name, model_version, dataset) so re-running is idempotent.
Every evaluated run is logged to MLflow.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from voc.evaluation.cost import monthly_cost_at_1m, self_hosted_cost_per_1k
from voc.evaluation.latency import measure_latency
from voc.evaluation.metrics import compute_metrics
from voc.models.registry import configure

PREDICT_FN = Callable[[list[str]], Sequence[str]]

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "model_comparison.csv"
MD_PATH = RESULTS_DIR / "model_comparison.md"
KEY = ["model_name", "model_version", "dataset"]


def evaluate_and_log(
    predict_fn: PREDICT_FN,
    texts: Sequence[str],
    y_true: Sequence[str],
    *,
    model_name: str,
    model_version: str,
    dataset: str,
    usd_per_hour: float | None = None,
    latency_kwargs: dict[str, Any] | None = None,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score ``texts``, measure latency, compute cost, log to MLflow, return a row."""
    configure()
    latency = measure_latency(predict_fn, texts, **(latency_kwargs or {}))
    y_pred = list(predict_fn(list(texts)))
    metrics = compute_metrics(y_true, y_pred)

    preds_per_hour = float(latency["batch_throughput_per_s"]) * 3600.0
    cost_per_1k = self_hosted_cost_per_1k(preds_per_hour, usd_per_hour)

    row: dict[str, Any] = {
        "model_name": model_name,
        "model_version": model_version,
        "dataset": dataset,
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "single_p50_ms": latency["single_p50_ms"],
        "single_p95_ms": latency["single_p95_ms"],
        "single_p99_ms": latency["single_p99_ms"],
        "batch_p50_ms": latency["batch_p50_ms"],
        "batch_throughput_per_s": latency["batch_throughput_per_s"],
        "preds_per_hour": preds_per_hour,
        "assumed_usd_per_hour": usd_per_hour,
        "cost_per_1k_usd": cost_per_1k,
        "monthly_cost_usd": monthly_cost_at_1m(cost_per_1k),
    }
    for label, cls in metrics["per_class"].items():  # type: ignore[union-attr]
        row[f"precision_{label}"] = cls["precision"]
        row[f"recall_{label}"] = cls["recall"]
        row[f"f1_{label}"] = cls["f1"]

    with mlflow.start_run(run_name=f"{model_name}-{model_version}-{dataset}") as run:
        mlflow.log_params(
            {
                "model_name": model_name,
                "model_version": model_version,
                "dataset": dataset,
                "n": metrics["n"],
                **(extra_params or {}),
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "single_p50_ms": latency["single_p50_ms"],
                "single_p95_ms": latency["single_p95_ms"],
                "single_p99_ms": latency["single_p99_ms"],
                "batch_throughput_per_s": latency["batch_throughput_per_s"],
                **({"cost_per_1k_usd": cost_per_1k} if cost_per_1k is not None else {}),
            }
        )
        row["run_id"] = run.info.run_id

    return row


def upsert_comparison_row(
    row: dict[str, Any], csv_path: Path = CSV_PATH, md_path: Path = MD_PATH
) -> pd.DataFrame:
    """Insert or replace a row by its key, then rewrite the CSV and Markdown."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    if not frame.empty and all(k in frame.columns for k in KEY):
        mask = pd.Series(True, index=frame.index)
        for key in KEY:
            mask &= frame[key].astype(str) == str(row[key])
        frame = frame.loc[~mask]

    new_row = pd.DataFrame([row])
    frame = pd.concat([frame, new_row], ignore_index=True)
    frame = frame.sort_values(["dataset", "model_name", "model_version"]).reset_index(drop=True)
    frame.to_csv(csv_path, index=False)
    md_path.write_text(render_markdown(frame), encoding="utf-8")
    return frame


def render_markdown(frame: pd.DataFrame) -> str:
    """Render a compact headline comparison table plus per-class detail."""
    if frame.empty:
        return "# Model comparison\n\nNo runs yet. Run `make eval-tfidf`.\n"

    headline_cols = [
        c
        for c in [
            "model_name",
            "model_version",
            "dataset",
            "n",
            "accuracy",
            "macro_f1",
            "single_p50_ms",
            "single_p95_ms",
            "cost_per_1k_usd",
            "monthly_cost_usd",
        ]
        if c in frame.columns
    ]
    shown = frame[headline_cols].copy()
    for col in ["accuracy", "macro_f1"]:
        if col in shown:
            shown[col] = shown[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "n/a")
    for col in ["single_p50_ms", "single_p95_ms"]:
        if col in shown:
            shown[col] = shown[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
    for col in ["cost_per_1k_usd", "monthly_cost_usd"]:
        if col in shown:
            shown[col] = shown[col].map(lambda v: f"{v:.6f}" if pd.notna(v) else "n/a")

    lines = [
        "# Model comparison (measured)",
        "",
        "Headline metric is macro-F1; accuracy is shown beside it. Latency is",
        "single-request p50/p95 (ms). Cost uses the stated instance-price",
        "assumption in configs/pricing.yaml (edit it to your provider).",
        "",
        _to_md(shown),
        "",
    ]

    detail_cols = [c for c in frame.columns if c.startswith(("precision_", "recall_", "f1_"))]
    if detail_cols:
        detail = frame[["model_name", "model_version", "dataset", *detail_cols]].copy()
        for col in detail_cols:
            detail[col] = detail[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "n/a")
        lines += ["## Per-class metrics", "", _to_md(detail), ""]

    return "\n".join(lines)


def _to_md(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(str(val) for val in record) + " |"
        for record in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def _main() -> None:
    """Rebuild Markdown from an existing CSV (no-op if no CSV yet)."""
    if not CSV_PATH.exists():
        print(f"no {CSV_PATH} yet — run a model evaluation first")
        return
    frame = pd.read_csv(CSV_PATH)
    MD_PATH.write_text(render_markdown(frame), encoding="utf-8")
    print(f"wrote {MD_PATH} from {CSV_PATH} ({len(frame)} rows)")


if __name__ == "__main__":
    _main()
