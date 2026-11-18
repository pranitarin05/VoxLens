"""Phase 4: shared evaluation harness + TF-IDF baseline (no MLflow needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from voc.evaluation.compare import render_markdown, upsert_comparison_row
from voc.evaluation.cost import (
    llm_cost_per_1k,
    monthly_cost_at_1m,
    self_hosted_cost_per_1k,
)
from voc.evaluation.latency import measure_latency
from voc.evaluation.metrics import compute_metrics
from voc.models.tfidf import build_pipeline


def test_metrics_known_values() -> None:
    y_true = ["negative", "negative", "neutral", "positive", "positive"]
    y_pred = ["negative", "positive", "neutral", "positive", "positive"]
    result = compute_metrics(y_true, y_pred)
    assert result["n"] == 5
    assert result["accuracy"] == pytest.approx(0.8)
    assert set(result["per_class"]) == {"negative", "neutral", "positive"}
    assert result["per_class"]["neutral"]["f1"] == pytest.approx(1.0)
    assert result["per_class"]["negative"]["support"] == 2
    assert len(result["confusion_matrix"]) == 3


def test_metrics_length_mismatch() -> None:
    with pytest.raises(ValueError):
        compute_metrics(["negative"], ["negative", "positive"])


def test_metrics_empty() -> None:
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_self_hosted_cost_math() -> None:
    cost = self_hosted_cost_per_1k(predictions_per_hour=3600.0, usd_per_hour=0.10)
    assert cost == pytest.approx(0.10 / 3600.0 * 1000.0)
    assert monthly_cost_at_1m(cost) == pytest.approx(cost * 1000.0)


def test_cost_none_when_price_unset() -> None:
    assert self_hosted_cost_per_1k(3600.0, None) is None
    assert monthly_cost_at_1m(None) is None
    assert llm_cost_per_1k(100, 10, None, None) is None


def test_llm_cost_math() -> None:
    cost = llm_cost_per_1k(
        avg_input_tokens=100.0,
        avg_output_tokens=5.0,
        input_per_mtok_usd=0.59,
        output_per_mtok_usd=0.79,
    )
    expected = (100 / 1_000_000 * 0.59 + 5 / 1_000_000 * 0.79) * 1000
    assert cost == pytest.approx(expected)


def test_latency_protocol_keys() -> None:
    calls = {"n": 0}

    def fake_predict(texts: list[str]) -> list[str]:
        calls["n"] += 1
        return ["neutral"] * len(texts)

    result = measure_latency(
        fake_predict,
        ["a", "b", "c"],
        warmup=2,
        n_single=5,
        batch_size=32,
        n_batches=2,
    )
    for key in ("single_p50_ms", "single_p95_ms", "single_p99_ms"):
        assert key in result
        assert result[key] >= 0.0
    assert result["batch_size"] == 32
    assert result["batch_throughput_per_s"] > 0.0
    assert calls["n"] == 2 + 5 + 2


def test_upsert_is_idempotent(tmp_path) -> None:
    csv_path = tmp_path / "cmp.csv"
    md_path = tmp_path / "cmp.md"
    row = {
        "model_name": "tfidf",
        "model_version": "v1",
        "dataset": "llm_test_1500",
        "n": 1500,
        "accuracy": 0.8,
        "macro_f1": 0.7,
        "single_p50_ms": 1.0,
        "single_p95_ms": 2.0,
        "cost_per_1k_usd": 0.01,
        "monthly_cost_usd": 10.0,
    }
    upsert_comparison_row(row, csv_path=csv_path, md_path=md_path)
    updated = dict(row, accuracy=0.81)
    frame = upsert_comparison_row(updated, csv_path=csv_path, md_path=md_path)
    assert len(frame) == 1
    assert frame.iloc[0]["accuracy"] == pytest.approx(0.81)
    md = md_path.read_text(encoding="utf-8")
    assert "Model comparison" in md
    assert "macro_f1" in md


def test_render_markdown_empty() -> None:
    text = render_markdown(pd.DataFrame())
    assert "No runs yet" in text


def test_tfidf_pipeline_trains_and_predicts() -> None:
    docs = ["great good love"] * 10 + ["bad hate terrible"] * 10
    labels = ["positive"] * 10 + ["negative"] * 10
    pipeline = build_pipeline(1.0)
    pipeline.fit(docs, labels)
    preds = pipeline.predict(["great good", "bad hate"])
    assert list(preds) == ["positive", "negative"]
    proba = pipeline.predict_proba(["great good"])[0]
    assert np.isclose(float(np.sum(proba)), 1.0, atol=1e-5)
