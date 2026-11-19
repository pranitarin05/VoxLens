"""Phase 6: LLM client cache, spend cap, parsing and cost arithmetic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voc.evaluation.cost import llm_cost_per_1k
from voc.evaluation.metrics import compute_metrics
from voc.models.llm_zero_shot import (
    PARSE_FAILURE,
    LLMClient,
    LLMError,
    SpendCapExceeded,
    build_prompt,
    cache_key,
    parse_label,
    usage_cost_usd,
)

PRICING = {
    "llm_model": "test-model",
    "llm_input_per_mtok_usd": 0.075,
    "llm_output_per_mtok_usd": 0.30,
}


def make_client(tmp_path: Path, transport, cap: float = 1.0, **kwargs) -> LLMClient:
    return LLMClient(
        model="test-model",
        provider="mock",
        max_spend_usd=cap,
        transport=transport,
        pricing=PRICING,
        cache_path=tmp_path / "responses.jsonl",
        spend_path=tmp_path / "spend.json",
        min_interval_s=0.0,
        sleep=lambda _: None,
        **kwargs,
    )


def test_prompt_and_cache_key_are_deterministic() -> None:
    assert build_prompt("Great phone") == build_prompt("Great phone")
    assert cache_key("m", "p") == cache_key("m", "p")
    assert cache_key("m", "p") != cache_key("m", "q")


def test_parse_label_accepts_punctuation_and_rejects_unknown() -> None:
    assert parse_label(" Positive. ") == "positive"
    assert parse_label("negative") == "negative"
    assert parse_label("banana") == PARSE_FAILURE
    assert parse_label("") == PARSE_FAILURE


def test_usage_cost_uses_measured_tokens() -> None:
    assert usage_cost_usd(1_000_000, 0, PRICING) == pytest.approx(0.075)
    assert usage_cost_usd(0, 1_000_000, PRICING) == pytest.approx(0.30)
    assert usage_cost_usd(200, 4, PRICING) == pytest.approx(200 / 1e6 * 0.075 + 4 / 1e6 * 0.30)


def test_disk_cache_makes_reruns_free(tmp_path: Path) -> None:
    calls: list[str] = []

    def transport(prompt: str, api_key: str) -> tuple[str, int, int]:
        calls.append(prompt)
        return "positive", 100, 1

    client = make_client(tmp_path, transport)
    first = client.predict("I love this")
    second = client.predict("I love this")

    assert len(calls) == 1
    assert first.cached is False
    assert second.cached is True
    assert second.cost_usd == 0.0
    assert second.label == "positive"
    assert client.cache_hits == 1

    reopened = make_client(tmp_path, transport)
    third = reopened.predict("I love this")
    assert third.cached is True
    assert len(calls) == 1


def test_parse_failure_is_recorded_and_counts_as_wrong(tmp_path: Path) -> None:
    client = make_client(tmp_path, lambda prompt, key: ("banana", 10, 1))
    prediction = client.predict("I love this")
    assert prediction.label == PARSE_FAILURE

    metrics = compute_metrics(["positive", "negative"], [PARSE_FAILURE, "negative"])
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_spend_cap_aborts_after_call_and_persists(tmp_path: Path) -> None:
    def transport(prompt: str, api_key: str) -> tuple[str, int, int]:
        return "neutral", 1_000_000, 0

    client = make_client(tmp_path, transport, cap=0.01)
    with pytest.raises(SpendCapExceeded):
        client.predict("something")
    saved = json.loads((tmp_path / "spend.json").read_text(encoding="utf-8"))
    assert saved["spent_usd"] == pytest.approx(0.075)


def test_spend_cap_blocks_before_any_new_call(tmp_path: Path) -> None:
    (tmp_path / "spend.json").write_text(json.dumps({"spent_usd": 2.0}), encoding="utf-8")

    def transport(prompt: str, api_key: str) -> tuple[str, int, int]:
        raise AssertionError("provider must not be called when the cap is reached")

    client = make_client(tmp_path, transport, cap=1.0)
    with pytest.raises(SpendCapExceeded):
        client.predict("anything")


def test_missing_cap_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("voc.models.llm_zero_shot.load_dotenv", lambda **kwargs: None)
    monkeypatch.delenv("MAX_LLM_SPEND_USD", raising=False)
    with pytest.raises(LLMError):
        LLMClient(
            model="test-model",
            provider="mock",
            transport=lambda prompt, key: ("neutral", 1, 1),
            pricing=PRICING,
            cache_path=tmp_path / "r.jsonl",
            spend_path=tmp_path / "s.json",
        )


def test_llm_cost_per_1k_matches_manual_arithmetic() -> None:
    cost = llm_cost_per_1k(400.0, 1.0, 0.075, 0.30)
    expected = (400 / 1e6 * 0.075 + 1 / 1e6 * 0.30) * 1000
    assert cost == pytest.approx(expected)
