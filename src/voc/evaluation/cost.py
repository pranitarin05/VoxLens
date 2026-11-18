"""PRD 7.7: cost per 1k and monthly cost calculators.

Self-hosted cost uses a *stated, configurable assumption* for the instance
price (configs/pricing.yaml). LLM cost uses real per-million-token prices.
Nothing here measures money; it applies arithmetic to measured throughput and
configured/assumed prices. If a price is unset the cost is returned as None so
callers never report a made-up number.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PRICING = Path("configs/pricing.yaml")
PREDICTIONS_PER_MONTH = 1_000_000


def load_pricing(path: Path = DEFAULT_PRICING) -> dict[str, object]:
    """Load pricing.yaml (values may be None until filled)."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def self_hosted_cost_per_1k(
    predictions_per_hour: float, usd_per_hour: float | None
) -> float | None:
    """PRD 7.7: (instance $/h ÷ measured preds/h) × 1,000."""
    if usd_per_hour is None or predictions_per_hour <= 0:
        return None
    return (usd_per_hour / predictions_per_hour) * 1000.0


def llm_cost_per_1k(
    avg_input_tokens: float,
    avg_output_tokens: float,
    input_per_mtok_usd: float | None,
    output_per_mtok_usd: float | None,
) -> float | None:
    """LLM cost per 1,000 calls from measured average tokens × price per Mtok."""
    if input_per_mtok_usd is None or output_per_mtok_usd is None:
        return None
    per_call = (avg_input_tokens / 1_000_000.0) * input_per_mtok_usd + (
        avg_output_tokens / 1_000_000.0
    ) * output_per_mtok_usd
    return per_call * 1000.0


def monthly_cost_at_1m(cost_per_1k_usd: float | None) -> float | None:
    """PRD 7.7: monthly cost at 1M predictions = cost/1k × 1,000."""
    if cost_per_1k_usd is None:
        return None
    return cost_per_1k_usd * (PREDICTIONS_PER_MONTH / 1000.0)
