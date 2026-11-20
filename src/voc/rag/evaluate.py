"""PRD 10/11: RAG and SQL-router evaluation on the 60-question gold set.

Metrics
- analytical: numeric correctness of whitelisted-SQL facts vs gold facts computed
  by independent SQL in ``scripts/build_rag_gold.py``.
- qualitative: retrieval hit@8 (a gold key point appears in a retrieved snippet)
  and LLM-judge faithfulness of the grounded answer.
- unanswerable: abstention accuracy and false-answer rate.

The 30 qualitative references are draft key points; per-answer faithfulness and
20 manual spot checks still require owner review.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
import typer

from voc.db.load import database_url
from voc.models.llm_zero_shot import LLMClient, LLMError, SpendCapExceeded
from voc.rag.answer import (
    AnswerResult,
    RAGParseError,
    answer_from_docs,
    build_snippets,
    parse_json_object,
    phrase_analysis,
)
from voc.rag.embed import load_embedder
from voc.rag.retrieve import retrieve
from voc.rag.router import list_aspects, list_categories, run_analysis

app = typer.Typer(add_completion=False, help="PRD 10: evaluate RAG on the gold set.")

GOLD_PATH = Path("eval/rag_gold.jsonl")
MD_PATH = Path("results/rag_eval.md")
JSON_PATH = Path("results/rag_eval.json")

JUDGE_SYSTEM = (
    "You verify whether an answer is fully supported by numbered review snippets. "
    "Every factual claim in the answer must be traceable to a cited snippet. "
    "Ignore style. Reply with one JSON object: "
    '{"faithful": boolean, "unsupported_claims": [string]}'
)


@dataclass
class ItemResult:
    """One evaluated gold item."""

    id: str
    kind: str
    question: str
    route: str
    latency_ms: float
    cost_usd: float
    correct: bool | None = None
    hit_at_8: bool | None = None
    abstained: bool | None = None
    faithful: bool | None = None
    error: str | None = None
    answer: str = ""


@dataclass
class Metrics:
    """Aggregated metrics per set."""

    counts: dict[str, float] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)


def compare_facts(intent: str, expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Compare the gold facts subset with router facts."""
    if intent in {"counts"}:
        return actual.get("counts") == expected.get("counts") and actual.get(
            "total"
        ) == expected.get("total")
    if intent == "trend":
        keys = ("points", "first_month", "last_month", "latest")
        return all(actual.get(key) == expected.get(key) for key in keys)
    if intent == "compare":
        rows_expected = expected.get("rows", [])
        rows_actual = actual.get("rows", [])
        if len(rows_expected) != len(rows_actual):
            return False
        return all(
            exp["category"] == act["category"]
            and exp["label"] == act["label"]
            and exp["count"] == act["count"]
            and abs(float(exp["share"]) - float(act["share"])) < 1e-9
            for exp, act in zip(rows_expected, rows_actual, strict=True)
        )
    if intent == "aspects":
        return actual.get("aspects") == expected.get("aspects")
    if intent == "top_products":
        return actual.get("products") == expected.get("products")
    return False


def hit_at_k(key_points: list[str], docs: list[Any]) -> bool:
    """True when any gold key point appears in any retrieved snippet."""
    lowered = [doc.text.lower() for doc in docs]
    return any(point.lower() in text for point in key_points for text in lowered)


def judge_faithfulness(
    client: LLMClient, question: str, docs: list[Any], answer: AnswerResult
) -> tuple[bool | None, str | None]:
    """Ask the LLM judge; returns (faithful or None, error)."""
    prompt = (
        f"Question: {question}\n\n"
        f"Snippets:\n{build_snippets(docs)}\n\n"
        f"Answer to verify: {answer.answer}\n"
        f"Cited snippets: {answer.citations}\n"
        "Return the JSON object now."
    )
    try:
        result = client.complete(
            prompt,
            system_prompt=JUDGE_SYSTEM,
            max_completion_tokens=256,
            cache_tag="rag-judge-v1",
        )
        payload = parse_json_object(result.content)
        return bool(payload.get("faithful", False)), None
    except (RAGParseError, LLMError, SpendCapExceeded) as exc:
        return None, f"judge: {exc}"


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy)."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def evaluate(limit: int | None = None) -> tuple[list[ItemResult], LLMClient]:
    """Run the gold set; returns per-item results and the LLM client."""
    gold = [json.loads(line) for line in GOLD_PATH.read_text(encoding="utf-8").splitlines() if line]
    if limit:
        gold = gold[:limit]
    client = LLMClient()
    embedder = load_embedder("cpu")
    results: list[ItemResult] = []
    with psycopg.connect(database_url()) as conn:
        categories = list_categories(conn)
        aspects = list_aspects(conn)
        for item in gold:
            started = time.perf_counter()
            result = ItemResult(
                id=item["id"],
                kind=item["kind"],
                question=item["question"],
                route="",
                latency_ms=0.0,
                cost_usd=0.0,
            )
            try:
                if item["kind"] == "analytical":
                    analysis = run_analysis(conn, item["question"], categories, aspects)
                    if analysis is None:
                        result.route = "rag"
                        result.correct = False
                    else:
                        result.route = "sql"
                        result.correct = analysis.intent == item["expected"]["intent"] and (
                            compare_facts(
                                item["expected"]["intent"],
                                item["expected"]["facts"],
                                analysis.facts,
                            )
                        )
                        phrased = phrase_analysis(client, item["question"], analysis)
                        result.answer = phrased.answer
                        result.cost_usd += phrased.cost_usd
                    result.latency_ms = round((time.perf_counter() - started) * 1000, 3)
                elif item["kind"] == "qualitative":
                    vector = embedder.encode([item["question"]], normalize_embeddings=True)[0]
                    docs = retrieve(conn, vector, keep=8)
                    result.route = "rag"
                    result.hit_at_8 = hit_at_k(item["key_points"], docs)
                    answer = answer_from_docs(client, item["question"], docs)
                    result.abstained = answer.abstained
                    result.answer = answer.answer
                    result.cost_usd += answer.cost_usd
                    result.latency_ms = round((time.perf_counter() - started) * 1000, 3)
                    faithful, error = judge_faithfulness(client, item["question"], docs, answer)
                    result.faithful = faithful
                    result.error = error
                else:
                    vector = embedder.encode([item["question"]], normalize_embeddings=True)[0]
                    docs = retrieve(conn, vector, keep=8)
                    result.route = "rag"
                    answer = answer_from_docs(client, item["question"], docs)
                    result.abstained = answer.abstained
                    result.answer = answer.answer
                    result.cost_usd += answer.cost_usd
                    result.latency_ms = round((time.perf_counter() - started) * 1000, 3)
            except (LLMError, SpendCapExceeded, psycopg.Error) as exc:
                result.error = f"{type(exc).__name__}: {exc}"
            if not result.latency_ms:
                result.latency_ms = round((time.perf_counter() - started) * 1000, 3)
            results.append(result)
    return results, client


def aggregate(results: list[ItemResult], kinds: str) -> Metrics:
    """Metrics for one gold set."""
    subset = [item for item in results if item.kind == kinds]
    metrics = Metrics(latencies=[item.latency_ms for item in subset])
    n = len(subset)
    metrics.counts["n"] = n
    metrics.counts["errors"] = sum(1 for item in subset if item.error)
    metrics.counts["cost_usd"] = sum(item.cost_usd for item in subset)
    if kinds == "analytical":
        metrics.counts["numeric_accuracy"] = (
            sum(1 for item in subset if item.correct) / n if n else 0.0
        )
        metrics.counts["sql_route_rate"] = (
            sum(1 for item in subset if item.route == "sql") / n if n else 0.0
        )
    elif kinds == "qualitative":
        metrics.counts["hit_at_8"] = sum(1 for item in subset if item.hit_at_8) / n if n else 0.0
        answered = [item for item in subset if item.abstained is False]
        metrics.counts["answer_rate"] = len(answered) / n if n else 0.0
        judged = [item for item in subset if item.faithful is not None]
        metrics.counts["faithfulness"] = (
            sum(1 for item in judged if item.faithful) / len(judged) if judged else 0.0
        )
        metrics.counts["judged"] = len(judged)
    else:
        metrics.counts["abstention_accuracy"] = (
            sum(1 for item in subset if item.abstained) / n if n else 0.0
        )
    return metrics


def render_markdown(results: list[ItemResult], client: LLMClient, elapsed_s: float) -> str:
    """Markdown report (no fabricated numbers: everything below is computed)."""
    lines = [
        "# RAG evaluation (Phase 9)",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"Gold set: `{GOLD_PATH.as_posix()}` ({len(results)} questions)",
        f"Wall time: {elapsed_s:.1f}s | LLM live calls: {client.live_calls} | "
        f"cache hits: {client.cache_hits} | Spend this run: ${client.run_spend_usd:.4f}",
        f"Spend ledger total: ${client.spent_usd:.4f} of ${client.max_spend_usd:.2f} cap",
        "",
        "> Status: DRAFT. The 30 qualitative reference key points and the "
        "faithfulness labels were produced by the agent and still require owner "
        "review, as does the PRD's manual spot check of 20 answers.",
        "",
        "## Definitions",
        "- **Numeric accuracy**: router facts equal independently computed SQL gold "
        "facts (exact counts/shares, tolerance 1e-9 for shares).",
        "- **Hit@8**: at least one gold key point string appears in one of the 8 "
        "retrieved review snippets.",
        "- **Faithfulness**: an LLM judge finds every claim supported by the cited "
        "snippets; judge parse failures are excluded from the denominator (`judged`).",
        "- **Abstention**: the unanswerable question produced `abstained=true`.",
        "",
        "## Metrics",
        "",
        "| Set | n | Metric | Value |",
        "| --- | --- | --- | --- |",
    ]
    for kind in ("analytical", "qualitative", "unanswerable"):
        metrics = aggregate(results, kind)
        n = int(metrics.counts["n"])
        if kind == "analytical":
            pairs = [
                ("Numeric accuracy", f"{metrics.counts['numeric_accuracy']:.1%}"),
                ("SQL route rate", f"{metrics.counts['sql_route_rate']:.1%}"),
            ]
        elif kind == "qualitative":
            pairs = [
                ("Hit@8", f"{metrics.counts['hit_at_8']:.1%}"),
                ("Answer rate", f"{metrics.counts['answer_rate']:.1%}"),
                (
                    "Faithfulness",
                    f"{metrics.counts['faithfulness']:.1%} "
                    f"({int(metrics.counts['judged'])} judged)",
                ),
            ]
        else:
            pairs = [
                ("Abstention accuracy", f"{metrics.counts['abstention_accuracy']:.1%}"),
            ]
        for label, value in pairs:
            lines.append(f"| {kind} | {n} | {label} | {value} |")
        p50 = percentile(metrics.latencies, 50)
        p95 = percentile(metrics.latencies, 95)
        lines.append(f"| {kind} | {n} | Latency p50 / p95 (ms) | {p50:.0f} / {p95:.0f} |")
        lines.append(f"| {kind} | {n} | Cost (USD) | ${metrics.counts['cost_usd']:.4f} |")
    lines.extend(
        [
            "",
            "## Analytical details",
            "",
            "| id | correct | route | question |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in results:
        if item.kind != "analytical":
            continue
        mark = "yes" if item.correct else "**no**"
        lines.append(f"| {item.id} | {mark} | {item.route} | {item.question} |")
    lines.extend(
        [
            "",
            "## Qualitative details",
            "",
            "| id | hit@8 | abstained | faithful | latency ms | question |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in results:
        if item.kind != "qualitative":
            continue
        faithful = "n/a" if item.faithful is None else ("yes" if item.faithful else "**no**")
        lines.append(
            f"| {item.id} | {'yes' if item.hit_at_8 else 'no'} | {item.abstained} | "
            f"{faithful} | {item.latency_ms:.0f} | {item.question} |"
        )
    lines.extend(
        [
            "",
            "## Unanswerable details",
            "",
            "| id | abstained | latency ms | question |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in results:
        if item.kind != "unanswerable":
            continue
        lines.append(f"| {item.id} | {item.abstained} | {item.latency_ms:.0f} | {item.question} |")
    errors = [item for item in results if item.error]
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {item.id}: {item.error}" for item in errors)
    lines.append("")
    return "\n".join(lines)


def summary(results: list[ItemResult], client: LLMClient, elapsed_s: float) -> dict[str, Any]:
    """Compact JSON summary used by the dashboard later."""
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "gold_path": GOLD_PATH.as_posix(),
        "status": "draft-pending-owner-review",
        "n_questions": len(results),
        "wall_time_s": round(elapsed_s, 1),
        "run_spend_usd": client.run_spend_usd,
        "ledger_spend_usd": client.spent_usd,
        "metrics": {
            kind: aggregate(results, kind).counts
            for kind in ("analytical", "qualitative", "unanswerable")
        },
        "latency_p50_ms": percentile([item.latency_ms for item in results], 50),
        "latency_p95_ms": percentile([item.latency_ms for item in results], 95),
        "errors": [{"id": item.id, "error": item.error} for item in results if item.error],
        "items": [
            {
                "id": item.id,
                "kind": item.kind,
                "route": item.route,
                "correct": item.correct,
                "hit_at_8": item.hit_at_8,
                "abstained": item.abstained,
                "faithful": item.faithful,
                "latency_ms": item.latency_ms,
                "cost_usd": item.cost_usd,
            }
            for item in results
        ],
    }


@app.command()
def main(limit: int | None = typer.Option(None, help="Evaluate only the first N items.")) -> None:
    """Run the gold set and write results/rag_eval.{md,json}."""
    started = time.perf_counter()
    results, client = evaluate(limit)
    elapsed = time.perf_counter() - started
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text(render_markdown(results, client, elapsed), encoding="utf-8")
    JSON_PATH.write_text(
        json.dumps(summary(results, client, elapsed), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    for kind in ("analytical", "qualitative", "unanswerable"):
        metrics = aggregate(results, kind).counts
        printable = {key: round(value, 4) for key, value in metrics.items()}
        print(kind, printable)
    print(f"wrote {MD_PATH} and {JSON_PATH} in {elapsed:.1f}s")


if __name__ == "__main__":
    app()
