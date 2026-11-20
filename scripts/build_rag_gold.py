"""Build the Phase 9 RAG gold set with SQL-computed ground truth.

Analytical expected facts are computed here with ad-hoc SQL (independent of the
router templates) so the evaluation can catch template regressions. The 30
qualitative references are draft key points and stay pending owner review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg

from voc.db.load import database_url

OUT = Path("eval/rag_gold.jsonl")

QUALITATIVE = [
    ("What do customers say about battery life?", ["battery", "charge"]),
    ("How do customers describe the screen?", ["screen", "display"]),
    ("What do reviewers say about the camera?", ["camera", "photo", "picture"]),
    ("How do customers describe the shipping experience?", ["shipping", "delivery", "arrived"]),
    (
        "Do customers think the product is good value for the price?",
        ["price", "value", "expensive", "cheap"],
    ),
    ("What quality issues do reviewers mention?", ["quality", "broke", "flimsy"]),
    ("What do customers say about how the product fits?", ["fit", "size"]),
    ("Is the product comfortable to wear?", ["comfortable", "comfort"]),
    ("How long do customers say the product lasts?", ["last", "durable", "broke"]),
    (
        "What are customers saying about customer service?",
        ["customer service", "support", "return"],
    ),
    ("How do customers describe the packaging?", ["packaging", "box"]),
    ("What do reviewers say about sound quality?", ["sound", "audio", "volume"]),
    ("How does the product perform in daily use?", ["works", "performance", "slow"]),
    ("Is the product easy to use?", ["easy", "simple"]),
    (
        "What do customers think about the design?",
        ["design", "look", "beautiful", "cute"],
    ),
    ("What do customers say about the companion app?", ["app"]),
    ("What do customers like about their purchase?", ["love", "great", "happy", "like"]),
    (
        "What negative experiences do reviewers describe?",
        ["disappointed", "bad", "poor", "waste"],
    ),
    ("What do customers say about the watch strap?", ["strap", "band", "wrist"]),
    ("What do customers say about the effect on their skin?", ["skin", "face"]),
    ("Do customers recommend the product to others?", ["recommend"]),
    ("What do customers say about the size?", ["size", "small", "large"]),
    ("How do customers describe the color they received?", ["color", "colour"]),
    ("What do customers say about the material?", ["material", "fabric", "plastic"]),
    ("Do reviewers report the product breaking?", ["broke", "broken", "break"]),
    (
        "What do customers say about the instructions?",
        ["instructions", "manual", "directions"],
    ),
    ("Do customers feel the product is worth the money?", ["worth", "money", "value"]),
    ("What do customers say about the charging cable?", ["cable", "charger", "charging"]),
    ("What do customers mention about gift packaging?", ["gift", "packaging"]),
    ("What do customers say about the weight?", ["weight", "heavy", "light"]),
]

UNANSWERABLE = [
    "What do customers say about the company's carbon-neutral shipping pledge?",
    "How do reviewers feel about the lifetime warranty policy?",
    "What do customers think about the new eco-friendly packaging initiative?",
    "Do reviewers discuss the brand's factory conditions in Vietnam?",
    "What do customers say about the subscription refill program?",
    "How do reviewers describe the augmented-reality try-on feature?",
    "What do customers think about the product's Bluetooth 6.0 support?",
    "Do reviewers mention the CEO's public statement about privacy?",
    "What do customers say about the free repair service after five years?",
    "How do reviewers rate the carbon offset program?",
]

ANALYTICAL: list[tuple[str, dict[str, Any]]] = [
    (
        "How many negative reviews are there in All_Beauty?",
        {"intent": "counts", "sentiment": "negative", "category": "All_Beauty"},
    ),
    (
        "How many positive reviews are there in Amazon_Fashion?",
        {"intent": "counts", "sentiment": "positive", "category": "Amazon_Fashion"},
    ),
    (
        "How many neutral reviews are there in All_Beauty?",
        {"intent": "counts", "sentiment": "neutral", "category": "All_Beauty"},
    ),
    (
        "How many reviews are there in Amazon_Fashion?",
        {"intent": "counts", "category": "Amazon_Fashion"},
    ),
    (
        "How many negative reviews were there in 2022?",
        {"intent": "counts", "sentiment": "negative", "year": 2022},
    ),
    (
        "How many positive reviews were there in 2020?",
        {"intent": "counts", "sentiment": "positive", "year": 2020},
    ),
    (
        "How many negative reviews were there in All_Beauty in 2021?",
        {"intent": "counts", "sentiment": "negative", "category": "All_Beauty", "year": 2021},
    ),
    (
        "Show the monthly sentiment trend in All_Beauty.",
        {"intent": "trend", "category": "All_Beauty"},
    ),
    (
        "Show the monthly sentiment trend in Amazon_Fashion.",
        {"intent": "trend", "category": "Amazon_Fashion"},
    ),
    ("Show the sentiment trend over time in 2022.", {"intent": "trend", "year": 2022}),
    ("Compare sentiment across categories.", {"intent": "compare"}),
    (
        "Compare negative sentiment across categories.",
        {"intent": "compare", "sentiment": "negative"},
    ),
    ("Compare sentiment across categories in 2021.", {"intent": "compare", "year": 2021}),
    (
        "What are the top aspects in negative reviews?",
        {"intent": "aspects", "sentiment": "negative"},
    ),
    (
        "What are the top aspects in positive reviews?",
        {"intent": "aspects", "sentiment": "positive"},
    ),
    ("What are the top aspects for All_Beauty?", {"intent": "aspects", "category": "All_Beauty"}),
    (
        "What are the top aspects in negative reviews for All_Beauty?",
        {"intent": "aspects", "sentiment": "negative", "category": "All_Beauty"},
    ),
    ("Which product has the most negative reviews?", {"intent": "top_products"}),
    (
        "Which product has the most negative reviews in All_Beauty?",
        {"intent": "top_products", "category": "All_Beauty"},
    ),
    (
        "Which product has the most negative reviews in Amazon_Fashion?",
        {"intent": "top_products", "category": "Amazon_Fashion"},
    ),
]


def _counts(conn: psycopg.Connection, spec: dict[str, Any]) -> dict[str, Any]:
    where = ["TRUE"]
    params: list[Any] = []
    if "sentiment" in spec:
        where.append("r.label = %s")
        params.append(spec["sentiment"])
    if "category" in spec:
        where.append("p.category = %s")
        params.append(spec["category"])
    if "year" in spec:
        where.append("date_part('year', r.reviewed_at) = %s")
        params.append(spec["year"])
    sql = (
        "SELECT r.label, count(*)::int FROM reviews r "
        "JOIN products p ON p.product_id = r.product_id "
        f"WHERE {' AND '.join(where)} GROUP BY r.label ORDER BY r.label"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        counts = {str(label): int(count) for label, count in cur.fetchall()}
    return {"counts": counts, "total": sum(counts.values())}


def _trend(conn: psycopg.Connection, spec: dict[str, Any]) -> dict[str, Any]:
    where = ["r.reviewed_at IS NOT NULL"]
    params: list[Any] = []
    if "category" in spec:
        where.append("p.category = %s")
        params.append(spec["category"])
    if "year" in spec:
        where.append("date_part('year', r.reviewed_at) = %s")
        params.append(spec["year"])
    sql = (
        "SELECT to_char(date_trunc('month', r.reviewed_at), 'YYYY-MM'), r.label, "
        "count(*)::int FROM reviews r JOIN products p ON p.product_id = r.product_id "
        f"WHERE {' AND '.join(where)} GROUP BY 1, r.label ORDER BY 1, r.label"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = [
            {"bucket": bucket, "label": label, "count": count}
            for bucket, label, count in cur.fetchall()
        ]
    buckets = sorted({row["bucket"] for row in rows})
    return {
        "points": len(rows),
        "first_month": buckets[0] if buckets else None,
        "last_month": buckets[-1] if buckets else None,
        "latest": [row for row in rows if buckets and row["bucket"] == buckets[-1]],
    }


def _compare(conn: psycopg.Connection, spec: dict[str, Any]) -> dict[str, Any]:
    where = ["TRUE"]
    params: list[Any] = []
    if "sentiment" in spec:
        where.append("r.label = %s")
        params.append(spec["sentiment"])
    if "year" in spec:
        where.append("date_part('year', r.reviewed_at) = %s")
        params.append(spec["year"])
    sql = (
        "SELECT p.category, r.label, count(*)::int FROM reviews r "
        "JOIN products p ON p.product_id = r.product_id "
        f"WHERE {' AND '.join(where)} GROUP BY 1, 2 ORDER BY 1, 2"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        raw = cur.fetchall()
    totals: dict[str, int] = {}
    for category, _label, count in raw:
        totals[category] = totals.get(category, 0) + int(count)
    rows = [
        {
            "category": category,
            "label": label,
            "count": int(count),
            "share": int(count) / totals[category],
        }
        for category, label, count in raw
    ]
    return {"rows": rows}


def _aspects(conn: psycopg.Connection, spec: dict[str, Any]) -> dict[str, Any]:
    where = ["TRUE"]
    params: list[Any] = []
    if "sentiment" in spec:
        where.append("a.sentiment = %s")
        params.append(spec["sentiment"])
    if "category" in spec:
        where.append("p.category = %s")
        params.append(spec["category"])
    sql = (
        "SELECT a.aspect, count(*)::int FROM aspects a "
        "JOIN reviews r ON r.review_id = a.review_id "
        "JOIN products p ON p.product_id = r.product_id "
        f"WHERE {' AND '.join(where)} GROUP BY a.aspect "
        "ORDER BY count(*) DESC, a.aspect LIMIT 5"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = [{"aspect": name, "count": count} for name, count in cur.fetchall()]
    return {"aspects": rows}


def _top_products(conn: psycopg.Connection, spec: dict[str, Any]) -> dict[str, Any]:
    where = ["r.label = 'negative'"]
    params: list[Any] = []
    if "category" in spec:
        where.append("p.category = %s")
        params.append(spec["category"])
    sql = (
        "SELECT r.product_id, p.title, count(*)::int FROM reviews r "
        "JOIN products p ON p.product_id = r.product_id "
        f"WHERE {' AND '.join(where)} GROUP BY r.product_id, p.title "
        "ORDER BY count(*) DESC, r.product_id LIMIT 5"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = [
            {"product_id": product_id, "title": title, "negative_count": count}
            for product_id, title, count in cur.fetchall()
        ]
    return {"products": rows}


BUILDERS = {
    "counts": _counts,
    "trend": _trend,
    "compare": _compare,
    "aspects": _aspects,
    "top_products": _top_products,
}


def main() -> None:
    lines: list[dict[str, Any]] = []
    with psycopg.connect(database_url()) as conn:
        for index, (question, key_points) in enumerate(QUALITATIVE, start=1):
            lines.append(
                {
                    "id": f"qual-{index:02d}",
                    "kind": "qualitative",
                    "question": question,
                    "key_points": key_points,
                    "review_status": "draft-pending-owner-review",
                }
            )
        for index, (question, spec) in enumerate(ANALYTICAL, start=1):
            expected = BUILDERS[spec["intent"]](conn, spec)
            lines.append(
                {
                    "id": f"ana-{index:02d}",
                    "kind": "analytical",
                    "question": question,
                    "expected": {"intent": spec["intent"], "facts": expected},
                }
            )
        for index, question in enumerate(UNANSWERABLE, start=1):
            lines.append(
                {
                    "id": f"unans-{index:02d}",
                    "kind": "unanswerable",
                    "question": question,
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(lines)} gold items to {OUT}")


if __name__ == "__main__":
    main()
