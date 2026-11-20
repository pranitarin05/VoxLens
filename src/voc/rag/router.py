"""PRD 10 router: analytical questions use whitelisted SQL, others use RAG.

The LLM never writes SQL. Intent detection and the template choice happen here;
filter values come from the database (categories) or validated literals
(sentiment, years, aspect names) and always travel as psycopg parameters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import psycopg

ANALYTICAL_PATTERN = re.compile(
    r"\b(how many|number of|count|trend|over time|increase|decrease|compare|comparison|"
    r"average|top|most|least|share|percentage|which product|by month|by year)\b",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
SENTIMENTS = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class Analysis:
    """A whitelisted SQL result plus the facts used to phrase the answer."""

    intent: str
    sql_name: str
    rows: list[dict[str, object]]
    facts: dict[str, object]
    summary: str
    params: dict[str, object] = field(default_factory=dict)


def route(question: str) -> str:
    """Return 'sql' for analytical questions, otherwise 'rag'."""
    return "sql" if ANALYTICAL_PATTERN.search(question) else "rag"


def detect_sentiment(question: str) -> str | None:
    """Return the first sentiment word present, if any."""
    lowered = question.lower()
    for sentiment in SENTIMENTS:
        if re.search(rf"\b{sentiment}\b", lowered):
            return sentiment
    return None


def detect_category(question: str, categories: list[str]) -> str | None:
    """Match a known category, tolerating spaces instead of underscores."""
    lowered = question.lower().replace("_", " ")
    for category in categories:
        if category.lower().replace("_", " ") in lowered:
            return category
    return None


def detect_year(question: str) -> int | None:
    """Extract a 4-digit year in 2000-2099."""
    match = YEAR_PATTERN.search(question)
    return int(match.group(1)) if match else None


def detect_aspect(question: str, aspects: list[str]) -> str | None:
    """Match an aspect name in its underscore or spaced form."""
    lowered = question.lower().replace("_", " ")
    for aspect in aspects:
        if aspect.lower().replace("_", " ") in lowered:
            return aspect
    return None


def list_categories(conn: psycopg.Connection) -> list[str]:
    """Distinct product categories (used for safe filter extraction)."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT category FROM products ORDER BY category")
        return [row[0] for row in cur.fetchall()]


def list_aspects(conn: psycopg.Connection) -> list[str]:
    """Distinct aspect names (used for safe filter extraction)."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT aspect FROM aspects ORDER BY aspect")
        return [row[0] for row in cur.fetchall()]


def _summary_counts(facts: dict[str, object]) -> str:
    counts = facts.get("counts", {})
    total = facts.get("total", 0)
    parts = ", ".join(f"{label}: {count}" for label, count in counts.items())
    return f"Counts by sentiment ({parts}); total {total} matching reviews."


def counts_analysis(
    conn: psycopg.Connection,
    *,
    sentiment: str | None,
    category: str | None,
    year: int | None,
) -> Analysis:
    """Counts per sentiment under optional category/year filters."""
    clauses = ["TRUE"]
    params: list[object] = []
    if sentiment is not None:
        clauses.append("r.label = %s")
        params.append(sentiment)
    if category is not None:
        clauses.append("p.category = %s")
        params.append(category)
    if year is not None:
        clauses.append("date_part('year', r.reviewed_at) = %s")
        params.append(year)
    sql = (
        "SELECT r.label, count(*)::int FROM reviews r "
        "JOIN products p ON p.product_id = r.product_id "
        f"WHERE {' AND '.join(clauses)} GROUP BY r.label ORDER BY r.label"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = [{"label": label, "count": count} for label, count in cur.fetchall()]
    counts = {str(row["label"]): int(row["count"]) for row in rows}
    facts: dict[str, object] = {"counts": counts, "total": sum(counts.values())}
    if sentiment:
        facts["sentiment"] = sentiment
    if category:
        facts["category"] = category
    if year:
        facts["year"] = year
    return Analysis(
        intent="counts",
        sql_name="count_by_sentiment",
        rows=rows,
        facts=facts,
        summary=_summary_counts(facts),
        params={"sentiment": sentiment, "category": category, "year": year},
    )


def trend_analysis(conn: psycopg.Connection, *, category: str | None, year: int | None) -> Analysis:
    """Monthly sentiment counts under optional filters."""
    clauses = ["r.reviewed_at IS NOT NULL"]
    params: list[object] = []
    if category is not None:
        clauses.append("p.category = %s")
        params.append(category)
    if year is not None:
        clauses.append("date_part('year', r.reviewed_at) = %s")
        params.append(year)
    sql = (
        "SELECT to_char(date_trunc('month', r.reviewed_at), 'YYYY-MM') AS bucket, "
        "r.label, count(*)::int FROM reviews r "
        "JOIN products p ON p.product_id = r.product_id "
        f"WHERE {' AND '.join(clauses)} "
        "GROUP BY bucket, r.label ORDER BY bucket, r.label"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = [
            {"bucket": bucket, "label": label, "count": count}
            for bucket, label, count in cur.fetchall()
        ]
    buckets = sorted({str(row["bucket"]) for row in rows})
    facts: dict[str, object] = {
        "points": len(rows),
        "first_month": buckets[0] if buckets else None,
        "last_month": buckets[-1] if buckets else None,
        "latest": [row for row in rows if buckets and row["bucket"] == buckets[-1]],
    }
    if category:
        facts["category"] = category
    if year:
        facts["year"] = year
    summary = (
        f"Monthly sentiment counts from {facts['first_month']} to {facts['last_month']} "
        f"({facts['points']} bucket rows). Latest month: {facts['latest']}."
    )
    return Analysis(
        intent="trend",
        sql_name="sentiment_trend_month",
        rows=rows,
        facts=facts,
        summary=summary,
        params={"category": category, "year": year},
    )


def aspects_analysis(
    conn: psycopg.Connection,
    *,
    sentiment: str | None,
    category: str | None,
    aspect: str | None,
    limit: int = 5,
) -> Analysis:
    """Top aspects (optionally one aspect, sentiment and category)."""
    clauses = ["TRUE"]
    params: list[object] = []
    if sentiment is not None:
        clauses.append("a.sentiment = %s")
        params.append(sentiment)
    if category is not None:
        clauses.append("p.category = %s")
        params.append(category)
    if aspect is not None:
        clauses.append("a.aspect = %s")
        params.append(aspect)
    params.append(limit)
    sql = (
        "SELECT a.aspect, count(*)::int FROM aspects a "
        "JOIN reviews r ON r.review_id = a.review_id "
        "JOIN products p ON p.product_id = r.product_id "
        f"WHERE {' AND '.join(clauses)} GROUP BY a.aspect "
        "ORDER BY count(*) DESC, a.aspect LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = [{"aspect": name, "count": count} for name, count in cur.fetchall()]
    facts: dict[str, object] = {"aspects": rows}
    if sentiment:
        facts["sentiment"] = sentiment
    if category:
        facts["category"] = category
    if aspect:
        facts["aspect"] = aspect
    listed = ", ".join(f"{row['aspect']}: {row['count']}" for row in rows)
    summary = f"Top aspects ({listed}); filters: sentiment={sentiment}, category={category}."
    return Analysis(
        intent="aspects",
        sql_name="top_aspects",
        rows=rows,
        facts=facts,
        summary=summary,
        params={
            "sentiment": sentiment,
            "category": category,
            "aspect": aspect,
            "limit": limit,
        },
    )


def compare_categories_analysis(
    conn: psycopg.Connection, *, sentiment: str | None, year: int | None
) -> Analysis:
    """Sentiment counts and shares per category."""
    clauses = ["TRUE"]
    params: list[object] = []
    if sentiment is not None:
        clauses.append("r.label = %s")
        params.append(sentiment)
    if year is not None:
        clauses.append("date_part('year', r.reviewed_at) = %s")
        params.append(year)
    sql = (
        "SELECT p.category, r.label, count(*)::int AS n, "
        "count(*)::float / sum(count(*)) OVER (PARTITION BY p.category) AS share "
        "FROM reviews r JOIN products p ON p.product_id = r.product_id "
        f"WHERE {' AND '.join(clauses)} "
        "GROUP BY p.category, r.label ORDER BY p.category, r.label"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = [
            {"category": category, "label": label, "count": count, "share": float(share)}
            for category, label, count, share in cur.fetchall()
        ]
    facts: dict[str, object] = {"rows": rows}
    if sentiment:
        facts["sentiment"] = sentiment
    if year:
        facts["year"] = year
    summary = "Category comparison: " + "; ".join(
        f"{row['category']} {row['label']} {row['count']} ({row['share']:.1%})" for row in rows
    )
    return Analysis(
        intent="compare",
        sql_name="compare_categories",
        rows=rows,
        facts=facts,
        summary=summary,
        params={"sentiment": sentiment, "year": year},
    )


def top_products_analysis(
    conn: psycopg.Connection, *, category: str | None, limit: int = 5
) -> Analysis:
    """Products with the most negative reviews."""
    clauses = ["r.label = 'negative'"]
    params: list[object] = []
    if category is not None:
        clauses.append("p.category = %s")
        params.append(category)
    params.append(limit)
    sql = (
        "SELECT r.product_id, p.title, count(*)::int FROM reviews r "
        "JOIN products p ON p.product_id = r.product_id "
        f"WHERE {' AND '.join(clauses)} GROUP BY r.product_id, p.title "
        "ORDER BY count(*) DESC, r.product_id LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = [
            {"product_id": product_id, "title": title, "negative_count": count}
            for product_id, title, count in cur.fetchall()
        ]
    facts: dict[str, object] = {"products": rows}
    if category:
        facts["category"] = category
    summary = "Products with most negative reviews: " + "; ".join(
        f"{row['product_id']} ({row['negative_count']})" for row in rows
    )
    return Analysis(
        intent="top_products",
        sql_name="top_products_negative",
        rows=rows,
        facts=facts,
        summary=summary,
        params={"category": category, "limit": limit},
    )


def run_analysis(
    conn: psycopg.Connection,
    question: str,
    categories: list[str],
    aspects: list[str],
    filters: dict[str, object] | None = None,
) -> Analysis | None:
    """Pick and run a whitelisted template; None means fall back to RAG."""
    lowered = question.lower()
    category = detect_category(question, categories)
    sentiment = detect_sentiment(question)
    year = detect_year(question)
    aspect = detect_aspect(question, aspects)
    if filters:
        category = filters.get("category") or category  # type: ignore[assignment]
        sentiment = filters.get("sentiment") or sentiment  # type: ignore[assignment]
    category = str(category) if category is not None else None
    sentiment = str(sentiment) if sentiment is not None else None

    if re.search(r"\bcompare|comparison\b", lowered):
        return compare_categories_analysis(conn, sentiment=sentiment, year=year)
    if re.search(r"\btrend|over time|by month|increase|decrease\b", lowered):
        return trend_analysis(conn, category=category, year=year)
    if re.search(r"\baspect|complaint|praise\b", lowered) or aspect is not None:
        return aspects_analysis(conn, sentiment=sentiment, category=category, aspect=aspect)
    if re.search(r"\bwhich product|top product|most negative product\b", lowered):
        return top_products_analysis(conn, category=category)
    if re.search(r"\bhow many|number of|count\b", lowered):
        return counts_analysis(conn, sentiment=sentiment, category=category, year=year)
    return None


def date_range(question: str) -> tuple[date | None, date | None]:
    """Reserved for date-range questions; years are handled as filters today."""
    return None, None
