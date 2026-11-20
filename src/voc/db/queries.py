"""Whitelisted parameterised SQL templates used by stats endpoints and the RAG router.

No SQL is ever built from free-form user text: filter values travel as psycopg
parameters and the only interpolated token is the ``date_trunc`` unit, which is
looked up in ``GRANULARITY`` (an unknown value raises ``KeyError``).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

GRANULARITY: dict[str, str] = {"day": "day", "week": "week", "month": "month"}


def trend_query(
    *,
    category: str | None = None,
    product_id: str | None = None,
    start: date | None = None,
    end: date | None = None,
    granularity: str = "month",
) -> tuple[str, list[object]]:
    """Return (sql, params) for sentiment counts bucketed by day/week/month."""
    unit = GRANULARITY[granularity]
    clauses = ["r.reviewed_at IS NOT NULL"]
    params: list[object] = [unit]
    if category is not None:
        clauses.append("p.category = %s")
        params.append(category)
    if product_id is not None:
        clauses.append("r.product_id = %s")
        params.append(product_id)
    if start is not None:
        clauses.append("r.reviewed_at >= %s")
        params.append(start)
    if end is not None:
        clauses.append("r.reviewed_at < %s::date + interval '1 day'")
        params.append(end)
    sql = (
        "SELECT date_trunc(%s, r.reviewed_at) AS bucket, r.label, count(*)::int AS count "
        "FROM reviews r JOIN products p ON p.product_id = r.product_id "
        f"WHERE {' AND '.join(clauses)} "
        "GROUP BY bucket, r.label ORDER BY bucket, r.label"
    )
    return sql, params


def top_aspects_query(
    *,
    sentiment: str | None = None,
    category: str | None = None,
    limit: int = 10,
) -> tuple[str, list[object]]:
    """Return (sql, params) for the most frequent aspects under optional filters."""
    clauses = ["TRUE"]
    params: list[object] = []
    if sentiment is not None:
        clauses.append("a.sentiment = %s")
        params.append(sentiment)
    if category is not None:
        clauses.append("p.category = %s")
        params.append(category)
    params.append(limit)
    sql = (
        "WITH agg AS ("
        "  SELECT a.aspect, count(*)::int AS n "
        "  FROM aspects a "
        "  JOIN reviews r ON r.review_id = a.review_id "
        "  JOIN products p ON p.product_id = r.product_id "
        f"  WHERE {' AND '.join(clauses)} "
        "  GROUP BY a.aspect"
        ") "
        "SELECT aspect, n, n::float / sum(n) OVER () AS share "
        "FROM agg ORDER BY n DESC, aspect LIMIT %s"
    )
    return sql, params


def insert_api_requests_sql() -> str:
    """Insert one audit row per scored text into api_requests."""
    return (
        "INSERT INTO api_requests (endpoint, model_name, latency_ms, label, text_len) "
        "VALUES (%s, %s, %s, %s, %s)"
    )


def fetch_trend(conn, **kwargs) -> list[dict[str, object]]:  # noqa: ANN001
    """Execute the trend template and return JSON-ready rows."""
    sql, params = trend_query(**kwargs)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [
            {
                "bucket": bucket.date().isoformat() if hasattr(bucket, "date") else str(bucket),
                "label": label,
                "count": count,
            }
            for bucket, label, count in cur.fetchall()
        ]


def fetch_top_aspects(conn, **kwargs) -> list[dict[str, object]]:  # noqa: ANN001
    """Execute the top-aspects template and return JSON-ready rows."""
    sql, params = top_aspects_query(**kwargs)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [
            {"aspect": aspect, "count": count, "share": float(share)}
            for aspect, count, share in cur.fetchall()
        ]


def insert_api_requests(conn, records: Sequence[tuple[str, str, float, str, int]]) -> None:  # noqa: ANN001
    """Insert audit rows (endpoint, model, latency_ms, label, text_len) in one batch."""
    if not records:
        return
    with conn.cursor() as cur:
        cur.executemany(insert_api_requests_sql(), list(records))
