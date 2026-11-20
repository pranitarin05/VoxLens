"""PRD 10: pgvector retrieval with optional filters (top 20, keep best 8)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import psycopg
from pgvector.psycopg import register_vector


@dataclass(frozen=True)
class RetrievedReview:
    """One retrieved review with cosine similarity and metadata."""

    review_id: str
    product_id: str
    category: str
    label: str
    text: str
    reviewed_at: object | None
    score: float

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["reviewed_at"] = (
            self.reviewed_at.isoformat() if hasattr(self.reviewed_at, "isoformat") else None
        )
        return data


def filter_clauses(
    *,
    category: str | None = None,
    product_id: str | None = None,
    sentiment: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> tuple[list[str], list[object]]:
    """Whitelisted WHERE fragments; values always travel as parameters."""
    clauses: list[str] = []
    params: list[object] = []
    if category is not None:
        clauses.append("p.category = %s")
        params.append(category)
    if product_id is not None:
        clauses.append("r.product_id = %s")
        params.append(product_id)
    if sentiment is not None:
        clauses.append("r.label = %s")
        params.append(sentiment)
    if start is not None:
        clauses.append("r.reviewed_at >= %s")
        params.append(start)
    if end is not None:
        clauses.append("r.reviewed_at < %s::date + interval '1 day'")
        params.append(end)
    return clauses, params


def retrieve(
    conn: psycopg.Connection,
    query_vector: Sequence[float] | np.ndarray,
    *,
    top_k: int = 20,
    keep: int = 8,
    category: str | None = None,
    product_id: str | None = None,
    sentiment: str | None = None,
    start: date | None = None,
    end: date | None = None,
    table: str = "review_embeddings",
) -> list[RetrievedReview]:
    """Cosine search over pgvector, returning the best ``keep`` of ``top_k``."""
    register_vector(conn)
    vector = np.asarray(query_vector, dtype=np.float32)
    clauses, params = filter_clauses(
        category=category,
        product_id=product_id,
        sentiment=sentiment,
        start=start,
        end=end,
    )
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        "SELECT r.review_id, r.product_id, p.category, r.label, r.text, r.reviewed_at, "
        "1 - (e.embedding <=> %s) AS score "
        "FROM review_embeddings e "
        "JOIN reviews r ON r.review_id = e.review_id "
        "JOIN products p ON p.product_id = r.product_id "
        f"{where} "
        "ORDER BY e.embedding <=> %s "
        "LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, [vector, *params, vector, top_k])
        rows = cur.fetchall()
    results = [
        RetrievedReview(
            review_id=row[0],
            product_id=row[1],
            category=row[2],
            label=row[3],
            text=row[4],
            reviewed_at=row[5],
            score=float(row[6]),
        )
        for row in rows
    ]
    return results[:keep]
