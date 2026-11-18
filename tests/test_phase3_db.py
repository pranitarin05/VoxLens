"""FR-4: migrations, COPY loader, count equality, analytics query headers."""

from __future__ import annotations

from pathlib import Path

import pytest
from psycopg import Connection

from voc.db.load import (
    ANALYTICS_DIR,
    MIGRATIONS_DIR,
    apply_migrations,
    database_url,
    run_analytics,
)

ROOT = Path(__file__).resolve().parents[1]
ANALYTICS = ROOT / "sql" / "analytics"


def _connect() -> Connection:
    import psycopg

    return psycopg.connect(database_url())


@pytest.fixture(scope="module")
def conn() -> Connection:
    if not MIGRATIONS_DIR.exists():
        pytest.skip("migrations dir missing")
    connection = _connect()
    yield connection
    connection.close()


def test_migrations_idempotent(conn: Connection) -> None:
    first = apply_migrations(MIGRATIONS_DIR, conn=conn)
    second = apply_migrations(MIGRATIONS_DIR, conn=conn)
    assert isinstance(first, list) and isinstance(second, list)
    assert second == [], f"re-run must apply nothing, got {second}"
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM schema_migrations")
        assert cur.fetchone()[0] >= 1
        cur.execute("SELECT to_regclass('reviews')")
        assert cur.fetchone()[0] is not None


def test_analytics_files_have_business_question_header() -> None:
    files = sorted(ANALYTICS.glob("*.sql"))
    assert len(files) >= 8, f"PRD FR-4 needs ≥8 analytics queries, found {len(files)}"
    for path in files:
        head = path.read_text(encoding="utf-8")[:400]
        assert "-- Business question:" in head, f"{path.name} missing Business question"


def test_analytics_execute_without_error(conn: Connection) -> None:
    apply_migrations(MIGRATIONS_DIR, conn=conn)
    results = run_analytics(ANALYTICS_DIR, conn=conn, limit_rows=3)
    assert len(results) >= 8
    for name, rows in results.items():
        assert isinstance(rows, list), name


@pytest.mark.integration
def test_counts_match_parquet_when_loaded(conn: Connection) -> None:
    """Acceptance: if reviews table is loaded, counts equal parquet counts."""
    import pandas as pd

    parquet = ROOT / "data" / "processed" / "reviews.parquet"
    if not parquet.exists():
        pytest.skip("reviews.parquet not present")
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM reviews")
        db_reviews = cur.fetchone()[0]
    if db_reviews == 0:
        pytest.skip("reviews table empty — run make db-load first")
    parquet_rows = len(pd.read_parquet(parquet, columns=["review_id"]))
    assert db_reviews == parquet_rows
