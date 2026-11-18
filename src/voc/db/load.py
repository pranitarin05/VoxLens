"""FR-4: versioned SQL migrations + COPY bulk loader for products/reviews.

Migrations are plain .sql files in sql/migrations/, applied in filename order
and recorded in schema_migrations (idempotent: re-run skips applied files).
COPY uses psycopg 3 cursor.copy (text format) from parquet-backed DataFrames.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import pandas as pd
import psycopg
import typer
from dotenv import load_dotenv

app = typer.Typer(help="FR-4: apply migrations and COPY-load parquet into Postgres.")

MIGRATIONS_DIR = Path("sql/migrations")
ANALYTICS_DIR = Path("sql/analytics")
DEFAULT_PARQUET = Path("data/processed/reviews.parquet")
REVIEW_COLUMNS = (
    "review_id",
    "product_id",
    "rating",
    "label",
    "title",
    "text",
    "clean_text",
    "reviewed_at",
    "split",
)


def database_url() -> str:
    """Load .env and return a psycopg connection string (no +driver suffix)."""
    load_dotenv()
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL not set — copy .env.example to .env")
    return url.replace("postgresql+psycopg://", "postgresql://")


def apply_migrations(
    migrations_dir: Path = MIGRATIONS_DIR,
    conn: psycopg.Connection | None = None,
) -> list[str]:
    """Apply pending .sql files in order; return filenames applied this call."""
    owns = conn is None
    if conn is None:
        conn = psycopg.connect(database_url())
    applied: list[str] = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                  filename TEXT PRIMARY KEY,
                  applied_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            cur.execute("SELECT filename FROM schema_migrations")
            done = {row[0] for row in cur.fetchall()}
            for path in sorted(migrations_dir.glob("*.sql")):
                if path.name in done:
                    continue
                sql_text = path.read_text(encoding="utf-8")
                with conn.transaction():
                    cur.execute(sql_text)
                    cur.execute(
                        "INSERT INTO schema_migrations (filename) VALUES (%s)",
                        (path.name,),
                    )
                applied.append(path.name)
        conn.commit()
    finally:
        if owns:
            conn.close()
    return applied


def _ts_or_none(value: object) -> object:
    """Convert pandas Timestamp/NaT to psycopg-friendly UTC datetime or None."""
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC").to_pydatetime()
    return ts.tz_convert("UTC").to_pydatetime()


def load_parquet(
    parquet_path: Path = DEFAULT_PARQUET,
    conn: psycopg.Connection | None = None,
    batch_size: int = 50_000,
) -> dict[str, int]:
    """Bulk-load reviews.parquet into products + reviews via COPY (idempotent).

    Products: ON CONFLICT (product_id) DO NOTHING.
    Reviews: ON CONFLICT (review_id) DO NOTHING so re-runs never duplicate.
    Returns {"products": n, "reviews": n} rows newly inserted.
    """
    owns = conn is None
    if conn is None:
        conn = psycopg.connect(database_url())
    frame = pd.read_parquet(parquet_path)
    if "split" not in frame.columns:
        raise ValueError("reviews.parquet missing split column — run make splits first")

    products = (
        frame.dropna(subset=["product_id"])
        .groupby("product_id", as_index=False)
        .agg(category=("category", "first"), title=("title", "first"))
    )

    written = {"products": 0, "reviews": 0}
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS tmp_products")
            cur.execute("DROP TABLE IF EXISTS tmp_reviews")
            cur.execute(
                """
                CREATE TEMP TABLE tmp_products (
                  product_id TEXT, category TEXT, title TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TEMP TABLE tmp_reviews (
                  review_id TEXT, product_id TEXT, rating SMALLINT, label TEXT,
                  title TEXT, text TEXT, clean_text TEXT,
                  reviewed_at TIMESTAMPTZ, split TEXT
                )
                """
            )
        conn.commit()

        for start in range(0, len(products), batch_size):
            chunk = products.iloc[start : start + batch_size]
            with (
                conn.cursor() as cur,
                cur.copy("COPY tmp_products (product_id, category, title) FROM STDIN") as copy,
            ):
                for product_id, category, title in chunk.itertuples(index=False, name=None):
                    copy.write_row(
                        (
                            product_id,
                            category,
                            None if pd.isna(title) else title,
                        )
                    )
            conn.commit()

        for start in range(0, len(frame), batch_size):
            chunk = frame.iloc[start : start + batch_size]
            with (
                conn.cursor() as cur,
                cur.copy(
                    """
                    COPY tmp_reviews (
                      review_id, product_id, rating, label,
                      title, text, clean_text, reviewed_at, split
                    ) FROM STDIN
                    """
                ) as copy,
            ):
                for row in chunk.loc[:, list(REVIEW_COLUMNS)].itertuples(index=False, name=None):
                    (
                        review_id,
                        product_id,
                        rating,
                        label,
                        title,
                        text,
                        clean,
                        reviewed,
                        split,
                    ) = row
                    copy.write_row(
                        (
                            review_id,
                            product_id,
                            int(rating),
                            label,
                            None if pd.isna(title) else title,
                            text,
                            clean,
                            _ts_or_none(reviewed),
                            split,
                        )
                    )
            conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO products (product_id, category, title)
                SELECT product_id, category, title FROM tmp_products
                ON CONFLICT (product_id) DO NOTHING
                """
            )
            written["products"] = cur.rowcount
            cur.execute(
                """
                INSERT INTO reviews (
                  review_id, product_id, rating, label,
                  title, text, clean_text, reviewed_at, split
                )
                SELECT
                  review_id, product_id, rating, label,
                  title, text, clean_text, reviewed_at, split
                FROM tmp_reviews
                ON CONFLICT (review_id) DO NOTHING
                """
            )
            written["reviews"] = cur.rowcount
        conn.commit()
    finally:
        if owns:
            conn.close()
    return written


def count_tables(conn: psycopg.Connection | None = None) -> dict[str, int]:
    """Row counts for acceptance: Postgres counts must equal parquet counts."""
    owns = conn is None
    if conn is None:
        conn = psycopg.connect(database_url())
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM products")
            products = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM reviews")
            reviews = cur.fetchone()[0]
        return {"products": int(products), "reviews": int(reviews)}
    finally:
        if owns:
            conn.close()


def run_analytics(
    analytics_dir: Path = ANALYTICS_DIR,
    conn: psycopg.Connection | None = None,
    limit_rows: int = 5,
) -> dict[str, list[tuple]]:
    """Execute every whitelisted analytics .sql file once; return sample rows.

    Files are static SQL from the repo (never LLM-generated at runtime) — see
    AGENTS rule 7. Each must start with a `-- Business question:` comment.
    """
    owns = conn is None
    if conn is None:
        conn = psycopg.connect(database_url())
    results: dict[str, list[tuple]] = {}
    try:
        with conn.cursor() as cur:
            for path in sorted(analytics_dir.glob("*.sql")):
                sql_text = path.read_text(encoding="utf-8").strip()
                if "-- Business question:" not in sql_text.splitlines()[0] and (
                    "-- Business question:" not in sql_text[:400]
                ):
                    raise ValueError(f"{path.name} missing Business question header")
                cur.execute(sql_text)
                rows = cur.fetchmany(limit_rows)
                results[path.name] = rows
        conn.rollback()  # read-only pass; do not leave a transaction open
    finally:
        if owns:
            conn.close()
    return results


@app.command()
def migrate(
    migrations_dir: Annotated[Path, typer.Option("--migrations-dir")] = MIGRATIONS_DIR,
) -> None:
    """Apply pending SQL migrations (idempotent)."""
    applied = apply_migrations(migrations_dir)
    if applied:
        typer.echo(f"applied: {applied}")
    else:
        typer.echo("no pending migrations")


@app.command()
def load(
    parquet: Annotated[Path, typer.Option("--parquet")] = DEFAULT_PARQUET,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 50_000,
) -> None:
    """COPY-load reviews.parquet into products + reviews (idempotent)."""
    written = load_parquet(parquet, batch_size=batch_size)
    typer.echo(f"inserted products={written['products']} reviews={written['reviews']}")
    typer.echo(f"table counts: {count_tables()}")


@app.command()
def counts() -> None:
    """Print products/reviews row counts."""
    typer.echo(count_tables())


@app.command()
def analytics(
    analytics_dir: Annotated[Path, typer.Option("--dir")] = ANALYTICS_DIR,
) -> None:
    """Run every sql/analytics/*.sql once and print first rows of each."""
    results = run_analytics(analytics_dir)
    for name, rows in results.items():
        typer.echo(f"=== {name} ({len(rows)} sample rows) ===")
        for row in rows:
            typer.echo(f"  {row}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
