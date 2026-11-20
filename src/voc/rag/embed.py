"""PRD 10: stratified 200k sample + MiniLM embeddings into pgvector.

The sample is deterministic (seed) and stratified by (category, label); quotas
use the largest-remainder method so the sample matches the corpus mix. Encoding
is idempotent: review ids already in ``review_embeddings`` are skipped, inserts
use ON CONFLICT DO NOTHING, and the HNSW index is created after the load.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import psycopg
import typer
from pgvector.psycopg import register_vector

from voc.db.load import database_url

app = typer.Typer(add_completion=False, help="PRD 10: embed reviews into pgvector.")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_LIMIT = 200_000
DEFAULT_SEED = 42
SAMPLE_PATH = Path("data/processed/rag_sample.parquet")
SAMPLE_COLUMNS = ["review_id", "product_id", "category", "label", "text", "reviewed_at"]


def allocate_quotas(counts: dict[tuple[str, str], int], total: int) -> dict[tuple[str, str], int]:
    """Proportional quotas capped by stratum size (largest remainder, stable)."""
    corpus = sum(counts.values())
    if corpus == 0:
        raise ValueError("no reviews to sample")
    if total >= corpus:
        return dict(counts)
    exact = {key: value * total / corpus for key, value in counts.items()}
    quotas = {key: int(np.floor(value)) for key, value in exact.items()}
    remainders = {key: exact[key] - quotas[key] for key in counts}
    order = sorted(counts, key=lambda key: (-remainders[key], key))
    for key in order[: total - sum(quotas.values())]:
        quotas[key] += 1
    return {key: min(quotas[key], counts[key]) for key in counts}


def stratum_counts(conn: psycopg.Connection) -> dict[tuple[str, str], int]:
    """Count reviews per (category, label)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT p.category, r.label, count(*) FROM reviews r "
            "JOIN products p ON p.product_id = r.product_id "
            "GROUP BY p.category, r.label ORDER BY p.category, r.label"
        )
        return {(category, label): int(count) for category, label, count in cur.fetchall()}


def build_sample(
    conn: psycopg.Connection, limit: int = DEFAULT_LIMIT, seed: int = DEFAULT_SEED
) -> pd.DataFrame:
    """Deterministically pick a stratified sample, ordered by a seeded hash."""
    quotas = allocate_quotas(stratum_counts(conn), limit)
    seed_key = str(seed)
    parts: list[pd.DataFrame] = []
    with conn.cursor() as cur:
        for (category, label), quota in sorted(quotas.items()):
            if quota <= 0:
                continue
            cur.execute(
                "SELECT r.review_id, r.product_id, p.category, r.label, r.text, r.reviewed_at "
                "FROM reviews r JOIN products p ON p.product_id = r.product_id "
                "WHERE p.category = %s AND r.label = %s "
                "ORDER BY md5(r.review_id || %s) LIMIT %s",
                (category, label, seed_key, quota),
            )
            rows = cur.fetchall()
            parts.append(pd.DataFrame(rows, columns=SAMPLE_COLUMNS))
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SAMPLE_COLUMNS)
    return frame


def load_embedder(device: str | None = None):
    """Load the MiniLM sentence encoder (GPU when available)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device=device)


def encode(model, texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Return L2-normalised float32 vectors (cosine == dot product)."""
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def existing_ids(conn: psycopg.Connection) -> set[str]:
    """Ids already embedded (used to make re-runs cheap and idempotent)."""
    with conn.cursor() as cur:
        cur.execute("SELECT review_id FROM review_embeddings")
        return {row[0] for row in cur.fetchall()}


def insert_embeddings(
    conn: psycopg.Connection,
    review_ids: list[str],
    vectors: np.ndarray,
    batch_size: int = 1000,
) -> int:
    """Insert embeddings with ON CONFLICT DO NOTHING; return rows attempted."""
    register_vector(conn)
    inserted = 0
    with conn.cursor() as cur:
        for start in range(0, len(review_ids), batch_size):
            rows = [
                (review_ids[index], vectors[index])
                for index in range(start, min(start + batch_size, len(review_ids)))
            ]
            cur.executemany(
                "INSERT INTO review_embeddings (review_id, embedding) VALUES (%s, %s) "
                "ON CONFLICT (review_id) DO NOTHING",
                rows,
            )
            inserted += len(rows)
    conn.commit()
    return inserted


def create_hnsw_index(conn: psycopg.Connection) -> None:
    """Create the cosine HNSW index after the bulk load (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_review_embeddings_hnsw "
            "ON review_embeddings USING hnsw (embedding vector_cosine_ops)"
        )
    conn.commit()


@app.command("build")
def build(
    limit: Annotated[int, typer.Option(help="sample size")] = DEFAULT_LIMIT,
    seed: Annotated[int, typer.Option()] = DEFAULT_SEED,
    batch_size: Annotated[int, typer.Option()] = 256,
    device: Annotated[str | None, typer.Option(help="cuda, cpu or auto")] = None,
    rebuild_sample: Annotated[bool, typer.Option(help="ignore cached sample parquet")] = False,
    skip_index: Annotated[bool, typer.Option(help="skip HNSW creation")] = False,
) -> None:
    """Build the sample, embed missing rows, insert and index."""
    started = time.perf_counter()
    conn = psycopg.connect(database_url())
    try:
        if rebuild_sample or not SAMPLE_PATH.exists():
            print(f"sampling {limit} reviews (seed={seed})...")
            frame = build_sample(conn, limit=limit, seed=seed)
            SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(SAMPLE_PATH, index=False)
            print(f"wrote {SAMPLE_PATH} rows={len(frame)}")
        else:
            frame = pd.read_parquet(SAMPLE_PATH)
            print(f"reusing {SAMPLE_PATH} rows={len(frame)}")

        done = existing_ids(conn)
        todo = frame[~frame["review_id"].isin(done)]
        print(f"existing={len(done)} to_embed={len(todo)}")
        if len(todo):
            model = load_embedder(device)
            vectors = encode(model, todo["text"].fillna("").tolist(), batch_size)
            attempted = insert_embeddings(conn, todo["review_id"].tolist(), vectors)
            print(f"inserted/attempted={attempted} (ON CONFLICT skips duplicates)")
        if not skip_index:
            create_hnsw_index(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM review_embeddings")
            total = cur.fetchone()[0]
        elapsed = time.perf_counter() - started
        print(f"embeddings_total={total} elapsed_s={elapsed:.1f}")
    finally:
        conn.close()


if __name__ == "__main__":
    app()
