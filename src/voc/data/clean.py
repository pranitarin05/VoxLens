"""FR-2: Pandas reference cleaner (PRD 4.4) writing parquet to data/processed/.

Rules (PRD 4.4): strip HTML, normalise whitespace, drop empty or <10-char text,
drop exact duplicates on (product id, text), keep original text and clean_text.
review_id follows D11; labels come from configs/labels.yaml (PRD 4.3).
"""

from __future__ import annotations

import hashlib
import html
import re
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import yaml

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

app = typer.Typer(help="FR-2: clean raw jsonl.gz into reviews_clean.parquet.")

REQUIRED_COLUMNS = ("rating", "title", "text", "asin", "parent_asin", "user_id", "timestamp")
OUTPUT_COLUMNS = (
    "review_id",
    "product_id",
    "category",
    "rating",
    "label",
    "title",
    "text",
    "clean_text",
    "reviewed_at",
)
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUT = Path("data/processed/reviews_clean.parquet")
DEFAULT_LABELS = Path("configs/labels.yaml")


def load_labels(path: Path | None = None) -> dict[int, str]:
    """Load rating → sentiment mapping from configs/labels.yaml."""
    labels_path = path or DEFAULT_LABELS
    with labels_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return {int(rating): str(label) for rating, label in config["rating_to_sentiment"].items()}


def strip_html(value: str) -> str:
    """Unescape entities, remove tags, collapse whitespace, strip ends."""
    text = html.unescape(value)
    text = TAG_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def _segment(value: object) -> str:
    """Stringify a hash segment; None/NaN/NaT → empty string (D11 format)."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if value is pd.NaT or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, pd.Timestamp):
        return str(value.value // 10**6)  # ns → unix ms, matches raw JSON ints
    if isinstance(value, str):
        return value
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def make_review_id(user_id: object, product_id: object, timestamp: object, text: str) -> str:
    """Deterministic review id (D11): sha256(user|product|timestamp|text)[:32]."""
    payload = f"{_segment(user_id)}|{_segment(product_id)}|{_segment(timestamp)}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def clean_chunk(chunk: pd.DataFrame, category: str, labels: dict[int, str]) -> pd.DataFrame:
    """Apply row-local cleaning rules to one chunk (no cross-row state)."""
    missing = [column for column in REQUIRED_COLUMNS if column not in chunk.columns]
    if missing:
        raise ValueError(f"raw chunk missing columns: {missing}")

    frame = chunk.copy()
    product = frame["parent_asin"].fillna(frame["asin"])
    frame = frame.assign(product_id=product)
    frame = frame[frame["product_id"].notna() & (frame["product_id"].astype(str) != "")]

    raw_text = frame["text"].fillna("").astype(str)
    clean_text = raw_text.map(strip_html)
    keep = clean_text.str.len() >= 10
    frame = frame.assign(text=raw_text, clean_text=clean_text)
    frame = frame[keep]

    rating = pd.to_numeric(frame["rating"], errors="coerce")
    frame = frame.assign(rating=rating)
    frame = frame[frame["rating"].isin([1, 2, 3, 4, 5])]
    frame = frame.assign(
        rating=frame["rating"].astype("int64"),
        label=frame["rating"].map(labels),
    )
    frame = frame[frame["label"].notna()]

    # Keep raw unix-ms for the D11 hash even if pandas inferred datetimes.
    ts_ms = frame["timestamp"].map(
        lambda value: (
            value.value // 10**6
            if isinstance(value, pd.Timestamp)
            else (None if pd.isna(value) else int(value))
        )
    )
    frame = frame.assign(
        review_id=[
            make_review_id(user, product_id, timestamp, text)
            for user, product_id, timestamp, text in zip(
                frame["user_id"],
                frame["product_id"],
                ts_ms,
                frame["text"],
                strict=True,
            )
        ],
        reviewed_at=pd.to_datetime(ts_ms, unit="ms", utc=True, errors="coerce"),
        category=category,
        title=frame["title"].fillna("").astype(str).map(strip_html),
    )
    return frame.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)


def clean_file(
    path: Path,
    category: str,
    labels: dict[int, str],
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Stream one raw category file through clean_chunk, then global exact dedupe."""
    parts: list[pd.DataFrame] = []
    reader = pd.read_json(path, lines=True, compression="infer", chunksize=chunksize)
    for chunk in reader:
        parts.append(clean_chunk(chunk, category, labels))
    if not parts:
        raise ValueError(f"no rows read from {path}")
    frame = pd.concat(parts, ignore_index=True)
    before = len(frame)
    frame = frame.drop_duplicates(subset=["product_id", "text"], keep="first")
    frame = frame.reset_index(drop=True)
    frame.attrs["rows_in"] = before
    frame.attrs["rows_deduped"] = before - len(frame)
    return frame


def clean_categories(
    raw_dir: Path,
    out_path: Path,
    labels: dict[int, str],
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Clean all selected raw files into a single deduped DataFrame and write parquet."""
    selected = categories or [
        path.name.removesuffix(".jsonl.gz") for path in sorted(raw_dir.glob("*.jsonl.gz"))
    ]
    frames = []
    for category in selected:
        source = raw_dir / f"{category}.jsonl.gz"
        if not source.exists():
            raise FileNotFoundError(f"raw file missing: {source} (run ingest first)")
        typer.echo(f"clean {category}: {source}")
        frames.append(clean_file(source, category, labels))
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["product_id", "text"], keep="first")
    combined = combined.reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    typer.echo(
        f"wrote {out_path}: {len(combined)} rows "
        f"(pre-dedupe across categories {before}, dropped {before - len(combined)})"
    )
    return combined


@app.command()
def run(
    raw_dir: Annotated[Path, typer.Option("--raw-dir")] = DEFAULT_RAW_DIR,
    out: Annotated[Path, typer.Option("--out")] = DEFAULT_OUT,
    labels_file: Annotated[Path, typer.Option("--labels-file")] = DEFAULT_LABELS,
    categories: Annotated[
        list[str] | None,
        typer.Argument(help="Subset of categories (default: all on disk)."),
    ] = None,
) -> None:
    """Clean raw category files into a deduped parquet dataset."""
    labels = load_labels(labels_file)
    clean_categories(raw_dir, out, labels, list(categories) if categories else None)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
