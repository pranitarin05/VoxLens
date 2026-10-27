"""Deterministic hash-based 80/10/10 split and fixed evaluation file builders.

PRD 4.5: split = f(hash(review_id)) after dedupe, stored once, never recomputed.
PRD 4.6: train_200k, val_20k, test_50k, llm_test_1500 — stratified, seed 42.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import yaml
from sklearn.model_selection import train_test_split

app = typer.Typer(help="Assign split column and build fixed evaluation parquet files.")

SPLIT_RATIOS = (0.80, 0.90)  # cumulative: train < 0.80 <= val < 0.90 <= test
EVAL_SPECS = (
    ("train_200k", "train", 200_000),
    ("val_20k", "val", 20_000),
    ("test_50k", "test", 50_000),
    ("llm_test_1500", "test", 1_500),
)
DEFAULT_SOURCE = Path("data/processed/reviews_clean.parquet")
DEFAULT_OUT_DIR = Path("data/processed")
DEFAULT_SEED_FILE = Path("configs/training.yaml")


def load_seed(path: Path | None = None) -> int:
    """Read the project seed from configs/training.yaml (PRD seed 42)."""
    seed_path = path or DEFAULT_SEED_FILE
    with seed_path.open(encoding="utf-8") as handle:
        return int(yaml.safe_load(handle)["seed"])


def split_from_id(review_id: str) -> str:
    """Map a review_id to train/val/test via first 8 hex digits mod 10000."""
    bucket = int(review_id[:8], 16) % 10_000
    train_cut = int(SPLIT_RATIOS[0] * 10_000)
    val_cut = int(SPLIT_RATIOS[1] * 10_000)
    if bucket < train_cut:
        return "train"
    if bucket < val_cut:
        return "val"
    return "test"


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the split column (idempotent: same ids always yield same splits)."""
    if "review_id" not in df.columns:
        raise ValueError("frame missing review_id")
    out = df.copy()
    out["split"] = [split_from_id(value) for value in out["review_id"]]
    return out


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Up to n rows stratified by label; returns all rows if already ≤ n."""
    if n <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= n:
        return df.reset_index(drop=True)
    try:
        sample, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    except ValueError:
        # Rare: a class has <2 members in this split — fall back to plain sample.
        sample = df.sample(n=n, random_state=seed)
    return sample.reset_index(drop=True)


def build_eval_files(df: pd.DataFrame, out_dir: Path, seed: int) -> dict[str, int]:
    """Write the four fixed evaluation files (PRD 4.6); returns row counts written."""
    if "split" not in df.columns:
        raise ValueError("frame missing split column — run assign_splits first")
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, split_name, size in EVAL_SPECS:
        subset = df[df["split"] == split_name]
        sample = stratified_sample(subset, size, seed)
        target = out_dir / f"{name}.parquet"
        sample.to_parquet(target, index=False)
        counts[name] = len(sample)
        typer.echo(f"wrote {target}: {len(sample)} rows (from {split_name}={len(subset)})")
    return counts


@app.command()
def run(
    source: Annotated[
        Path, typer.Option("--source", help="Cleaned parquet without split column.")
    ] = DEFAULT_SOURCE,
    out_dir: Annotated[Path, typer.Option("--out-dir")] = DEFAULT_OUT_DIR,
    seed_file: Annotated[Path, typer.Option("--seed-file")] = DEFAULT_SEED_FILE,
) -> None:
    """Add split column to reviews_clean.parquet and write the four eval files."""
    seed = load_seed(seed_file)
    frame = pd.read_parquet(source)
    frame = assign_splits(frame)
    main_out = out_dir / "reviews.parquet"
    frame.to_parquet(main_out, index=False)
    typer.echo(f"wrote {main_out}: {len(frame)} rows with split column")
    for split_name in ("train", "val", "test"):
        subset = frame[frame["split"] == split_name]
        typer.echo(f"  {split_name}: {len(subset)} ({len(subset) / len(frame):.1%})")
    build_eval_files(frame, out_dir, seed)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
