"""FR-1: CLI that downloads raw Amazon Reviews 2023 files into data/raw/.

Re-running skips files already present and prints row counts per category.
Source layout follows D7 (mcauleylab.ucsd.edu direct jsonl.gz links).
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Annotated

import httpx
import typer

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories"
DEFAULT_CATEGORIES: tuple[str, ...] = ("Amazon_Fashion", "All_Beauty")
DEFAULT_RAW_DIR = Path("data/raw")

app = typer.Typer(help="FR-1: download raw Amazon Reviews 2023 category files.")


def download_category(category: str, raw_dir: Path) -> Path:
    """Download one category file; skip if a non-empty file already exists (idempotent)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{category}.jsonl.gz"
    if dest.exists() and dest.stat().st_size > 0:
        typer.echo(f"skip  {category}: already present ({dest.stat().st_size} bytes)")
        return dest

    url = f"{BASE_URL}/{category}.jsonl.gz"
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        head = client.head(url)
        head.raise_for_status()
        expected = int(head.headers.get("content-length", "0"))
        typer.echo(f"get   {category}: {url} ({expected} bytes)")
        tmp = dest.with_suffix(".gz.part")
        with client.stream("GET", url) as response:
            response.raise_for_status()
            written = 0
            with tmp.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1 << 20):
                    handle.write(chunk)
                    written += len(chunk)
        if expected and written != expected:
            tmp.unlink(missing_ok=True)
            raise typer.BadParameter(
                f"size mismatch for {category}: got {written}, expected {expected}"
            )
        tmp.replace(dest)
    typer.echo(f"ok    {category}: {dest} ({dest.stat().st_size} bytes)")
    return dest


def count_rows(path: Path) -> int:
    """Count newline-delimited JSON rows in a gzipped file without decoding fields."""
    rows = 0
    with gzip.open(path, "rb") as handle:
        for _ in handle:
            rows += 1
    return rows


@app.command()
def run(
    categories: Annotated[
        list[str] | None,
        typer.Argument(help="Categories to ensure on disk (default: Fashion + Beauty)."),
    ] = None,
    raw_dir: Annotated[Path, typer.Option("--raw-dir", help="Output directory.")] = DEFAULT_RAW_DIR,
) -> None:
    """Download missing category files, then print row counts for each."""
    selected = list(categories) if categories else list(DEFAULT_CATEGORIES)
    counts: dict[str, int] = {}
    for category in selected:
        path = download_category(category, raw_dir)
        counts[category] = count_rows(path)
    typer.echo("row counts:")
    for category, rows in counts.items():
        typer.echo(f"  {category}: {rows}")
    typer.echo(f"total: {sum(counts.values())}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
