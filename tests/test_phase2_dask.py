"""FR-3: Dask vs Pandas exact parity and small-batch benchmark helpers."""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from voc.data.clean import clean_chunk, load_labels
from voc.data.clean_spark import canonicalize_strings, clean_file_dask, parity_check

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "All_Beauty.jsonl.gz"
FIXTURE = Path(__file__).parent / "fixtures" / "reviews_10.jsonl"


@pytest.fixture(scope="module")
def labels() -> dict[int, str]:
    return load_labels(ROOT / "configs" / "labels.yaml")


def _write_sample(source: Path, dest: Path, rows: int) -> Path:
    """Copy the first `rows` lines of a gzipped jsonl into dest (idempotent)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    written = 0
    tmp = dest.with_suffix(".part.gz")
    with gzip.open(source, "rb") as src, gzip.open(tmp, "wb") as dst:
        for line in src:
            dst.write(line)
            written += 1
            if written >= rows:
                break
    tmp.replace(dest)
    return dest


def test_dask_matches_pandas_on_fixture(tmp_path: Path, labels: dict[int, str]) -> None:
    """Same 10-row fixture through both engines → exact DataFrame equality."""
    from voc.data.clean import clean_file as clean_file_pandas

    gz = tmp_path / "Fixture.jsonl.gz"
    with FIXTURE.open("rb") as src, gzip.open(gz, "wb") as dst:
        dst.write(src.read())

    pandas_df = canonicalize_strings(
        clean_file_pandas(gz, "Fixture", labels, chunksize=10).reset_index(drop=True)
    )
    dask_df = clean_file_dask(gz, "Fixture", labels, npartitions=2).compute()
    dask_df = canonicalize_strings(dask_df.reset_index(drop=True)).loc[:, list(pandas_df.columns)]

    pd.testing.assert_frame_equal(
        pandas_df, dask_df, check_dtype=True, check_exact=True, check_like=False
    )
    assert len(pandas_df) == 7  # same drops as Phase 1 fixture


def test_parity_check_helper_on_fixture(tmp_path: Path, labels: dict[int, str]) -> None:
    gz = tmp_path / "Fixture.jsonl.gz"
    with FIXTURE.open("rb") as src, gzip.open(gz, "wb") as dst:
        dst.write(src.read())
    report = parity_check(gz, "Fixture", labels, rows=10)
    assert report["equal"] is True, report["detail"]
    assert report["pandas_rows"] == report["dask_rows"] == 7


@pytest.mark.integration
def test_parity_10k_real_data(tmp_path: Path, labels: dict[int, str]) -> None:
    """FR-3 acceptance on 10k real All_Beauty rows (skipped if raw data absent)."""
    if not RAW.exists():
        pytest.skip("data/raw/All_Beauty.jsonl.gz not present")
    sample = _write_sample(RAW, tmp_path / "sample10k.jsonl.gz", 10_000)
    report = parity_check(sample, "All_Beauty", labels, rows=10_000)
    assert report["equal"] is True, report["detail"]
    assert report["pandas_rows"] == report["dask_rows"]


def test_clean_chunk_shared_by_both_engines(labels: dict[int, str]) -> None:
    """Dask path imports the same clean_chunk as Pandas (no duplicated rules)."""
    import voc.data.clean_spark as spark_mod

    assert spark_mod.clean_chunk is clean_chunk
