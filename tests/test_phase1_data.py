"""Phase 1 tests: FR-1 skip logic, FR-2 cleaning rules, splits, eval file sizes."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from voc.data.clean import (
    clean_chunk,
    clean_file,
    load_labels,
    make_review_id,
    strip_html,
)
from voc.data.ingest import count_rows, download_category
from voc.data.splits import assign_splits, build_eval_files, split_from_id, stratified_sample

FIXTURE = Path(__file__).parent / "fixtures" / "reviews_10.jsonl"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def labels() -> dict[int, str]:
    return load_labels(ROOT / "configs" / "labels.yaml")


@pytest.fixture(scope="module")
def raw_frame(labels: dict[int, str]) -> pd.DataFrame:
    return pd.read_json(FIXTURE, lines=True)


def test_label_mapping_matches_prd(labels: dict[int, str]) -> None:
    assert labels == {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}


def test_strip_html_entities_and_whitespace() -> None:
    assert strip_html("Hello <b>world</b>") == "Hello world"
    assert strip_html("a &amp; b") == "a & b"
    assert strip_html("line1\n\n  line2") == "line1 line2"
    assert strip_html("<p>Terrible quality</p><br/>The fabric tore") == (
        "Terrible quality The fabric tore"
    )


def test_make_review_id_is_deterministic_and_frozen() -> None:
    first = make_review_id("U1", "B001", 1600000000000, "hello text here")
    second = make_review_id("U1", "B001", 1600000000000, "hello text here")
    assert first == second
    assert len(first) == 32
    assert all(c in "0123456789abcdef" for c in first)
    # Frozen D11 format: sha256("U1|B001|1600000000000|hello text here")[:32]
    expected = hashlib.sha256(b"U1|B001|1600000000000|hello text here").hexdigest()[:32]
    assert first == expected
    # Null user/ts become empty segments
    null_id = make_review_id(None, "B001", None, "hello text here")
    assert null_id == hashlib.sha256(b"|B001||hello text here").hexdigest()[:32]
    # pd.Timestamp input hashes as unix ms (same as raw JSON int)
    ts = pd.Timestamp("2020-09-13 12:26:40", tz="UTC")
    assert ts.value // 10**6 == 1_600_000_000_000
    assert make_review_id("U1", "B001", ts, "x") == make_review_id(
        "U1", "B001", 1_600_000_000_000, "x"
    )


def test_clean_chunk_covers_prd_44_and_fixture(
    labels: dict[int, str], raw_frame: pd.DataFrame
) -> None:
    cleaned = clean_chunk(raw_frame, "TestCat", labels)
    # Row-local drops only: empty text (U2) and <10 chars (U3) → 8 rows.
    # Exact (product_id, text) dedupe happens in clean_file, not here.
    assert len(cleaned) == 8
    assert set(cleaned["label"]) <= {"negative", "neutral", "positive"}
    by_rating = cleaned.groupby("rating")["label"].first().to_dict()
    assert by_rating[1] == "negative"
    assert by_rating[2] == "negative"
    assert by_rating[3] == "neutral"
    assert by_rating[4] == "positive"
    assert by_rating[5] == "positive"
    assert "<b>" not in cleaned["clean_text"].tolist()[0]
    assert not cleaned["clean_text"].str.contains("<").any()
    assert cleaned["clean_text"].str.len().min() >= 10
    assert "B005EEEEEE" in set(cleaned["product_id"])
    assert cleaned["review_id"].is_unique
    assert set(cleaned["category"]) == {"TestCat"}
    assert str(cleaned["reviewed_at"].dtype).startswith("datetime64")


def test_clean_file_drops_exact_product_text_duplicates(
    tmp_path: Path, labels: dict[int, str]
) -> None:
    gz_path = tmp_path / "Fixture.jsonl.gz"
    with FIXTURE.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        dst.write(src.read())
    frame = clean_file(gz_path, "Fixture", labels)
    # 10 raw − empty − short − exact dup (U9 same product+text as U1) = 7
    assert len(frame) == 7
    b001 = frame[frame["product_id"] == "B001AAAAAA"]
    assert len(b001) == 1
    assert frame["review_id"].is_unique


def test_ingest_skips_existing_and_counts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    existing = raw_dir / "Amazon_Fashion.jsonl.gz"
    with gzip.open(existing, "wt", encoding="utf-8") as handle:
        for i in range(3):
            handle.write(f'{{"rating": 5, "text": "row {i} is fine"}}\n')
    path = download_category("Amazon_Fashion", raw_dir)
    assert path == existing
    assert count_rows(path) == 3


def test_split_from_id_boundaries_and_determinism() -> None:
    seen: dict[str, str | None] = {"train": None, "val": None, "test": None}
    for i in range(100_000):
        rid = hashlib.sha256(f"probe-{i}".encode()).hexdigest()[:32]
        split = split_from_id(rid)
        if seen[split] is None:
            seen[split] = rid
        if all(seen.values()):
            break
    assert all(seen.values()), "failed to find sample ids for all splits"
    for split, rid in seen.items():
        assert rid is not None
        assert split_from_id(rid) == split
    unique_ids = [value for value in seen.values() if value is not None]
    a = assign_splits(pd.DataFrame({"review_id": unique_ids}))
    b = assign_splits(pd.DataFrame({"review_id": unique_ids}))
    assert a["split"].tolist() == b["split"].tolist()
    # Function property: each review_id maps to exactly one split (no leakage)
    for rid in unique_ids:
        assert split_from_id(rid) == split_from_id(rid)
    assert len(set(unique_ids)) == len(unique_ids)


def test_stratified_sample_sizes_and_caps() -> None:
    rows = []
    for i in range(300):
        rows.append(
            {
                "review_id": hashlib.sha256(f"s{i}".encode()).hexdigest()[:32],
                "label": ("positive", "negative", "neutral")[i % 3],
            }
        )
    frame = pd.DataFrame(rows)
    sample = stratified_sample(frame, 30, seed=42)
    assert len(sample) == 30
    counts = sample["label"].value_counts().to_dict()
    assert counts == {"positive": 10, "negative": 10, "neutral": 10}
    assert len(stratified_sample(frame, 1_000, seed=42)) == 300
    again = stratified_sample(frame, 30, seed=42)
    assert sample["review_id"].tolist() == again["review_id"].tolist()


def test_build_eval_files_writes_expected_sizes(tmp_path: Path) -> None:
    rows = []
    for i in range(2_000):
        rows.append(
            {
                "review_id": hashlib.sha256(f"e{i}".encode()).hexdigest()[:32],
                "label": ("positive", "negative", "neutral")[i % 3],
                "text": f"review number {i} with enough characters",
            }
        )
    frame = assign_splits(pd.DataFrame(rows))
    counts = build_eval_files(frame, tmp_path, seed=42)
    train_n = int((frame["split"] == "train").sum())
    val_n = int((frame["split"] == "val").sum())
    test_n = int((frame["split"] == "test").sum())
    assert counts["train_200k"] == min(200_000, train_n)
    assert counts["val_20k"] == min(20_000, val_n)
    assert counts["test_50k"] == min(50_000, test_n)
    assert counts["llm_test_1500"] == min(1_500, test_n)
    for name in counts:
        assert (tmp_path / f"{name}.parquet").exists()
    llm = pd.read_parquet(tmp_path / "llm_test_1500.parquet")
    assert llm["review_id"].is_unique
    assert set(llm["review_id"]) <= set(frame.loc[frame["split"] == "test", "review_id"])


def test_split_assign_never_moves_rows(raw_frame: pd.DataFrame, labels: dict[int, str]) -> None:
    cleaned = clean_chunk(raw_frame, "TestCat", labels)
    with_split = assign_splits(cleaned)
    assert len(with_split) == len(cleaned)
    assert set(with_split["split"]) <= {"train", "val", "test"}
    assert with_split["review_id"].tolist() == cleaned["review_id"].tolist()
