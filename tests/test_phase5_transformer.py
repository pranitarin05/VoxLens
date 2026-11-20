"""Phase 5: transformer training helpers that need no GPU or model download."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from voc.models.transformer import (  # noqa: E402
    ID2LABEL,
    LABEL2ID,
    ReviewDataset,
    _compute_metrics,
    class_weights_from_labels,
)


def test_label_maps_are_inverse() -> None:
    assert LABEL2ID == {"negative": 0, "neutral": 1, "positive": 2}
    for label, idx in LABEL2ID.items():
        assert ID2LABEL[idx] == label


def test_class_weights_inverse_frequency() -> None:
    labels = ["negative", "negative", "neutral", "positive"]
    weights = class_weights_from_labels(labels)
    assert isinstance(weights, torch.Tensor)
    got = weights.tolist()
    # counts [2,1,1]; total 4; w_i = 4 / (3 * count_i)
    assert got[0] == pytest.approx(4 / (3 * 2))
    assert got[1] == pytest.approx(4 / (3 * 1))
    assert got[2] == pytest.approx(4 / (3 * 1))


def test_review_dataset_encodes_labels() -> None:
    def fake_tokenizer(text: str, truncation: bool, max_length: int):  # noqa: ARG001
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    dataset = ReviewDataset(
        ["a", "b", "c"], ["positive", "negative", "neutral"], fake_tokenizer, 16
    )
    assert len(dataset) == 3
    item = dataset[0]
    assert item["labels"] == LABEL2ID["positive"]
    assert item["input_ids"] == [1, 2, 3]
    assert dataset[2]["labels"] == LABEL2ID["neutral"]


def test_compute_metrics_from_logits() -> None:
    logits = np.array(
        [
            [3.0, 0.1, 0.1],  # negative
            [0.1, 3.0, 0.1],  # neutral
            [0.1, 0.1, 3.0],  # positive
            [3.0, 0.1, 0.1],  # negative
        ]
    )
    labels = np.array([0, 1, 2, 0])
    result = _compute_metrics((logits, labels))
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)
