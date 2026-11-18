"""FR-6: accuracy, macro-F1, per-class P/R/F1, confusion matrix.

One shared implementation used by every model so the comparison table is
apples-to-apples. Label order is fixed (negative, neutral, positive).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

LABELS: tuple[str, str, str] = ("negative", "neutral", "positive")


def compute_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS
) -> dict[str, object]:
    """Return accuracy, macro-F1, per-class P/R/F1, support and confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: y_true={len(y_true)} y_pred={len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("empty input")

    label_list = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_list, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_list)

    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(label_list)
    }

    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "labels": label_list,
        "confusion_matrix": cm.tolist(),
    }
