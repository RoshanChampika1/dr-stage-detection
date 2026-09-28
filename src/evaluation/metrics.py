"""Classification metrics for DR staging.

- Accuracy, precision, recall and F1 (per class, macro and weighted average)
- Quadratic Weighted Kappa (QWK): agreement between predicted and true stage
  that penalises big mistakes (stage 0 vs 4) more than small ones (2 vs 3).
  It is the standard metric for DR grading and was the official metric of
  the Kaggle DR competitions.
- Binary DR detection (any DR = stage > 0): accuracy, sensitivity,
  specificity and ROC AUC, derived from the 5-class prediction. This covers
  "classify diabetic retinopathy as well as the stage".
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None, num_classes: int = 5
) -> dict:
    """Compute all stage-level and binary metrics.

    Args:
        y_true: True stages, shape (N,).
        y_pred: Predicted stages, shape (N,).
        probs: Softmax probabilities, shape (N, num_classes). Needed for AUC.

    Returns:
        Flat dict of scalar metrics plus per-class lists.
    """
    labels = list(range(num_classes))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
    }

    # Binary: DR present (stage 1-4) vs no DR (stage 0).
    bt, bp = (y_true > 0).astype(int), (y_pred > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(bt, bp, labels=[0, 1]).ravel()
    out["binary_accuracy"] = float((tp + tn) / max(len(bt), 1))
    out["binary_sensitivity"] = float(tp / max(tp + fn, 1))
    out["binary_specificity"] = float(tn / max(tn + fp, 1))
    if probs is not None and len(np.unique(bt)) == 2:
        out["binary_auc"] = float(roc_auc_score(bt, 1.0 - probs[:, 0]))
    return out
