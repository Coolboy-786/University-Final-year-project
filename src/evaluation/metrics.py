"""Per-class and aggregate metrics (Section 9).

Computes per-class precision, recall, F1 plus macro- and
weighted-averaged variants from a run's predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Compute per-class precision, recall, and F1.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels, shape ``(N,)``.
    y_pred : np.ndarray
        Predicted integer labels, shape ``(N,)``.
    class_names : list[str]
        Class name for each integer index (length == number of classes).

    Returns
    -------
    pd.DataFrame
        Columns: ``class``, ``precision``, ``recall``, ``f1``, ``support``.
        One row per class, sorted by class index.
    """
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "class": class_names,
            "precision": precision.astype(float),
            "recall": recall.astype(float),
            "f1": f1.astype(float),
            "support": support.astype(int),
        }
    )


def compute_aggregate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute macro- and weighted-averaged aggregate metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.

    Returns
    -------
    dict[str, float]
        Keys: ``accuracy``, ``macro_precision``, ``macro_recall``,
        ``macro_f1``, ``weighted_precision``, ``weighted_recall``,
        ``weighted_f1``.
    """
    acc = float(accuracy_score(y_true, y_pred))

    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": float(p_mac),
        "macro_recall": float(r_mac),
        "macro_f1": float(f1_mac),
        "weighted_precision": float(p_wt),
        "weighted_recall": float(r_wt),
        "weighted_f1": float(f1_wt),
    }
