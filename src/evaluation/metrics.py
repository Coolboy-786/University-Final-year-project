"""Per-class and aggregate metrics (Section 9).

Computes per-class precision, recall, F1 plus macro- and
weighted-averaged variants from a run's predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
        Class name for each integer index.

    Returns
    -------
    pd.DataFrame
        Columns: ``class``, ``precision``, ``recall``, ``f1``, ``support``.
    """
    raise NotImplementedError


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
    raise NotImplementedError
