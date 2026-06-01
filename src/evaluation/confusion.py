"""Confusion matrix computation and plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Compute a labelled confusion matrix.

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
        Square DataFrame with class names as both index and columns.
        Rows = true class, columns = predicted class.
    """
    raise NotImplementedError


def plot_confusion_matrix(
    cm: pd.DataFrame,
    output_path: Path,
    normalize: bool = True,
    figsize: tuple[int, int] = (14, 12),
) -> plt.Figure:
    """Plot a confusion matrix heatmap and save to disk.

    Parameters
    ----------
    cm : pd.DataFrame
        Output of :func:`compute_confusion_matrix`.
    output_path : Path
        File path for the saved PNG.
    normalize : bool
        If True, normalise rows to sum to 1 (shows recall per class).
    figsize : tuple[int, int]
        Matplotlib figure size in inches.

    Returns
    -------
    plt.Figure
        The generated figure (also saved to ``output_path``).
    """
    raise NotImplementedError
