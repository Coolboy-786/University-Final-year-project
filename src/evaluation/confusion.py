"""Confusion matrix computation and plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


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
        Square DataFrame indexed and columned by class name.
        Rows = true class, columns = predicted class.
    """
    labels = list(range(len(class_names)))
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=class_names, columns=class_names)


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
        File path for the saved PNG (parent directories created if absent).
    normalize : bool
        If ``True``, normalise each row to sum to 1 (shows recall per class).
    figsize : tuple[int, int]
        Matplotlib figure size in inches.

    Returns
    -------
    plt.Figure
        The generated figure (also written to ``output_path``).
    """
    data = cm.to_numpy().astype(float)
    fmt = ".2f"
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.where(row_sums > 0, data / row_sums, 0.0)
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        xticklabels=list(cm.columns),
        yticklabels=list(cm.index),
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Confusion matrix" + (" (normalised)" if normalize else ""), fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
