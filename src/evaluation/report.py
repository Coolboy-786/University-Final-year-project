"""Generate paper-ready tables and figures from evaluation results (Section 3).

Produces:
  - Table (b): main results across models.
  - Table (c): router on/off ablation.
  - Table (d): per-class precision/recall/F1.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_main_results_table(results: list[dict[str, object]]) -> pd.DataFrame:
    """Assemble Table (b): main results across models/preprocessing combos.

    Parameters
    ----------
    results : list[dict[str, object]]
        One dict per experiment row; expected keys: ``model``,
        ``preprocessing``, ``accuracy``, ``macro_f1``, ``weighted_f1``.

    Returns
    -------
    pd.DataFrame
        Formatted for direct LaTeX export via ``to_latex()``.
    """
    raise NotImplementedError


def build_ablation_table(results: list[dict[str, object]]) -> pd.DataFrame:
    """Assemble Table (c): router ablation results.

    Parameters
    ----------
    results : list[dict[str, object]]
        Subset of experiments comparing ``routed`` vs. uniform strategies.

    Returns
    -------
    pd.DataFrame
        Formatted for direct LaTeX export.
    """
    raise NotImplementedError


def build_per_class_table(per_class_df: pd.DataFrame) -> pd.DataFrame:
    """Format Table (d): per-class precision/recall/F1.

    Parameters
    ----------
    per_class_df : pd.DataFrame
        Output of :func:`~src.evaluation.metrics.compute_per_class_metrics`.

    Returns
    -------
    pd.DataFrame
        Rounded and sorted by class name for LaTeX export.
    """
    raise NotImplementedError


def export_latex(df: pd.DataFrame, output_path: Path, caption: str = "", label: str = "") -> None:
    """Write a DataFrame to a ``.tex`` file as a LaTeX table.

    Parameters
    ----------
    df : pd.DataFrame
        Table to export.
    output_path : Path
        Destination ``.tex`` file.
    caption : str
        LaTeX table caption.
    label : str
        LaTeX table label (for ``\\ref{}``).
    """
    raise NotImplementedError
