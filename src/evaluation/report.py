"""Generate paper-ready tables and figures from evaluation results (Section 3).

Produces:
  - Table (b): main results across models.
  - Table (c): router on/off ablation.
  - Table (d): per-class precision/recall/F1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_MAIN_COLS = ["model", "preprocessing", "accuracy", "macro_f1", "weighted_f1"]
_DISPLAY_COLS = ["Model", "Preprocessing", "Accuracy", "Macro F1", "Weighted F1"]
_ROUND = 4


def build_main_results_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble Table (b): main results across models and preprocessing combos.

    Parameters
    ----------
    results : list[dict[str, Any]]
        One dict per experiment row; expected keys: ``model``,
        ``preprocessing``, ``accuracy``, ``macro_f1``, ``weighted_f1``.

    Returns
    -------
    pd.DataFrame
        Sorted by ``macro_f1`` descending, rounded to 4 d.p., with
        display-friendly column names. Ready for ``to_latex()``.
    """
    df = pd.DataFrame(results)[_MAIN_COLS].copy()
    df = df.sort_values("macro_f1", ascending=False).round(_ROUND)
    df.columns = pd.Index(_DISPLAY_COLS)
    return df.reset_index(drop=True)


def build_ablation_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Assemble Table (c): router ablation results (routed vs. fixed strategies).

    Parameters
    ----------
    results : list[dict[str, Any]]
        Subset of experiments comparing ``routed`` against other preprocessing
        modes; same schema as :func:`build_main_results_table`.

    Returns
    -------
    pd.DataFrame
        Sorted by ``preprocessing`` then ``macro_f1`` descending.
        Ready for ``to_latex()``.
    """
    df = pd.DataFrame(results)[_MAIN_COLS].copy()
    df = df.sort_values(
        ["preprocessing", "macro_f1"], ascending=[True, False]
    ).round(_ROUND)
    df.columns = pd.Index(_DISPLAY_COLS)
    return df.reset_index(drop=True)


def build_per_class_table(per_class_df: pd.DataFrame) -> pd.DataFrame:
    """Format Table (d): per-class precision/recall/F1.

    Parameters
    ----------
    per_class_df : pd.DataFrame
        Output of :func:`~src.evaluation.metrics.compute_per_class_metrics`.
        Expected columns: ``class``, ``precision``, ``recall``, ``f1``,
        ``support``.

    Returns
    -------
    pd.DataFrame
        Sorted alphabetically by class, rounded to 4 d.p.
        Ready for ``to_latex()``.
    """
    df = per_class_df.copy()
    df = df.sort_values("class").round(_ROUND)
    df.columns = pd.Index(["Class", "Precision", "Recall", "F1", "Support"])
    return df.reset_index(drop=True)


def export_latex(
    df: pd.DataFrame,
    output_path: Path,
    caption: str = "",
    label: str = "",
) -> None:
    """Write a DataFrame to a ``.tex`` file as a LaTeX table.

    Wraps the tabular in a ``table`` float when ``caption`` or ``label`` is
    given. Uses manual string assembly to avoid the jinja2 dependency
    introduced by ``DataFrame.to_latex(caption=, label=)`` in pandas 2.x.

    Parameters
    ----------
    df : pd.DataFrame
        Table to export.
    output_path : Path
        Destination ``.tex`` file (parent directories created if absent).
    caption : str
        LaTeX ``\\caption{...}`` text.
    label : str
        LaTeX ``\\label{...}`` identifier (for ``\\ref{}``).
    """
    tabular = df.to_latex(index=False, escape=True, float_format=f"%.{_ROUND}f")

    if caption or label:
        parts = ["\\begin{table}[htbp]", "\\centering"]
        if caption:
            parts.append(f"\\caption{{{caption}}}")
        if label:
            parts.append(f"\\label{{{label}}}")
        parts.append(tabular.strip())
        parts.append("\\end{table}")
        latex_str = "\n".join(parts)
    else:
        latex_str = tabular

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str, encoding="utf-8")
