"""Threshold-based class pruning (Section 6.3).

Drops any class that fails the configured thresholds, then returns the
surviving manifest and class list.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def count_per_class(manifest: pd.DataFrame) -> pd.Series:
    """Count images per class in a manifest.

    Parameters
    ----------
    manifest : pd.DataFrame
        Must contain a ``class_name`` column.

    Returns
    -------
    pd.Series
        Index is class name, values are image counts.
    """
    raise NotImplementedError


def compute_fractions(counts: pd.Series) -> pd.Series:
    """Compute each class's fraction of the total dataset.

    Parameters
    ----------
    counts : pd.Series
        Output of :func:`count_per_class`.

    Returns
    -------
    pd.Series
        Same index as ``counts``, values in [0, 1].
    """
    raise NotImplementedError


def apply_threshold(
    manifest: pd.DataFrame,
    min_instances: int = 100,
    min_fraction: float = 0.005,
    prune_rule: str = "either",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Drop under-represented classes from the manifest.

    Parameters
    ----------
    manifest : pd.DataFrame
        Must contain a ``class_name`` column.
    min_instances : int
        Absolute floor; classes with fewer images are dropped.
    min_fraction : float
        Fractional floor; classes below this share are dropped.
    prune_rule : str
        ``'either'``: drop if fails EITHER threshold.
        ``'both'``: drop only if fails BOTH thresholds.

    Returns
    -------
    tuple[pd.DataFrame, list[str], list[str]]
        ``(pruned_manifest, surviving_classes, dropped_classes)``
        where ``surviving_classes`` is sorted alphabetically.

    Raises
    ------
    ValueError
        If ``prune_rule`` is not ``'either'`` or ``'both'``.
    """
    raise NotImplementedError
