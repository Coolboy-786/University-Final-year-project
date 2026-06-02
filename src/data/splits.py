"""Stratified 70/15/15 train/val/test splits, seed-locked at 42.

Splits are written to ``data/splits.csv`` once and never regenerated unless
explicitly requested — every downstream run reads the same file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.paths import get_data_root

logger = logging.getLogger(__name__)

TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15
SEED: int = 42


def make_splits(
    manifest: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = SEED,
) -> pd.DataFrame:
    """Create stratified splits and return the manifest with a ``split`` column.

    Parameters
    ----------
    manifest : pd.DataFrame
        Must contain ``filepath`` and ``label`` columns.
    train_frac : float
        Fraction of data for training.
    val_frac : float
        Fraction of data for validation; test gets the remainder.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Input manifest with an added ``split`` column whose values are
        ``'train'``, ``'val'``, or ``'test'``.
    """
    df = manifest.copy()

    train_idx, tmp_idx = train_test_split(
        df.index,
        train_size=train_frac,
        stratify=df["label"],
        random_state=seed,
    )

    tmp_df = df.loc[tmp_idx]
    val_of_tmp = val_frac / (1.0 - train_frac)
    val_idx, test_idx = train_test_split(
        tmp_df.index,
        train_size=val_of_tmp,
        stratify=tmp_df["label"],
        random_state=seed,
    )

    df["split"] = ""
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    logger.info(
        "Split counts — train: %d  val: %d  test: %d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    return df


def save_splits(splits_df: pd.DataFrame, output_path: Path) -> None:
    """Write the splits DataFrame to a CSV file.

    Parameters
    ----------
    splits_df : pd.DataFrame
        Output of :func:`make_splits`.
    output_path : Path
        Destination path (e.g. ``data/splits.csv``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(output_path, index=False)
    logger.info("Splits written to %s", output_path)


def load_splits(splits_path: Path) -> pd.DataFrame:
    """Load a previously saved splits CSV.

    Parameters
    ----------
    splits_path : Path
        Path to the splits CSV written by :func:`save_splits`.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least ``filepath``, ``label``, and ``split`` columns.

    Raises
    ------
    FileNotFoundError
        If the splits file does not exist (run ``make_splits`` first).
    """
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_path}. "
            "Generate it with make_splits() first."
        )
    return pd.read_csv(splits_path)


def verify_no_leakage(splits_df: pd.DataFrame) -> None:
    """Assert that no image path appears in more than one split.

    Parameters
    ----------
    splits_df : pd.DataFrame
        Output of :func:`make_splits` or :func:`load_splits`.

    Raises
    ------
    AssertionError
        If any path is found in multiple splits.
    """
    counts = splits_df.groupby("filepath")["split"].nunique()
    leakers = counts[counts > 1]
    assert leakers.empty, (
        f"{len(leakers)} filepath(s) appear in multiple splits: "
        f"{leakers.index.tolist()[:5]}"
    )
