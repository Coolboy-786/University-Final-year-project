"""Tests for data splitting — no-leakage and stratification checks."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.splits import make_splits, verify_no_leakage


def _make_manifest(n_per_class: int = 50) -> pd.DataFrame:
    classes = ["acne", "melanoma", "psoriasis"]
    rows = []
    for cls in classes:
        for i in range(n_per_class):
            rows.append({"processed_path": f"data/{cls}/{i}.jpg", "class_name": cls})
    return pd.DataFrame(rows)


def test_splits_cover_all_samples() -> None:
    manifest = _make_manifest()
    split_df = make_splits(manifest)
    assert len(split_df) == len(manifest)


def test_splits_have_three_partitions() -> None:
    manifest = _make_manifest()
    split_df = make_splits(manifest)
    assert set(split_df["split"].unique()) == {"train", "val", "test"}


def test_no_leakage() -> None:
    manifest = _make_manifest()
    split_df = make_splits(manifest)
    verify_no_leakage(split_df)  # raises AssertionError if paths overlap


def test_approximate_fractions() -> None:
    manifest = _make_manifest(n_per_class=200)
    split_df = make_splits(manifest)
    total = len(split_df)
    train_frac = (split_df["split"] == "train").sum() / total
    assert 0.65 <= train_frac <= 0.75
