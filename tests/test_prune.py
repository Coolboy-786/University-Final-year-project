"""Tests for threshold-based class pruning."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.prune import apply_threshold, compute_fractions, count_per_class


def _make_manifest(class_counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    for cls, n in class_counts.items():
        rows.extend([{"class_name": cls, "processed_path": f"data/{cls}/{i}.jpg"} for i in range(n)])
    return pd.DataFrame(rows)


def test_count_per_class() -> None:
    manifest = _make_manifest({"acne": 50, "melanoma": 200})
    counts = count_per_class(manifest)
    assert counts["acne"] == 50
    assert counts["melanoma"] == 200


def test_apply_threshold_drops_small_class() -> None:
    manifest = _make_manifest({"acne": 50, "melanoma": 500})
    pruned, surviving, dropped = apply_threshold(manifest, min_instances=100)
    assert "melanoma" in surviving
    assert "acne" in dropped


def test_apply_threshold_invalid_rule() -> None:
    manifest = _make_manifest({"acne": 200})
    with pytest.raises(ValueError):
        apply_threshold(manifest, prune_rule="invalid")


def test_verify_no_leakage_placeholder() -> None:
    # Placeholder — full implementation in splits.py
    pass
