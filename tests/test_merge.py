"""Tests for src/data/sources.py and src/data/merge.py."""

from __future__ import annotations

import pandas as pd

from src.data.merge import one_hot_row_to_label
from src.data.sources import CANONICAL_LABELS, DX_TO_CANONICAL


class TestCanonicalLabels:
    def test_has_seven_classes(self) -> None:
        assert len(CANONICAL_LABELS) == 7

    def test_no_duplicates(self) -> None:
        assert len(CANONICAL_LABELS) == len(set(CANONICAL_LABELS))


class TestDxToCanonical:
    def test_every_dx_code_maps(self) -> None:
        for dx in ("mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"):
            assert dx in DX_TO_CANONICAL, f"missing dx code: {dx}"
            assert DX_TO_CANONICAL[dx]

    def test_all_values_in_canonical(self) -> None:
        canonical_set = set(CANONICAL_LABELS)
        for dx, label in DX_TO_CANONICAL.items():
            assert label in canonical_set, (
                f"{dx!r} maps to {label!r} which is not in CANONICAL_LABELS"
            )

    def test_covers_canonical_labels(self) -> None:
        assert set(DX_TO_CANONICAL.values()) == set(CANONICAL_LABELS)


class TestOneHotToLabel:
    def test_mel_row(self) -> None:
        row = pd.Series({"MEL": 1.0, "NV": 0, "BCC": 0, "AKIEC": 0, "BKL": 0, "DF": 0, "VASC": 0})
        assert one_hot_row_to_label(row) == "melanoma"

    def test_vasc_row(self) -> None:
        row = pd.Series({"MEL": 0, "NV": 0, "BCC": 0, "AKIEC": 0, "BKL": 0, "DF": 0, "VASC": 1.0})
        assert one_hot_row_to_label(row) == "vascular_lesion"

    def test_bkl_row(self) -> None:
        row = pd.Series({"MEL": 0, "NV": 0, "BCC": 0, "AKIEC": 0, "BKL": 1, "DF": 0, "VASC": 0})
        assert one_hot_row_to_label(row) == "benign_keratosis"

    def test_all_zeros_returns_none(self) -> None:
        row = pd.Series({"MEL": 0, "NV": 0, "BCC": 0, "AKIEC": 0, "BKL": 0, "DF": 0, "VASC": 0})
        assert one_hot_row_to_label(row) is None

    def test_every_one_hot_column_resolves(self) -> None:
        for col in ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"):
            row = pd.Series({c: (1.0 if c == col else 0.0) for c in
                             ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")})
            label = one_hot_row_to_label(row)
            assert label == DX_TO_CANONICAL[col.lower()]
