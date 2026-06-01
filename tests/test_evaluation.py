"""Tests for evaluation metrics, confusion matrix, and report tables."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_aggregate_metrics, compute_per_class_metrics
from src.evaluation.confusion import compute_confusion_matrix
from src.evaluation.report import (
    build_ablation_table,
    build_main_results_table,
    build_per_class_table,
    export_latex,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CLASSES = ["acne", "melanoma", "psoriasis"]
_N = 90  # 30 per class


@pytest.fixture()
def perfect_preds() -> tuple[np.ndarray, np.ndarray]:
    """y_true == y_pred for all samples."""
    y = np.repeat([0, 1, 2], _N // 3)
    return y, y.copy()


@pytest.fixture()
def noisy_preds() -> tuple[np.ndarray, np.ndarray]:
    """y_pred with some errors."""
    rng = np.random.default_rng(0)
    y_true = np.repeat([0, 1, 2], _N // 3)
    y_pred = y_true.copy()
    noise_idx = rng.choice(len(y_true), size=10, replace=False)
    y_pred[noise_idx] = rng.integers(0, 3, size=10)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# compute_per_class_metrics
# ---------------------------------------------------------------------------


def test_per_class_shape(perfect_preds: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = perfect_preds
    df = compute_per_class_metrics(y_true, y_pred, _CLASSES)
    assert df.shape == (len(_CLASSES), 5)
    assert list(df.columns) == ["class", "precision", "recall", "f1", "support"]


def test_per_class_perfect_scores(perfect_preds: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = perfect_preds
    df = compute_per_class_metrics(y_true, y_pred, _CLASSES)
    assert (df["precision"] == 1.0).all()
    assert (df["recall"] == 1.0).all()
    assert (df["f1"] == 1.0).all()


def test_per_class_support_sums_to_n(perfect_preds: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = perfect_preds
    df = compute_per_class_metrics(y_true, y_pred, _CLASSES)
    assert df["support"].sum() == len(y_true)


def test_per_class_class_names_match(perfect_preds: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = perfect_preds
    df = compute_per_class_metrics(y_true, y_pred, _CLASSES)
    assert list(df["class"]) == _CLASSES


# ---------------------------------------------------------------------------
# compute_aggregate_metrics
# ---------------------------------------------------------------------------


def test_aggregate_keys(perfect_preds: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = perfect_preds
    result = compute_aggregate_metrics(y_true, y_pred)
    expected_keys = {
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1",
    }
    assert set(result.keys()) == expected_keys


def test_aggregate_perfect(perfect_preds: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = perfect_preds
    result = compute_aggregate_metrics(y_true, y_pred)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["weighted_f1"] == pytest.approx(1.0)


def test_aggregate_noisy_accuracy_below_1(
    noisy_preds: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred = noisy_preds
    result = compute_aggregate_metrics(y_true, y_pred)
    assert result["accuracy"] < 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0


# ---------------------------------------------------------------------------
# compute_confusion_matrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_shape(perfect_preds: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = perfect_preds
    cm = compute_confusion_matrix(y_true, y_pred, _CLASSES)
    assert cm.shape == (len(_CLASSES), len(_CLASSES))


def test_confusion_matrix_labels(perfect_preds: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = perfect_preds
    cm = compute_confusion_matrix(y_true, y_pred, _CLASSES)
    assert list(cm.index) == _CLASSES
    assert list(cm.columns) == _CLASSES


def test_confusion_matrix_perfect_is_diagonal(
    perfect_preds: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred = perfect_preds
    cm = compute_confusion_matrix(y_true, y_pred, _CLASSES)
    off_diag = cm.to_numpy() - np.diag(np.diag(cm.to_numpy()))
    assert off_diag.sum() == 0


# ---------------------------------------------------------------------------
# report tables
# ---------------------------------------------------------------------------

_RESULTS = [
    {"model": "MobileNetV2", "preprocessing": "none", "accuracy": 0.85, "macro_f1": 0.83, "weighted_f1": 0.84},
    {"model": "ShuffleNet",  "preprocessing": "routed", "accuracy": 0.80, "macro_f1": 0.78, "weighted_f1": 0.79},
    {"model": "Ensemble",    "preprocessing": "routed", "accuracy": 0.90, "macro_f1": 0.89, "weighted_f1": 0.89},
]


def test_main_results_table_shape() -> None:
    df = build_main_results_table(_RESULTS)
    assert df.shape == (3, 5)


def test_main_results_table_sorted_by_f1() -> None:
    df = build_main_results_table(_RESULTS)
    f1_vals = df["Macro F1"].tolist()
    assert f1_vals == sorted(f1_vals, reverse=True)


def test_ablation_table_shape() -> None:
    df = build_ablation_table(_RESULTS)
    assert df.shape == (3, 5)


def test_per_class_table_sorted(perfect_preds: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = perfect_preds
    raw = compute_per_class_metrics(y_true, y_pred, ["psoriasis", "acne", "melanoma"])
    df = build_per_class_table(raw)
    assert list(df["Class"]) == sorted(["psoriasis", "acne", "melanoma"])


def test_export_latex_writes_file(perfect_preds: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = perfect_preds
    raw = compute_per_class_metrics(y_true, y_pred, _CLASSES)
    df = build_per_class_table(raw)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "table.tex"
        export_latex(df, out, caption="Test", label="tab:test")
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "tabular" in content
        assert "Test" in content
