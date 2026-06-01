"""Shape and dtype tests for segmentation modules."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.segment_identity import segment_identity
from src.preprocessing.segment_otsu import segment_otsu
from src.preprocessing.segment_adaptive import segment_adaptive


@pytest.fixture()
def rgb_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)


def test_identity_returns_same_array(rgb_image: np.ndarray) -> None:
    result = segment_identity(rgb_image)
    assert result is rgb_image


def test_otsu_output_shape_and_dtype(rgb_image: np.ndarray) -> None:
    result = segment_otsu(rgb_image)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_adaptive_output_shape_and_dtype(rgb_image: np.ndarray) -> None:
    result = segment_adaptive(rgb_image)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_adaptive_raises_on_even_block_size(rgb_image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        segment_adaptive(rgb_image, block_size=10)
