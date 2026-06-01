"""Shape, dtype, and behaviour tests for segmentation modules and pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.segment_adaptive import segment_adaptive
from src.preprocessing.segment_identity import segment_identity
from src.preprocessing.segment_otsu import segment_otsu


@pytest.fixture()
def rgb_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# segment_identity
# ---------------------------------------------------------------------------


def test_identity_returns_same_array(rgb_image: np.ndarray) -> None:
    result = segment_identity(rgb_image)
    assert result is rgb_image


def test_identity_preserves_dtype(rgb_image: np.ndarray) -> None:
    assert segment_identity(rgb_image).dtype == np.uint8


# ---------------------------------------------------------------------------
# segment_otsu
# ---------------------------------------------------------------------------


def test_otsu_output_shape_and_dtype(rgb_image: np.ndarray) -> None:
    result = segment_otsu(rgb_image)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_otsu_raises_on_invalid_input() -> None:
    bad = np.zeros((224, 224), dtype=np.uint8)  # 2-D, not 3-channel
    with pytest.raises(ValueError):
        segment_otsu(bad)


def test_otsu_output_is_subset_of_input(rgb_image: np.ndarray) -> None:
    result = segment_otsu(rgb_image)
    # Every output pixel must equal the input pixel or be zero
    assert np.all((result == rgb_image) | (result == 0))


# ---------------------------------------------------------------------------
# segment_adaptive
# ---------------------------------------------------------------------------


def test_adaptive_output_shape_and_dtype(rgb_image: np.ndarray) -> None:
    result = segment_adaptive(rgb_image)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_adaptive_raises_on_even_block_size(rgb_image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        segment_adaptive(rgb_image, block_size=10)


def test_adaptive_raises_on_invalid_input() -> None:
    bad = np.zeros((224, 224), dtype=np.uint8)
    with pytest.raises(ValueError):
        segment_adaptive(bad)


def test_adaptive_output_is_subset_of_input(rgb_image: np.ndarray) -> None:
    result = segment_adaptive(rgb_image)
    assert np.all((result == rgb_image) | (result == 0))


# ---------------------------------------------------------------------------
# PreprocessingPipeline
# ---------------------------------------------------------------------------


def test_pipeline_none_returns_identical_array(rgb_image: np.ndarray) -> None:
    pipe = PreprocessingPipeline(mode="none")
    result = pipe(rgb_image)
    assert result is rgb_image


def test_pipeline_otsu_valid_output(rgb_image: np.ndarray) -> None:
    pipe = PreprocessingPipeline(mode="otsu")
    result = pipe(rgb_image)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_pipeline_adaptive_valid_output(rgb_image: np.ndarray) -> None:
    pipe = PreprocessingPipeline(mode="adaptive")
    result = pipe(rgb_image)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_pipeline_routed_raises_without_ckpt() -> None:
    with pytest.raises(ValueError, match="router_ckpt"):
        PreprocessingPipeline(mode="routed")


def test_pipeline_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        PreprocessingPipeline(mode="unknown")  # type: ignore[arg-type]
