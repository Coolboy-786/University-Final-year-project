"""Adaptive thresholding segmentation for body-part images."""

from __future__ import annotations

import cv2
import numpy as np


def segment_adaptive(
    image: np.ndarray,
    block_size: int = 11,
    c: int = 2,
) -> np.ndarray:
    """Apply adaptive Gaussian thresholding to isolate the lesion region.

    Parameters
    ----------
    image : np.ndarray
        HxWx3 uint8 RGB image.
    block_size : int
        Neighbourhood size for the adaptive threshold (must be odd).
    c : int
        Constant subtracted from the weighted mean.

    Returns
    -------
    np.ndarray
        HxWx3 uint8 RGB image with background masked to zero.

    Raises
    ------
    ValueError
        If ``block_size`` is even or the input is not a 3-channel uint8 image.
    """
    if block_size % 2 == 0:
        raise ValueError(f"block_size must be odd, got {block_size}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected HxWx3 uint8 RGB image")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )
    return np.where(mask[:, :, np.newaxis] > 0, image, 0).astype(np.uint8)
