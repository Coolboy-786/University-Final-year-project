"""Otsu thresholding segmentation for full-body images."""

from __future__ import annotations

import cv2
import numpy as np


def segment_otsu(image: np.ndarray) -> np.ndarray:
    """Apply Otsu thresholding to isolate the skin lesion region.

    Converts the image to grayscale, computes the Otsu threshold, creates a
    binary mask, and returns the image with the background zeroed out.

    Parameters
    ----------
    image : np.ndarray
        HxWx3 uint8 RGB image.

    Returns
    -------
    np.ndarray
        HxWx3 uint8 RGB image with background masked to zero.

    Raises
    ------
    ValueError
        If the input is not a 3-channel uint8 image.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected HxWx3 uint8 RGB image")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.where(mask[:, :, np.newaxis] > 0, image, 0).astype(np.uint8)
