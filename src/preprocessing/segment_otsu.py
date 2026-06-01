"""Otsu thresholding segmentation for full-body images."""

from __future__ import annotations

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
    raise NotImplementedError
