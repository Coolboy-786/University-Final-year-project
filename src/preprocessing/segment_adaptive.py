"""Adaptive thresholding segmentation for body-part images."""

from __future__ import annotations

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
    raise NotImplementedError
