"""Identity (no-op) segmentation for skin-only images."""

from __future__ import annotations

import numpy as np


def segment_identity(image: np.ndarray) -> np.ndarray:
    """Return the image unchanged (identity transform).

    Used for ``skin_only`` images where the entire frame is already the
    region of interest and no masking is needed.

    Parameters
    ----------
    image : np.ndarray
        HxWx3 uint8 RGB image.

    Returns
    -------
    np.ndarray
        The same array, unmodified.
    """
    return image
