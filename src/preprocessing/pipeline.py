"""Conditional segmentation dispatcher (Section 7.2).

Selects and applies the appropriate segmentation method based on the
configured ``mode``. The ``mode`` parameter is what the router ablation
switches over.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np

from src.preprocessing.router import ImageType, ImageTypeRouter
from src.preprocessing.segment_adaptive import segment_adaptive
from src.preprocessing.segment_identity import segment_identity
from src.preprocessing.segment_otsu import segment_otsu

PreprocessMode = Literal["none", "otsu", "adaptive", "routed"]


class PreprocessingPipeline:
    """Dispatches an image to a segmentation method.

    mode='none'     -> identity (no segmentation)
    mode='otsu'     -> Otsu thresholding for all images
    mode='adaptive' -> adaptive thresholding for all images
    mode='routed'   -> use router to pick per image

    Parameters
    ----------
    mode : PreprocessMode
        Segmentation strategy.
    router_ckpt : Path or None
        Path to the router checkpoint; required when ``mode='routed'``.
    device : str
        Torch device for the router (only used when ``mode='routed'``).

    Raises
    ------
    ValueError
        If ``mode='routed'`` and no ``router_ckpt`` is provided, or if
        ``mode`` is not one of the recognised values.
    """

    def __init__(
        self,
        mode: PreprocessMode = "none",
        router_ckpt: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        raise NotImplementedError

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single image.

        Parameters
        ----------
        image : np.ndarray
            HxWx3 uint8 RGB image.

        Returns
        -------
        np.ndarray
            Preprocessed HxWx3 uint8 RGB image.
        """
        raise NotImplementedError

    def _route(self, image: np.ndarray) -> np.ndarray:
        """Route ``image`` to the correct segmentation method via the router.

        Parameters
        ----------
        image : np.ndarray
            HxWx3 uint8 RGB image.

        Returns
        -------
        np.ndarray
            Segmented HxWx3 uint8 RGB image.
        """
        raise NotImplementedError
