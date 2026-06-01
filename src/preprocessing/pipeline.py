"""Conditional segmentation dispatcher (Section 7.2).

Selects and applies the appropriate segmentation method based on the
configured ``mode``. The ``mode`` parameter is what the router ablation
switches over.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

from src.preprocessing.segment_adaptive import segment_adaptive
from src.preprocessing.segment_identity import segment_identity
from src.preprocessing.segment_otsu import segment_otsu

if TYPE_CHECKING:
    # Deferred so torch/torchvision are only imported when mode='routed'
    from src.preprocessing.router import ImageTypeRouter

PreprocessMode = Literal["none", "otsu", "adaptive", "routed"]

_VALID_MODES: frozenset[str] = frozenset({"none", "otsu", "adaptive", "routed"})


class PreprocessingPipeline:
    """Dispatches an image to the appropriate segmentation method.

    +-----------+------------------------------------------------------------+
    | mode      | behaviour                                                  |
    +===========+============================================================+
    | ``none``  | identity — no segmentation                                 |
    +-----------+------------------------------------------------------------+
    | ``otsu``  | Otsu thresholding for every image                          |
    +-----------+------------------------------------------------------------+
    | ``adaptive`` | adaptive Gaussian thresholding for every image          |
    +-----------+------------------------------------------------------------+
    | ``routed``| router predicts image type; dispatches per-image           |
    +-----------+------------------------------------------------------------+

    Parameters
    ----------
    mode : PreprocessMode
        Segmentation strategy.
    router_ckpt : Path or None
        Path to the router checkpoint. Required when ``mode='routed'``.
    device : str
        Torch device for the router (only used when ``mode='routed'``).

    Raises
    ------
    ValueError
        If ``mode`` is unrecognised, or if ``mode='routed'`` is requested
        without a ``router_ckpt``.
    """

    def __init__(
        self,
        mode: PreprocessMode = "none",
        router_ckpt: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown preprocessing mode {mode!r}. "
                f"Valid choices: {sorted(_VALID_MODES)}"
            )
        if mode == "routed" and router_ckpt is None:
            raise ValueError("router_ckpt must be provided when mode='routed'")

        self._mode: PreprocessMode = mode
        self._router: Optional[ImageTypeRouter] = None

        if mode == "routed":
            # Deferred import so torch is not required for non-routed modes
            from src.preprocessing.router import ImageTypeRouter as _Router

            self._router = _Router(router_ckpt, device=device)  # type: ignore[arg-type]

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
        if self._mode == "otsu":
            return segment_otsu(image)
        if self._mode == "adaptive":
            return segment_adaptive(image)
        if self._mode == "routed":
            return self._route(image)
        return segment_identity(image)  # "none"

    def _route(self, image: np.ndarray) -> np.ndarray:
        """Route ``image`` to the correct segmentation method via the router.

        Mapping: FULL_BODY → Otsu, BODY_PART → adaptive, SKIN_ONLY → identity.

        Parameters
        ----------
        image : np.ndarray
            HxWx3 uint8 RGB image.

        Returns
        -------
        np.ndarray
            Segmented HxWx3 uint8 RGB image.
        """
        from src.preprocessing.router import ImageType

        assert self._router is not None
        image_type = self._router.predict(image)
        if image_type == ImageType.FULL_BODY:
            return segment_otsu(image)
        if image_type == ImageType.BODY_PART:
            return segment_adaptive(image)
        return segment_identity(image)  # SKIN_ONLY
