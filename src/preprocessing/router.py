"""VGG19-based image-type classifier (Section 7.1).

Predicts whether an input image is ``full_body``, ``body_part``, or
``skin_only`` and is used by the routing pipeline to dispatch images to the
appropriate segmentation method.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

IMAGE_SIZE: int = 224
ROUTER_CLASSES: list[str] = ["full_body", "body_part", "skin_only"]


class ImageType(str, Enum):
    """Categorical output of the router."""

    FULL_BODY = "full_body"
    BODY_PART = "body_part"
    SKIN_ONLY = "skin_only"


class RouterModel(pl.LightningModule):
    """VGG19 fine-tuned for 3-class image-type prediction.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet weights for the VGG19 backbone.
    lr : float
        Learning rate for the Adam optimiser.
    """

    def __init__(self, pretrained: bool = True, lr: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x : Tensor
            Batch of images, shape ``(N, 3, 224, 224)``, normalised to ImageNet stats.

        Returns
        -------
        Tensor
            Raw logits, shape ``(N, 3)``.
        """
        raise NotImplementedError

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        raise NotImplementedError

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        raise NotImplementedError

    def configure_optimizers(self) -> torch.optim.Optimizer:
        raise NotImplementedError


class ImageTypeRouter:
    """Inference wrapper around a trained :class:`RouterModel`.

    Parameters
    ----------
    checkpoint_path : Path
        Path to a Lightning ``.ckpt`` file.
    device : str
        Torch device string (``'cpu'``, ``'cuda'``, etc.).
    """

    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        raise NotImplementedError

    def predict(self, image: np.ndarray) -> ImageType:
        """Classify a single RGB image.

        Parameters
        ----------
        image : np.ndarray
            HxWx3 uint8 array in RGB order.

        Returns
        -------
        ImageType
            Predicted image type.
        """
        raise NotImplementedError

    def predict_batch(self, images: list[np.ndarray]) -> list[ImageType]:
        """Classify a batch of RGB images.

        Parameters
        ----------
        images : list[np.ndarray]
            List of HxWx3 uint8 arrays in RGB order.

        Returns
        -------
        list[ImageType]
            Predicted image types, one per input image.
        """
        raise NotImplementedError
