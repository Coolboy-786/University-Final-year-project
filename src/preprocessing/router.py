"""VGG19-based image-type classifier (Section 7.1).

Predicts whether an input image is ``full_body``, ``body_part``, or
``skin_only`` and is used by the routing pipeline to dispatch images to the
appropriate segmentation method.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import VGG19_Weights, vgg19

from src.models.base import BaseClassifier

IMAGE_SIZE: int = 224
ROUTER_CLASSES: list[str] = ["full_body", "body_part", "skin_only"]

# ImageNet normalisation constants
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class ImageType(str, Enum):
    """Categorical output of the router."""

    FULL_BODY = "full_body"
    BODY_PART = "body_part"
    SKIN_ONLY = "skin_only"


class RouterModel(BaseClassifier):
    """VGG19 fine-tuned for 3-class image-type prediction.

    Extends :class:`~src.models.base.BaseClassifier` so it inherits the
    standard training/validation loop and torchmetrics logging.

    Parameters
    ----------
    pretrained : bool
        Load ImageNet weights for the VGG19 backbone.
    lr : float
        Learning rate for the Adam optimiser.
    """

    def __init__(self, pretrained: bool = True, lr: float = 1e-4) -> None:
        super().__init__(num_classes=len(ROUTER_CLASSES), lr=lr)
        self.save_hyperparameters()

        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = vgg19(weights=weights)
        self.backbone.classifier[6] = nn.Linear(4096, len(ROUTER_CLASSES))

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x : Tensor
            Batch of images, shape ``(N, 3, 224, 224)``, normalised to
            ImageNet statistics.

        Returns
        -------
        Tensor
            Raw logits, shape ``(N, 3)``.
        """
        return self.backbone(x)  # type: ignore[no-any-return]


class ImageTypeRouter:
    """Inference wrapper around a trained :class:`RouterModel`.

    Handles preprocessing (resize to 224×224, ImageNet normalisation) and
    converts logit argmax to an :class:`ImageType` enum value.

    Parameters
    ----------
    checkpoint_path : Path
        Path to a Lightning ``.ckpt`` file produced by training
        :class:`RouterModel`.
    device : str
        Torch device string (``'cpu'``, ``'cuda'``, etc.).
    """

    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        self._device = torch.device(device)
        model = cast(
            RouterModel,
            RouterModel.load_from_checkpoint(
                str(checkpoint_path), map_location=self._device
            ),
        )
        model.eval()
        self._model = model

        # ImageNet normalisation tensors — precomputed for efficiency
        self._mean = torch.tensor(_MEAN, device=self._device).view(3, 1, 1)
        self._std = torch.tensor(_STD, device=self._device).view(3, 1, 1)

    def _preprocess(self, image: np.ndarray) -> Tensor:
        """Convert a single HxWx3 uint8 RGB array to a normalised (1,3,H,W) tensor."""
        t = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        t = (t - self._mean) / self._std
        return t.unsqueeze(0).to(self._device)

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
        with torch.no_grad():
            logits = self._model(self._preprocess(image))
            idx = int(logits.argmax(dim=1).item())
        return ImageType(ROUTER_CLASSES[idx])

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
        with torch.no_grad():
            batch = torch.cat([self._preprocess(img) for img in images], dim=0)
            indices = self._model(batch).argmax(dim=1).tolist()
        return [ImageType(ROUTER_CLASSES[int(i)]) for i in indices]
