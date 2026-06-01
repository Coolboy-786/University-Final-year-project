"""VGG19 classifier (Section 7.5 / thesis §10.3.1).

ImageNet-pretrained VGG19 with a task-specific linear head. Used as a
single-model baseline and, via a separate instantiation, as the image-type
router backbone. Trained with Adam lr=1e-3 and StepLR(step_size=20, gamma=0.01)
for 40 epochs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import VGG19_Weights, vgg19

from src.models.base import BaseClassifier


class VGG19Classifier(BaseClassifier):
    """VGG19 fine-tuned for skin disease classification.

    All layers are trainable from the start. The final classifier layer
    (``classifier[6]``) is replaced with a ``Linear(4096, num_classes)`` head.
    Input size must be 224×224.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    pretrained : bool
        If ``True``, load ``VGG19_Weights.IMAGENET1K_V1`` weights.
    lr : float
        Initial Adam learning rate (thesis §10.3.1 uses 1e-3).
    step_size : int
        StepLR epoch interval (thesis §10.3.1 uses 20).
    gamma : float
        StepLR multiplicative decay factor (thesis §10.3.1 uses 0.01).
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        lr: float = 1e-3,
        step_size: int = 20,
        gamma: float = 0.01,
    ) -> None:
        super().__init__(num_classes=num_classes, lr=lr)
        self.save_hyperparameters()
        self._step_size = step_size
        self._gamma = gamma

        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = vgg19(weights=weights)
        # Replace the 1000-class output layer with a task-specific head
        self.backbone.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits.

        Parameters
        ----------
        x : Tensor
            Shape ``(N, 3, 224, 224)``.

        Returns
        -------
        Tensor
            Logits, shape ``(N, num_classes)``.
        """
        return self.backbone(x)  # type: ignore[no-any-return]

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore[override]
        """Adam + StepLR(step_size=20, gamma=0.01) per thesis §10.3.1.

        Returns
        -------
        dict[str, Any]
            Lightning optimiser configuration dict with ``optimizer`` and
            ``lr_scheduler`` keys.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self._step_size, gamma=self._gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
