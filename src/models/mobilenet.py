"""MobileNetV2 classifier (Section 7.3 / thesis §10.3.2).

ImageNet-pretrained MobileNetV2 from torchvision with the last 20 layers
unfrozen and a GAP → Dense(num_classes) head. Optimised with Adam lr=1e-4
for 15 epochs.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.models.base import BaseClassifier


class MobileNetV2Classifier(BaseClassifier):
    """MobileNetV2 fine-tuned for skin disease classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    pretrained : bool
        Load ImageNet weights.
    unfreeze_last_n : int
        Number of layers from the end to unfreeze for fine-tuning.
        Matches the thesis value of 20.
    lr : float
        Adam learning rate.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        unfreeze_last_n: int = 20,
        lr: float = 1e-4,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.save_hyperparameters()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits.

        Parameters
        ----------
        x : Tensor
            Shape ``(N, 3, 224, 224)``.

        Returns
        -------
        Tensor
            Shape ``(N, num_classes)``.
        """
        raise NotImplementedError

    def configure_optimizers(self) -> Any:
        """Adam optimiser at ``lr=1e-4`` with no scheduler (thesis §8.3)."""
        raise NotImplementedError
