"""VGG19 classifier (Section 7.5 / thesis §10.3.1).

ImageNet-pretrained VGG19 with a softmax head. Used as a single-model
baseline and (via a separate instantiation) as the image-type router backbone.
Trained with Adam lr=1e-3 and StepLR(step_size=20, gamma=0.01) for 40 epochs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseClassifier


class VGG19Classifier(BaseClassifier):
    """VGG19 fine-tuned for skin disease classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    pretrained : bool
        Load ImageNet weights.
    lr : float
        Initial Adam learning rate.
    step_size : int
        StepLR epoch interval.
    gamma : float
        StepLR multiplicative factor.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        lr: float = 1e-3,
        step_size: int = 20,
        gamma: float = 0.01,
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
        """Adam + StepLR(step_size=20, gamma=0.01) (thesis §8.3)."""
        raise NotImplementedError
