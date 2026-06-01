"""MobileNetV2 classifier (Section 7.3 / thesis §10.3.2).

ImageNet-pretrained MobileNetV2 from torchvision with the last 20 parameter
tensors unfrozen and the classifier head replaced. Optimised with Adam lr=1e-4
for 15 epochs (thesis §10.3.2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from src.models.base import BaseClassifier


class MobileNetV2Classifier(BaseClassifier):
    """MobileNetV2 fine-tuned for skin disease classification.

    Loads ImageNet-1K weights, freezes the full backbone, then unfreezes the
    last ``unfreeze_last_n`` parameter tensors (including the new head) for
    fine-tuning. Input size must be 224×224.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    pretrained : bool
        If ``True``, load ``MobileNet_V2_Weights.IMAGENET1K_V1`` weights.
    unfreeze_last_n : int
        Number of parameter tensors from the end to unfreeze (thesis §10.3.2
        uses 20).
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
        super().__init__(num_classes=num_classes, lr=lr)
        self.save_hyperparameters()

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v2(weights=weights)

        # Freeze the entire network first
        for param in backbone.parameters():
            param.requires_grad = False

        # Replace the 1000-class head with a task-specific linear layer
        last_channel: int = backbone.last_channel  # 1280
        backbone.classifier[1] = nn.Linear(last_channel, num_classes)

        # Unfreeze the last `unfreeze_last_n` parameter tensors (includes new head)
        all_params = list(backbone.parameters())
        for param in all_params[-unfreeze_last_n:]:
            param.requires_grad = True

        self.backbone = backbone

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Adam optimiser over trainable parameters at lr=1e-4 (thesis §10.3.2).

        Returns
        -------
        torch.optim.Optimizer
            Adam optimiser restricted to parameters with ``requires_grad=True``.
        """
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=self.lr)
