"""VGG19 classifier (Section 7.5 / thesis §10.3.1).

ImageNet-pretrained VGG19 with a task-specific linear head. Used as a
single-model baseline and, via a separate instantiation, as the image-type
router backbone. Trained with Adam lr=1e-3 and StepLR(step_size=20, gamma=0.01)
for 40 epochs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
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
    use_class_weights : bool
        If ``True``, compute balanced class weights from the training split
        at fit-start and use them in the cross-entropy loss. Addresses
        HAM10000 class imbalance (melanocytic_nevus dominates at ~67%).
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        lr: float = 1e-3,
        step_size: int = 20,
        gamma: float = 0.01,
        use_class_weights: bool = True,
    ) -> None:
        super().__init__(num_classes=num_classes, lr=lr)
        self.save_hyperparameters()
        self._step_size = step_size
        self._gamma = gamma
        self.use_class_weights = use_class_weights

        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = vgg19(weights=weights)
        # Replace the 1000-class output layer with a task-specific head
        self.backbone.classifier[6] = nn.Linear(4096, num_classes)

        # Uniform weights by default; overwritten in on_fit_start with balanced weights
        self.register_buffer(
            "class_weights", torch.ones(num_classes, dtype=torch.float)
        )

    def on_fit_start(self) -> None:
        """Compute balanced class weights from the training split labels."""
        if not self.use_class_weights:
            return
        dm = getattr(self.trainer, "datamodule", None)
        if dm is None or getattr(dm, "_train_dataset", None) is None:
            return
        labels = np.array([item[1] for item in dm._train_dataset._items])
        classes = np.arange(self.num_classes)
        w = compute_class_weight("balanced", classes=classes, y=labels)
        self.class_weights = torch.tensor(w, dtype=torch.float, device=self.device)

    def _shared_step(
        self, batch: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Cross-entropy with balanced class weighting."""
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        preds = logits.argmax(dim=1)
        return loss, preds, labels

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
