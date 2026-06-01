"""Base LightningModule shared by all classification models."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class BaseClassifier(pl.LightningModule):
    """Abstract base for all skin-disease classifiers.

    Subclasses must implement :meth:`forward` and
    :meth:`configure_optimizers`.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes, average="macro"),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits for a batch of images.

        Parameters
        ----------
        x : Tensor
            Batch of images, shape ``(N, 3, H, W)``.

        Returns
        -------
        Tensor
            Logits, shape ``(N, num_classes)``.
        """
        raise NotImplementedError

    def _shared_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Compute loss and predictions for a batch.

        Parameters
        ----------
        batch : tuple[Tensor, Tensor]
            ``(images, labels)`` where ``labels`` are integer class indices.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(loss, predictions, labels)``.
        """
        raise NotImplementedError

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        raise NotImplementedError

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        raise NotImplementedError

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        raise NotImplementedError

    def configure_optimizers(self) -> Any:
        """Return optimiser (and optional scheduler).

        Must be implemented by each subclass because learning rates and
        schedules differ across models (thesis §10.3).
        """
        raise NotImplementedError
