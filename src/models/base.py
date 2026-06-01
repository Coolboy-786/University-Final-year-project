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

    Subclasses must implement :meth:`forward`. :meth:`configure_optimizers`
    provides a default Adam optimiser; subclasses may override for custom
    schedules.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    lr : float
        Learning rate for the default Adam optimiser.
    """

    def __init__(self, num_classes: int, lr: float = 1e-3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes, average="macro"),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            }
        )
        self.train_metrics: MetricCollection = metrics.clone(prefix="train/")
        self.val_metrics: MetricCollection = metrics.clone(prefix="val/")
        self.test_metrics: MetricCollection = metrics.clone(prefix="test/")

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
        """Compute cross-entropy loss and hard predictions for one batch.

        Parameters
        ----------
        batch : tuple[Tensor, Tensor]
            ``(images, labels)`` pair delivered by the dataloader.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(loss, predictions, labels)`` — all scalar/1-D tensors.
        """
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        return loss, preds, labels

    def training_step(  # type: ignore[override]
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Forward pass + loss + metric logging for one training batch.

        Parameters
        ----------
        batch : tuple[Tensor, Tensor]
            ``(images, labels)`` mini-batch.
        batch_idx : int
            Index of the current batch within the epoch.

        Returns
        -------
        Tensor
            Scalar cross-entropy loss used for back-propagation.
        """
        loss, preds, labels = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics(preds, labels), on_step=False, on_epoch=True)
        return loss

    def validation_step(  # type: ignore[override]
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        """Forward pass + loss + metric logging for one validation batch.

        Parameters
        ----------
        batch : tuple[Tensor, Tensor]
            ``(images, labels)`` mini-batch.
        batch_idx : int
            Index of the current batch within the epoch.
        """
        loss, preds, labels = self._shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log_dict(self.val_metrics(preds, labels), sync_dist=True)

    def test_step(  # type: ignore[override]
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        """Forward pass + loss + metric logging for one test batch.

        Parameters
        ----------
        batch : tuple[Tensor, Tensor]
            ``(images, labels)`` mini-batch.
        batch_idx : int
            Index of the current batch within the epoch.
        """
        loss, preds, labels = self._shared_step(batch)
        self.log("test/loss", loss, sync_dist=True)
        self.log_dict(self.test_metrics(preds, labels), sync_dist=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return Adam optimiser with ``self.lr``.

        Returns
        -------
        torch.optim.Optimizer
            Adam optimiser over all model parameters.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
