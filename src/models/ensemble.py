"""Average ensemble of MobileNetV2 and ShuffleNet (Section 7.6 / thesis §10.3.4).

Averages the softmax outputs of the two trained models. No trainable
parameters — evaluation only.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseClassifier
from src.models.mobilenet import MobileNetV2Classifier
from src.models.shufflenet import ShuffleNetClassifier


class AverageEnsemble(BaseClassifier):
    """Average ensemble over MobileNetV2 and ShuffleNet softmax outputs.

    This module has no trainable parameters. Both constituent models are
    loaded from checkpoints and frozen.

    Parameters
    ----------
    mobilenet_ckpt : Path
        Checkpoint path for the trained :class:`MobileNetV2Classifier`.
    shufflenet_ckpt : Path
        Checkpoint path for the trained :class:`ShuffleNetClassifier`.
    num_classes : int
        Number of output classes (must match both checkpoints).
    """

    def __init__(
        self,
        mobilenet_ckpt: Path,
        shufflenet_ckpt: Path,
        num_classes: int,
    ) -> None:
        super().__init__(num_classes=num_classes)
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Compute averaged softmax probabilities.

        Parameters
        ----------
        x : Tensor
            Shape ``(N, 3, 224, 224)``.

        Returns
        -------
        Tensor
            Averaged softmax probabilities, shape ``(N, num_classes)``.
            These are probabilities, not logits.
        """
        raise NotImplementedError

    def configure_optimizers(self) -> None:
        """Not applicable — ensemble has no trainable parameters."""
        return None
