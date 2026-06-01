"""Average ensemble of MobileNetV2 and ShuffleNet (Section 7.6 / thesis §10.3.4).

Averages the softmax outputs of the two trained models. No trainable
parameters — evaluation only.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseClassifier
from src.models.mobilenet import MobileNetV2Classifier
from src.models.shufflenet import ShuffleNetClassifier


class AverageEnsemble(BaseClassifier):
    """Average ensemble over MobileNetV2 and ShuffleNet softmax outputs.

    Both constituent models are loaded from checkpoints and fully frozen. The
    module itself has no trainable parameters and is used for evaluation only.

    Parameters
    ----------
    mobilenet_ckpt : Path
        Path to the checkpoint for the trained :class:`MobileNetV2Classifier`.
    shufflenet_ckpt : Path
        Path to the checkpoint for the trained :class:`ShuffleNetClassifier`.
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

        self.mobilenet: MobileNetV2Classifier = cast(
            MobileNetV2Classifier,
            MobileNetV2Classifier.load_from_checkpoint(str(mobilenet_ckpt)),
        )
        self.shufflenet: ShuffleNetClassifier = cast(
            ShuffleNetClassifier,
            ShuffleNetClassifier.load_from_checkpoint(str(shufflenet_ckpt)),
        )

        for model in (self.mobilenet, self.shufflenet):
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

    def forward(self, x: Tensor) -> Tensor:
        """Return element-wise average of both models' softmax probabilities.

        Parameters
        ----------
        x : Tensor
            Shape ``(N, 3, 224, 224)``.

        Returns
        -------
        Tensor
            Averaged softmax probabilities, shape ``(N, num_classes)``.
            Values are probabilities (sum to 1), not raw logits.
        """
        p_mobile = F.softmax(self.mobilenet(x), dim=1)
        p_shuffle = F.softmax(self.shufflenet(x), dim=1)
        return (p_mobile + p_shuffle) * 0.5

    def configure_optimizers(self) -> None:  # type: ignore[override]
        """Not applicable — ensemble has no trainable parameters."""
        return None
