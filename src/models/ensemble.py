"""Average ensemble of MobileNetV2, ShuffleNet, and VGG19 (thesis §10.3.4).

Averages the softmax outputs of all three trained models. No trainable
parameters — evaluation only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseClassifier
from src.models.clip_classifier import CLIPClassifier
from src.models.mobilenet import MobileNetV2Classifier
from src.models.shufflenet import ShuffleNetClassifier
from src.models.vgg19 import VGG19Classifier


class AverageEnsemble(BaseClassifier):
    """Average ensemble over MobileNetV2, ShuffleNet, VGG19, and/or CLIP softmax outputs.

    All constituent models are loaded from checkpoints and fully frozen. The
    module itself has no trainable parameters and is used for evaluation only.

    Parameters
    ----------
    mobilenet_ckpt : Path
        Path to the trained :class:`MobileNetV2Classifier` checkpoint.
    shufflenet_ckpt : Path
        Path to the trained :class:`ShuffleNetClassifier` checkpoint.
    vgg19_ckpt : Path or None
        Path to the trained :class:`VGG19Classifier` checkpoint.
        When None the ensemble skips VGG19.
    clip_ckpt : Path or None
        Path to the trained :class:`CLIPClassifier` checkpoint.
        When None the ensemble skips CLIP.
    num_classes : int
        Number of output classes (must match all checkpoints).
    """

    def __init__(
        self,
        mobilenet_ckpt: Path,
        shufflenet_ckpt: Path,
        num_classes: int,
        vgg19_ckpt: Optional[Path] = None,
        clip_ckpt: Optional[Path] = None,
    ) -> None:
        super().__init__(num_classes=num_classes)

        self.mobilenet: MobileNetV2Classifier = cast(
            MobileNetV2Classifier,
            MobileNetV2Classifier.load_from_checkpoint(
                str(mobilenet_ckpt), map_location="cpu", weights_only=False
            ),
        )
        self.shufflenet: ShuffleNetClassifier = cast(
            ShuffleNetClassifier,
            ShuffleNetClassifier.load_from_checkpoint(
                str(shufflenet_ckpt), map_location="cpu", weights_only=False
            ),
        )
        self.vgg19: Optional[VGG19Classifier] = None
        if vgg19_ckpt is not None:
            self.vgg19 = cast(
                VGG19Classifier,
                VGG19Classifier.load_from_checkpoint(
                    str(vgg19_ckpt), map_location="cpu", weights_only=False
                ),
            )
        self.clip: Optional[CLIPClassifier] = None
        if clip_ckpt is not None:
            self.clip = cast(
                CLIPClassifier,
                CLIPClassifier.load_from_checkpoint(
                    str(clip_ckpt), map_location="cpu", weights_only=False
                ),
            )

        members = [self.mobilenet, self.shufflenet]
        for opt in (self.vgg19, self.clip):
            if opt is not None:
                members.append(opt)
        for model in members:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

    def forward(self, x: Tensor) -> Tensor:
        """Return element-wise average of all models' softmax probabilities.

        Parameters
        ----------
        x : Tensor
            Shape ``(N, 3, 224, 224)``.

        Returns
        -------
        Tensor
            Averaged softmax probabilities, shape ``(N, num_classes)``.
        """
        probs = [
            F.softmax(self.mobilenet(x), dim=1),
            F.softmax(self.shufflenet(x), dim=1),
        ]
        if self.vgg19 is not None:
            probs.append(F.softmax(self.vgg19(x), dim=1))
        if self.clip is not None:
            probs.append(F.softmax(self.clip(x), dim=1))
        n = len(probs)
        return sum(probs) / n  # type: ignore[return-value]

    def configure_optimizers(self) -> None:  # type: ignore[override]
        """Not applicable — ensemble has no trainable parameters."""
        return None
