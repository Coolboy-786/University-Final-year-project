"""CLIP ViT-B/32 classifier (thesis §10.3.4 / replacement for VGG19).

Uses the OpenAI CLIP vision encoder (ViT-B/32) as a frozen feature extractor
with a trainable linear classification head. Input images arrive ImageNet-
normalised from the shared DataModule; this module re-normalises them to CLIP's
own statistics before passing to the encoder.

Trained with Adam lr=1e-4 for up to 20 epochs with early stopping.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from transformers import CLIPVisionModel

from src.models.base import BaseClassifier

# CLIP ViT-B/32 was trained with these channel statistics
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# DataModule applies ImageNet normalisation before handing batches to the model
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class CLIPClassifier(BaseClassifier):
    """CLIP ViT-B/32 visual encoder + linear head for skin disease classification.

    The encoder is fully frozen; only the linear head is trained. Re-normalises
    inputs from ImageNet statistics to CLIP statistics internally so the shared
    DataModule requires no changes.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    model_name : str
        HuggingFace model identifier for the CLIP vision encoder.
    freeze_encoder : bool
        If ``True`` (default), freeze all encoder parameters and train only
        the linear head.
    lr : float
        Adam learning rate for the trainable parameters.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze_encoder: bool = True,
        lr: float = 1e-4,
    ) -> None:
        super().__init__(num_classes=num_classes, lr=lr)
        self.save_hyperparameters()

        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        hidden_size: int = self.encoder.config.hidden_size  # 768 for ViT-B/32

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.head = nn.Linear(hidden_size, num_classes)

        # Buffers for ImageNet → CLIP re-normalisation (moved to device automatically)
        self.register_buffer(
            "_im_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_im_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_cl_mean", torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_cl_std", torch.tensor(_CLIP_STD).view(1, 3, 1, 1)
        )

    def _renormalize(self, x: Tensor) -> Tensor:
        """Convert ImageNet-normalised tensor to CLIP normalisation."""
        x = x * self._im_std + self._im_mean  # type: ignore[operator]
        return (x - self._cl_mean) / self._cl_std  # type: ignore[operator]

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits.

        Parameters
        ----------
        x : Tensor
            ImageNet-normalised images, shape ``(N, 3, 224, 224)``.

        Returns
        -------
        Tensor
            Logits, shape ``(N, num_classes)``.
        """
        x = self._renormalize(x)
        outputs = self.encoder(pixel_values=x)
        pooled: Tensor = outputs.pooler_output  # (N, hidden_size)
        return self.head(pooled)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Adam over trainable parameters (head only when encoder is frozen).

        Returns
        -------
        torch.optim.Optimizer
            Adam optimiser restricted to ``requires_grad=True`` parameters.
        """
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable, lr=self.lr)
