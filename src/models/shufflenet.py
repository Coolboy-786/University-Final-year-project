"""ShuffleNetV1 from scratch (Section 7.4 / thesis §10.3.3).

Three stages of shuffle units with channel-shuffle operations. Trained from
random initialisation (no pretrained weights) with Adam for 20 epochs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseClassifier


class ChannelShuffle(nn.Module):
    """Channel shuffle operation for grouped convolutions.

    Parameters
    ----------
    groups : int
        Number of groups used in the preceding grouped convolution.
    """

    def __init__(self, groups: int) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        """Shuffle channels across groups.

        Parameters
        ----------
        x : Tensor
            Shape ``(N, C, H, W)``.

        Returns
        -------
        Tensor
            Same shape as input with channels shuffled.
        """
        raise NotImplementedError


class ShuffleUnit(nn.Module):
    """A single ShuffleNet unit (with or without stride).

    Parameters
    ----------
    in_channels : int
        Input channel count.
    out_channels : int
        Output channel count.
    groups : int
        Number of groups for grouped 1x1 convolutions.
    stride : int
        Stride for the depthwise 3x3 convolution; 2 for downsampling units.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class ShuffleNetClassifier(BaseClassifier):
    """ShuffleNetV1 for skin disease classification.

    Architecture: conv1 → maxpool → stage2 → stage3 → stage4 → gap → fc.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    groups : int
        Number of groups for grouped convolutions (3 by default).
    stage_channels : list[int]
        Output channels for each of the 4 stages (including conv1).
    stage_repeats : list[int]
        Number of shuffle units in stages 2, 3, and 4.
    lr : float
        Adam learning rate.
    """

    def __init__(
        self,
        num_classes: int,
        groups: int = 3,
        stage_channels: list[int] | None = None,
        stage_repeats: list[int] | None = None,
        lr: float = 1e-3,
    ) -> None:
        super().__init__(num_classes=num_classes)
        self.save_hyperparameters()
        if stage_channels is None:
            stage_channels = [24, 240, 480, 960]
        if stage_repeats is None:
            stage_repeats = [3, 7, 3]
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
        """Adam optimiser at default lr with no scheduler (thesis §8.3)."""
        raise NotImplementedError
