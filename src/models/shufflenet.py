"""ShuffleNetV1 from scratch (Section 7.4 / thesis §10.3.3).

Three stages of ShuffleUnits with grouped 1×1 convolutions and channel-shuffle
operations between them. Trained from random initialisation with Adam for 20
epochs. Default channel configuration matches ShuffleNet-g3 (thesis §10.3.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseClassifier


class ChannelShuffle(nn.Module):
    """Channel shuffle for grouped convolution outputs.

    Reshapes ``(N, C, H, W)`` into ``(N, g, C/g, H, W)``, transposes the
    group and channel axes, then flattens back — effectively shuffling channel
    information across groups so that subsequent grouped convolutions can mix
    features from all groups.

    Parameters
    ----------
    groups : int
        Number of groups used in the immediately preceding grouped convolution.
    """

    def __init__(self, groups: int) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        """Shuffle channels across groups.

        Parameters
        ----------
        x : Tensor
            Shape ``(N, C, H, W)``.  ``C`` must be divisible by ``groups``.

        Returns
        -------
        Tensor
            Same shape as input with channels permuted across groups.
        """
        n, c, h, w = x.shape
        g = self.groups
        return (
            x.view(n, g, c // g, h, w)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(n, c, h, w)
        )


class ShuffleUnit(nn.Module):
    """A single ShuffleNetV1 unit.

    When ``stride=1`` the unit applies an additive identity residual (spatial
    resolution and channel count are preserved). When ``stride=2`` it
    downsamples via a 3×3 average-pool shortcut and concatenates the shortcut
    with the main branch output, increasing the channel count from
    ``in_channels`` to ``out_channels``.

    Parameters
    ----------
    in_channels : int
        Input feature-map channels.
    out_channels : int
        Output feature-map channels.
    groups : int
        Number of groups for grouped 1×1 pointwise convolutions.
    stride : int
        Stride for the depthwise 3×3 convolution.  Use ``2`` for the first
        unit of each stage to halve spatial resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.stride = stride
        bottleneck = max(out_channels // 4, 1)

        if stride == 2:
            # Concat shortcut: branch produces (out - in) channels,
            # shortcut keeps in_channels via avg-pool → concat = out_channels
            branch_out = out_channels - in_channels
            self.shortcut: nn.Module = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            # Additive shortcut: branch output must equal in_channels
            branch_out = out_channels
            self.shortcut = nn.Identity()

        # The first grouped 1×1 conv should not use groups when the input comes
        # directly from stage-1 (24 channels) — the channel count is already
        # small and groups=1 is used in the original paper for this case.
        first_groups = 1 if in_channels == 24 else groups

        self.branch = nn.Sequential(
            # Pointwise grouped conv: compress to bottleneck
            nn.Conv2d(in_channels, bottleneck, 1, groups=first_groups, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            # Channel shuffle before depthwise conv
            ChannelShuffle(first_groups),
            # Depthwise 3×3 conv (groups == channels → fully depthwise)
            nn.Conv2d(
                bottleneck, bottleneck, 3,
                stride=stride, padding=1, groups=bottleneck, bias=False,
            ),
            nn.BatchNorm2d(bottleneck),
            # Pointwise grouped conv: expand to branch_out
            nn.Conv2d(bottleneck, branch_out, 1, groups=groups, bias=False),
            nn.BatchNorm2d(branch_out),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the shuffle unit.

        Parameters
        ----------
        x : Tensor
            Input feature map, shape ``(N, in_channels, H, W)``.

        Returns
        -------
        Tensor
            Output feature map, shape ``(N, out_channels, H', W')`` where
            ``H' = H // stride``.
        """
        if self.stride == 2:
            out = torch.cat([self.branch(x), self.shortcut(x)], dim=1)
        else:
            out = self.branch(x) + self.shortcut(x)
        return self.relu(out)


class ShuffleNetClassifier(BaseClassifier):
    """ShuffleNetV1 trained from scratch for skin disease classification.

    Architecture (default g=3 configuration, thesis §10.3.3):

    * **Stage 1** — Conv2d(3→24, 3×3, s=2) + BN + ReLU + MaxPool(3×3, s=2)
    * **Stage 2** — 3 ShuffleUnits (first stride=2), output 240 ch
    * **Stage 3** — 7 ShuffleUnits (first stride=2), output 480 ch
    * **Stage 4** — 3 ShuffleUnits (first stride=2), output 960 ch
    * **Head** — GlobalAvgPool → Flatten → Linear(960, num_classes)

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    groups : int
        Number of groups for grouped convolutions (default 3, thesis §10.3.3).
    stage_channels : list[int] or None
        Output channels for ``[conv1, stage2, stage3, stage4]`` (4 values).
        Defaults to ``[24, 240, 480, 960]``.
    stage_repeats : list[int] or None
        Total shuffle units per stage including the stride-2 unit (3 values).
        Defaults to ``[3, 7, 3]``.
    lr : float
        Adam learning rate (default 1e-3, thesis §10.3.3).
    """

    def __init__(
        self,
        num_classes: int,
        groups: int = 3,
        stage_channels: list[int] | None = None,
        stage_repeats: list[int] | None = None,
        lr: float = 1e-3,
    ) -> None:
        super().__init__(num_classes=num_classes, lr=lr)
        self.save_hyperparameters()

        if stage_channels is None:
            stage_channels = [24, 240, 480, 960]
        if stage_repeats is None:
            stage_repeats = [3, 7, 3]

        if len(stage_channels) != 4:
            raise ValueError("stage_channels must have exactly 4 values")
        if len(stage_repeats) != 3:
            raise ValueError("stage_repeats must have exactly 3 values")

        # Stage 1: standard convolutional entry block
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, stage_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stage_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Stages 2–4: repeated ShuffleUnits
        self.stage2 = self._make_stage(
            stage_channels[0], stage_channels[1], stage_repeats[0], groups
        )
        self.stage3 = self._make_stage(
            stage_channels[1], stage_channels[2], stage_repeats[1], groups
        )
        self.stage4 = self._make_stage(
            stage_channels[2], stage_channels[3], stage_repeats[2], groups
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_channels[3], num_classes)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        repeats: int,
        groups: int,
    ) -> nn.Sequential:
        """Build one stage of shuffle units.

        Parameters
        ----------
        in_channels : int
            Input channels to the stage.
        out_channels : int
            Output channels after the stride-2 downsampling unit.
        repeats : int
            Total units in the stage (first has stride=2, rest stride=1).
        groups : int
            Grouped convolution groups.

        Returns
        -------
        nn.Sequential
            Sequential container of :class:`ShuffleUnit` modules.
        """
        layers: list[nn.Module] = [
            ShuffleUnit(in_channels, out_channels, groups=groups, stride=2)
        ]
        for _ in range(repeats - 1):
            layers.append(
                ShuffleUnit(out_channels, out_channels, groups=groups, stride=1)
            )
        return nn.Sequential(*layers)

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
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.fc(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Adam optimiser for training from scratch (thesis §10.3.3).

        Returns
        -------
        torch.optim.Optimizer
            Adam optimiser over all parameters.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
