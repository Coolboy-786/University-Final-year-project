"""DataModule that serves pre-cached feature tensors instead of images.

Used for CLIP linear-probe training: features are extracted once by
``scripts/cache_clip_features.py`` and loaded here as TensorDatasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class CLIPFeatureDataModule(pl.LightningDataModule):
    """LightningDataModule over pre-cached CLIP feature tensors.

    Parameters
    ----------
    features_dir : str
        Directory containing ``{split}_features.pt`` and
        ``{split}_labels.pt`` files written by
        ``scripts/cache_clip_features.py``.
    batch_size : int
        DataLoader batch size. Can be much larger than image batches
        since features are small float tensors (768-dim).
    num_workers : int
        DataLoader worker count.
    """

    def __init__(
        self,
        features_dir: str,
        batch_size: int = 512,
        num_workers: int = 2,
    ) -> None:
        super().__init__()
        self._dir = Path(features_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train: Optional[TensorDataset] = None
        self._val:   Optional[TensorDataset] = None
        self._test:  Optional[TensorDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self._train = TensorDataset(
                torch.load(self._dir / "train_features.pt", weights_only=True),
                torch.load(self._dir / "train_labels.pt",   weights_only=True),
            )
            self._val = TensorDataset(
                torch.load(self._dir / "val_features.pt", weights_only=True),
                torch.load(self._dir / "val_labels.pt",   weights_only=True),
            )
        if stage in ("test", "predict", None):
            self._test = TensorDataset(
                torch.load(self._dir / "test_features.pt", weights_only=True),
                torch.load(self._dir / "test_labels.pt",   weights_only=True),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
