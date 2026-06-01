"""PyTorch Lightning DataModule for the merged skin disease corpus."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.paths import get_data_root

logger = logging.getLogger(__name__)


class SkinDiseaseDataset(Dataset[tuple[torch.Tensor, int]]):
    """Map-style dataset for one split of the skin disease corpus.

    Parameters
    ----------
    manifest : list[tuple[str, int]]
        List of ``(image_path, class_index)`` tuples.
    transform : callable or None
        Albumentations or torchvision transform applied to each image.
    """

    def __init__(
        self,
        manifest: list[tuple[str, int]],
        transform: Optional[object] = None,
    ) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        raise NotImplementedError


class SkinDiseaseDataModule(pl.LightningDataModule):
    """LightningDataModule wrapping the merged skin disease corpus.

    Parameters
    ----------
    data_root : str
        Root of ``data/processed/``.
    class_mapping_path : str
        Path to ``src/data/class_mapping.json``.
    splits_path : str
        Path to ``data/splits.csv``.
    batch_size : int
        DataLoader batch size.
    num_workers : int
        DataLoader worker count.
    pin_memory : bool
        Whether to pin DataLoader memory.
    image_size : int
        Expected spatial dimension (square).
    """

    def __init__(
        self,
        data_root: str = "",
        class_mapping_path: str = "src/data/class_mapping.json",
        splits_path: str = "",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 224,
        storage_backend: str = "local",  # passed from Hydra config; get_data_root() reads env
        storage: object = None,  # path profile map from config; resolution handled by paths.py
    ) -> None:
        super().__init__()
        root = get_data_root()
        self.data_root = Path(data_root) if data_root else root / "processed"
        self.class_mapping_path = Path(class_mapping_path)
        self.splits_path = Path(splits_path) if splits_path else root / "splits.csv"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        self._class_mapping: Optional[dict[str, int]] = None

    @property
    def num_classes(self) -> int:
        """Number of classes determined from the class mapping file."""
        raise NotImplementedError

    def prepare_data(self) -> None:
        """Verify that splits and class mapping files exist.

        Called only on rank-0 in distributed training.
        """
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate train/val/test datasets.

        Parameters
        ----------
        stage : str or None
            One of ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``.
        """
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, int]]:
        """Return the training DataLoader with augmentation transforms."""
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, int]]:
        """Return the validation DataLoader with evaluation-only transforms."""
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, int]]:
        """Return the test DataLoader with evaluation-only transforms."""
        raise NotImplementedError
