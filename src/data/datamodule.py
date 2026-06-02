"""PyTorch Lightning DataModule for the merged skin disease corpus."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.splits import load_splits
from src.training.augment import get_eval_transforms, get_train_transforms
from src.utils.paths import get_data_root

logger = logging.getLogger(__name__)


class SkinDiseaseDataset(Dataset[tuple[torch.Tensor, int]]):
    """Map-style dataset for one split of the skin disease corpus.

    Parameters
    ----------
    items : list[tuple[str, int]]
        List of ``(absolute_image_path, class_index)`` tuples.
    transform : callable or None
        Albumentations transform applied to each image.
    """

    def __init__(
        self,
        items: list[tuple[str, int]],
        transform: Optional[object] = None,
    ) -> None:
        self._items = items
        self._transform = transform

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path_str, label_idx = self._items[idx]
        img = np.array(Image.open(path_str).convert("RGB"))
        if self._transform is not None:
            img = self._transform(image=img)["image"]
        return img, label_idx


class SkinDiseaseDataModule(pl.LightningDataModule):
    """LightningDataModule wrapping the merged skin disease corpus.

    Parameters
    ----------
    data_root : str
        Base directory used to resolve relative filepaths stored in splits.csv.
        Defaults to ``get_data_root()``.
    class_mapping_path : str
        Path to ``src/data/class_mapping.json`` (label → int index).
    splits_path : str
        Path to ``data/splits.csv`` produced by :func:`src.data.splits.make_splits`.
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
        storage_backend: str = "local",
        storage: object = None,
        min_instances: int = 100,
        min_fraction: float = 0.005,
        prune_rule: str = "either",
    ) -> None:
        super().__init__()
        root = get_data_root()
        self._data_root = root  # filepaths in manifest are relative to get_data_root()
        self.class_mapping_path = Path(class_mapping_path)
        self.splits_path = Path(splits_path) if splits_path else root / "splits.csv"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        self._class_mapping: Optional[dict[str, int]] = None
        self._train_dataset: Optional[SkinDiseaseDataset] = None
        self._val_dataset: Optional[SkinDiseaseDataset] = None
        self._test_dataset: Optional[SkinDiseaseDataset] = None

    @property
    def num_classes(self) -> int:
        """Number of classes determined from the class mapping file."""
        if self._class_mapping is None:
            mapping = json.loads(self.class_mapping_path.read_text())
            self._class_mapping = mapping
        return len(self._class_mapping)

    def prepare_data(self) -> None:
        """Verify splits and class mapping files exist (rank-0 only)."""
        if not self.splits_path.exists():
            raise FileNotFoundError(
                f"Splits file missing: {self.splits_path}. "
                "Run src.data.splits.make_splits() first."
            )
        if not self.class_mapping_path.exists():
            self._build_class_mapping()

    def _build_class_mapping(self) -> None:
        """Derive label→index mapping from splits CSV and write class_mapping.json."""
        splits_df = load_splits(self.splits_path)
        labels = sorted(splits_df["label"].unique().tolist())
        mapping = {label: idx for idx, label in enumerate(labels)}
        self.class_mapping_path.parent.mkdir(parents=True, exist_ok=True)
        self.class_mapping_path.write_text(json.dumps(mapping, indent=2))
        logger.info(
            "Wrote class mapping (%d classes) to %s", len(mapping), self.class_mapping_path
        )
        self._class_mapping = mapping

    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate train/val/test datasets.

        Parameters
        ----------
        stage : str or None
            One of ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``.
        """
        if self._class_mapping is None:
            self._class_mapping = json.loads(self.class_mapping_path.read_text())

        splits_df = load_splits(self.splits_path)

        def _make_items(split_name: str) -> list[tuple[str, int]]:
            rows = splits_df[splits_df["split"] == split_name]
            items = []
            for _, row in rows.iterrows():
                abs_path = str(self._data_root / row["filepath"])
                label_idx = self._class_mapping[row["label"]]
                items.append((abs_path, label_idx))
            return items

        if stage in ("fit", None):
            self._train_dataset = SkinDiseaseDataset(
                _make_items("train"),
                transform=get_train_transforms(self.image_size),
            )
            self._val_dataset = SkinDiseaseDataset(
                _make_items("val"),
                transform=get_eval_transforms(self.image_size),
            )

        if stage in ("validate", None):
            if self._val_dataset is None:
                self._val_dataset = SkinDiseaseDataset(
                    _make_items("val"),
                    transform=get_eval_transforms(self.image_size),
                )

        if stage in ("test", "predict", None):
            self._test_dataset = SkinDiseaseDataset(
                _make_items("test"),
                transform=get_eval_transforms(self.image_size),
            )

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, int]]:
        """Return the training DataLoader with augmentation transforms."""
        return DataLoader(
            self._train_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, int]]:
        """Return the validation DataLoader with evaluation-only transforms."""
        return DataLoader(
            self._val_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, int]]:
        """Return the test DataLoader with evaluation-only transforms."""
        return DataLoader(
            self._test_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
