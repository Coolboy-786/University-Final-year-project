"""Lightning callbacks shared across training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.utils.paths import get_data_root


def get_early_stopping(patience: int = 5, monitor: str = "val/f1") -> EarlyStopping:
    """Return an EarlyStopping callback watching val macro-F1.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping (thesis §8.3).
    monitor : str
        Metric to monitor.

    Returns
    -------
    EarlyStopping
        Configured callback.
    """
    raise NotImplementedError


def get_model_checkpoint(
    dirpath: Optional[Path] = None,
    monitor: str = "val/f1",
    filename: str = "best-{epoch:02d}-{val/f1:.4f}",
) -> ModelCheckpoint:
    """Return a ModelCheckpoint callback saving the best val macro-F1 model.

    Parameters
    ----------
    dirpath : Path or None
        Directory for saved checkpoints. Defaults to ``<data_root>/checkpoints/``
        so weights land on Drive when ``STORAGE_BACKEND=colab``.
    monitor : str
        Metric to monitor.
    filename : str
        Checkpoint filename template.

    Returns
    -------
    ModelCheckpoint
        Configured callback.
    """
    if dirpath is None:
        dirpath = get_data_root() / "checkpoints"
    raise NotImplementedError
