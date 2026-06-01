"""Lightning callbacks shared across training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.utils.paths import get_data_root


def get_early_stopping(patience: int = 5, monitor: str = "val/f1") -> EarlyStopping:
    """Return an EarlyStopping callback watching val macro-F1.

    Stops training when ``monitor`` does not improve for ``patience`` consecutive
    epochs (thesis §8.3).

    Parameters
    ----------
    patience : int
        Epochs with no improvement before stopping.
    monitor : str
        Metric name to watch; must be logged by the model.

    Returns
    -------
    EarlyStopping
        Configured callback.
    """
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode="max",
        verbose=True,
    )


def get_model_checkpoint(
    dirpath: Optional[Path] = None,
    monitor: str = "val/f1",
    filename: str = "best-{epoch:02d}-{val/f1:.4f}",
) -> ModelCheckpoint:
    """Return a ModelCheckpoint callback saving the best val macro-F1 model.

    Checkpoints are written to ``<data_root>/checkpoints/`` by default so they
    persist on Drive when ``STORAGE_BACKEND=colab``.

    Parameters
    ----------
    dirpath : Path or None
        Directory for saved checkpoints. Defaults to
        ``get_data_root() / "checkpoints"``.
    monitor : str
        Metric name to watch.
    filename : str
        Checkpoint filename template (Lightning format string).

    Returns
    -------
    ModelCheckpoint
        Configured callback.
    """
    if dirpath is None:
        dirpath = get_data_root() / "checkpoints"
    return ModelCheckpoint(
        dirpath=str(dirpath),
        filename=filename,
        monitor=monitor,
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
