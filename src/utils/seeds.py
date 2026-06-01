"""Reproducibility utilities (Section 8.4)."""

from __future__ import annotations

import os
import random

import numpy as np
import pytorch_lightning as pl
import torch

GLOBAL_SEED: int = 42


def set_all_seeds(seed: int = GLOBAL_SEED) -> None:
    """Set seeds for Python, NumPy, PyTorch, and ``PYTHONHASHSEED``.

    Also enables ``torch.use_deterministic_algorithms(True)`` where feasible
    and calls ``pl.seed_everything`` to cover Lightning's internal RNGs.

    Parameters
    ----------
    seed : int
        Seed value; defaults to the project-wide seed of 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass  # a small number of ops lack deterministic implementations
    pl.seed_everything(seed, workers=True)
