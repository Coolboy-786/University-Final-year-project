"""Reproducibility utilities (Section 8.4)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

GLOBAL_SEED: int = 42


def set_all_seeds(seed: int = GLOBAL_SEED) -> None:
    """Set seeds for Python, NumPy, PyTorch, and ``PYTHONHASHSEED``.

    Also enables ``torch.use_deterministic_algorithms(True)`` where feasible.

    Parameters
    ----------
    seed : int
        Seed value; defaults to the project-wide seed of 42.
    """
    raise NotImplementedError
