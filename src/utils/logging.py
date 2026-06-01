"""Experiment logging helpers (W&B + TensorBoard fallback)."""

from __future__ import annotations

import logging
from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def get_logger(
    cfg: DictConfig,
    run_name: Optional[str] = None,
) -> Optional[pl.loggers.Logger]:
    """Instantiate a Lightning logger from Hydra config.

    Returns a W&B logger when ``cfg.logging.wandb.enabled`` is True,
    falls back to TensorBoard when ``cfg.logging.tensorboard.enabled`` is True,
    and returns None otherwise.

    Parameters
    ----------
    cfg : DictConfig
        Top-level Hydra config.
    run_name : str or None
        Optional human-readable run name; auto-generated if omitted.

    Returns
    -------
    pl.loggers.Logger or None
        Configured logger, or None to use Lightning's default.
    """
    raise NotImplementedError


def log_config_snapshot(
    logger_obj: Optional[pl.loggers.Logger],
    cfg: DictConfig,
) -> None:
    """Log a full config snapshot and git SHA to the experiment tracker.

    Parameters
    ----------
    logger_obj : pl.loggers.Logger or None
        Active logger; no-op if None.
    cfg : DictConfig
        Top-level Hydra config to serialise.
    """
    raise NotImplementedError
