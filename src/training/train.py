"""Entry point for a single training run.

Hydra composes the config; this module instantiates the DataModule, model,
callbacks, and Trainer, then calls ``trainer.fit``.
"""

from __future__ import annotations

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.utils.seeds import set_all_seeds


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> float:
    """Run one training experiment from a Hydra config.

    Parameters
    ----------
    cfg : DictConfig
        Composed Hydra config (data + model + preprocessing + trainer).

    Returns
    -------
    float
        Best validation macro-F1 achieved during training.
    """
    raise NotImplementedError


if __name__ == "__main__":
    train()
