"""Entry point for a single training run.

Hydra composes the config; this module instantiates the DataModule, model,
preprocessing pipeline, callbacks, and Trainer, then calls ``trainer.fit``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.training.callbacks import get_early_stopping, get_model_checkpoint
from src.utils.seeds import set_all_seeds

log = logging.getLogger(__name__)

# Keys present in model configs for training metadata but not constructor params
_MODEL_EXTRA_KEYS: frozenset[str] = frozenset({"optimizer", "scheduler", "trainer_overrides"})


def _resolve_num_classes(cfg: DictConfig) -> int:
    """Read num_classes from the class mapping JSON, or fall back to 23.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.

    Returns
    -------
    int
        Number of classes.
    """
    mapping_path = Path(cfg.data.class_mapping_path)
    if mapping_path.exists():
        with mapping_path.open(encoding="utf-8") as f:
            mapping = json.load(f)
        n: int = mapping.get("n_classes") or len(mapping.get("classes", {}))
        return n
    log.warning(
        "class_mapping.json not found at %s; defaulting num_classes=23. "
        "Run src.data.merge first.",
        mapping_path,
    )
    return 23


def _build_model(cfg: DictConfig, num_classes: int) -> pl.LightningModule:
    """Instantiate the model, injecting num_classes and stripping config-only keys.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.
    num_classes : int
        Number of output classes resolved from the class mapping.

    Returns
    -------
    pl.LightningModule
        Instantiated model.
    """
    model_dict: dict[str, Any] = OmegaConf.to_container(  # type: ignore[assignment]
        cfg.model, resolve=True, throw_on_missing=False
    )
    for key in _MODEL_EXTRA_KEYS:
        model_dict.pop(key, None)
    model_dict["num_classes"] = num_classes
    return hydra.utils.instantiate(OmegaConf.create(model_dict))  # type: ignore[return-value]


def _build_trainer_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """Merge base trainer config with any model-specific overrides.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.

    Returns
    -------
    dict[str, Any]
        Keyword arguments for ``pl.Trainer``.
    """
    kwargs: dict[str, Any] = OmegaConf.to_container(cfg.trainer, resolve=True)  # type: ignore[assignment]
    overrides = OmegaConf.select(cfg, "model.trainer_overrides")
    if overrides is not None:
        extra: dict[str, Any] = OmegaConf.to_container(overrides, resolve=True)  # type: ignore[assignment]
        kwargs.update(extra)
    return kwargs


def _build_logger(cfg: DictConfig) -> pl.loggers.Logger:
    """Build a W&B or TensorBoard logger from config.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.

    Returns
    -------
    pl.loggers.Logger
        Configured logger.
    """
    model_name = cfg.model._target_.rsplit(".", 1)[-1]
    preprocess_mode = OmegaConf.select(cfg, "preprocessing.mode") or "none"
    run_name = f"{model_name}__{preprocess_mode}"

    if cfg.logging.wandb.enabled:
        from pytorch_lightning.loggers import WandbLogger

        return WandbLogger(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity or None,
            name=run_name,
        )

    from pytorch_lightning.loggers import TensorBoardLogger

    return TensorBoardLogger(save_dir="logs", name=run_name)


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
    set_all_seeds(cfg.seed)

    num_classes = _resolve_num_classes(cfg)
    log.info("num_classes=%d", num_classes)

    # DataModule — constructed from cfg.data; uses get_data_root() internally
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data, _recursive_=False
    )

    # Model
    model = _build_model(cfg, num_classes)
    log.info("Model: %s", model.__class__.__name__)

    # Preprocessing pipeline (wired into DataModule by Person 1's implementation)
    preprocess_pipeline = hydra.utils.instantiate(cfg.preprocessing)
    log.info("Preprocessing mode: %s", OmegaConf.select(cfg, "preprocessing.mode"))

    # Attach pipeline to datamodule if it exposes the attribute
    if hasattr(datamodule, "preprocessing_pipeline"):
        datamodule.preprocessing_pipeline = preprocess_pipeline  # type: ignore[attr-defined]

    trainer_kwargs = _build_trainer_kwargs(cfg)
    logger = _build_logger(cfg)
    callbacks = [get_early_stopping(patience=5), get_model_checkpoint()]

    trainer = pl.Trainer(
        **trainer_kwargs,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule)

    best_f1: float = float(trainer.callback_metrics.get("val/f1", 0.0))
    log.info("Best val/f1: %.4f", best_f1)
    return best_f1


if __name__ == "__main__":
    train()
