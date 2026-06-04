"""Train CLIP linear probe on pre-cached features.

Run cache_clip_features.py first, then:
    !python scripts/train_clip_probe.py

Trains only the linear head (CLIPClassifier.forward detects 2-D feature
input and bypasses the frozen encoder). Saves a full checkpoint containing
both encoder weights and trained head so evaluate.py can load it unchanged.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.utils.class_weight import compute_class_weight

from src.data.feature_datamodule import CLIPFeatureDataModule
from src.models.clip_classifier import CLIPClassifier
from src.training.callbacks import get_early_stopping, get_model_checkpoint
from src.utils.seeds import set_all_seeds

log = logging.getLogger(__name__)

_FEATURES_DIR = "/content/drive/MyDrive/FYP_Thesis/data/clip_features"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CLIP linear probe on cached features."
    )
    parser.add_argument("--features-dir", default=_FEATURES_DIR)
    parser.add_argument("--num-classes",  type=int,   default=7)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--max-epochs",   type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=512)
    args = parser.parse_args()

    set_all_seeds(42)

    dm = CLIPFeatureDataModule(
        features_dir=args.features_dir,
        batch_size=args.batch_size,
    )
    dm.setup("fit")

    model = CLIPClassifier(
        num_classes=args.num_classes,
        freeze_encoder=True,
        lr=args.lr,
    )

    # Compute balanced class weights from cached training labels
    train_labels = dm._train.tensors[1].numpy()  # type: ignore[union-attr]
    weights = compute_class_weight(
        "balanced", classes=np.arange(args.num_classes), y=train_labels
    )
    model.class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"class weights: {model.class_weights.numpy().round(3).tolist()}")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[
            get_early_stopping(patience=10),
            get_model_checkpoint(filename="clip-probe-{epoch:02d}-{val/f1:.4f}"),
        ],
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    trainer.fit(model, dm)

    best_f1 = float(trainer.callback_metrics.get("val/f1", 0.0))
    log.info("Best val/f1: %.4f", best_f1)
    print(f"Best val/f1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
