"""Augmentation pipelines matching thesis §10.1.

Training:  rotation ±10°, horizontal flip, width/height shift 0.1,
           zoom 0.1, shear 0.1 — all via albumentations.
Eval/test: resize to 224×224 + ImageNet normalisation only.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Return the training augmentation pipeline (thesis §10.1).

    Parameters
    ----------
    image_size : int
        Target spatial dimension for the square crop/resize.

    Returns
    -------
    A.Compose
        Albumentations composition ready for use in a Dataset.
    """
    raise NotImplementedError


def get_eval_transforms(image_size: int = 224) -> A.Compose:
    """Return the evaluation/test transform (resize + normalise only).

    Parameters
    ----------
    image_size : int
        Target spatial dimension.

    Returns
    -------
    A.Compose
        Albumentations composition.
    """
    raise NotImplementedError
