"""Augmentation pipelines matching thesis §10.1.

Training:  rotation ±10°, horizontal flip, width/height shift 0.1,
           zoom 0.1, shear 0.1 — all via albumentations.
Eval/test: resize to 224×224 + ImageNet normalisation only.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet channel statistics
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Return the training augmentation pipeline (thesis §10.1).

    Applies, in order:
    * Resize to ``image_size × image_size``
    * Horizontal flip (p=0.5)
    * Rotation ±10° (p=0.5)
    * Shift (0.1) + zoom/scale (0.1) via ShiftScaleRotate (p=0.5)
    * Shear ±6° ≈ 0.1 rad via Affine (p=0.5)
    * ImageNet normalisation
    * HWC→CHW tensor conversion

    Parameters
    ----------
    image_size : int
        Target spatial dimension for the square resize.

    Returns
    -------
    A.Compose
        Albumentations composition ready for use in a Dataset ``__getitem__``.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=0, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=0,
                border_mode=0,
                p=0.5,
            ),
            A.Affine(shear=(-6, 6), p=0.5),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> A.Compose:
    """Return the evaluation/test transform (resize + normalise only).

    No geometric or colour augmentation — deterministic and invertible up to
    the normalisation step.

    Parameters
    ----------
    image_size : int
        Target spatial dimension.

    Returns
    -------
    A.Compose
        Albumentations composition.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )
