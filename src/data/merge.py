"""Build the merged corpus from raw downloaded sources.

Reads images from ``data/raw/``, applies the class name mapping, de-duplicates
by perceptual hash, writes a flat ``data/processed/`` directory tree, and
emits ``src/data/class_mapping.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import get_data_root

logger = logging.getLogger(__name__)

# Candidate classes from thesis §5 (sorted alphabetically for stable indexing)
CANDIDATE_CLASSES: list[str] = [
    "acne",
    "actinic-keratosis",
    "basal-cell-carcinoma",
    "chickenpox",
    "cowpox",
    "dermatitis",
    "eczema",
    "erythema-multiforme",
    "granuloma",
    "herpes",
    "hidradenitis-suppurativa",
    "lupus",
    "measles",
    "melanocytic-nevi",
    "melanoma",
    "molluscum",
    "monkeypox",
    "normal-skin",
    "psoriasis",
    "rosacea",
    "tinea",
    "vasculitis",
    "warts",
]


def build_manifest(raw_root: Path) -> pd.DataFrame:
    """Scan raw download directories and build an image manifest.

    Parameters
    ----------
    raw_root : Path
        Root of raw downloads (``data/raw/``).

    Returns
    -------
    pd.DataFrame
        Columns: ``path``, ``source``, ``raw_label``, ``class_name``.
        ``class_name`` is ``None`` for images that could not be mapped.
    """
    raise NotImplementedError


def deduplicate(manifest: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate images using perceptual hashing.

    Parameters
    ----------
    manifest : pd.DataFrame
        Output of :func:`build_manifest`.

    Returns
    -------
    pd.DataFrame
        Manifest with duplicate rows removed. Keeps one representative per
        perceptual-hash cluster (deterministic: smallest ``path`` lexicographically).
    """
    raise NotImplementedError


def copy_to_processed(
    manifest: pd.DataFrame,
    processed_root: Path,
    image_size: int = 224,
) -> pd.DataFrame:
    """Copy and resize images into the processed directory tree.

    Parameters
    ----------
    manifest : pd.DataFrame
        De-duplicated manifest.
    processed_root : Path
        Destination root (``data/processed/``).
    image_size : int
        Target edge length for square resize.

    Returns
    -------
    pd.DataFrame
        Manifest updated with new ``processed_path`` column.
    """
    raise NotImplementedError


def write_class_mapping(
    surviving_classes: list[str],
    thresholds: dict[str, Any],
    output_path: Path,
) -> dict[str, int]:
    """Write the stable class-name → index mapping to JSON.

    Parameters
    ----------
    surviving_classes : list[str]
        Alphabetically sorted list of classes that passed pruning.
    thresholds : dict[str, Any]
        Threshold parameters used for pruning (stored in the file for provenance).
    output_path : Path
        Destination path (``src/data/class_mapping.json``).

    Returns
    -------
    dict[str, int]
        The written mapping.
    """
    raise NotImplementedError


def run_merge(
    raw_root: Path,
    processed_root: Path,
    mapping_path: Path,
    min_instances: int = 100,
    min_fraction: float = 0.005,
    prune_rule: str = "either",
    image_size: int = 224,
) -> pd.DataFrame:
    """Full merge pipeline: scan → deduplicate → copy → prune → write mapping.

    Parameters
    ----------
    raw_root : Path
        Raw downloads root.
    processed_root : Path
        Processed output root.
    mapping_path : Path
        Destination for ``class_mapping.json``.
    min_instances : int
        Absolute minimum images per class.
    min_fraction : float
        Minimum fraction of total dataset per class.
    prune_rule : str
        ``'either'`` (drop if fails either threshold) or ``'both'``.
    image_size : int
        Target edge length for resizing.

    Returns
    -------
    pd.DataFrame
        Final manifest of all kept images.
    """
    raise NotImplementedError
