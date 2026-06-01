"""Generate all paper figures (Section 3).

Produces:
  (a) Pipeline overview diagram
  (b) Router confusion matrix
  (c) Sample segmentation outputs across image types
  (d) Main experiment confusion matrix
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

FIGURES_DIR = Path("paper/figures")


def make_pipeline_diagram(output_path: Path) -> None:
    """Render the pipeline overview figure (Figure a).

    Parameters
    ----------
    output_path : Path
        Destination PNG path.
    """
    raise NotImplementedError


def make_router_confusion_matrix(
    predictions_csv: Path,
    output_path: Path,
) -> None:
    """Render the router confusion matrix (Figure b).

    Parameters
    ----------
    predictions_csv : Path
        CSV with ``true_type`` and ``pred_type`` columns from router evaluation.
    output_path : Path
        Destination PNG path.
    """
    raise NotImplementedError


def make_segmentation_samples(
    sample_images: list[Path],
    output_path: Path,
) -> None:
    """Render sample segmentation outputs for all three image types (Figure c).

    Parameters
    ----------
    sample_images : list[Path]
        Paths to representative images (one per image type).
    output_path : Path
        Destination PNG path.
    """
    raise NotImplementedError


def make_main_confusion_matrix(
    predictions_csv: Path,
    class_mapping_path: Path,
    output_path: Path,
) -> None:
    """Render the main experiment confusion matrix (Figure d).

    Parameters
    ----------
    predictions_csv : Path
        CSV with ``y_true`` and ``y_pred`` columns from the ensemble run.
    class_mapping_path : Path
        Path to ``src/data/class_mapping.json``.
    output_path : Path
        Destination PNG path.
    """
    raise NotImplementedError


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    make_pipeline_diagram(FIGURES_DIR / "pipeline.png")
    make_segmentation_samples([], FIGURES_DIR / "segmentation_samples.png")
