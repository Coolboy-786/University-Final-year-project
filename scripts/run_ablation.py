"""Router on/off ablation script (Section 8.2).

Runs experiments 5–8 from the ablation grid:
  5. Ensemble + routed
  6. Ensemble + otsu only
  7. Ensemble + adaptive only
  8. Ensemble + none

Collects results and writes the ablation table to ``paper/figures/``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ABLATION_RUNS: list[dict[str, str]] = [
    {"preprocessing": "routed",        "label": "Routed (proposed)"},
    {"preprocessing": "otsu_only",     "label": "Otsu only"},
    {"preprocessing": "adaptive_only", "label": "Adaptive only"},
    {"preprocessing": "none",          "label": "No segmentation"},
]


def run_ablation(output_dir: Path) -> None:
    """Execute all ablation runs and collect results.

    Parameters
    ----------
    output_dir : Path
        Directory where per-run output directories are created.
    """
    raise NotImplementedError


if __name__ == "__main__":
    run_ablation(Path("outputs/ablation"))
