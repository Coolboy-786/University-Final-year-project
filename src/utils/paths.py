"""Resolve data root based on STORAGE_BACKEND environment variable."""

from __future__ import annotations

import os
from pathlib import Path

_BACKENDS: dict[str, Path] = {
    "local": Path("data"),
    "colab": Path("/content/drive/MyDrive/FYP_Thesis (1)/data"),
}


def get_data_root() -> Path:
    """Return the data root directory for the active storage backend.

    Reads ``STORAGE_BACKEND`` from the environment (default: ``local``).
    Set ``STORAGE_BACKEND=colab`` when running on Google Colab with Drive mounted.
    """
    backend = os.environ.get("STORAGE_BACKEND", "local")
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown STORAGE_BACKEND {backend!r}. Valid choices: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[backend]
