"""Canonical label space and per-dataset label mappings."""

from __future__ import annotations

from dataclasses import dataclass


CANONICAL_LABELS: list[str] = [
    "melanoma",
    "melanocytic_nevus",
    "basal_cell_carcinoma",
    "actinic_keratosis",
    "benign_keratosis",
    "dermatofibroma",
    "vascular_lesion",
]


DX_TO_CANONICAL: dict[str, str] = {
    "mel": "melanoma",
    "nv": "melanocytic_nevus",
    "bcc": "basal_cell_carcinoma",
    "akiec": "actinic_keratosis",
    "bkl": "benign_keratosis",
    "df": "dermatofibroma",
    "vasc": "vascular_lesion",
}


@dataclass(frozen=True)
class DatasetSource:
    """Registered dataset and its native-label → canonical-label mapping."""

    name: str
    label_mapping: dict[str, str]


DATASETS: dict[str, DatasetSource] = {
    "HAM10000": DatasetSource(name="HAM10000", label_mapping=DX_TO_CANONICAL),
    "ISIC2018_test": DatasetSource(name="ISIC2018_test", label_mapping=DX_TO_CANONICAL),
}
