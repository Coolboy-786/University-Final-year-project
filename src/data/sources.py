"""Registry of public dataset sources with native-label to canonical-class mappings."""

from __future__ import annotations

from dataclasses import dataclass


CANONICAL_CLASSES: list[str] = [
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
    "psoriasis",
    "rosacea",
    "tinea",
    "vasculitis",
    "warts",
    "normal-skin",
]


@dataclass
class DataSource:
    """One public dataset source.

    Parameters
    ----------
    name : str
        Unique identifier used as directory name under ``data/raw/``.
    url : str
        Primary download URL or API base.
    license : str
        SPDX expression or free-text license description.
    native_labels : list[str]
        Label strings as they appear in the source dataset.
    label_mapping : dict[str, str]
        Maps each native label to a canonical class name from CANONICAL_CLASSES.
        Native labels absent from this dict have no canonical equivalent and are
        excluded from the merged corpus.
    requires_auth : bool
        True when download requires an API key or account (Kaggle, DermNet scrape).
    """

    name: str
    url: str
    license: str
    native_labels: list[str]
    label_mapping: dict[str, str]
    requires_auth: bool = False


SOURCES: dict[str, DataSource] = {
    "HAM10000": DataSource(
        name="HAM10000",
        url="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T",
        license="CC-BY-NC 4.0",
        native_labels=["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        label_mapping={
            "akiec": "actinic-keratosis",
            "bcc": "basal-cell-carcinoma",
            "mel": "melanoma",
            "nv": "melanocytic-nevi",
            # bkl (benign keratosis), df (dermatofibroma), vasc (vascular lesions)
            # have no canonical equivalent and are omitted intentionally.
        },
        requires_auth=False,
    ),
    "ISIC": DataSource(
        name="ISIC",
        url="https://api.isic-archive.com/api/v2/",
        license="CC-BY-NC 4.0",
        native_labels=[
            "melanoma",
            "nevus",
            "basal cell carcinoma",
            "actinic keratosis",
            "benign keratosis",
            "dermatofibroma",
            "vascular lesion",
            "squamous cell carcinoma",
            "seborrheic keratosis",
        ],
        label_mapping={
            "melanoma": "melanoma",
            "nevus": "melanocytic-nevi",
            "basal cell carcinoma": "basal-cell-carcinoma",
            "actinic keratosis": "actinic-keratosis",
        },
        requires_auth=False,
    ),
    "MSLD": DataSource(
        name="MSLD",
        url="https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset",
        license="CC-BY 4.0",
        native_labels=["Monkeypox", "Chickenpox", "Measles", "Normal"],
        label_mapping={
            "Monkeypox": "monkeypox",
            "Chickenpox": "chickenpox",
            "Measles": "measles",
        },
        requires_auth=True,
    ),
    "DermNet": DataSource(
        name="DermNet",
        url="https://dermnetnz.org/image-library",
        license="All rights reserved — images not redistributable",
        native_labels=[
            "Acne",
            "Dermatitis",
            "Eczema",
            "Erythema multiforme",
            "Granuloma",
            "Herpes simplex",
            "Herpes zoster",
            "Hidradenitis suppurativa",
            "Lupus",
            "Molluscum contagiosum",
            "Psoriasis",
            "Rosacea",
            "Tinea",
            "Vasculitis",
            "Warts",
        ],
        label_mapping={
            "Acne": "acne",
            "Dermatitis": "dermatitis",
            "Eczema": "eczema",
            "Erythema multiforme": "erythema-multiforme",
            "Granuloma": "granuloma",
            "Herpes simplex": "herpes",
            "Herpes zoster": "herpes",
            "Hidradenitis suppurativa": "hidradenitis-suppurativa",
            "Lupus": "lupus",
            "Molluscum contagiosum": "molluscum",
            "Psoriasis": "psoriasis",
            "Rosacea": "rosacea",
            "Tinea": "tinea",
            "Vasculitis": "vasculitis",
            "Warts": "warts",
        },
        requires_auth=True,
    ),
    "KaggleAcne": DataSource(
        name="KaggleAcne",
        url="https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset",
        license="Unknown — verify before use",
        native_labels=[
            "Acne and Rosacea Photos",
            "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
            "Atopic Dermatitis Photos",
            "Psoriasis pictures Lichen Planus and related diseases",
            "Tinea Ringworm Candidiasis and other Fungal Infections",
            "Warts Molluscum and other Viral Infections",
        ],
        label_mapping={
            "Acne and Rosacea Photos": "acne",
            "Atopic Dermatitis Photos": "eczema",
            "Psoriasis pictures Lichen Planus and related diseases": "psoriasis",
            "Tinea Ringworm Candidiasis and other Fungal Infections": "tinea",
            "Warts Molluscum and other Viral Infections": "warts",
        },
        requires_auth=True,
    ),
    "NormalSkin": DataSource(
        name="NormalSkin",
        url="https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset",
        license="CC-BY 4.0",
        native_labels=["normal", "oily", "dry"],
        label_mapping={
            "normal": "normal-skin",
        },
        requires_auth=True,
    ),
}

# Reverse index: canonical class name → list of source names that provide it.
# Each source appears at most once per class even when multiple native labels
# share the same canonical target (e.g. "Herpes simplex" and "Herpes zoster").
CLASS_TO_SOURCES: dict[str, list[str]] = {}
for _src in SOURCES.values():
    _seen_canonical: set[str] = set()
    for _canonical in _src.label_mapping.values():
        if _canonical not in _seen_canonical:
            CLASS_TO_SOURCES.setdefault(_canonical, []).append(_src.name)
            _seen_canonical.add(_canonical)


def get_source(name: str) -> DataSource:
    """Return a DataSource by its unique name.

    Parameters
    ----------
    name : str
        The ``name`` field of the desired source.

    Returns
    -------
    DataSource
        The matching registry entry.

    Raises
    ------
    KeyError
        If no source with the given name is registered.
    """
    try:
        return SOURCES[name]
    except KeyError:
        raise KeyError(f"No source registered with name: {name!r}") from None


def sources_for_class(class_name: str) -> list[DataSource]:
    """Return all registered sources that provide a given canonical class.

    Parameters
    ----------
    class_name : str
        Canonical disease class name (e.g. ``'melanoma'``).

    Returns
    -------
    list[DataSource]
        Possibly empty list of matching sources.
    """
    return [SOURCES[n] for n in CLASS_TO_SOURCES.get(class_name, [])]
