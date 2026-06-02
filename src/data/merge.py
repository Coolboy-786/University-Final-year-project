"""Merge HAM10000 + ISIC2018 test images into a single manifest.

Reads raw archives from ``<data_root>/raw/HAM10000/``, extracts into
``<data_root>/interim/images/`` (skipping anything already extracted), maps
labels to the canonical space, and writes
``<data_root>/interim/merged_manifest.csv``.

CLI: ``python -m src.data.merge [--dry-run]``.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data.sources import DX_TO_CANONICAL
from src.utils.paths import get_data_root


HAM_IMAGE_ZIPS: tuple[str, ...] = (
    "HAM10000_images_part_1.zip",
    "HAM10000_images_part_2.zip",
)
ISIC_TEST_ZIP = "ISIC2018_Task3_Test_Images.zip"

HAM_METADATA = "HAM10000_metadata.tab"
ISIC_TEST_GT = "ISIC2018_Task3_Test_GroundTruth.tab"

ONE_HOT_COLUMNS: tuple[str, ...] = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")


def extract_zip(zip_path: Path, dest_dir: Path) -> int:
    """Extract *zip_path* into *dest_dir*, skipping members that already exist.

    Returns the number of newly-extracted files.
    """
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            target = dest_dir / member.filename
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as out:
                out.write(src.read())
            extracted += 1
    return extracted


def read_tab_table(path: Path) -> pd.DataFrame:
    """Read a .tab file as TSV; fall back to comma if it parses as a single column."""
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")
    return df


def one_hot_row_to_label(row: "pd.Series[object]") -> str | None:
    """Return the canonical label for the one-hot column equal to 1 in *row*."""
    for col in ONE_HOT_COLUMNS:
        if col not in row.index:
            continue
        try:
            if float(row[col]) == 1.0:
                return DX_TO_CANONICAL[col.lower()]
        except (TypeError, ValueError):
            continue
    return None


def derive_ham_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map HAM10000 metadata rows to ``(image_id, label)``."""
    return pd.DataFrame(
        {
            "image_id": df["image_id"].astype(str),
            "label": df["dx"].astype(str).map(DX_TO_CANONICAL),
        }
    )


def derive_isic_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map ISIC2018 ground-truth rows to ``(image_id, label)`` via the one-hot columns."""
    image_col = next(c for c in df.columns if c.lower() == "image")
    labels = df.apply(one_hot_row_to_label, axis=1)
    return pd.DataFrame({"image_id": df[image_col].astype(str), "label": labels})


def index_images(images_dir: Path) -> dict[str, Path]:
    """Return ``{image_stem: filepath}`` for every ``*.jpg`` under *images_dir*."""
    return {p.stem: p for p in images_dir.rglob("*.jpg")}


def build_manifest(data_root: Path) -> tuple[pd.DataFrame, int]:
    """Build the merged manifest dataframe and return ``(df, dropped_count)``."""
    raw_dir = data_root / "raw" / "HAM10000"
    images_dir = data_root / "interim" / "images"

    ham_df = read_tab_table(raw_dir / HAM_METADATA)
    ham_labels = derive_ham_labels(ham_df)
    ham_labels["source"] = "HAM10000"

    isic_df = read_tab_table(raw_dir / ISIC_TEST_GT)
    isic_labels = derive_isic_labels(isic_df)
    isic_labels["source"] = "ISIC2018_test"

    combined = pd.concat([ham_labels, isic_labels], ignore_index=True)

    index = index_images(images_dir)
    resolved_paths = combined["image_id"].map(index)
    missing_mask = resolved_paths.isna() | combined["label"].isna()
    dropped = int(missing_mask.sum())

    kept = combined[~missing_mask].copy()
    kept["filepath"] = resolved_paths[~missing_mask].apply(
        lambda p: str(p.relative_to(data_root))
    )

    manifest = kept[["filepath", "label", "source"]].reset_index(drop=True)
    return manifest, dropped


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog="python -m src.data.merge")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline but do not write the manifest CSV.",
    )
    args = parser.parse_args(argv)

    data_root = get_data_root()
    raw_dir = data_root / "raw" / "HAM10000"
    interim_dir = data_root / "interim"
    images_dir = interim_dir / "images"
    manifest_path = interim_dir / "merged_manifest.csv"

    images_dir.mkdir(parents=True, exist_ok=True)

    for zname in (*HAM_IMAGE_ZIPS, ISIC_TEST_ZIP):
        zpath = raw_dir / zname
        if not zpath.exists():
            print(f"WARNING: missing zip {zpath}", file=sys.stderr)
            continue
        n = extract_zip(zpath, images_dir)
        print(f"Extracted {n} new files from {zname}")

    manifest, dropped = build_manifest(data_root)

    if dropped:
        print(f"Dropped {dropped} rows (no matching image file or unmapped label)")

    if args.dry_run:
        print("--dry-run: not writing manifest")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
        print(f"Wrote manifest: {manifest_path} ({len(manifest)} rows)")

    print(f"\nTotal images: {len(manifest)}")

    print("\nPer-source counts:")
    for src, n in manifest["source"].value_counts().items():
        print(f"  {src}: {n}")

    print("\nPer-class counts:")
    class_counts = Counter(manifest["label"])
    width = max((len(c) for c in class_counts), default=0)
    for label, n in sorted(class_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label.ljust(width)}  {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
