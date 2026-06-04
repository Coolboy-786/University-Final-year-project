"""Cache CLIP ViT-B/32 pooled features for all splits.

Run once before training the linear probe:
    !python scripts/cache_clip_features.py

Writes to <output_dir>/:
    train_features.pt  train_labels.pt
    val_features.pt    val_labels.pt
    test_features.pt   test_labels.pt

Each features file is a float32 tensor of shape (N, 768).
Each labels file is a long tensor of shape (N,).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPVisionModel

from src.data.splits import load_splits
from src.training.augment import get_eval_transforms
from src.utils.paths import get_data_root

_MODEL_NAME = "openai/clip-vit-base-patch32"
_DRIVE_FEATURES = Path("/content/drive/MyDrive/FYP_Thesis/data/clip_features")

_IM_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
_IM_STD  = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
_CL_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
_CL_STD  = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)


def _renorm(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    im_mean = _IM_MEAN.to(device)
    im_std  = _IM_STD.to(device)
    cl_mean = _CL_MEAN.to(device)
    cl_std  = _CL_STD.to(device)
    x = x * im_std + im_mean
    return (x - cl_mean) / cl_std


class _SplitDataset(Dataset):
    def __init__(self, items: list[tuple[str, int]], image_size: int = 224) -> None:
        self._items = items
        self._transform = get_eval_transforms(image_size)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self._items[idx]
        img = np.array(Image.open(path).convert("RGB"))
        img = self._transform(image=img)["image"]
        return img, label


def _build_items(
    splits_df,
    split_name: str,
    data_root: Path,
    class_mapping: dict[str, int],
) -> list[tuple[str, int]]:
    rows = splits_df[splits_df["split"] == split_name]
    return [
        (str(data_root / row["filepath"]), class_mapping[row["label"]])
        for _, row in rows.iterrows()
    ]


@torch.no_grad()
def _extract(
    encoder: CLIPVisionModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder.eval()
    all_feats, all_labels = [], []
    for images, labels in loader:
        images = _renorm(images.to(device), device)
        feats = encoder(pixel_values=images).pooler_output.cpu()
        all_feats.append(feats)
        all_labels.append(
            labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
        )
    return torch.cat(all_feats), torch.cat(all_labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache CLIP ViT-B/32 features for all data splits."
    )
    parser.add_argument("--output-dir", default=str(_DRIVE_FEATURES))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("Loading CLIP encoder...")
    encoder = CLIPVisionModel.from_pretrained(_MODEL_NAME).to(device)

    root = get_data_root()
    splits_df = load_splits(root / "splits.csv")
    class_mapping: dict[str, int] = json.loads(
        (Path("src/data/class_mapping.json")).read_text(encoding="utf-8")
    )

    for split in ("train", "val", "test"):
        items = _build_items(splits_df, split, root, class_mapping)
        loader = DataLoader(
            _SplitDataset(items),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print(f"Extracting {split} ({len(items)} images)...")
        feats, labels = _extract(encoder, loader, device)
        torch.save(feats,  output_dir / f"{split}_features.pt")
        torch.save(labels, output_dir / f"{split}_labels.pt")
        print(f"  saved {feats.shape}  dtype={feats.dtype}")

    print(f"\nAll features cached to {output_dir}/")


if __name__ == "__main__":
    main()
