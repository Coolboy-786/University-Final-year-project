"""Test-set evaluation for all trained models.

Loads each checkpoint, runs inference on the test split, and writes per-model
outputs under outputs/eval/{model_name}/:
  predictions.csv  — y_true, y_pred integer columns
  confusion.png    — normalised confusion matrix heatmap
  per_class.csv    — per-class precision/recall/F1/support
  metrics.json     — aggregate accuracy/macro_f1/weighted_f1

Usage (Colab, default Drive checkpoint paths):
    !python scripts/evaluate.py

Override checkpoint paths:
    !python scripts/evaluate.py \\
        --vgg19   /path/to/vgg19.ckpt \\
        --mobile  /path/to/mobile.ckpt \\
        --shuffle /path/to/shuffle.ckpt

Skip ensemble:
    !python scripts/evaluate.py --no-ensemble
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.datamodule import SkinDiseaseDataModule
from src.evaluation.confusion import compute_confusion_matrix, plot_confusion_matrix
from src.evaluation.metrics import compute_aggregate_metrics, compute_per_class_metrics
from src.models.ensemble import AverageEnsemble
from src.models.mobilenet import MobileNetV2Classifier
from src.models.shufflenet import ShuffleNetClassifier
from src.models.vgg19 import VGG19Classifier

# PL ModelCheckpoint with monitor="val/f1" creates a subdirectory because of
# the "/" in the metric name: checkpoints/best-epoch=XX-val/f1=Y.ckpt
_CKPT_DIR = Path("/content/drive/MyDrive/FYP_Thesis/data/checkpoints")
_DEFAULT_VGG19   = _CKPT_DIR / "best-epoch=00-val" / "f1=0.6570.ckpt"
_DEFAULT_MOBILE  = _CKPT_DIR / "best-epoch=01-val" / "f1=0.5674.ckpt"
_DEFAULT_SHUFFLE = _CKPT_DIR / "best-epoch=05-val" / "f1=0.5288.ckpt"

_OUTPUT_ROOT = Path("outputs/eval")


def _collect_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_pred.append(preds)
            all_true.append(labels.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def evaluate_model(
    name: str,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
    output_root: Path,
) -> dict[str, float]:
    out_dir = output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{name}] running inference...")
    y_true, y_pred = _collect_predictions(model, dataloader, device)

    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        out_dir / "predictions.csv", index=False
    )

    cm = compute_confusion_matrix(y_true, y_pred, class_names)
    plot_confusion_matrix(cm, out_dir / "confusion.png", normalize=True)

    per_class = compute_per_class_metrics(y_true, y_pred, class_names)
    per_class.to_csv(out_dir / "per_class.csv", index=False)

    agg = compute_aggregate_metrics(y_true, y_pred)
    (out_dir / "metrics.json").write_text(json.dumps(agg, indent=2))

    print(
        f"[{name}] acc={agg['accuracy']:.4f}  "
        f"macro_f1={agg['macro_f1']:.4f}  "
        f"weighted_f1={agg['weighted_f1']:.4f}"
    )
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all trained models on the test split."
    )
    parser.add_argument("--vgg19",       default=str(_DEFAULT_VGG19),
                        help="Path to VGG19 checkpoint")
    parser.add_argument("--mobile",      default=str(_DEFAULT_MOBILE),
                        help="Path to MobileNetV2 checkpoint")
    parser.add_argument("--shuffle",     default=str(_DEFAULT_SHUFFLE),
                        help="Path to ShuffleNet checkpoint")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Skip ensemble evaluation")
    parser.add_argument("--two-model", action="store_true",
                        help="Run MobileNet+ShuffleNet ensemble only (no VGG19)")
    parser.add_argument("--output-dir",  default=str(_OUTPUT_ROOT),
                        help="Root directory for evaluation outputs")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dm = SkinDiseaseDataModule()
    dm.prepare_data()
    dm.setup("test")
    test_loader = dm.test_dataloader()

    class_mapping: dict[str, int] = json.loads(
        dm.class_mapping_path.read_text(encoding="utf-8")
    )
    class_names = [k for k, _ in sorted(class_mapping.items(), key=lambda x: x[1])]
    print(f"classes ({len(class_names)}): {class_names}")

    vgg19_ckpt   = Path(args.vgg19)
    mobile_ckpt  = Path(args.mobile)
    shuffle_ckpt = Path(args.shuffle)

    results: dict[str, dict[str, float]] = {}

    for name, cls, ckpt in [
        ("vgg19",      VGG19Classifier,       vgg19_ckpt),
        ("mobilenet",  MobileNetV2Classifier, mobile_ckpt),
        ("shufflenet", ShuffleNetClassifier,  shuffle_ckpt),
    ]:
        if not ckpt.exists():
            print(f"[{name}] checkpoint not found: {ckpt} — skipped")
            continue
        model = cls.load_from_checkpoint(str(ckpt), weights_only=False, map_location="cpu")
        results[name] = evaluate_model(
            name, model, test_loader, class_names, device, output_root
        )

    if not args.no_ensemble:
        if args.two_model:
            missing = [p for p in (mobile_ckpt, shuffle_ckpt) if not p.exists()]
            if missing:
                print(f"2-model ensemble skipped — missing: {[str(p) for p in missing]}")
            else:
                ensemble = AverageEnsemble(
                    mobilenet_ckpt=mobile_ckpt,
                    shufflenet_ckpt=shuffle_ckpt,
                    vgg19_ckpt=None,
                    num_classes=len(class_names),
                )
                results["ensemble_2model"] = evaluate_model(
                    "ensemble_2model", ensemble, test_loader, class_names, device, output_root
                )
        else:
            missing = [p for p in (vgg19_ckpt, mobile_ckpt, shuffle_ckpt) if not p.exists()]
            if missing:
                print(f"ensemble skipped — missing: {[str(p) for p in missing]}")
            else:
                ensemble = AverageEnsemble(
                    mobilenet_ckpt=mobile_ckpt,
                    shufflenet_ckpt=shuffle_ckpt,
                    vgg19_ckpt=vgg19_ckpt,
                    num_classes=len(class_names),
                )
                results["ensemble"] = evaluate_model(
                    "ensemble", ensemble, test_loader, class_names, device, output_root
                )

    print("\n=== test-set summary ===")
    summary = pd.DataFrame(
        [{"model": k, **v} for k, v in results.items()]
    ).set_index("model")
    print(summary[["accuracy", "macro_f1", "weighted_f1"]].to_string())
    summary.to_csv(output_root / "summary.csv")
    print(f"\noutputs → {output_root}/")


if __name__ == "__main__":
    main()
