# PLAN.md — Image-Type-Aware Skin Disease Classification

Working title: **"Image-Type-Aware Segmentation Routing for Multi-Class Skin Disease Classification with a Lightweight CNN Ensemble"**

This document is the single source of truth for the research paper and code. It is designed to track the original B.Tech thesis (VNIT Nagpur, 2023) faithfully — the goal is a clean reproduction with the minimum additions required for peer review, not a redesign.

---

## 1. Project Overview

We are rebuilding a thesis on multi-class skin disease classification using CNNs. The original reported 93.2% test accuracy on a 23-class merged dataset using an average ensemble of MobileNetV2 and ShuffleNet, with a novel preprocessing step that routed images to different segmentation methods based on image type.

The original code, weights, and dataset are not available. We rebuild from scratch using public data sources, but we preserve the thesis's methodology, model choices, and training procedure wherever possible.

### Locked decisions

- **Framework:** PyTorch + PyTorch Lightning.
- **Dataset:** rebuilt from public sources (we do not have the original).
- **Class count:** target 23, but apply a threshold-based filter to drop under-represented classes from the final corpus. See Section 6.
- **Ablations:** one (router on vs off). Everything else matches the thesis.

---

## 2. Research Contributions

In order of novelty:

1. **Image-type-aware segmentation routing.** A lightweight classifier predicts whether an input is a *full-body*, *body-part*, or *skin-only* image and dispatches it to the appropriate segmentation method (Otsu / adaptive thresholding / identity). Not present in the surveyed prior work.
2. **A lightweight CNN ensemble (MobileNetV2 + ShuffleNet) for multi-class skin disease classification.**
3. **A reconstructed, documented multi-class dataset** assembled from public sources with per-class provenance.

---

## 3. Paper Target & Narrative

**Venue type:** IEEE / Springer conference or workshop. Candidates (verify deadlines closer to submission):

- IEEE CBMS (Computer-Based Medical Systems)
- IEEE EMBC
- IEEE ISBI / ICIP medical imaging workshops
- MICCAI workshops (ISIC Skin Image Analysis is the most relevant)
- Springer MIUA

**Page budget:** 6–8 pages, double-column IEEE format.

**Section sketch:**

1. Introduction — motivation, gaps, contributions (1 page)
2. Related Work — extends the thesis's 5-paper lit survey (~1 page)
3. Method — router, segmentation pipeline, backbones, ensemble (1.5–2 pages, with pipeline figure)
4. Dataset — provenance table + threshold pruning rule (0.75 page)
5. Experiments — main results + router ablation (1.5 pages)
6. Discussion & limitations (0.5 page)
7. Conclusion (0.25 page)
8. References

**Tables:** (a) dataset composition and sources, (b) main results across models, (c) router on/off ablation, (d) per-class precision/recall/F1.

**Figures:** (a) pipeline overview, (b) router confusion matrix, (c) sample segmentation outputs across image types, (d) main confusion matrix.

---

## 4. Repository Structure

```
skin-disease-classification/
├── PLAN.md                       # this file
├── README.md
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
├── configs/                      # Hydra configs
│   ├── config.yaml
│   ├── data/
│   │   └── merged.yaml
│   ├── model/
│   │   ├── mobilenet.yaml
│   │   ├── shufflenet.yaml
│   │   ├── vgg19.yaml
│   │   └── ensemble.yaml
│   ├── preprocessing/
│   │   ├── none.yaml
│   │   ├── otsu_only.yaml
│   │   ├── adaptive_only.yaml
│   │   └── routed.yaml
│   └── experiment/
├── src/
│   ├── data/
│   │   ├── sources.py            # registry of public dataset sources
│   │   ├── download.py
│   │   ├── merge.py              # builds the merged corpus
│   │   ├── prune.py              # applies the threshold filter
│   │   ├── splits.py             # stratified 70/15/15, seed locked
│   │   ├── datamodule.py         # Lightning DataModule
│   │   └── class_mapping.json    # stable label ↔ index mapping (generated)
│   ├── preprocessing/
│   │   ├── router.py             # VGG19-based image-type classifier
│   │   ├── segment_otsu.py
│   │   ├── segment_adaptive.py
│   │   ├── segment_identity.py
│   │   └── pipeline.py           # conditional dispatcher
│   ├── models/
│   │   ├── mobilenet.py
│   │   ├── shufflenet.py
│   │   ├── vgg19.py
│   │   ├── ensemble.py
│   │   └── base.py               # LightningModule base
│   ├── training/
│   │   ├── train.py
│   │   ├── callbacks.py
│   │   └── augment.py
│   ├── evaluation/
│   │   ├── metrics.py            # per-class P/R/F1, macro/weighted
│   │   ├── confusion.py
│   │   └── report.py             # generates the paper-ready tables
│   └── utils/
│       ├── seeds.py
│       └── logging.py
├── scripts/
│   ├── run_ablation.py           # router on vs off
│   └── make_paper_figures.py
├── notebooks/                    # exploratory; not load-bearing
├── tests/                        # pytest
└── paper/
    ├── main.tex
    ├── refs.bib
    └── figures/
```

---

## 5. Technology Stack

- **Python** 3.11+
- **PyTorch** 2.x + **PyTorch Lightning**
- **timm** for MobileNetV2 (and optionally a ShuffleNetV2 reference)
- **Hydra** for config management
- **Weights & Biases** for experiment tracking (TensorBoard fallback)
- **albumentations** for augmentation
- **OpenCV** for segmentation
- **scikit-learn** for metrics
- **matplotlib** + **seaborn** for figures
- **pytest** for tests
- **ruff** + **black** + **pre-commit** for code hygiene

---

## 6. Dataset Construction

This is the most fragile part of the rebuild. The plan is deterministic, documented, and threshold-driven so the final class list is a function of the data, not the author.

### 6.1 Candidate classes (target: 23, from thesis)

acne, actinic-keratosis, basal-cell-carcinoma, chickenpox, cowpox, dermatitis, eczema, erythema-multiforme, granuloma, herpes, hidradenitis-suppurativa, lupus, measles, melanocytic-nevi, melanoma, molluscum, monkeypox, psoriasis, rosacea, tinea, vasculitis, warts, normal-skin.

### 6.2 Public sources to scrape/download

`src/data/sources.py` is a registry mapping each candidate class to one or more public sources:

- **HAM10000** (via ISIC): melanocytic-nevi, melanoma, basal-cell-carcinoma, actinic-keratosis, benign-keratosis, vascular-lesions, dermatofibroma.
- **ISIC Archive** (full): augments HAM10000 for the same classes.
- **Monkeypox Skin Lesion Dataset (MSLD)** on Kaggle: monkeypox, chickenpox, measles, "others".
- **DermNet NZ** image collection: eczema, psoriasis, rosacea, tinea, warts, herpes, dermatitis, molluscum, and several rarer conditions.
- **Kaggle "Skin Diseases Image Dataset"** and similar aggregated Kaggle releases: acne, additional eczema/psoriasis samples.
- **Curated normal-skin** sources (mixed Kaggle + manually verified).

Each source entry records: URL, license, expected class mapping, and a checksum once downloaded.

### 6.3 Threshold-based pruning

After downloading and de-duplicating, the merge step counts instances per class. Any class with fewer than `min_instances` images is dropped. Defaults (configurable in `configs/data/merged.yaml`):

```yaml
min_instances: 100         # absolute floor
min_fraction: 0.005        # 0.5% of total dataset size
prune_rule: "either"       # drop class if it fails EITHER threshold
```

Classes likely to fail the threshold based on public availability: cowpox, smallpox (not in thesis anyway), hidradenitis-suppurativa, granuloma, vasculitis, erythema-multiforme, lupus, possibly molluscum and measles. The final class count will likely land between **17 and 21**. This is fine — the paper reports the threshold and the resulting list explicitly.

### 6.4 Stable class index mapping

After pruning, the surviving class names are sorted alphabetically and assigned integer indices `0..N-1`. The mapping is written to `src/data/class_mapping.json` and is the canonical reference for every train/eval/inference operation. Example:

```json
{
  "version": 1,
  "n_classes": 19,
  "threshold": {"min_instances": 100, "min_fraction": 0.005, "rule": "either"},
  "classes": {
    "acne": 0,
    "actinic-keratosis": 1,
    "basal-cell-carcinoma": 2,
    "...": "..."
  }
}
```

Once committed, this file is not edited by hand. Re-running `merge.py` with different thresholds produces a new versioned mapping.

### 6.5 Splits

Stratified 70/15/15 (train/val/test), seed 42. Saved to `splits.csv` so every downstream run reads exactly the same splits. No re-stratification per experiment.

### 6.6 Router training data

The router needs its own small labelled dataset (~400 images) tagged as `full_body / body_part / skin_only`. Drawn as a stratified sample from the merged corpus and manually labelled. Labels stored in `data/router_labels.csv`.

---

## 7. Module Specifications

### 7.1 `preprocessing/router.py`

- Input: RGB image, resized to 224×224.
- Output: one of `{full_body, body_part, skin_only}`.
- Model: VGG19 ImageNet-pretrained, 3-class head.
- Trained on the router dataset (Section 6.6).
- Saved as a Lightning checkpoint.

### 7.2 `preprocessing/pipeline.py`

```python
class PreprocessingPipeline:
    """Dispatches an image to a segmentation method.

    mode='none'     → identity
    mode='otsu'     → Otsu thresholding for all images
    mode='adaptive' → adaptive thresholding for all images
    mode='routed'   → use router to pick per image
    """
```

The `mode` parameter is what the ablation switches over.

### 7.3 `models/mobilenet.py`

- MobileNetV2 from `torchvision.models` (ImageNet pretrained).
- Last 20 layers unfrozen (matches thesis).
- Head: GAP → Dense(num_classes), softmax via cross-entropy loss.
- Adam, lr=1e-4.

### 7.4 `models/shufflenet.py`

ShuffleNetV1 from scratch, architecture per thesis §10.3.3. Three stages plus output layer, channel-shuffle in the shuffle units.

### 7.5 `models/vgg19.py`

VGG19 ImageNet-pretrained, softmax head. Used both as a single-model baseline and (via a separate instantiation) as the router.

### 7.6 `models/ensemble.py`

Average ensemble of MobileNetV2 and ShuffleNet softmax outputs. No trainable parameters.

---

## 8. Experimental Plan

### 8.1 Main results (mirrors thesis)

Single dataset (the merged corpus), single seed for the main table:

| # | Model | Preprocessing | Notes |
|---|---|---|---|
| 1 | MobileNetV2 | none | thesis baseline |
| 2 | ShuffleNet  | none | thesis baseline |
| 3 | VGG19       | none | thesis baseline |
| 4 | Ensemble    | none | thesis baseline |
| 5 | Ensemble    | routed | **proposed (thesis headline)** |

### 8.2 Router ablation (the one new experiment)

| # | Model | Preprocessing | Notes |
|---|---|---|---|
| 5 | Ensemble | routed | proposed |
| 6 | Ensemble | otsu only | uniform Otsu baseline |
| 7 | Ensemble | adaptive only | uniform adaptive baseline |
| 8 | Ensemble | none | no segmentation baseline |

Compared head-to-head, this answers "does the routing actually help vs. picking one method?".

### 8.3 Hyperparameters

Locked across runs (match thesis where specified):

- Batch size: 32
- Optimiser: Adam, lr=1e-4 (MobileNetV2), default lr (ShuffleNet), 1e-3 with 0.01/20 decay (VGG19)
- Epochs: MobileNetV2 = 15, ShuffleNet = 20, VGG19 = 40
- Augmentation: rotation ±10°, horizontal flip, width/height shift 0.1, zoom 0.1, shear 0.1
- Input size: 224×224
- Loss: cross-entropy
- Early stopping: patience 5 on val macro-F1

### 8.4 Reproducibility

- Seeds: `torch`, `numpy`, `random`, `PYTHONHASHSEED` all set to 42
- `torch.use_deterministic_algorithms(True)` where feasible
- Splits and class mapping committed to the repo
- Every run logs: config snapshot, git SHA, hardware info, full metrics

---

## 9. Evaluation Protocol

For every run, report:

- Overall accuracy
- Macro-averaged precision, recall, F1
- Weighted-averaged precision, recall, F1
- Per-class precision, recall, F1 (table in supplementary if long)
- Confusion matrix (PNG + CSV)

The thesis reported aggregate accuracy/precision/recall only. Per-class metrics and the confusion matrix are added because they're standard and cost nothing extra.

---

## 10. Build Order

1. **Repo scaffolding** — pyproject, configs, empty modules with type signatures and docstrings.
2. **Data acquisition** — `sources.py` registry, `download.py`, raw class counts.
3. **Merge + prune** — apply threshold rule, generate `class_mapping.json`, plot final class distribution.
4. **DataModule + splits** — stratified split, fixed seed, sanity check (no leakage, label consistency).
5. **Augmentation pipeline** — visualise augmented samples.
6. **MobileNetV2 baseline, no preprocessing** — first end-to-end run; debug here if it doesn't train.
7. **ShuffleNet from scratch** — verify convergence.
8. **VGG19 baseline.**
9. **Ensemble wrapper.**
10. **Segmentation modules** — Otsu, adaptive, identity; visual sanity check.
11. **Router** — label the router dataset, train VGG19 router, evaluate.
12. **Routed pipeline** — wire together.
13. **Run main results table (Section 8.1).**
14. **Run router ablation (Section 8.2).**
15. **Paper figures + draft.**

Checkpoints to stop and reassess: after step 3 (final class list known), step 6 (first model trains), step 13 (main results in).

---

## 11. Coding Standards

- Type hints everywhere; `mypy --strict` on `src/`.
- NumPy-style docstrings on public functions.
- No magic numbers; everything tunable goes in a Hydra config.
- Share library functions across train/eval/ablate — no duplication.
- Tests for data loading, splits (no leakage), preprocessing (shapes/dtypes), and the threshold pruning rule.
- Pre-commit must pass before any commit.
- Soft limit ~300 lines per file.

---

## 12. Open Questions / Decisions Needed

- **Router labels.** Who labels the ~400 router-training images? This is the one manual bottleneck.
- **Compute budget.** Free Colab/Kaggle GPUs are enough for individual runs. The main + ablation grid is 8 runs total — should be manageable.
- **DermNet licensing.** DermNet images are not freely redistributable. The released code repo can include data *loading* scripts but not the images themselves. The paper must note that the dataset cannot be republished in full, only the assembly recipe.
- **Comparison to prior work.** Identify 3–4 recent (2023–2025) skin classification papers and cite their reported numbers in the related work / introduction — useful even though we don't run them ourselves.

---

## 13. References to the Original Thesis

The thesis PDF is the starting point. Sections most relevant to the implementation:

- **§5** — original dataset construction and class list (we target the same 23, modulo pruning)
- **§10.1** — augmentation parameters (replicated exactly in Section 8.3 above)
- **§10.2** — segmentation routing logic
- **§10.3.1–10.3.3** — VGG19, MobileNetV2, and ShuffleNet specifications
- **§10.3.4** — ensemble averaging
