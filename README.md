# Image-Type-Aware Skin Disease Classification

Reproduction of a B.Tech thesis (VNIT Nagpur, 2023) on multi-class skin disease
classification using a lightweight CNN ensemble with image-type-aware segmentation
routing.

See `PLAN.md` for the full specification, build order, and open questions.

## Getting Started

### Track A — Local Setup

```bash
pip install -e ".[dev]"
pre-commit install

# Download raw data
python -m src.data.download

# Build processed corpus
python -m src.data.merge

# Train
python -m src.training.train model=mobilenet preprocessing=none
```

The data lands in `data/raw/`, `data/interim/`, and `data/processed/`.
Checkpoints land in `data/checkpoints/`.

### Track B — Google Colab Setup

Open [notebooks/00_colab_setup.ipynb](notebooks/00_colab_setup.ipynb) in Colab
and run all cells in order. The notebook will:

1. Mount Google Drive
2. Clone (or update) this repo
3. Install dependencies
4. Symlink `repo/data` → Drive folder
5. Export `STORAGE_BACKEND=colab` so all paths resolve to Drive

**Expected Drive folder structure:**

```
MyDrive/
└── FYP_Thesis/
    └── data/
        ├── raw/
        │   └── HAM10000/
        │       ├── manifest.csv        ← tracked in git
        │       └── ...zips / images
        ├── interim/                    ← generated
        ├── processed/                  ← generated; copied to /content/ for fast I/O
        │   └── <class>/
        └── splits.csv                  ← generated; lock after first run
```

Set your Kaggle API token as a Colab secret named `KAGGLE_API_TOKEN` before running.

## Train

```bash
python -m src.training.train model=mobilenet preprocessing=none
```

## Ablation

```bash
python scripts/run_ablation.py
```
