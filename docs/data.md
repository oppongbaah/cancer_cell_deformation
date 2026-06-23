# Data Layout

Full documentation of the `data/` directory — see also `data/README.md` for per-cell tables and the contamination firewall details.

---

## Directory structure

```
data/
├── raw/                            Source recordings — never modified
│   ├── high/                       High HER2-expression cells (9 videos)
│   ├── low/                        Low HER2-expression cells (5 videos)
│   ├── cell 1 low 50 increments.mp4
│   └── legacy/20210825_cancer/     223 unlabelled JPGs from 2021 microscope
│
├── frames/                         Raw extracted frames (1228×922 greyscale JPEG)
│   ├── 01_annotate_pool/           U-Net training source
│   ├── 02_unet_holdout/            U-Net test source
│   ├── 03_mask_factory/            Inference → classifier training
│   └── 04_clf_arena/               Final classifier evaluation
│
├── masks/                          Hand-drawn binary masks (1228×922 PNG)
│   ├── 01_annotate_pool/           U-Net training labels
│   └── 02_unet_holdout/            U-Net test ground truth
│
└── preprocessed/                   256×256 pipeline-ready data
    ├── train/01_annotate_pool/
    │   ├── inputs/                 Preprocessed frames → U-Net training inputs
    │   └── labels/                 Resized masks       → U-Net training labels
    ├── evaluate/02_unet_holdout/
    │   ├── inputs/                 Preprocessed frames → U-Net evaluation inputs
    │   └── labels/                 Resized masks       → U-Net evaluation labels
    ├── inference/
    │   ├── 03_mask_factory/inputs/ Preprocessed frames → classifier training
    │   └── 04_clf_arena/inputs/    Preprocessed frames → final evaluation
    └── comparisons/                Pipeline stage figures (reporting)
```

---

## Pipeline role of each folder

| Folder | ML role | HER2 label | Masks | Touch when |
|--------|---------|-----------|-------|-----------|
| `raw/` | Source — never modified | In filename | None | Never |
| `frames/01_annotate_pool/` | U-Net training source | No | Hand-drawn (required) | Annotate now |
| `frames/02_unet_holdout/` | U-Net test source | No | Hand-drawn (required) | After U-Net training |
| `frames/03_mask_factory/` | Classifier training source | Yes | U-Net predicted | After U-Net training |
| `frames/04_clf_arena/` | Classifier test source | Yes | U-Net predicted | Final evaluation only |
| `masks/01_annotate_pool/` | U-Net training labels | No | Binary, hand-drawn | During annotation |
| `masks/02_unet_holdout/` | U-Net test labels | No | Binary, hand-drawn | After U-Net training |
| `preprocessed/train/.../inputs/` | U-Net training inputs (256×256) | No | — | After preprocessing |
| `preprocessed/train/.../labels/` | U-Net training labels (256×256) | No | Resized binary | After preprocessing |
| `preprocessed/evaluate/.../inputs/` | U-Net evaluation inputs (256×256) | No | — | After holdout annotation |
| `preprocessed/evaluate/.../labels/` | U-Net evaluation labels (256×256) | No | Resized binary | After holdout annotation |
| `preprocessed/inference/03_.../inputs/` | Classifier training inputs | Yes | — | After U-Net training |
| `preprocessed/inference/04_.../inputs/` | Classifier test inputs | Yes | — | Final evaluation only |
| `preprocessed/comparisons/` | Reporting figures | — | — | On demand (`--compare`) |

---

## Pool descriptions

### `01_annotate_pool` — 243 images (U-Net training)

Source for U-Net training masks. Annotate every image before running preprocessing.

| Subset | Count | Notes |
|--------|-------|-------|
| Legacy cancer images (2021) | 223 | Filename: `YYYYMMDDHHMMSS.jpg` — no HER2 label |
| Domain-adapt samples | 20 | Prefix: `domain_<her2>_<cell_id>_<frame>.jpg` |

Domain-adapt frames come from cells also in `03_mask_factory`. The U-Net trains on them, then runs inference on the full set from those cells.

### `02_unet_holdout` — 101 images (U-Net test)

Held out from all training. Annotate **after** U-Net training is complete.

| Cell | HER2 | Frames |
|------|------|--------|
| cell9 | high | 50 |
| cell6 | low | 51 |

The DSC score from evaluating on this pool is the thesis-reportable segmentation metric.

### `03_mask_factory` — 557 images (classifier training)

U-Net inference pool. Each predicted mask → 15 morphological features → one classifier training sample.

| HER2 | Cells | Frames |
|------|-------|--------|
| high | 1, 2, 3, 4, 6, 7, 8 | 355 |
| low | 1, 2, 3, 4 | 202 |
| **Total** | | **557** |

!!! note "Class imbalance"
    Training set is ~64 % high / 36 % low. Use `class_weight='balanced'` on all classifiers.

### `04_clf_arena` — 101 images (classifier test)

Completely held out. These cells were never annotated and never seen by any model during training.

| Cell | HER2 | Frames |
|------|------|--------|
| cell10 | high | 50 |
| cell5 | low | 51 |

---

## Contamination firewall

The data split ensures no training signal leaks into final evaluation:

- **Legacy images** have no HER2 label — safe for U-Net training; cannot contaminate the classifier.
- **`02_unet_holdout`** cells were never used during U-Net training. Annotated masks give an unbiased segmentation DSC.
- **`04_clf_arena`** cells were never annotated, never used to generate classifier training features, and never included in cross-validation. They are the single honest test of classifier generalisation.

---

## Regenerating frames

```bash
celldform-organize-dataset                                    # defaults
celldform-organize-dataset --every-n 100 --domain-adapt-n 10 # denser sampling
celldform-organize-dataset --no-clean                         # skip wiping data/frames/
```
