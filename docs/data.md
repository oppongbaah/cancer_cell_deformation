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
│   └── legacy/20210825_cancer/     220 unlabelled JPGs from 2021 microscope
│
├── frames/                         Raw extracted frames (1228×922 greyscale JPEG)
│   ├── 01_annotate_pool/           U-Net training source
│   ├── 02_unet_holdout/            U-Net test source
│   │   └── legacy_holdout/         20 reserved legacy frames — separate diagnostic DSC
│   ├── 03_mask_factory/            Inference → classifier training
│   └── 04_clf_arena/               Final classifier evaluation
│
├── masks/                          Hand-drawn masks (1228×922 PNG)
│   ├── 01_annotate_pool/           U-Net training labels (binary, 0/255)
│   ├── 01_annotate_pool_multiclass/  Experimental 3-class labels (0/1/2), same 40 frames
│   ├── 02_unet_holdout/            U-Net test ground truth
│   └── 02_unet_holdout/legacy_holdout/  Ground truth for the legacy holdout frames
│
└── preprocessed/                   256×256 pipeline-ready data
    ├── train/01_annotate_pool/
    │   ├── inputs/                 Preprocessed frames → U-Net training inputs
    │   └── labels/                 Resized masks       → U-Net training labels
    ├── evaluate/02_unet_holdout/
    │   ├── inputs/                 Preprocessed frames → U-Net evaluation inputs
    │   └── labels/                 Resized masks       → U-Net evaluation labels
    ├── evaluate/02_unet_holdout/legacy_holdout/
    │   ├── inputs/                 Preprocessed legacy holdout frames → separate DSC
    │   └── labels/                 Resized legacy holdout masks
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
| `frames/02_unet_holdout/legacy_holdout/` | Legacy-domain diagnostic source | No | Hand-drawn (required) | After U-Net training |
| `frames/03_mask_factory/` | Classifier training source | Yes | U-Net predicted | After U-Net training |
| `frames/04_clf_arena/` | Classifier test source | Yes | U-Net predicted | Final evaluation only |
| `masks/01_annotate_pool/` | U-Net training labels | No | Binary, hand-drawn | During annotation |
| `masks/02_unet_holdout/` | U-Net test labels | No | Binary, hand-drawn | After U-Net training |
| `masks/02_unet_holdout/legacy_holdout/` | Legacy-domain diagnostic labels | No | Binary, hand-drawn | After U-Net training |
| `preprocessed/train/.../inputs/` | U-Net training inputs (256×256) | No | — | After preprocessing |
| `preprocessed/train/.../labels/` | U-Net training labels (256×256) | No | Resized binary | After preprocessing |
| `preprocessed/evaluate/.../inputs/` | U-Net evaluation inputs (256×256) | No | — | After holdout annotation |
| `preprocessed/evaluate/.../labels/` | U-Net evaluation labels (256×256) | No | Resized binary | After holdout annotation |
| `preprocessed/inference/03_.../inputs/` | Classifier training inputs | Yes | — | After U-Net training |
| `preprocessed/inference/04_.../inputs/` | Classifier test inputs | Yes | — | Final evaluation only |
| `preprocessed/comparisons/` | Reporting figures | — | — | On demand (`--compare`) |

---

## Pool descriptions

### `01_annotate_pool` — 240 images (U-Net training)

Source for U-Net training masks. Annotate every image before running preprocessing.

| Subset | Count | Notes |
|--------|-------|-------|
| Legacy cancer images (2021) | 220 | Filename: `YYYYMMDDHHMMSS.jpg` — no HER2 label |
| Domain-adapt samples | 20 | Prefix: `domain_<her2>_<cell_id>_<frame>.jpg` |

Domain-adapt frames come from cells also in `03_mask_factory`. The U-Net trains on them, then runs inference on the full set from those cells.

!!! note "Multiclass labels"
    The same 40 currently-annotated frames also have an experimental 3-class labeling (`data/masks/01_annotate_pool_multiclass/`: 0=background, 1=trapped cell, 2=other/decoy object) used for the binary-vs-multiclass ablation in [Training Results](training.md). `scripts/annotate.py --multiclass` seeds label 1 from the existing binary mask, so only decoy objects need to be painted.

### `02_unet_holdout` — 101 images (U-Net test)

Held out from all training. Annotate **after** U-Net training is complete.

| Cell | HER2 | Frames |
|------|------|--------|
| cell9 | high | 50 |
| cell6 | low | 51 |

The DSC score from evaluating on this pool is the thesis-reportable segmentation metric.

!!! note "`02_unet_holdout/legacy_holdout/` — 20 images, evaluated separately"
    A diagnostic subset nested under `02_unet_holdout/` but **never merged** into the DSC above. Reserved unannotated from `01_annotate_pool` (evenly sampled across the legacy frames unannotated at the time) before annotated training data included any real-cell frames. Since training was legacy-only while `02_unet_holdout` is real-cell-only, one blended DSC couldn't separate "weak segmentation" from "domain gap." Evaluating this subset on its own (own `--image-dir`/`--mask-dir` pair; `evaluate_unet.py --pool 02_unet_holdout` doesn't recurse into it) gives a second, comparable DSC — high legacy-holdout DSC + low real-cell-holdout DSC points to the domain gap specifically.

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
