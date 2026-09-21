# Dataset — Cancer Cell Deformation (Optical Tweezers)

This directory contains all raw and processed data for the dual-beam optical
tweezer cancer cell deformation study.  The experiment quantifies how HER2
expression level affects the mechanical deformability of breast cancer cells
trapped by optical tweezers at 50 incrementally increasing laser power/trap
current steps.

---

## Directory structure

```
data/
├── raw/                            Source recordings (never modified)
│   ├── high/                       High HER2-expression cells (9 cells)
│   ├── low/                        Low HER2-expression cells (5 cells)
│   ├── cell 1 low 50 increments.mp4   (one low cell stored at root)
│   └── legacy/
│       └── 20210825_cancer/        2021 microscope images (220 JPGs)
│
├── frames/                         Raw extracted frames (1228×922 grayscale JPEG)
│   ├── 01_annotate_pool/           U-Net training source  — annotate to produce masks
│   ├── 02_unet_holdout/            U-Net test source      — annotate after training only
│   │   └── legacy_holdout/         20 reserved legacy frames — separate diagnostic DSC (domain-gap check)
│   ├── 03_mask_factory/            Inference pool         — U-Net inference → classifier training
│   └── 04_clf_arena/               Classifier test source — touch only at final evaluation
│
├── masks/                          Hand-drawn masks (1228×922 PNG)
│   ├── 01_annotate_pool/           U-Net training labels (binary, 0=bg / 255=cell)
│   ├── 01_annotate_pool_multiclass/  Experimental 3-class labels (0=bg / 1=trapped cell / 2=other/decoy), same 40 frames
│   ├── 02_unet_holdout/            U-Net test ground truth (annotate after training)
│   └── 02_unet_holdout/legacy_holdout/  Ground truth for the 20 legacy holdout frames
│
└── preprocessed/                   256×256 pipeline-ready data (built by preprocess_frames.py)
    ├── train/
    │   └── 01_annotate_pool/
    │       ├── inputs/             Preprocessed frames  → U-Net training inputs
    │       └── labels/             Resized masks        → U-Net training labels
    ├── evaluate/
    │   ├── 02_unet_holdout/
    │   │   ├── inputs/             Preprocessed frames  → U-Net evaluation inputs
    │   │   └── labels/             Resized masks        → U-Net evaluation labels
    │   └── 02_unet_holdout/legacy_holdout/
    │       ├── inputs/             Preprocessed legacy holdout frames → separate DSC
    │       └── labels/             Resized legacy holdout masks
    ├── inference/
    │   ├── 03_mask_factory/
    │   │   └── inputs/             Preprocessed frames  → U-Net inference → classifier training
    │   └── 04_clf_arena/
    │       └── inputs/             Preprocessed frames  → final classifier evaluation
    └── comparisons/                Pipeline stage figures (for thesis reporting)
```

---

## Pipeline role of each folder

| Folder | ML role | HER2 label | Masks | Touch when |
|--------|---------|-----------|-------|-----------|
| `raw/` | Source data — never modified | In filename | None | Never |
| `frames/01_annotate_pool/` | U-Net training source (220 images) | No | Hand-drawn (required) | Annotate now |
| `frames/02_unet_holdout/` | U-Net test source | No | Hand-drawn (required) | Annotate after training |
| `frames/02_unet_holdout/legacy_holdout/` | Legacy-domain diagnostic test source (20 images) | No | Hand-drawn (required) | Annotate after training |
| `frames/03_mask_factory/` | Classifier training source | Yes (high/low) | U-Net predicted | After U-Net training |
| `frames/04_clf_arena/` | Classifier test source | Yes (high/low) | U-Net predicted | Final evaluation only |
| `masks/01_annotate_pool/` | U-Net training labels | No | Binary, hand-drawn | During annotation |
| `masks/02_unet_holdout/` | U-Net test labels | No | Binary, hand-drawn | After U-Net training |
| `masks/02_unet_holdout/legacy_holdout/` | Legacy-domain diagnostic labels | No | Binary, hand-drawn | After U-Net training |
| `preprocessed/train/.../inputs/` | U-Net training inputs (256×256) | No | — | After annotation + preprocessing |
| `preprocessed/train/.../labels/` | U-Net training labels (256×256) | No | Resized binary | After annotation + preprocessing |
| `preprocessed/evaluate/.../inputs/` | U-Net evaluation inputs (256×256) | No | — | After holdout annotation |
| `preprocessed/evaluate/.../labels/` | U-Net evaluation labels (256×256) | No | Resized binary | After holdout annotation |
| `preprocessed/inference/03_.../inputs/` | Classifier training inputs (256×256) | Yes (high/low) | — | After U-Net training |
| `preprocessed/inference/04_.../inputs/` | Classifier test inputs (256×256) | Yes (high/low) | — | Final evaluation only |
| `preprocessed/comparisons/` | Pipeline stage figures | — | — | On demand (`--compare`) |

### Key rules
- **`04_clf_arena` is a contamination firewall** — these two cells must never be seen by any model or cross-validation fold until the very last evaluation step.
- **`02_unet_holdout` annotation must happen after training** — annotating it before training and accidentally including it in training data would invalidate the U-Net test DSC score. The same applies to `02_unet_holdout/legacy_holdout/`.
- **`legacy_holdout` is evaluated as its own DSC, never merged with the real-cell holdout DSC** — it exists specifically to separate "the segmentation is weak" from "there's a domain gap between legacy microscope images and the real optical-tweezer setup." Blending the two into one number would hide that distinction. It's nested under `frames/02_unet_holdout/` for discoverability only; `preprocess_frames.py`/`evaluate_unet.py` list files non-recursively, so the subfolder is never picked up by a plain `--pool 02_unet_holdout` run.
- **`03_mask_factory` has no hand-drawn masks** — masks come entirely from U-Net inference.  The HER2 label comes from the cell ID in the filename, not from any annotation.
- **Legacy images (`01_annotate_pool`) have no HER2 label** — safe to use as U-Net training data without any risk of classifier contamination.

---

## Raw data

### High HER2-expression videos (`raw/high/`)

| File | Frames | Duration |
|------|--------|----------|
| cell 1 high 50 increments.mp4 | 10,213 | ~464 s |
| cell 2 high 50 increments.mp4 | ~10,000 | ~455 s |
| cell 3 high 50 increments.mp4 | ~10,000 | ~455 s |
| cell 4 high 50 increments.mp4 | ~10,000 | ~455 s |
| cell 6 high 50 increments.mp4 | ~10,000 | ~455 s |
| cell 7 high 50 increments.mp4 | ~10,000 | ~455 s |
| cell 8 high 50 increments.mp4 | ~10,000 | ~455 s |
| cell 9 high 50 increments.mp4 | ~10,000 | ~455 s |
| cell 10 high 50 increments.mp4 | ~10,000 | ~455 s |

### Low HER2-expression videos (`raw/low/` and `raw/`)

| File | Frames | Duration |
|------|--------|----------|
| cell 1 low 50 increments.mp4 (raw/) | ~10,000 | ~455 s |
| cell 2 low 50 increments.mp4 | ~10,000 | ~455 s |
| cell 3 low 50 increments.mp4 | ~10,000 | ~455 s |
| cell 4 low 50 increments.mp4 | ~10,000 | ~455 s |
| cell 5 low 50 increments.mp4 | ~10,000 | ~455 s |
| cell 6 low 50 increments.mp4 | ~10,000 | ~455 s |

**Native resolution:** 2456 × 1842 px @ 22 fps  
**Laser protocol:** 50 incremental steps per cell (~200 frames per step)

### Legacy cancer images (`raw/legacy/20210825_cancer/`)

220 JPGs acquired 2021-08-25 with an earlier microscope setup.  
**Resolution:** 1228 × 922 px  
**HER2 label:** none (unlabeled)  
**Use:** U-Net annotation only — the absence of HER2 labels means these images
can never contaminate the downstream classifier evaluation regardless of how
many are used for segmentation training.

---

## Processed frames (`frames/`)

All extracted frames are saved as **grayscale JPEG at 1228 × 922 px** (half
the native video resolution; identical to the legacy image size).  Videos are
sampled at **every 200th frame** (~1 frame per laser increment), giving ~50
temporally-diverse frames per cell.

### 01_annotate_pool — 220 images (+ 1 combined manifest CSV)

Source for U-Net training masks.  **Annotate every image in this folder** with
a segmentation tool (e.g. LabelMe, Napari) before running `celldform-train-unet`.

| Subset | Count | Notes |
|--------|-------|-------|
| Legacy cancer (2021) | 200 | Filename format: `YYYYMMDDHHMMSS.jpg` (20 of the original 220 legacy JPGs were reserved into `02_unet_holdout/legacy_holdout/` before annotation, not annotated here) |
| Domain-adapt samples (2025) | 20 | Filename prefix: `domain_<her2>_<cell_id>_<frame>.jpg` |

Domain-adapt cells: high-cell1, high-cell2, low-cell2, low-cell3 (5 frames each).  
These are drawn from the same cells as `03_mask_factory` — the U-Net trains on
them, then later runs inference on the full set of frames from those cells.

All 220 frames are annotated in the experimental 3-class labeling in
`masks/01_annotate_pool_multiclass/` (0=background, 1=trapped cell, 2=other/decoy
object), created with `scripts/annotation/annotate.py --multiclass`. This is now the frozen
label scheme for U-Net training (binary annotation stalled at 43/220 and is not
being continued) — see the binary-vs-multiclass ablation in `docs/training.md`
and the decision note in `CLAUDE.md`.

### 02_unet_holdout — 101 images

Held out from U-Net training.  **Annotate these separately** after training is
complete to get an unbiased test Dice score.

| Cell | HER2 | Frames |
|------|------|--------|
| cell9 | high | 50 |
| cell6 | low | 51 |

Filename format: `<her2>_<cell_id>_<frame_index:06d>.jpg`

#### 02_unet_holdout/legacy_holdout — 20 images

A diagnostic subset, nested under `02_unet_holdout/` but evaluated as a **separate**
DSC from the 101 real-cell frames above. Reserved (unannotated) from `01_annotate_pool`
before any of them were labeled, evenly sampled across the 179 legacy frames that were
still unannotated at the time, for temporal spread rather than 20 adjacent frames.

Rationale: at the time this was created, all 41 annotated training masks were
legacy-domain and none were real-cell-domain, while `02_unet_holdout` is 100%
real-cell-domain. A single blended holdout DSC couldn't distinguish "the U-Net
is bad at segmentation" from "the U-Net hasn't crossed the legacy → real-tweezer
domain gap." Evaluating this legacy subset on its own answers that: a *low*
legacy-holdout DSC would point to a genuine segmentation problem; a *high*
legacy-holdout DSC alongside a *low* real-cell-holdout DSC would point specifically
to the domain gap.

Filename format: unchanged legacy format, `YYYYMMDDHHMMSS.jpg`.

### 03_mask_factory — 557 images

The bulk inference pool.  After U-Net training, run inference on every image
here to generate binary cell masks.  Each mask → 15 morphological features
(Stage 4) → one classifier training sample (Stage 5).

| Cell | HER2 | Frames |
|------|------|--------|
| cell1 | high | 52 |
| cell2 | high | 50 |
| cell3 | high | 50 |
| cell4 | high | 50 |
| cell6 | high | 50 |
| cell7 | high | 52 |
| cell8 | high | 51 |
| **High subtotal** | | **355** |
| cell1 | low | 50 |
| cell2 | low | 50 |
| cell3 | low | 51 |
| cell4 | low | 51 |
| **Low subtotal** | | **202** |
| **Total** | | **557** |

Expected classifier training samples: ~355 high, ~202 low.  
Use `class_weight='balanced'` in all classifiers to compensate for the 64:36
class ratio.  Use **cell-level cross-validation** (leave-one-cell-out), not
frame-level, since frames from the same cell are correlated.

### 04_clf_arena — 101 images

Completely held out — **do not use until the classifier is fully trained**.
These two cells were never annotated and never seen by any model during training.

| Cell | HER2 | Frames |
|------|------|--------|
| cell10 | high | 50 |
| cell5 | low | 51 |

---

## Cell split summary

| Cell | HER2 | Role |
|------|------|------|
| cell1 | high | mask_factory + domain_adapt |
| cell2 | high | mask_factory + domain_adapt |
| cell3 | high | mask_factory |
| cell4 | high | mask_factory |
| cell6 | high | mask_factory |
| cell7 | high | mask_factory |
| cell8 | high | mask_factory |
| cell9 | high | **unet_holdout** |
| cell10 | high | **clf_arena** |
| cell1 | low | mask_factory |
| cell2 | low | mask_factory + domain_adapt |
| cell3 | low | mask_factory + domain_adapt |
| cell4 | low | mask_factory |
| cell5 | low | **clf_arena** |
| cell6 | low | **unet_holdout** |

---

## Contamination firewall

The split is designed so that no data seen during model training can bias the
final evaluation metrics:

- **Legacy images → U-Net training only.** No HER2 label exists, so they
  cannot appear as classifier samples.
- **unet_holdout cells** are never used during U-Net training.  Their
  annotated masks give an unbiased segmentation test score.
- **clf_arena cells** are never used during U-Net training, never used to
  generate classifier training features, and never used in cross-validation.
  They are the single honest evaluation of classifier generalisation.

---

## Regenerating frames

To rebuild `frames/` from scratch (e.g. after changing `--every-n`):

```bash
celldform-organize-dataset
# or with overrides:
celldform-organize-dataset --every-n 100 --domain-adapt-n 10
# without wiping existing frames:
celldform-organize-dataset --no-clean
```

All logic lives in `celldform/acquisition/organiser.py`.
