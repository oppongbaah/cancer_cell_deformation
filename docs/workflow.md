# Workflow Guide

End-to-end steps from raw video to trained models and evaluation results.

---

## Overview

```
Stage 0          Stage 1        Annotation     Stage 2           Stage 3          Evaluation
─────────────    ────────────   ────────────   ───────────────   ──────────────   ──────────────
Organise    →   Extract    →   Annotate   →   Preprocess    →   Train U-Net  →   Evaluate
raw videos       frames         masks          (manual)          + Classify        holdout set
```

---

## Step 0 — Organise raw data

Build the four pipeline-ready frame folders from `data/raw/`.

```bash
celldform-organize-dataset
```

Outputs:

| Folder | Images | Purpose |
|--------|--------|---------|
| `data/frames/01_annotate_pool/` | 248 | U-Net training source — annotate these |
| `data/frames/02_unet_holdout/` | 101 | U-Net test — annotate **after** training |
| `data/frames/03_mask_factory/` | 557 | U-Net inference → classifier training |
| `data/frames/04_clf_arena/` | 101 | Final classifier evaluation only |

---

## Step 1 — Annotate training images

Open each image in napari, paint the cell body on the mask layer, and press `S` to save.

```bash
python scripts/annotate.py
```

**Keybindings:**

| Key | Action |
|-----|--------|
| `2` | Paint brush |
| `3` | Eraser |
| `[` / `]` | Smaller / larger brush |
| `Ctrl+Z` | Undo |
| `S` | Save mask and advance |
| `N` | Skip (no save) |
| `P` | Previous image |

The script skips images that already have a saved mask — safe to stop and resume at any time.

Toggle the `enhanced` layer in the napari layer list to see the CLAHE-boosted view for easier boundary identification.

---

## Step 2 — Validate masks

Confirm every image has a clean binary mask before preprocessing.

```bash
python scripts/validate_masks.py
```

Validates the masks that are present — does not require every frame to have a mask. Checks performed per mask found:

- **Readable** — valid image file
- **Binary** — pixel values are strictly 0 or 255
- **Non-empty** — at least one cell pixel (255) present

Exits with code `0` if all present masks pass, `1` if any issues found. Safe to run incrementally while annotation is still in progress.

---

## Step 3 — Preprocess training data

Resize frames to 256×256 and resize masks to match. **This must be run before training.**

```bash
python scripts/preprocess_frames.py --masks
```

Only frames that have a corresponding mask are processed — frames without a mask are skipped automatically. This means you can run this step incrementally as you annotate more images; re-run it after each annotation session to keep the preprocessed data in sync.

Outputs:

```
data/preprocessed/train/01_annotate_pool/
  inputs/    ← 256×256 preprocessed frames  (U-Net training inputs)
  labels/    ← 256×256 binary masks         (U-Net training labels)
```

Optional — generate pipeline stage figures for thesis reporting:

```bash
python scripts/preprocess_frames.py --masks --compare --n-compare 10
```

---

## Step 4 — Train U-Net

```bash
python scripts/train_unet.py --config configs/default.yaml
# or
celldform-train-unet --config configs/default.yaml
```

Training reads from `data/preprocessed/train/01_annotate_pool/`. The dataset is automatically split into train / val / test (75 % / 15 % / 10 %).

Checkpoints are saved to `checkpoints/`:

| File | Contents |
|------|----------|
| `unet_best.pt` | Best checkpoint by validation DSC |
| `unet_last.pt` | Final epoch checkpoint |
| `config.yaml` | Config snapshot for reproducibility |

---

## Step 5 — Evaluate on holdout set

!!! warning
    Complete Steps 1–4 before annotating `02_unet_holdout`. Annotating earlier risks biasing the reported DSC.

```bash
# 5a — annotate holdout images
python scripts/annotate.py --pool 02_unet_holdout

# 5b — validate holdout masks
python scripts/validate_masks.py --pool 02_unet_holdout

# 5c — preprocess holdout data
python scripts/preprocess_frames.py --masks --pool 02_unet_holdout

# 5d — evaluate and report DSC
python scripts/evaluate_unet.py --checkpoint checkpoints/unet_best.pt
```

The DSC reported by `evaluate_unet.py` is the **thesis-reportable segmentation score** — these cells were never seen during training or hyperparameter tuning.

---

## Step 6 — Run U-Net inference on mask factory

Generate predicted masks for the classifier training pool.

```bash
python scripts/preprocess_frames.py --pool 03_mask_factory

celldform-infer data/preprocessed/inference/03_mask_factory/inputs \
    --unet-ckpt checkpoints/unet_best.pt \
    --output-dir data/predicted_masks/03_mask_factory
```

---

## Step 7 — Train classifier

Feature extraction and classifier training (SVM / RF / DT) using the predicted masks from Step 6.

```bash
python scripts/infer.py \
    --unet-ckpt checkpoints/unet_best.pt \
    --classifier-ckpt checkpoints/svm.pkl
```

!!! note "Cell-level cross-validation"
    Use leave-one-cell-out or group-k-fold with `groups=cell_id`. Never use frame-level CV — frames from the same cell are correlated.

---

## Step 8 — Final classifier evaluation

!!! danger "04_clf_arena firewall"
    Do not preprocess or touch `04_clf_arena` until the classifier is fully trained and frozen.

```bash
python scripts/preprocess_frames.py --pool 04_clf_arena

celldform-infer data/preprocessed/inference/04_clf_arena/inputs \
    --unet-ckpt checkpoints/unet_best.pt \
    --classifier-ckpt checkpoints/svm.pkl \
    --output-dir results/clf_arena
```
