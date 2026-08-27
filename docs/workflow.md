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
| `data/frames/01_annotate_pool/` | 428 | U-Net training source — annotate these. 220 fully annotated as of 2026-08-10 (multiclass scheme frozen); grown by 208 domain-adapt frames on 2026-08-19 to close the domain gap (see [Training Results](training.md#domain-gap)) — those 208 are the current annotation priority |
| `data/frames/02_unet_holdout/` | 101 | U-Net test — annotate **after** training |
| `data/frames/03_mask_factory/` | 557 | U-Net inference → classifier training |
| `data/frames/04_clf_arena/` | 101 | Final classifier evaluation only |

---

## Step 1 — Annotate training images

Open each image in napari, paint the cell body on the mask layer, and press `S` to save.

```bash
python scripts/annotate.py
python scripts/annotate.py --multiclass   # experimental 3-class labels (see docs/scripts.md)
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
| `B` | Previous image |
| `P` | Toggle paint mode (napari default) |

The viewer loads every image in the pool, in order — including already-annotated ones, so you can browse back with `B` to review or fix earlier masks — but opens on the first not-yet-annotated image, so it's still safe to stop and resume at any time.

Toggle the `enhanced` layer in the napari layer list to see the CLAHE-boosted view for easier boundary identification.

---

## Step 2 — Validate masks

Confirm every image has a clean binary mask before preprocessing.

```bash
python scripts/validate_masks.py
python scripts/validate_masks.py --multiclass   # validates 0/1/2 label masks instead
```

Validates the masks that are present — does not require every frame to have a mask. Checks performed per mask found:

- **Readable** — valid image file
- **Binary** — pixel values are strictly 0 or 255 (`--multiclass`: 0, 1, or 2)
- **Non-empty** — at least one cell pixel (255) present (`--multiclass`: at least one trapped-cell pixel, label 1)

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

!!! note "Which config to use"
    `configs/multiclass_experiment_256.yaml` is the frozen label scheme/resolution (decided 2026-08-10) and the config actually being trained on — its checkpoint lives at `checkpoints/multiclass_experiment_256/unet_best.pt`, **not** `checkpoints/unet_best.pt` (that path is `configs/default.yaml`'s binary/256×256 checkpoint, kept as a baseline but not the primary one). `configs/binary_experiment_{128,256}.yaml` and `configs/multiclass_experiment.yaml` (128×128) are the historical ablation that made that decision — see [Training Results](training.md) for the comparison and the domain-gap finding since. `configs/multiclass_experiment_256_domain_oversample.yaml` is an A/B variant (`training.domain_oversample_weight: 8.0`, own checkpoint dir) — tested, did not help, kept for the record.

---

## Step 5 — Evaluate on holdout set

!!! warning
    Complete Steps 1–4 before annotating `02_unet_holdout`. Annotating earlier risks biasing the reported DSC.

!!! danger "Deferred until the domain-adapt retrain looks solid (as of 2026-08-19)"
    `02_unet_holdout` is the one-shot, thesis-reportable number, so it's deliberately **not** the next step right now. The current priority is annotating the 208 new domain-adapt frames in `01_annotate_pool` (Steps 1–4 above, `--multiclass`) and checking the legacy-vs-domain-adapt DSC split on the internal validation/test data — the same breakdown used in [Domain Gap](training.md#domain-gap) — to see whether growing the pool closed the gap. Only come back to this step once that number looks solid; see [Training Results — Next Steps](training.md#next-steps).

```bash
# 5a — annotate holdout images
python scripts/annotate.py --pool 02_unet_holdout

# 5b — validate holdout masks
python scripts/validate_masks.py --pool 02_unet_holdout

# 5c — preprocess holdout data
python scripts/preprocess_frames.py --masks --pool 02_unet_holdout

# 5d — evaluate and report DSC (--config must match the checkpoint's architecture —
#      multiclass_experiment_256.yaml is the frozen scheme, out_channels=3)
python scripts/evaluate_unet.py --checkpoint checkpoints/multiclass_experiment_256/unet_best.pt \
    --config configs/multiclass_experiment_256.yaml
```

The DSC reported by `evaluate_unet.py` is the **thesis-reportable segmentation score** — these cells were never seen during training or hyperparameter tuning.

!!! warning "Expect a lower number than the 0.943 training-time DSC"
    Training validation was measured on a legacy-dominated split; on the 5 held-out real-cell frames available so far, DSC drops to 0.538 (recall 0.386). This step's result — on the full 101-frame real-cell holdout — is expected to land well below 0.943 and is the number that should actually drive whether the U-Net gets frozen. See [Domain Gap](training.md#domain-gap) for the full finding.

!!! note "Optional — legacy-domain diagnostic"
    `data/frames/02_unet_holdout/legacy_holdout/` holds 20 legacy frames reserved separately for a second, independent DSC — it isolates "the segmentation is weak" from "there's a domain gap between legacy microscope images and the real optical-tweezer setup." Repeat 5a–5d with `--image-dir data/frames/02_unet_holdout/legacy_holdout --mask-dir data/masks/02_unet_holdout/legacy_holdout` and separate `--output-dir`/`--mask-output-dir` for preprocessing. Report it as its own number — do not merge it with the DSC above.

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
