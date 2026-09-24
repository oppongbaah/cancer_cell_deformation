# Workflow Guide

## Experiment rules (from 2026-09-17)

These came out of measuring run-to-run variance for the first time. They are not
style preferences; two results have already been retracted for breaking them.

1. **One seed (42) while iterating**, `split_seed` pinned at 42 too. Three seeds on
   **both arms** before anything is reported as a result.
2. **A single-seed delta below ~0.16 is not a result.** Log it as *screened,
   inconclusive*. `scripts/training/compare_postswap.py` prints this verdict for you.
3. **Change one variable per run.** If a non-experimental setting moves — schedule,
   split, width, pool — re-run the baseline under it before quoting any delta.
4. **Check the split log line** at the top of every run. `high_cell10` is 50 frames
   against a ~45-frame validation target and lands in validation on **13% of seeds**;
   when it does, that run trained without its only small cell and is void.
5. **Prefer levers big enough to see.** The two effects that cleared the noise were a
   data change (+0.245 on `high-9` after the cell exchange) and a resolution change
   (+0.048 for 128→256). No hyperparameter has come close.

---

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

Build the three pipeline-ready frame folders from `data/raw/`. (There used to be a fourth, `04_clf_arena` — collapsed into `03_mask_factory` on 2026-09-03; see [Data Layout](data.md) if you run across a reference to it elsewhere.)

```bash
celldform-organize-dataset
```

Outputs — see [Data Layout](data.md) for the full breakdown, including the development-set/blind-test split inside `02_unet_holdout`:

| Folder | Images | Purpose |
|--------|--------|---------|
| `data/frames/01_annotate_pool/` | 445 | U-Net training source — **fully annotated as of 2026-09-10** (445/445; `low_cell5` was the last outstanding cell) |
| `data/frames/02_unet_holdout/` | 101 | Development set (`high_cell9` + `low_cell6`) — evaluated repeatedly to guide tuning, fully annotated |
| `data/frames/02_unet_holdout/final_test/` | 50 | Blind test (`high_cell10`) — reserved, deliberately unannotated, see [Data Layout — Cell roles](data.md#cell-roles) |
| `data/frames/03_mask_factory/` | 608 | U-Net inference → classifier training (12 cells; `high_cell10` removed 2026-09-10 — it is the blind test and must not reach classifier training) |

!!! warning "The current data on disk doesn't exactly match what a fresh `celldform-organize-dataset` run produces"
    The 2026-09-04 cell reallocation (`high_cell10` → blind test, `low_cell5` stays training) was done by moving files by hand, not by rerunning the organiser — see [Data Layout](data.md#regenerating-frames) before running `--no-clean` against the current pool.

---

## Step 1 — Annotate training images

Open each image in napari, paint the cell body on the mask layer, and press `S` to save.

```bash
python scripts/annotation/annotate.py
python scripts/annotation/annotate.py --multiclass   # experimental 3-class labels (see docs/scripts.md)
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
python scripts/annotation/validate_masks.py
python scripts/annotation/validate_masks.py --multiclass   # validates 0/1/2 label masks instead
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
python scripts/data/preprocess_frames.py --masks
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
python scripts/data/preprocess_frames.py --masks --compare --n-compare 10
```

---

## Step 4 — Train U-Net

```bash
python scripts/training/train_unet.py --config configs/default.yaml
# or
celldform-train-unet --config configs/default.yaml
```

Training reads from `data/preprocessed/train/01_annotate_pool/`. The dataset is automatically split into train / val (85 % / 15 % by default, `data.val_split` in the config) — val is a monitoring split only (checkpoint selection, early stopping), not the thesis-reportable score. That comes from `scripts/training/evaluate_unet.py` against the external `02_unet_holdout` pool, which is why there's no third "test" split here — it would be redundant with that external evaluation (removed 2026-09-04).

Checkpoints are saved to `checkpoints/`:

| File | Contents |
|------|----------|
| `unet_best.pt` | Best checkpoint by validation DSC |
| `unet_last.pt` | Final epoch checkpoint |
| `config.yaml` | Config snapshot for reproducibility |

!!! note "Which config to use"
    `configs/multiclass_experiment_256.yaml` is the frozen label scheme/resolution (decided 2026-08-10) and the config actually being trained on — its checkpoint lives at `checkpoints/multiclass_experiment_256/unet_best.pt`, **not** `checkpoints/unet_best.pt` (that path is `configs/default.yaml`'s binary/256×256 checkpoint, kept as a baseline but not the primary one). `configs/binary_experiment_{128,256}.yaml` and `configs/multiclass_experiment.yaml` (128×128) are the historical ablation that made that decision — see [Training Results](training.md) for the comparison and the domain-gap finding since. `configs/multiclass_experiment_256_domain_oversample.yaml` is an A/B variant (`training.domain_oversample_weight: 8.0`, own checkpoint dir) — tested, did not help, kept for the record.

---

## Step 5 — Evaluate on the development set

`02_unet_holdout`'s top level (`high_cell9` + `low_cell6`) is fully annotated as of 2026-09-03 and safe to evaluate against freely — see [Data Layout](data.md#three-tier-evaluation-terminology) for why this is called the **development set**, not a test set: every score from it is expected to shape a tuning decision, unlike the reserved blind test.

```bash
# 5a — validate development-set masks (already annotated — this just confirms nothing's broken)
python scripts/annotation/validate_masks.py --pool 02_unet_holdout --multiclass

# 5b — preprocess development-set data
python scripts/data/preprocess_frames.py --masks --pool 02_unet_holdout \
    --mask-dir data/masks/02_unet_holdout_multiclass \
    --output-dir data/preprocessed/evaluate/02_unet_holdout/inputs \
    --mask-output-dir data/preprocessed/evaluate/02_unet_holdout/labels

# 5c — evaluate: pooled, and per-cell (always check both — see the warning below)
python scripts/training/evaluate_unet.py --checkpoint checkpoints/multiclass_experiment_256_bf64_scale_cos/unet_best.pt \
    --config checkpoints/multiclass_experiment_256_bf64_scale_cos/config.yaml
python scripts/training/evaluate_unet.py --checkpoint checkpoints/multiclass_experiment_256_bf64_scale_cos/unet_best.pt \
    --config checkpoints/multiclass_experiment_256_bf64_scale_cos/config.yaml --filter high_cell9
python scripts/training/evaluate_unet.py --checkpoint checkpoints/multiclass_experiment_256_bf64_scale_cos/unet_best.pt \
    --config checkpoints/multiclass_experiment_256_bf64_scale_cos/config.yaml --filter low_cell6
```

Every run appends its full-precision metrics to `results/preswap/preswap_eval_log.jsonl` (`--no-save-results` to skip, `--results-log` to redirect) — that file, not a number copied into a doc or terminal scrollback, is the durable record.

!!! danger "Always check the per-cell breakdown, never trust the pooled number alone"
    With only two development cells, the pooled score is a frame-weighted blend, not an average, and it has hidden a real regression before: `bf64_scale` looked like a win pooled (0.525→0.541) but actually made `high_cell9` *worse* (0.428→0.368) while making `low_cell6` much better — the `--filter` flags above exist specifically so this doesn't get missed again. See [Training Results](training.md#the-2026-09-full-evaluation-sweep) for the full story.

!!! note "This is not the thesis-reportable blind-test number"
    The development set is evaluated repeatedly to guide tuning — that's the opposite of what a blind test needs. The one honest, one-shot generalization number comes from `02_unet_holdout/final_test/` (`high_cell10`), reserved and deliberately unannotated until the recipe is settled — see [Data Layout — Cell roles](data.md#cell-roles) for the full protocol.

!!! note "Optional — legacy-domain diagnostic"
    `data/frames/02_unet_holdout/legacy_holdout/` holds 20 legacy frames reserved separately for a second, independent DSC — it isolates "the segmentation is weak" from "there's a domain gap between legacy microscope images and the real optical-tweezer setup." Repeat 5a–5c with `--image-dir data/frames/02_unet_holdout/legacy_holdout --mask-dir data/masks/02_unet_holdout/legacy_holdout` and separate `--output-dir`/`--mask-output-dir` for preprocessing. Report it as its own number — do not merge it with the DSC above.

---

## Step 6 — Run U-Net inference on mask factory

Not yet started as of 2026-09-04 — the U-Net recipe is still being tuned (see [Training Results — Next Steps](training.md#next-steps)). This step generates predicted masks for the classifier training pool once it is.

```bash
python scripts/data/preprocess_frames.py --pool 03_mask_factory

celldform-infer data/preprocessed/inference/03_mask_factory/inputs \
    --unet-ckpt checkpoints/multiclass_experiment_256_bf64_scale_cos/unet_best.pt \
    --output-dir data/predicted_masks/03_mask_factory
```

!!! warning "`checkpoints/unet_best.pt` is not the right path"
    That default path belongs to `configs/default.yaml` (binary, 256×256) — not the frozen multiclass pipeline actually being trained. Use whatever checkpoint [Training Results](training.md) currently names as best (`checkpoints/multiclass_experiment_256_bf64_scale_cos/unet_best.pt` as of 2026-09-10), and update this doc when that changes.

!!! note "`high_cell10` is not in this pool"
    It was removed on 2026-09-10 — as the U-Net's blind test it must not also feed classifier training. The pool is 608 frames across 12 cells.

---

## Step 7 — Train classifier

Feature extraction and classifier training (SVM / RF / DT) using the predicted masks from Step 6.

```bash
python scripts/inference/infer.py \
    --unet-ckpt checkpoints/multiclass_experiment_256_bf64_scale_cos/unet_best.pt \
    --classifier-ckpt checkpoints/svm.pkl
```

!!! note "Cell-level cross-validation"
    Use leave-one-cell-out or group-k-fold with `groups=cell_id`. Never use frame-level CV — frames from the same cell are correlated.

---

## Step 8 — Final classifier evaluation

There is no separate classifier-only firewall pool anymore — `04_clf_arena` was retired 2026-09-03 when its role merged into `02_unet_holdout` (see [Data Layout](data.md)). The same three-tier discipline from Step 5 applies here too: use the development set (`high_cell9`/`low_cell6`) freely while building the classifier, and evaluate on the blind test (`high_cell10`) exactly once, at the same moment the U-Net's own blind test happens — not before.

```bash
python scripts/data/preprocess_frames.py --image-dir data/frames/02_unet_holdout/final_test \
    --output-dir data/preprocessed/inference/final_test/inputs

celldform-infer data/preprocessed/inference/final_test/inputs \
    --unet-ckpt checkpoints/multiclass_experiment_256_bf64_scale_cos/unet_best.pt \
    --classifier-ckpt checkpoints/svm.pkl \
    --output-dir results/final_test
```
