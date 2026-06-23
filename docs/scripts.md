# Scripts Reference

All scripts live under `scripts/` and are thin wrappers or standalone tools. They can be run directly with `python scripts/<name>.py`.

---

## `annotate.py` — napari annotation tool

Opens images from an annotation pool one at a time in napari. Paint the cell mask on the Labels layer, then press `S` to save and advance.

```bash
python scripts/annotate.py                             # 01_annotate_pool (default)
python scripts/annotate.py --pool 02_unet_holdout      # holdout set
python scripts/annotate.py --redo                      # re-annotate already-masked images
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pool` | `01_annotate_pool` | Pool folder name under `data/frames/` |
| `--image-dir` | `data/frames/<pool>` | Override source image directory |
| `--mask-dir` | `data/masks/<pool>` | Override mask output directory |
| `--redo` | off | Re-annotate images that already have masks |

**Keybindings:**

| Key | Action |
|-----|--------|
| `2` | Paint brush |
| `3` | Eraser |
| `[` / `]` | Smaller / larger brush |
| `Ctrl+Z` | Undo last stroke |
| `S` | Save mask → next image |
| `N` | Skip (no save) → next image |
| `P` | Previous image |

Masks are saved as uint8 PNGs with values 0 (background) / 255 (cell). Already-annotated images are skipped by default.

---

## `validate_masks.py` — mask validation gate

Validates the masks that are present in a pool. Iterates over masks in `data/masks/<pool>/` directly — does not require every frame to have a mask. Use as a pre-training gate.

```bash
python scripts/validate_masks.py                          # 01_annotate_pool
python scripts/validate_masks.py --pool 02_unet_holdout
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pool` | `01_annotate_pool` | Pool folder name |
| `--mask-dir` | `data/masks/<pool>` | Override mask directory |

**Checks performed on each mask that exists:**

| Check | Fail condition |
|-------|---------------|
| Readable | File cannot be decoded by OpenCV |
| Binary | Pixel values other than 0 or 255 |
| Non-empty | No cell pixels (all zeros) |

Exits with code `0` if all present masks pass, `1` if any issues are found. Reports `Valid: N/N (100%)` against the masks found, not against the full frame pool.

---

## `preprocess_frames.py` — manual preprocessing

Applies `PreprocessingPipeline` to frames and saves 256×256 PNGs. **Required before training.** Optionally resizes masks to match.

!!! note "Mask-gated processing"
    When `--masks` is passed, only frames that have a corresponding mask in `data/masks/<pool>/` are processed. Frames without a mask are skipped. The summary line reports how many were skipped.

```bash
python scripts/preprocess_frames.py --masks                        # train pool
python scripts/preprocess_frames.py --masks --pool 02_unet_holdout # evaluate pool
python scripts/preprocess_frames.py --pool 03_mask_factory          # inference pool
python scripts/preprocess_frames.py --pool 04_clf_arena             # inference pool
python scripts/preprocess_frames.py --masks --compare --n-compare 10
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pool` | `01_annotate_pool` | Pool folder name |
| `--stage` | auto-detected | `train`, `evaluate`, or `inference` |
| `--masks` | off | Also resize and save binary masks |
| `--compare` | off | Save pipeline stage comparison figures |
| `--n-compare` | `5` | Number of images to generate figures for |
| `--image-dir` | `data/frames/<pool>` | Override source directory |
| `--output-dir` | `data/preprocessed/<stage>/<pool>/inputs` | Override frame output |
| `--mask-dir` | `data/masks/<pool>` | Override source mask directory |
| `--mask-output-dir` | `data/preprocessed/<stage>/<pool>/labels` | Override mask output |

**Stage auto-detection:**

| Pool | Stage |
|------|-------|
| `01_annotate_pool` | `train` |
| `02_unet_holdout` | `evaluate` |
| `03_mask_factory` | `inference` |
| `04_clf_arena` | `inference` |

---

## `train_unet.py` — U-Net training

Trains the U-Net on preprocessed data. Expects already-preprocessed 256×256 PNGs — no preprocessing is applied internally.

```bash
python scripts/train_unet.py --config configs/default.yaml
python scripts/train_unet.py --config configs/default.yaml --seed 123
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `configs/default.yaml` | YAML config file |
| `--seed` | config value | Random seed override |

The script reads `data.frames_dir` and `data.masks_dir` from the config. Pairs frames and masks by filename stem. Saves best and last checkpoints to `checkpoint_dir`.

---

## `evaluate_unet.py` — holdout evaluation

Evaluates a trained U-Net on `02_unet_holdout` and reports the thesis-reportable segmentation metrics.

```bash
python scripts/evaluate_unet.py --checkpoint checkpoints/unet_best.pt
python scripts/evaluate_unet.py --checkpoint checkpoints/unet_best.pt --pool 02_unet_holdout
```

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | required | Path to trained `.pt` checkpoint |
| `--pool` | `02_unet_holdout` | Pool to evaluate |
| `--config` | `configs/default.yaml` | YAML config |
| `--batch-size` | `8` | Inference batch size |

**Metrics reported:** DSC, IoU, precision, recall, accuracy.

---

## `organize_dataset.py` — dataset organiser

Thin wrapper around `celldform.acquisition.organiser.organise()`.

```bash
python scripts/organize_dataset.py
python scripts/organize_dataset.py --every-n 100 --domain-adapt-n 10
```

Equivalent to `celldform-organize-dataset`. See [CLI Reference](cli.md).

---

## `extract_frames.py` — single video extraction

Thin wrapper around `FrameExtractor` for a single video.

```bash
python scripts/extract_frames.py --video data/raw/high/"cell 1 high 50 increments.mp4"
```

Equivalent to `celldform-extract-frames`. See [CLI Reference](cli.md).

---

## `infer.py` — full inference pipeline

Runs stages 1–6 on a video or directory of preprocessed images.

```bash
python scripts/infer.py \
    --video data/raw/high/"cell 1 high 50 increments.mp4" \
    --unet-ckpt checkpoints/unet_best.pt \
    --classifier-ckpt checkpoints/svm.pkl
```

Equivalent to `celldform-infer`. See [CLI Reference](cli.md).
