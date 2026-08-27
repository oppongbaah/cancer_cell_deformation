# Scripts Reference

All scripts live under `scripts/` and are thin wrappers or standalone tools. They can be run directly with `python scripts/<name>.py`.

---

## `annotate.py` — napari annotation tool

Opens every image in an annotation pool, in order, in napari — including already-annotated ones, so `N`/`B` can freely browse and correct earlier work rather than only stepping through what's left. The viewer opens on the first not-yet-annotated image (not the start of the pool); revisiting an annotated frame loads its saved mask instead of a blank one, and the window title shows `[saved]` for frames that already have a mask on disk. Paint the cell mask on the Labels layer, then press `S` to save and advance.

```bash
python scripts/annotate.py                             # 01_annotate_pool (default)
python scripts/annotate.py --pool 02_unet_holdout      # holdout set
python scripts/annotate.py --multiclass                # experimental 3-class labels (same 40 images)
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pool` | `01_annotate_pool` | Pool folder name under `data/frames/` |
| `--image-dir` | `data/frames/<pool>` | Override source image directory |
| `--mask-dir` | `data/masks/<pool>` | Override mask output directory |
| `--multiclass` | off | Annotate 3 classes (0=background, 1=trapped cell, 2=other/decoy object) instead of binary. Masks default to `data/masks/<pool>_multiclass/` and are seeded from the existing binary mask (as label 1) when one exists, so you only need to paint decoy objects |

**Keybindings:**

| Key | Action |
|-----|--------|
| `2` | Paint brush |
| `3` | Eraser |
| `[` / `]` | Smaller / larger brush |
| `Ctrl+Z` | Undo last stroke |
| `S` | Save mask → next image |
| `N` | Skip (no save) → next image |
| `B` | Previous image |
| `P` | Toggle paint mode (napari default — left alone) |
| `-` / `=` | Step the active paint label down/up (multiclass mode) |

!!! note "P vs. B"
    `P` used to mean "previous image" but napari's Labels layer binds it by default to "toggle preserve labels," which shadows any viewer-level binding while the mask layer is active. `B` is reclaimed for "previous" instead; `P` is left as napari's default paint-mode toggle.

Binary mode: masks are saved as uint8 PNGs with values 0 (background) / 255 (cell). Multiclass mode: masks are saved with raw label values 0/1/2 (not binarized) — use `scripts/preview_masks.py` to view them, since label values 1–2 are visually indistinguishable from background in a plain image viewer.

---

## `preview_masks.py` — colorized mask preview

Renders saved masks overlaid on their raw frames as real color PNGs — no napari needed. Multiclass masks (label ids 0/1/2) are nearly invisible in a plain image viewer; this renders label 1 (trapped cell) as green and label 2 (other/decoy) as red. Legacy binary masks (0/255) also work — foreground renders green.

```bash
python scripts/preview_masks.py                                    # 01_annotate_pool_multiclass (default)
python scripts/preview_masks.py --pool 01_annotate_pool            # binary masks
python scripts/preview_masks.py --alpha 0.5 --limit 10
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pool` | `01_annotate_pool_multiclass` | Pool folder name under `data/masks/` (raw frame pool is inferred by stripping `_multiclass`) |
| `--image-dir` | `data/frames/<inferred pool>` | Override raw frame directory |
| `--mask-dir` | `data/masks/<pool>` | Override mask directory |
| `--output-dir` | `data/masks/<pool>_preview` | Override output directory |
| `--alpha` | `0.45` | Overlay blend strength (0–1) |
| `--limit` | none | Only preview the first N masks |

---

## `validate_masks.py` — mask validation gate

Validates the masks that are present in a pool, and (by default) that every frame in the pool has one. Use as a pre-training gate.

```bash
python scripts/validate_masks.py                          # 01_annotate_pool
python scripts/validate_masks.py --pool 02_unet_holdout
python scripts/validate_masks.py --multiclass              # validates 01_annotate_pool_multiclass
python scripts/validate_masks.py --no-completeness         # skip the "every frame has a mask" check
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pool` | `01_annotate_pool` | Pool folder name (used for both `data/frames/<pool>` and `data/masks/<pool>`) |
| `--mask-dir` | `data/masks/<pool>` (or `<pool>_multiclass` with `--multiclass`) | Override mask directory |
| `--image-dir` | `data/frames/<pool>` | Override frame directory (completeness check) |
| `--no-completeness` | off | Skip the "every pool image has a mask" check entirely |
| `--multiclass` | off | Validate 3-class masks (0/1/2) instead of binary (0/255) |

**Checks performed:**

| Check | Fail condition (binary) | Fail condition (`--multiclass`) |
|-------|---------------|---------------|
| Present | Frame in the pool has no corresponding mask file (skippable with `--no-completeness`) | same |
| Readable | File cannot be decoded by OpenCV | same |
| Binary/label-valid | Pixel values other than 0 or 255 | Pixel values other than 0, 1, or 2 |
| Non-empty | No foreground pixels (255) at all | No foreground pixels of **any** class (neither label 1 nor label 2) — a mask with only a decoy object (label 2) and no trapped cell is a valid deliberate negative and passes |

Exits with code `0` if all checks pass, `1` if any issues are found. Reports `Valid: N/N (100%)` against the full pool count (or the mask count with `--no-completeness`).

---

## `preprocess_frames.py` — manual preprocessing

Applies `PreprocessingPipeline` to frames and saves PNGs (256×256 by default). **Required before training.** Optionally resizes masks to match.

!!! note "Mask-gated processing"
    When `--masks` is passed, only frames that have a corresponding mask in `data/masks/<pool>/` are processed. Frames without a mask are skipped. The summary line reports how many were skipped.

```bash
python scripts/preprocess_frames.py --masks                        # train pool
python scripts/preprocess_frames.py --masks --pool 02_unet_holdout # evaluate pool
python scripts/preprocess_frames.py --pool 03_mask_factory          # inference pool
python scripts/preprocess_frames.py --pool 04_clf_arena             # inference pool
python scripts/preprocess_frames.py --masks --compare --n-compare 10
python scripts/preprocess_frames.py --masks --target-size 128 128   # override output resolution
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pool` | `01_annotate_pool` | Pool folder name |
| `--stage` | auto-detected | `train`, `evaluate`, or `inference` |
| `--masks` | off | Also resize and save masks (binary or label, nearest-neighbour) |
| `--compare` | off | Save pipeline stage comparison figures |
| `--n-compare` | `5` | Number of images to generate figures for |
| `--image-dir` | `data/frames/<pool>` | Override source directory |
| `--output-dir` | `data/preprocessed/<stage>/<pool>/inputs` | Override frame output |
| `--mask-dir` | `data/masks/<pool>` | Override source mask directory |
| `--mask-output-dir` | `data/preprocessed/<stage>/<pool>/labels` | Override mask output |
| `--target-size` | `256 256` | Output spatial size `H W`. Both dims should stay divisible by 16 (U-Net's 4 downsampling steps) — the U-Net is fully convolutional, so no model code changes are needed, but whatever size is chosen here is what `CellDataset` will load during training |

**Stage auto-detection:**

| Pool | Stage |
|------|-------|
| `01_annotate_pool` | `train` |
| `02_unet_holdout` | `evaluate` |
| `03_mask_factory` | `inference` |
| `04_clf_arena` | `inference` |

---

## `train_unet.py` — U-Net training

Trains the U-Net on preprocessed data. Expects already-preprocessed PNGs at whatever size was used for preprocessing (256×256 by default) — no preprocessing is applied internally. Reads `unet.out_channels` from the config: `1` (default) trains binary segmentation; `>1` trains multiclass (`CellDataset` loads raw integer label masks instead of binarising, and `_DiceCELoss` replaces `_DiceBCELoss`).

```bash
python scripts/train_unet.py --config configs/multiclass_experiment_256.yaml   # frozen scheme
python scripts/train_unet.py --config configs/default.yaml --seed 123
python scripts/train_unet.py --config configs/multiclass_experiment_256.yaml --tag auto
python scripts/train_unet.py --config configs/multiclass_experiment_256.yaml --domain-oversample-weight 8.0
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `configs/default.yaml` | YAML config file |
| `--seed` | config value | Random seed override |
| `--domain-oversample-weight` | config value (`1.0` if unset) | Sampling weight for `domain_`-prefixed (real-cell) train frames vs. legacy frames — builds a `WeightedRandomSampler`. `1.0` = off/uniform; pass `1.0` explicitly to force it off even if the config enables it. See [Training Results — Domain Gap](training.md#domain-gap) (tested at weight=8.0: no improvement on the small real-cell test sample available so far). |
| `--tag` | none | Suffix appended to the saved training-curve filename (e.g. `--tag auto` → `unet_training_curves_v6_auto.png`), for distinguishing runs at a glance |

The script reads `data.frames_dir` and `data.masks_dir` from the config. Pairs frames and masks by filename stem. Saves best and last checkpoints to `checkpoint_dir`. After training, it reloads `unet_best.pt` (not necessarily the last epoch's weights, which can be past the peak validation DSC on small datasets) before running final evaluation and plotting the training curves.

---

## `evaluate_unet.py` — holdout evaluation

Evaluates a trained U-Net on `02_unet_holdout` and reports the thesis-reportable segmentation metrics.

```bash
python scripts/evaluate_unet.py --checkpoint checkpoints/multiclass_experiment_256/unet_best.pt \
    --config configs/multiclass_experiment_256.yaml   # frozen scheme — --config must match out_channels
python scripts/evaluate_unet.py --checkpoint checkpoints/unet_best.pt --pool 02_unet_holdout  # binary baseline
```

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | required | Path to trained `.pt` checkpoint |
| `--pool` | `02_unet_holdout` | Pool to evaluate |
| `--config` | `configs/default.yaml` | YAML config |
| `--batch-size` | `8` | Inference batch size |

**Metrics reported:** DSC, IoU, precision, recall, accuracy. When the checkpoint's config has `unet.out_channels > 1`, results are reported per class (`background`, `trapped_cell`, ...) instead of as a single flat table.

---

## `organize_dataset.py` — dataset organiser

Thin wrapper around `celldform.acquisition.organiser.organise()`.

```bash
python scripts/organize_dataset.py
python scripts/organize_dataset.py --every-n 100 --domain-adapt-n 10
python scripts/organize_dataset.py --no-clean --domain-adapt-n 20   # expand domain-adapt only, don't wipe data/frames/ — run 2026-08-19
```

Equivalent to `celldform-organize-dataset`. See [CLI Reference](cli.md). The `domain_adapt` cell list in `celldform/acquisition/organiser.py` covers all 11 `03_mask_factory` cells (as of 2026-08-10, extended from 4); `--domain-adapt-n` was raised 5→20/cell on 2026-08-19 — see [Training Results — Domain Gap](training.md#domain-gap) for why. Re-running with a larger N doesn't lose frames/masks from an earlier N — every `domain_*` file already on disk is kept and re-enumerated into the manifest, and legacy frames reserved in `02_unet_holdout/legacy_holdout/` are always excluded from the copy regardless of N (a contamination bug where this wasn't true was found and fixed the same day).

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
