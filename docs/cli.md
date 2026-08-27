# CLI Reference

All commands are registered in `pyproject.toml` and available after `pip install -e ".[dev]"`.

---

## `celldform-organize-dataset`

Build pipeline-ready frame folders from `data/raw/`.

```bash
celldform-organize-dataset
celldform-organize-dataset --raw-dir data/raw --every-n 100
celldform-organize-dataset --domain-adapt-n 10
celldform-organize-dataset --no-clean
celldform-organize-dataset --no-clean --domain-adapt-n 20   # run 2026-08-19, see Training Results — Domain Gap
celldform-organize-dataset --no-clean --config configs/multiclass_experiment_256.yaml  # same run, N from acquisition.domain_adapt_n
```

| Option | Default | Description |
|--------|---------|-------------|
| `--raw-dir` | `data/raw` | Root of raw data |
| `--frames-dir` | `data/frames` | Output root for frame folders |
| `--every-n` | `200` | Sample every N-th frame per video |
| `--config` | – | Optional celldform YAML config; its `acquisition.domain_adapt_n` supplies the `--domain-adapt-n` default |
| `--domain-adapt-n` | `5` (or `acquisition.domain_adapt_n` when `--config` is given) | Frames per cell copied into `01_annotate_pool` across all 11 `domain_adapt` cells (raised to 20 as of 2026-08-19; recorded as `acquisition.domain_adapt_n: 20` in `configs/default.yaml` and `configs/multiclass_experiment_256.yaml`). An explicit flag always beats the config value. Legacy frames reserved in `02_unet_holdout/legacy_holdout/` are always excluded from the pool, regardless of N. |
| `--no-clean` | off | Skip wiping `--frames-dir` before running |

---

## `celldform-extract-frames`

Extract frames from a single video.

```bash
celldform-extract-frames --video data/raw/high/"cell 1 high 50 increments.mp4"
celldform-extract-frames --video experiment.mp4 --every-n 5
celldform-extract-frames --video experiment.mp4 --target-fps 5.0
celldform-extract-frames --video experiment.mp4 --change-threshold 5.0
```

| Option | Default | Description |
|--------|---------|-------------|
| `--video` | required | Input video path |
| `--config` | `configs/default.yaml` | YAML config |
| `--output-dir` | config `data.frames_dir` | Output directory |
| `--every-n` | config value | Keep every N-th frame |
| `--target-fps` | config value | Resample to this rate |
| `--max-frames` | config value | Hard upper limit on saved frames |
| `--target-size W H` | config value | Resize each frame |
| `--format` | `png` | Output format: `png` or `tiff` |
| `--change-threshold` | disabled | Min centroid displacement (px) to save a frame |
| `--channel-width` | `0.30` | Channel strip width as fraction of image width |
| `--dark-percentile` | `20.0` | Bottom X% of intensities treated as cell candidates |
| `--min-circularity` | `0.60` | Minimum blob circularity to accept as a cell |

---

## `celldform-train-unet`

Train the U-Net segmentation model. Delegates directly to `scripts/train_unet.py`.

```bash
celldform-train-unet --config configs/multiclass_experiment_256.yaml   # frozen scheme
celldform-train-unet --config configs/default.yaml --seed 123
celldform-train-unet --config configs/multiclass_experiment_256.yaml --domain-oversample-weight 8.0
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `configs/default.yaml` | YAML config file |
| `--seed` | config `training.seed` | Random seed override |
| `--domain-oversample-weight` | config value (`1.0` if unset) | Oversample `domain_`-prefixed (real-cell) train frames relative to legacy ones via `WeightedRandomSampler`; `1.0` = off. See [Training Results — Domain Gap](training.md#domain-gap). |

All other training parameters (`epochs`, `device`, `loss_alpha`, `loss_pos_weight`, etc.) are set in the YAML config. Checkpoints written: `unet_best.pt`, `unet_last.pt`, and a `config.yaml` snapshot.

---

## `celldform-infer`

Run the full inference pipeline (stages 1–6) on a video or image folder.

```bash
celldform-infer data/preprocessed/inference/03_mask_factory/inputs \
    --unet-ckpt checkpoints/unet_best.pt

celldform-infer data/raw/high/"cell 1 high 50 increments.mp4" \
    --unet-ckpt checkpoints/unet_best.pt \
    --classifier-ckpt checkpoints/svm.pkl
```

| Option | Default | Description |
|--------|---------|-------------|
| `input` | required | Video file or directory of images |
| `--unet-ckpt` | required | Path to trained U-Net checkpoint (`.pt`) |
| `--classifier-ckpt` | none | Path to trained classifier (`.pkl`) |
| `--output-dir` | `results/` | Directory for inference outputs |
| `--device` | auto-detect | `cuda` or `cpu` |
| `--every-n` | `1` | Process every N-th frame |
