# CLI Reference

Every script in `scripts/` is also a console command named after it, so the whole pipeline runs
from the shell without remembering paths:

```bash
celldform-train-experiment postswap/experiment_1b_loss e1b_lovasz_s42
# is exactly
python scripts/training/train_experiment.py postswap/experiment_1b_loss e1b_lovasz_s42
```

- **Naming:** `scripts/<stage>/<name>.py` becomes `celldform-<name>` with `_` turned into `-`. Each
  command runs its script with the current python and passes every argument through unchanged, so
  options, defaults, output and exit codes are the script's own. `celldform-<command> --help` is the
  authoritative option list; the per-command sections below show it as of 2026-10-07.
- **Umbrella:** `celldform` on its own lists every command; `celldform <command> [args]` runs one
  (`celldform train-experiment ...` = `celldform-train-experiment ...`).
- **Install:** `pip install -e .` (editable, from the repo). The scripts are not part of the installed
  package, so a non-editable install has the commands but they report that `scripts/` is missing.
  `celldform-annotate` and `celldform-annotate-frame` also need `pip install -e ".[annotation]"` (napari).
- **Working directory:** like `python scripts/...`, commands resolve relative paths (configs, data,
  results) against the current directory, so run them from the repo root.
- **On the NSU cluster:** the training and evaluation commands have SLURM job scripts under
  `scripts/hpc/` that take the same arguments, e.g. `sbatch scripts/hpc/training/train_experiment.sh ...`.
  See [HPC Guide](hpc.md).
- **No command:** `scripts/hpc/` (submit those with `sbatch`) and `scripts/maintenance/` (one-off
  migrations; re-running them would move files again).

`tests/test_cli.py` keeps the commands, `scripts/` and `pyproject.toml` in sync: a new script without a
command, a command without a script, or a script without an argument parser fails the tests.

---

## Overview

**Data**

| Command | What it does |
|---|---|
| [`celldform-organize-dataset`](#celldform-organize-dataset) | Build pipeline-ready frame folders from `data/raw/` |
| [`celldform-extract-frames`](#celldform-extract-frames) | Data acquisition — extract frames from an optical tweezer video recording. |
| [`celldform-preprocess-frames`](#celldform-preprocess-frames) | Apply PreprocessingPipeline to annotated frames and save the outputs to disk. |
| [`celldform-build-randswap`](#celldform-build-randswap) | Build one randswap fold: draw the cell split from the config, then preprocess its data. |
| [`celldform-build-nnunet-dataset`](#celldform-build-nnunet-dataset) | Build the nnU-Net benchmark dataset from the randswap cross-validation -- same frames, same folds. |
| [`celldform-build-distill-data`](#celldform-build-distill-data) | Distillation data for one randswap round: our round-k training data + nnU-Net (teacher) targets. |

**Annotation**

| Command | What it does |
|---|---|
| [`celldform-annotate`](#celldform-annotate) | Interactive cell annotation tool — napari-based. |
| [`celldform-annotate-frame`](#celldform-annotate-frame) | Annotate or correct specific frames in napari, one after the other. |
| [`celldform-preview-masks`](#celldform-preview-masks) | Colorized masks — render saved masks as real color PNGs, viewable directly in any image viewer (no napari needed). |
| [`celldform-validate-masks`](#celldform-validate-masks) | Validate the masks that are present in the annotation pool. |

**Training**

| Command | What it does |
|---|---|
| [`celldform-train-unet`](#celldform-train-unet) | Train U-Net for cancer cell segmentation. |
| [`celldform-train-experiment`](#celldform-train-experiment) | Train + evaluate one run of an experiment, keeping every artefact inside that experiment's folder (2026-09-30 layout: one folder per experiment, mirrored under configs/… |
| [`celldform-train-cv-fold`](#celldform-train-cv-fold) | One randswap cross-validation round: draw + build the fold's data, train, score. |
| [`celldform-train-deploy-model`](#celldform-train-deploy-model) | The randswap DEPLOYMENT model -- the one RealTimePipeline runs on the camera: every cell trains, fixed length = the CV median best epoch. |
| [`celldform-train-nnunet-cv`](#celldform-train-nnunet-cv) | nnU-Net benchmark: the randswap cross-validation, round for round, with nnU-Net in place of our U-Net. |
| [`celldform-train-student-cv`](#celldform-train-student-cv) | Distillation student CV (Experiment 6): every randswap round, against our CV model at the same epoch. |
| [`celldform-distill-ensemble`](#celldform-distill-ensemble) | Distil the 4-seed ensemble into a single deployable U-Net. |

**Evaluation**

| Command | What it does |
|---|---|
| [`celldform-evaluate-unet`](#celldform-evaluate-unet) | Evaluate a trained U-Net on the holdout set and report segmentation metrics. |
| [`celldform-eval-native`](#celldform-eval-native) | Score a checkpoint on the testing set at NATIVE resolution (1228x922). |
| [`celldform-eval-testing-set`](#celldform-eval-testing-set) | Score one checkpoint on the testing set (the four held-out cells, on disk 02_unet_holdout / preprocessed postswap_devset): each cell, then all cells pooled. |
| [`celldform-eval-nnunet`](#celldform-eval-nnunet) | Score one round of the nnU-Net benchmark exactly as eval_native.py scores our U-Net. |
| [`celldform-ensemble-eval`](#celldform-ensemble-eval) | Evaluate an ensemble of checkpoints on the development set. |
| [`celldform-eval-ensemble-cv`](#celldform-eval-ensemble-cv) | Ensemble test (report: Pending 1, "the gate"): our U-Net + nnU-Net, averaged, on the randswap rounds. |
| [`celldform-summarize-cv`](#celldform-summarize-cv) | Summarise a randswap cross-validation: every round, every cell, one headline. |
| [`celldform-compare-postswap`](#celldform-compare-postswap) | Compare postswap runs against the series baseline, on the development set. |
| [`celldform-check-convergence`](#celldform-check-convergence) | Flag training runs that converged poorly, before their numbers enter a mean. |
| [`celldform-latency-benchmark`](#celldform-latency-benchmark) | Frame-to-mask latency: our U-Net against nnU-Net, measured the same way on the same frames. |

**Inference**

| Command | What it does |
|---|---|
| [`celldform-infer`](#celldform-infer) | End-to-end inference on optical tweezers video data. |
| [`celldform-predict-single-frame`](#celldform-predict-single-frame) | Quick sanity check — run a trained U-Net checkpoint on one raw frame and draw two outlines on top of it: a solid line around each predicted region (the actual, possibly… |
| [`celldform-harvest-factory`](#celldform-harvest-factory) | Segment every 03_mask_factory frame with the ensemble and extract classifier features. |

**Report figures and tables**

| Command | What it does |
|---|---|
| [`celldform-build-report-assets`](#celldform-build-report-assets) | Regenerate every figure and table in the current progress report. |
| [`celldform-make-report-tables`](#celldform-make-report-tables) | Generate the progress report's LaTeX tables from the data. |
| [`celldform-make-cell-gif`](#celldform-make-cell-gif) | Social-media GIF: one trapped cell across the laser-power ramp, raw beside the model's outline. |
| [`celldform-diagnose-false-positives`](#celldform-diagnose-false-positives) | Show what a checkpoint's false-positive region actually *is*, on the raw frame. |
| [`celldform-render-holdout-comparison`](#celldform-render-holdout-comparison) | Render the four-figure holdout comparison used in the thesis progress reports: a strip of N frames from one holdout cell, shown as (1) raw unannotated frames, (2) the mu… |
| [`celldform-view-soft-target`](#celldform-view-soft-target) | Look at a teacher target (.npy) from the distillation data: the frame, its label, and the teacher's per-class probabilities as heat maps. |
| [`celldform-plot-cell-sizes`](#celldform-plot-cell-sizes) | Per-cell trapped-cell size, measured from the annotation masks. |
| [`celldform-plot-cv`](#celldform-plot-cv) | Figure and tables for the randswap cross-validation (1 October report). |
| [`celldform-plot-data-map`](#celldform-plot-data-map) | Dataset Organization figure (randswap): where every cell and frame goes. |
| [`celldform-plot-experiments`](#celldform-plot-experiments) | Figures for the experiments chapter of the 1 October report. |
| [`celldform-plot-model-comparison`](#celldform-plot-model-comparison) | The four candidate segmenters side by side, on the same 15 cross-validated cells: one small panel per measure, every value labelled. |
| [`celldform-plot-nnunet-epochs`](#celldform-plot-nnunet-epochs) | nnU-Net at 100 against 1000 epochs on the same randswap rounds (8 October report). |
| [`celldform-plot-percell-metrics`](#celldform-plot-percell-metrics) | Grouped bar chart of per-cell holdout metrics, read straight out of results/preswap/preswap_eval_log.jsonl — one panel per evaluation scope (each development-set cell, p… |
| [`celldform-plot-pool-flow`](#celldform-plot-pool-flow) | Provenance diagram of the frame pools: which cells go where, what draws from what, and where the evaluation firewall sits. |
| [`celldform-plot-resize-vs-crop`](#celldform-plot-resize-vs-crop) | Experiment 3 figure: the same frames as each input variant sees them. |
| [`celldform-plot-split-change`](#celldform-plot-split-change) | Before/after of the train/validation/test split as two stacked bars. |
| [`celldform-plot-sweep-dsc`](#celldform-plot-sweep-dsc) | Horizontal bar chart ranking every checkpoint in results/preswap/preswap_eval_log.jsonl by its evaluated DSC on the development set. |
| [`celldform-plot-fe-dataset`](#celldform-plot-fe-dataset) | Dataset Organization figure for the feature-extraction reports (22 October report onward). |
| [`celldform-plot-features`](#celldform-plot-features) | Feature-extraction figures and tables (22 October report). |
| [`celldform-plot-final-models`](#celldform-plot-final-models) | "Final Models at a Glance" -- the closing page of the 8 October report (written 2026-10-21). |
| [`celldform-plot-high1-exclusion`](#celldform-plot-high1-exclusion) | Why high-1 is excluded from the features (22 October report, feature extraction). |
| [`celldform-plot-high10`](#celldform-plot-high10) | Why high-10 is over-predicted in leave-one-cell-out round 14 (8 October report, added 27 October). |
| [`celldform-plot-classification-thesis`](#celldform-plot-classification-thesis) | Classification chapter figures: every model's AUC and F1, per-cell P(high), the best model's chance null and its summary card. |
| [`celldform-plot-analysis-dataset`](#celldform-plot-analysis-dataset) | Dataset page of the analysis reports: what each cell contributes, and the leave-one-cell-out split. |
| [`celldform-plot-loocv`](#celldform-plot-loocv) | Leave-one-cell-out against the 5-fold cross-validation, for the U-Net and nnU-Net (15 October report). |
| [`celldform-plot-protocols`](#celldform-plot-protocols) | Dataset Organization figure, general form (15 October report): one rule, three protocols. |
| [`celldform-plot-region-selection`](#celldform-plot-region-selection) | How the extraction picks the trapped cell, and why it drops a frame -- drawn on the predicted masks (22 October report, feature extraction). |
| [`celldform-diagnose-mask-growth`](#celldform-diagnose-mask-growth) | Where do a model's extra pixels go? Compare several runs' saved predictions of ONE test cell. |

**Features and classification**

| Command | What it does |
|---|---|
| [`celldform-extract-features`](#celldform-extract-features) | Classifier features from the feature frames -- segment, validate every segmentation, then extract. |
| [`celldform-benchmark-cells`](#celldform-benchmark-cells) | Per-cell HER2 benchmark: seven classical classifiers on the per-cell features, leave-one-cell-out. |
| [`celldform-frames-cnn`](#celldform-frames-cnn) | Per-frame HER2 classifier (deep learning), judged per held-out cell. |
| [`celldform-power-resolution`](#celldform-power-resolution) | How many cells would a conclusive experiment need, and how small a deformation can the system resolve? |
| [`celldform-tune-cells`](#celldform-tune-cells) | Hyperparameter tuning of the classical classifiers by nested leave-one-cell-out cross-validation. |
| [`celldform-sequence-net`](#celldform-sequence-net) | Sequence networks on the deformation curves: can HER2 be told from HOW a cell deforms over the laser ramp? |
| [`celldform-feature-ranking`](#celldform-feature-ranking) | Feature rankings x k x classifiers, best-of-grid and nested choice, with permutation tests: does choosing features with the labels help? |
| [`celldform-combine-models`](#celldform-combine-models) | Frame- and step-level logistic regression and the soft-voting ensembles (equal-weight mean of P(high)), with permutation tests. |
| [`celldform-pca-models`](#celldform-pca-models) | PCA in place of feature selection: does compressing the features beat the selected 4? |
| [`celldform-deformation-regression`](#celldform-deformation-regression) | Deformation regression: does the strain grow with the laser power, and differently for high and low HER2 cells? |
| [`celldform-rate-of-deformation`](#celldform-rate-of-deformation) | Rate of deformation (RoD) per cell: measured over the full ramp, in real time, and predicted early. |

**HPC**

| Command | What it does |
|---|---|
| [`celldform-hpc-submit`](#celldform-hpc-submit) | Generate and submit the SLURM U-Net training job on the NSU cluster |

---

## Data

### `celldform-organize-dataset`

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

Implemented in the package (`celldform.acquisition.organiser`); `scripts/data/organize_dataset.py` is a thin wrapper around it.

---

### `celldform-extract-frames`

Data acquisition — extract frames from an optical tweezer video recording.

Script: `scripts/data/extract_frames.py`

```bash
celldform-extract-frames --video data/videos/experiment_01.mp4
celldform-extract-frames --video experiment.mp4 --output-dir data/frames
celldform-extract-frames --video experiment.mp4 --every-n 5
celldform-extract-frames --video experiment.mp4 --target-fps 10.0
celldform-extract-frames --video experiment.mp4 --max-frames 200
```

??? note "All options (`--help`)"
    ```text
    usage: extract-frames [-h] --video VIDEO [--config CONFIG]
                          [--output-dir OUTPUT_DIR] [--every-n EVERY_N]
                          [--target-fps TARGET_FPS] [--max-frames MAX_FRAMES]
                          [--target-size W H] [--format {png,tiff}]
                          [--change-threshold CHANGE_THRESHOLD]
                          [--dark-percentile DARK_PERCENTILE]
                          [--channel-width CHANNEL_WIDTH]
                          [--min-circularity MIN_CIRCULARITY]

    Extract frames from an optical tweezer video recording.

    options:
      -h, --help            show this help message and exit
      --video VIDEO         Path to the input video file (.mp4, .avi, …).
      --config CONFIG       Path to YAML config file (default:
                            configs/default.yaml).
      --output-dir OUTPUT_DIR
                            Directory to write extracted frames (overrides config
                            data.frames_dir).
      --every-n EVERY_N     Keep every N-th frame (overrides config
                            acquisition.every_n).
      --target-fps TARGET_FPS
                            Resample to this frame rate in Hz (mutually exclusive
                            with --every-n).
      --max-frames MAX_FRAMES
                            Stop after this many frames (overrides config
                            acquisition.max_frames).
      --target-size W H     Resize each frame to W H pixels (overrides config
                            preprocessing.target_size).
      --format {png,tiff}   Output image format (overrides config
                            acquisition.image_format).
      --change-threshold CHANGE_THRESHOLD
                            Minimum trapped-cell centroid displacement (pixels)
                            required to save a frame. A frame is skipped when the
                            trapped cell has not moved this far from its position
                            in the last saved frame. Overrides config
                            acquisition.change_threshold.
      --dark-percentile DARK_PERCENTILE
                            Bottom X% of channel pixel intensities treated as
                            candidate cells. Default: 20. Overrides config
                            acquisition.dark_percentile.
      --channel-width CHANNEL_WIDTH
                            Width of the vertical flow channel as a fraction of
                            image width, centred at the horizontal midpoint (e.g.
                            0.30 = ±15% from centre). Overrides config
                            acquisition.channel_width.
      --min-circularity MIN_CIRCULARITY
                            Minimum circularity (4π·area/perimeter², 0–1) for a
                            blob to be accepted as a cell. Default: 0.60.
                            Overrides config acquisition.min_circularity.
    ```

---

### `celldform-preprocess-frames`

Apply PreprocessingPipeline to annotated frames and save the outputs to disk.

Script: `scripts/data/preprocess_frames.py`

```bash
celldform-preprocess-frames --masks
celldform-preprocess-frames --masks --compare --n-compare 5
celldform-preprocess-frames --pool 02_unet_holdout --masks
celldform-preprocess-frames --masks --target-size 128 128
celldform-preprocess-frames --image-dir data/frames/01_annotate_pool \
 \
    --output-dir data/preprocessed/frames/01_annotate_pool \
 \
    --mask-dir   data/masks/01_annotate_pool
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-preprocess-frames [-h] [--pool POOL]
                                [--stage {train,evaluate,inference}]
                                [--image-dir IMAGE_DIR] [--output-dir OUTPUT_DIR]
                                [--compare] [--n-compare N_COMPARE]
                                [--compare-dir COMPARE_DIR] [--masks]
                                [--mask-dir MASK_DIR]
                                [--mask-output-dir MASK_OUTPUT_DIR]
                                [--crop-size CROP_SIZE] [--crop-center X Y]
                                [--legacy-crop-center X Y] [--target-size H W]

    Preprocess annotated frames and optionally save pipeline figures

    options:
      -h, --help            show this help message and exit
      --pool POOL           Pool folder name under data/frames/
      --stage {train,evaluate,inference}
                            Pipeline stage (default: auto-detected from --pool)
      --image-dir IMAGE_DIR
                            Override source image directory
      --output-dir OUTPUT_DIR
                            Override output directory for preprocessed frames
      --compare             Save pipeline stage comparison figures
      --n-compare N_COMPARE
                            Number of images to generate comparison figures for
                            (default: 5)
      --compare-dir COMPARE_DIR
                            Directory for comparison figures
      --masks               Also resize and save binary masks alongside inputs
      --mask-dir MASK_DIR   Source mask directory (default: data/masks/<pool>)
      --mask-output-dir MASK_OUTPUT_DIR
                            Override destination for resized masks
      --crop-size CROP_SIZE
                            Cut a native-resolution SIZE x SIZE window instead of
                            resizing the whole frame (experiment 3). Needs --crop-
                            center.
      --crop-center X Y     Crop centre in full-frame pixels for real-cell frames
      --legacy-crop-center X Y
                            Crop centre for legacy 2021 frames (their trap sits
                            elsewhere); defaults to --crop-center
      --target-size H W     Output spatial size (default: 256 256). Both dims
                            should be divisible by 16 for U-Net's 4 downsampling
                            steps.
    ```

---

### `celldform-build-randswap`

Build one randswap fold: draw the cell split from the config, then preprocess its data.

Script: `scripts/data/build_randswap.py`

```bash
celldform-build-randswap --config configs/randswap/<exp>/<exp>.yaml --fold 1
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-build-randswap [-h] --config CONFIG --fold FOLD

    Build one randswap fold: draw the cell split from the config, then preprocess
    its data.

    options:
      -h, --help       show this help message and exit
      --config CONFIG
      --fold FOLD      1..k, or 'deploy' (every cell trains: the deployment model)
    ```

---

### `celldform-build-nnunet-dataset`

Build the nnU-Net benchmark dataset from the randswap cross-validation -- same frames, same folds.

Script: `scripts/data/build_nnunet_dataset.py`

```bash
celldform-build-nnunet-dataset \
    --config configs/randswap/experiment_0_cv_baseline/cv_baseline.yaml \
    --splits results/randswap/experiment_0_cv_baseline --run cv_baseline
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-build-nnunet-dataset [-h] --config CONFIG --splits SPLITS --run RUN
                                   [--raw RAW] [--preprocessed PREPROCESSED]
                                   [--dataset-id DATASET_ID] [--name NAME]

    Build the nnU-Net benchmark dataset from the randswap cross-validation -- same
    frames, same folds.

    options:
      -h, --help            show this help message and exit
      --config CONFIG
      --splits SPLITS       results folder holding <run>_fold_<k>_split.json
      --run RUN
      --raw RAW
      --preprocessed PREPROCESSED
      --dataset-id DATASET_ID
      --name NAME
    ```

---

### `celldform-build-distill-data`

Distillation data for one randswap round: our round-k training data + nnU-Net (teacher) targets.

Script: `scripts/data/build_distill_data.py`

```bash
celldform-build-distill-data --round 4
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-build-distill-data [-h] --round ROUND [--every EVERY]
                                 [--offset OFFSET]

    Distillation data for one randswap round: our round-k training data + nnU-Net
    (teacher) targets.

    options:
      -h, --help       show this help message and exit
      --round ROUND
      --every EVERY
      --offset OFFSET
    ```

---

## Annotation

### `celldform-annotate`

Interactive cell annotation tool — napari-based.

Script: `scripts/annotation/annotate.py`

```bash
celldform-annotate
celldform-annotate --image-dir data/frames/01_annotate_pool \
 \
    --mask-dir  data/masks/01_annotate_pool
celldform-annotate --pool 02_unet_holdout   # annotate holdout set
celldform-annotate --multiclass             # 3-class experiment (same 40 images)
```

Needs the annotation extra (`pip install -e ".[annotation]"`) — it opens a napari window.

| Option | Default | Description |
|---|---|---|
| `--pool` | `01_annotate_pool` | Pool folder name under `data/frames/` and `data/masks/` |
| `--image-dir` | – | Explicit image folder (instead of `--pool`) |
| `--mask-dir` | – | Explicit mask folder (instead of `--pool`) |
| `--multiclass` | off | Annotate 3 classes (0 background, 1 trapped cell, 2 other/decoy) into `data/masks/<pool>_multiclass/` |

---

### `celldform-annotate-frame`

Annotate or correct specific frames in napari, one after the other.

Script: `scripts/annotation/annotate_frame.py`

```bash
celldform-annotate-frame data/frames/01_annotate_pool/frame_0000.jpg
celldform-annotate-frame --multiclass \
    domain_high_cell8_000000 domain_low_cell3_000000 domain_low_cell4_000000
celldform-annotate-frame frame_0000.jpg --mask-out custom/path.png
```

Needs the annotation extra (`pip install -e ".[annotation]"`) — it opens a napari window.

| Option | Default | Description |
|---|---|---|
| `images` | required | Image files: full paths, or bare names resolved against `data/frames/<pool>/` |
| `--pool` | `01_annotate_pool` | Pool used to resolve bare names and default mask paths |
| `--multiclass` | off | Edit 3-class masks, saved to `data/masks/<pool>_multiclass/` |
| `--mask-out` | mirrors `data/masks/<pool>[_multiclass]/` | Explicit output mask path (single image only) |

---

### `celldform-preview-masks`

Colorized masks — render saved masks as real color PNGs, viewable directly in any image viewer (no napari needed).

Script: `scripts/annotation/preview_masks.py`

```bash
celldform-preview-masks                                    # 01_annotate_pool_multiclass
celldform-preview-masks --pool 01_annotate_pool            # binary masks
celldform-preview-masks --mask-dir data/masks/01_annotate_pool_multiclass \
 \
    --output-dir data/masks/01_annotate_pool_multiclass_preview
celldform-preview-masks --limit 10
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-preview-masks [-h] [--pool POOL] [--mask-dir MASK_DIR]
                            [--output-dir OUTPUT_DIR] [--limit LIMIT]

    Render colorized mask previews (black background)

    options:
      -h, --help            show this help message and exit
      --pool POOL           Pool folder name under data/masks/
      --mask-dir MASK_DIR   Override mask directory
      --output-dir OUTPUT_DIR
                            Override output directory
      --limit LIMIT         Only preview the first N masks
    ```

---

### `celldform-validate-masks`

Validate the masks that are present in the annotation pool.

Script: `scripts/annotation/validate_masks.py`

```bash
celldform-validate-masks                          # default: 01_annotate_pool
celldform-validate-masks --pool 02_unet_holdout
celldform-validate-masks --image-dir data/frames/01_annotate_pool \
 \
    --mask-dir  data/masks/01_annotate_pool
celldform-validate-masks --multiclass \
 \
    --mask-dir  data/masks/01_annotate_pool_multiclass
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-validate-masks [-h] [--pool POOL] [--mask-dir MASK_DIR]
                             [--image-dir IMAGE_DIR] [--no-completeness]
                             [--multiclass]

    Validate annotation masks

    options:
      -h, --help            show this help message and exit
      --pool POOL           Pool folder name under data/masks/ (and data/frames/)
      --mask-dir MASK_DIR   Override mask directory
      --image-dir IMAGE_DIR
                            Override frame directory (checked for missing masks);
                            pass --no-completeness to skip this check entirely
      --no-completeness     Skip checking that every pool image has a mask
      --multiclass          Validate 3-class masks (0/1/2) instead of binary
                            (0/255)
    ```

---

## Training

### `celldform-train-unet`

Train U-Net for cancer cell segmentation.

Script: `scripts/training/train_unet.py` · SLURM job: `scripts/hpc/training/train_unet.sh`

```bash
celldform-train-unet --config configs/default.yaml
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-train-unet [-h] [--config CONFIG] [--seed SEED]
                         [--domain-oversample-weight DOMAIN_OVERSAMPLE_WEIGHT]
                         [--fold FOLD] [--results-dir RESULTS_DIR]
                         [--epochs EPOCHS] [--save-last] [--tag TAG]

    Train U-Net segmentation model.

    options:
      -h, --help            show this help message and exit
      --config CONFIG       Path to YAML config file.
      --seed SEED           Random seed (overrides config value).
      --domain-oversample-weight DOMAIN_OVERSAMPLE_WEIGHT
                            Sampling weight for domain_-prefixed (real-cell) train
                            frames vs. legacy frames (overrides
                            training.domain_oversample_weight). 1.0 = off/uniform
                            (default); pass 1.0 explicitly to force it off when
                            the config has it enabled.
      --fold FOLD           randswap: train fold K of the config's split section
                            (built first by scripts/data/build_randswap.py). Data,
                            validation cell and checkpoint folder all come from
                            that fold.
      --results-dir RESULTS_DIR
                            Where the training-curve PNG goes (e.g.
                            results/postswap/experiment_1b_loss). Versions are
                            numbered within this folder.
      --epochs EPOCHS       Stop after this many epochs. The LR schedule still
                            spans training.epochs, so a CV fold that peaked at
                            epoch E and a deployment-model run with --epochs E see
                            the same learning-rate trajectory. Required with
                            --fold deploy.
      --save-last           Also save the LAST epoch's weights as unet_last.pt
                            (e.g. a fixed-length run scored at its final epoch
                            rather than at the best validation epoch).
      --tag TAG             Suffix appended to the saved training-curve filename
                            (e.g. --tag auto -> unet_training_curves_v6_auto.png),
                            for distinguishing runs at a glance.
    ```

---

### `celldform-train-experiment`

Train + evaluate one run of an experiment, keeping every artefact inside that experiment's folder (2026-09-30 layout: one folder per experiment, mirrored under configs/ checkpoints/ logs/ results/).

Script: `scripts/training/train_experiment.py` · SLURM job: `scripts/hpc/training/train_experiment.sh`

```bash
celldform-train-experiment <experiment> <run> [extra train_unet.py args...]
celldform-train-experiment postswap/experiment_1b_loss e1b_lovasz_s42
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-train-experiment [-h] experiment run ...

    Train + evaluate one run of an experiment, keeping every artefact inside that
    experiment's folder (2026-09-30 layout: one folder per experiment, mirrored
    under configs/ checkpoints/ logs/ results/).

    positional arguments:
      experiment  e.g. postswap/experiment_1b_loss
      run         config name inside configs/<experiment>/, without .yaml
      train_args  passed on to train_unet.py

    options:
      -h, --help  show this help message and exit
    ```

---

### `celldform-train-cv-fold`

One randswap cross-validation round: draw + build the fold's data, train, score.

Script: `scripts/training/train_cv_fold.py` · SLURM job: `scripts/hpc/training/train_cv_fold.sh`

```bash
celldform-train-cv-fold <era>/<experiment> <run> <fold>
celldform-train-cv-fold randswap/experiment_0_cv_baseline cv_baseline 1
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-train-cv-fold [-h] experiment run fold

    One randswap cross-validation round: draw + build the fold's data, train,
    score.

    positional arguments:
      experiment  e.g. randswap/experiment_0_cv_baseline
      run         config name inside configs/<experiment>/, without .yaml
      fold        CV round, 1..5

    options:
      -h, --help  show this help message and exit
    ```

---

### `celldform-train-deploy-model`

The randswap DEPLOYMENT model -- the one RealTimePipeline runs on the camera: every cell trains, fixed length = the CV median best epoch.

Script: `scripts/training/train_deploy_model.py` · SLURM job: `scripts/hpc/training/train_deploy_model.sh`

```bash
celldform-train-deploy-model <era>/<experiment> <run>
celldform-train-deploy-model randswap/experiment_0_cv_baseline cv_baseline
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-train-deploy-model [-h] experiment run

    The randswap DEPLOYMENT model -- the one RealTimePipeline runs on the camera:
    every cell trains, fixed length = the CV median best epoch.

    positional arguments:
      experiment  e.g. randswap/experiment_0_cv_baseline
      run         config name inside configs/<experiment>/, without .yaml

    options:
      -h, --help  show this help message and exit
    ```

---

### `celldform-train-nnunet-cv`

nnU-Net benchmark: the randswap cross-validation, round for round, with nnU-Net in place of our U-Net.

Script: `scripts/training/train_nnunet_cv.py` · SLURM job: `scripts/hpc/training/train_nnunet_cv.sh`

```bash
celldform-train-nnunet-cv --setup          # dataset + plan/preprocess, once
celldform-train-nnunet-cv                  # train + predict + score rounds 1..5
celldform-train-nnunet-cv --rounds 3 4     # some rounds only
celldform-train-nnunet-cv --rounds 2 --continue-training   # resume an interrupted round
celldform-train-nnunet-cv --trainer nnUNetTrainer \
    --exp randswap/experiment_1b_nnunet_1000ep --rounds 1
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-train-nnunet-cv [-h] [--setup] [--rounds ROUNDS [ROUNDS ...]]
                              [--trainer TRAINER] [--continue-training]
                              [--exp EXP] [--no-summary] [-np NUM_PROCESSES]

    nnU-Net benchmark: the randswap cross-validation, round for round, with nnU-
    Net in place of our U-Net.

    options:
      -h, --help            show this help message and exit
      --setup               build the dataset and plan/preprocess, then stop
      --rounds ROUNDS [ROUNDS ...]
      --trainer TRAINER
      --continue-training
      --exp EXP             experiment folder for checkpoints/ and results/
                            (default randswap/experiment_1_nnunet); use a new one
                            for any other trainer
      --no-summary          skip summarize_cv.py at the end (rounds running as
                            parallel jobs)
      -np NUM_PROCESSES, --num-processes NUM_PROCESSES
    ```

---

### `celldform-train-student-cv`

Distillation student CV (Experiment 6): every randswap round, against our CV model at the same epoch.

Script: `scripts/training/train_student_cv.py` · SLURM job: `scripts/hpc/training/train_student_cv.sh`

```bash
celldform-train-student-cv                 # rounds 1..5
celldform-train-student-cv --rounds 2 3
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-train-student-cv [-h] [--rounds ROUNDS [ROUNDS ...]]

    Distillation student CV (Experiment 6): every randswap round, against our CV
    model at the same epoch.

    options:
      -h, --help            show this help message and exit
      --rounds ROUNDS [ROUNDS ...]
    ```

---

### `celldform-distill-ensemble`

Distil the 4-seed ensemble into a single deployable U-Net.

Script: `scripts/training/distill_ensemble.py` · SLURM job: `scripts/hpc/training/distill_ensemble.sh`

```bash
celldform-distill-ensemble                      # cache targets, then train
celldform-distill-ensemble --cache-only         # just build the targets
celldform-distill-ensemble --temperature 3 --lambda-hard 0.5
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-distill-ensemble [-h] [--config CONFIG] [--teachers [TEACHERS ...]]
                               [--cache CACHE] [--rebuild-cache] [--cache-only]
                               [--temperature TEMPERATURE]
                               [--lambda-hard LAMBDA_HARD] [--epochs EPOCHS]
                               [--run-name RUN_NAME]

    Distil the 4-seed ensemble into a single deployable U-Net.

    Why: the ensemble scores 0.859 evaluated against 0.809 for the mean single model,
    and no per-cell collapse survives it. But it is an inference-time construct --
    no checkpoint, 4x GPU per frame, and RealTimePipeline takes one UNet. Distillation
    trains one student on the ensemble's *soft* outputs, so the student learns the
    smoothed decision boundary the ensemble produces rather than only the hard masks.

    How it works
    ------------
    1. The teachers' averaged softmax is PRECOMPUTED over the training pool and cached
       to disk. Four teachers plus a student do not fit in 8 GB together, and caching
       also avoids 4 extra forward passes per batch per epoch.
    2. The student trains on
           L = lambda * CE(hard masks) + (1 - lambda) * T^2 * KL(student/T || teacher/T)
       Hard labels are kept because the training pool *is* annotated -- the ensemble is
       a second opinion here, not the only source of truth. T > 1 softens both
       distributions so the student sees the teacher's relative confidences, not just
       its argmax; the T^2 factor restores the gradient scale that softening removes.
    3. The split is the teachers' split (split_seed 42), so the student's validation
       slice is one no teacher trained on either.

    Usage
    -----
    python scripts/training/celldform-distill-ensemble                      # cache targets, then train
    python scripts/training/celldform-distill-ensemble --cache-only         # just build the targets
    python scripts/training/celldform-distill-ensemble --temperature 3 --lambda-hard 0.5

    options:
      -h, --help            show this help message and exit
      --config CONFIG
      --teachers [TEACHERS ...]
      --cache CACHE
      --rebuild-cache
      --cache-only
      --temperature TEMPERATURE
      --lambda-hard LAMBDA_HARD
                            Weight on the hard-label CE term; 1-lambda goes to the
                            KL term
      --epochs EPOCHS
      --run-name RUN_NAME
    ```

---

## Evaluation

### `celldform-evaluate-unet`

Evaluate a trained U-Net on the holdout set and report segmentation metrics.

Script: `scripts/evaluation/evaluate_unet.py` · SLURM job: `scripts/hpc/evaluation/evaluate_unet.sh`

```bash
celldform-evaluate-unet --checkpoint checkpoints/unet_best.pt
celldform-evaluate-unet --checkpoint checkpoints/unet_best.pt \
 \
    --pool 02_unet_holdout
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-evaluate-unet [-h] --checkpoint CHECKPOINT [--config CONFIG]
                            [--pool POOL] [--batch-size BATCH_SIZE]
                            [--filter FILTER] [--results-log RESULTS_LOG]
                            [--no-save-results]

    Evaluate U-Net on holdout set

    options:
      -h, --help            show this help message and exit
      --checkpoint CHECKPOINT
                            Path to trained .pt checkpoint
      --config CONFIG
      --pool POOL           Pool folder name under data/preprocessed/ (default:
                            02_unet_holdout)
      --batch-size BATCH_SIZE
      --filter FILTER       Only evaluate frames whose filename stem starts with
                            this prefix (e.g. --filter low_cell6 to isolate one
                            cell from a multi-cell pool)
      --results-log RESULTS_LOG
                            JSONL file to append this run's metrics to
      --no-save-results     Skip writing to --results-log
    ```

---

### `celldform-eval-native`

Score a checkpoint on the testing set at NATIVE resolution (1228x922).

Script: `scripts/evaluation/eval_native.py` · SLURM job: `scripts/hpc/evaluation/eval_native.sh`

```bash
celldform-eval-native --checkpoint CK --config CFG \
 \
    --inputs data/preprocessed/evaluate/postswap_devset_512/inputs \
 \
    --log results/postswap/<experiment>/eval_log_native.jsonl
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-eval-native [-h] --checkpoint CHECKPOINT [CHECKPOINT ...]
                          [--name NAME] --config CONFIG [--inputs INPUTS]
                          [--gt-dir GT_DIR] [--fold-dir FOLD_DIR] [--log LOG]
                          [--fold FOLD]

    Score a checkpoint on the testing set at NATIVE resolution (1228x922).

    options:
      -h, --help            show this help message and exit
      --checkpoint CHECKPOINT [CHECKPOINT ...]
                            one checkpoint, or several to score as an ensemble
                            (averaged softmax)
      --name NAME           run name in the log (default: checkpoint folder)
      --config CONFIG
      --inputs INPUTS       preprocessed testing-set inputs at the MODEL's
                            resolution
      --gt-dir GT_DIR
      --fold-dir FOLD_DIR   randswap fold folder: test inputs, native masks and
                            this fold's crop centre all come from its split.json
                            (overrides --inputs/--gt-dir)
      --log LOG
      --fold FOLD           CV round to record in the log (summarize_cv.py reads
                            it); needed when the checkpoint is not
                            <...>/fold_<k>/unet_best.pt, e.g. unet_last.pt
    ```

---

### `celldform-eval-testing-set`

Score one checkpoint on the testing set (the four held-out cells, on disk 02_unet_holdout / preprocessed postswap_devset): each cell, then all cells pooled.

Script: `scripts/evaluation/eval_testing_set.py` · SLURM job: `scripts/hpc/evaluation/eval_testing_set.sh`

```bash
celldform-eval-testing-set <checkpoint.pt> <config.yaml> <eval_log.jsonl> [pool]
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-eval-testing-set [-h] checkpoint config log [pool]

    Score one checkpoint on the testing set (the four held-out cells, on disk
    02_unet_holdout / preprocessed postswap_devset): each cell, then all cells
    pooled.

    positional arguments:
      checkpoint
      config
      log         eval log (JSONL) the scores are appended to
      pool

    options:
      -h, --help  show this help message and exit
    ```

---

### `celldform-eval-nnunet`

Score one round of the nnU-Net benchmark exactly as eval_native.py scores our U-Net.

Script: `scripts/evaluation/eval_nnunet.py` · SLURM job: `scripts/hpc/evaluation/eval_nnunet.sh`

```bash
celldform-eval-nnunet --round 1 \
    --pred-dir data/nnunet/predictions/round1 \
    --checkpoint checkpoints/randswap/experiment_1_nnunet/Dataset501_.../fold_0/checkpoint_final.pth \
    --log results/randswap/experiment_1_nnunet/eval_log_native.jsonl
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-eval-nnunet [-h] --round ROUND --pred-dir PRED_DIR --checkpoint
                          CHECKPOINT [--test-json TEST_JSON] [--log LOG]
                          [--trainer TRAINER]

    Score one round of the nnU-Net benchmark exactly as eval_native.py scores our
    U-Net.

    options:
      -h, --help            show this help message and exit
      --round ROUND         randswap round 1..k
      --pred-dir PRED_DIR
      --checkpoint CHECKPOINT
                            recorded in the log
      --test-json TEST_JSON
      --log LOG
      --trainer TRAINER     nnU-Net trainer, recorded in the log
    ```

---

### `celldform-ensemble-eval`

Evaluate an ensemble of checkpoints on the development set.

Script: `scripts/evaluation/ensemble_eval.py` · SLURM job: `scripts/hpc/evaluation/ensemble_eval.sh`

```bash
celldform-ensemble-eval                      # all postswap seeds
celldform-ensemble-eval --runs A B C
celldform-ensemble-eval --save-log           # append to the eval log
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-ensemble-eval [-h] [--runs [RUNS ...]] [--inputs INPUTS]
                            [--labels LABELS] [--ckpt-root CKPT_ROOT] [--log LOG]
                            [--save-log] [--name NAME]

    Evaluate an ensemble of checkpoints on the development set.

    Scores :class:`celldform.segmentation.ensemble.EnsembleUNet` -- the same class
    ``RealTimePipeline`` and ``infer.py`` run -- so the number reported here is the
    number the deployed model produces. The combination rule (average the per-class
    **softmax probabilities**, then argmax once) lives in that class and nowhere else;
    changing it there changes what this script measures, by construction.

    Averaging probabilities rather than voting on argmaxed masks keeps each model's
    confidence, so a model that is unsure about a pixel does not get the same say as
    one that is certain.

    The per-member columns re-run each member through its own ``UNet.predict``, which
    costs a second set of forward passes per frame (~2x). That is deliberate: both the
    ensemble row and the member rows then come from shipped inference code rather than
    from a copy of it written for this script.

    Why this exists: on the 2026-09-23 three-seed baseline, `high-1` scored 0.818 /
    0.833 / 0.507 across seeds (sd 0.184) while `high-9` sat at 0.043. That spread
    is a property of the training run, not of the cell — seed 44 is behind on three
    of four cells and on internal validation. An ensemble averages that variance out
    instead of reporting it.

    Costs 3x inference. Only Stage 3 uses the GPU, so this is the cheap part of the
    pipeline to triple.

    Usage
    -----
    python scripts/evaluation/celldform-ensemble-eval                      # all postswap seeds
    python scripts/evaluation/celldform-ensemble-eval --runs A B C
    python scripts/evaluation/celldform-ensemble-eval --save-log           # append to the eval log

    options:
      -h, --help            show this help message and exit
      --runs [RUNS ...]     Run names under
                            checkpoints/postswap/experiment_0a_seed_baseline.
                            Default: postswap_ftl_cw142_s42 postswap_ftl_cw142_s43
                            postswap_ftl_cw142_s44 postswap_ftl_cw142_s45
      --inputs INPUTS
      --labels LABELS
      --ckpt-root CKPT_ROOT
                            Experiment folder holding the runs (default: the seed
                            baseline)
      --log LOG             Eval log --save-log appends to
      --save-log            Append the ensemble's metrics to the postswap eval log
      --name NAME           Name recorded in the log
    ```

---

### `celldform-eval-ensemble-cv`

Ensemble test (report: Pending 1, "the gate"): our U-Net + nnU-Net, averaged, on the randswap rounds.

Script: `scripts/evaluation/eval_ensemble_cv.py` · SLURM job: `scripts/hpc/evaluation/eval_ensemble_cv.sh`

```bash
celldform-eval-ensemble-cv                 # all 5 rounds, writes the log + verdict
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-eval-ensemble-cv [-h] [--gate-only] [--tables TABLES]

    Ensemble test (report: Pending 1, "the gate"): our U-Net + nnU-Net, averaged,
    on the randswap rounds.

    options:
      -h, --help       show this help message and exit
      --gate-only      skip the 5-round GPU evaluation; recompute the verdict from
                       the existing log
      --tables TABLES  with --gate-only: also write <dir>/ensemble_gate.tex
    ```

---

### `celldform-summarize-cv`

Summarise a randswap cross-validation: every round, every cell, one headline.

Script: `scripts/evaluation/summarize_cv.py` · SLURM job: `scripts/hpc/evaluation/summarize_cv.sh`

```bash
celldform-summarize-cv --exp randswap/experiment_0_cv_baseline --run cv_baseline
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-summarize-cv [-h] --exp EXP --run RUN

    Summarise a randswap cross-validation: every round, every cell, one headline.

    options:
      -h, --help  show this help message and exit
      --exp EXP
      --run RUN
    ```

---

### `celldform-compare-postswap`

Compare postswap runs against the series baseline, on the development set.

Script: `scripts/evaluation/compare_postswap.py` · SLURM job: `scripts/hpc/evaluation/compare_postswap.sh`

```bash
celldform-compare-postswap                       # every run in the log
celldform-compare-postswap --runs A B            # just these, vs. baseline
celldform-compare-postswap --seeds 42 43 44      # pool seeds of one recipe
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-compare-postswap [-h] [--log LOG] [--runs [RUNS ...]]
                               [--baseline BASELINE] [--seeds [SEEDS ...]]

    Compare postswap runs against the series baseline, on the development set.

    Exists to enforce the rule in configs/postswap/README.md that a single-seed
    delta smaller than the noise floor is not a result. Measured spread on this
    pipeline is sd ~0.047 pooled (3 seeds, split pinned), so a difference between
    two single runs has SE ~ sd*sqrt(2) ~ 0.066 and needs roughly 2x that to be
    believable. Deltas under that threshold are printed as INCONCLUSIVE, however
    tempting they look -- this project has twice recorded a sub-threshold delta as
    a finding and had to retract it.

    Usage
    -----
    python scripts/evaluation/celldform-compare-postswap                       # every run in the log
    python scripts/evaluation/celldform-compare-postswap --runs A B            # just these, vs. baseline
    python scripts/evaluation/celldform-compare-postswap --seeds 42 43 44      # pool seeds of one recipe

    options:
      -h, --help           show this help message and exit
      --log LOG
      --runs [RUNS ...]    Run names to show
      --baseline BASELINE
      --seeds [SEEDS ...]  Run names to aggregate as seeds of one recipe (mean +/-
                           sd)
    ```

---

### `celldform-check-convergence`

Flag training runs that converged poorly, before their numbers enter a mean.

Script: `scripts/evaluation/check_convergence.py` · SLURM job: `scripts/hpc/evaluation/check_convergence.sh`

```bash
celldform-check-convergence                       # all postswap seeds
celldform-check-convergence --runs A B C
celldform-check-convergence --tolerance 0.015
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-check-convergence [-h] [--ckpt-root CKPT_ROOT] [--runs [RUNS ...]]
                                [--tolerance TOLERANCE]

    Flag training runs that converged poorly, before their numbers enter a mean.

    Internal validation carries **no signal for comparing recipes** — the 2026-09-10
    controlled re-run separated two checkpoints by 0.001 internally while the
    development set separated the same two by 0.066. But it does detect a run that
    simply trained badly, and that is a different question.

    The case this exists for: `postswap_ftl_cw142_s44` reached internal val 0.905
    against its siblings' 0.933 and 0.940, and went on to score 0.507 on `high-1`
    against their 0.818 and 0.833 — dragging the 3-seed mean down and roughly
    tripling the evaluated error bar. The bad run announced itself on a number that was
    already being printed; nothing was looking at it.

    Usage
    -----
    python scripts/evaluation/celldform-check-convergence                       # all postswap seeds
    python scripts/evaluation/celldform-check-convergence --runs A B C
    python scripts/evaluation/celldform-check-convergence --tolerance 0.015

    options:
      -h, --help            show this help message and exit
      --ckpt-root CKPT_ROOT
      --runs [RUNS ...]     Run names to compare (default: every seed of the
                            baseline recipe)
      --tolerance TOLERANCE
                            Flag a run this far below the best sibling (default
                            0.02)
    ```

---

### `celldform-latency-benchmark`

Frame-to-mask latency: our U-Net against nnU-Net, measured the same way on the same frames.

Script: `scripts/evaluation/latency_benchmark.py` · SLURM job: `scripts/hpc/evaluation/latency_benchmark.sh`

```bash
celldform-latency-benchmark \
    --unet checkpoints/randswap/experiment_0_cv_baseline/cv_baseline/fold_deploy/unet_deploy.pt \
    --config configs/randswap/experiment_0_cv_baseline/cv_baseline.yaml \
    --crop-center-from results/randswap/experiment_0_cv_baseline/cv_baseline_fold_deploy_split.json \
    --nnunet checkpoints/randswap/experiment_1_nnunet/Dataset501_CellTrapRandswap/nnUNetTrainer_100epochs__nnUNetPlans__2d \
    --fold 0 --out results/randswap/experiment_1_nnunet/latency_ms.json
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-latency-benchmark [-h] --unet UNET --config CONFIG
                                --crop-center-from CROP_CENTER_FROM --nnunet
                                NNUNET [--fold FOLD] [--student STUDENT]
                                [--frames FRAMES] [--n N] [--warmup WARMUP] --out
                                OUT

    Frame-to-mask latency: our U-Net against nnU-Net, measured the same way on the
    same frames.

    options:
      -h, --help            show this help message and exit
      --unet UNET
      --config CONFIG
      --crop-center-from CROP_CENTER_FROM
                            a split.json with crop_center
      --nnunet NNUNET       nnU-Net trainer output folder
      --fold FOLD
      --student STUDENT     a MegaNet (distilled student) checkpoint: same network
                            and crop as --unet
      --frames FRAMES
      --n N
      --warmup WARMUP
      --out OUT
    ```

---

## Inference

### `celldform-infer`

End-to-end inference on optical tweezers video data.

Script: `scripts/inference/infer.py`

```bash
celldform-infer \
 \
    --video data/videos/experiment_01.mp4 \
 \
    --unet-ckpt checkpoints/unet_best.pt \
 \
    --classifier-ckpt checkpoints/svm_clf.pkl \
 \
    --config configs/default.yaml \
 \
    --output-dir results/experiment_01
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-infer [-h] --video VIDEO --unet-ckpt UNET_CKPT
                    [--classifier-ckpt CLASSIFIER_CKPT] [--config CONFIG]
                    [--output-dir OUTPUT_DIR] [--save-masks] [--every-n EVERY_N]
                    [--max-frames MAX_FRAMES] [--device DEVICE]

    Run celldform inference on a video.

    options:
      -h, --help            show this help message and exit
      --video VIDEO         Path to input video.
      --unet-ckpt UNET_CKPT
                            U-Net checkpoint (.pt).
      --classifier-ckpt CLASSIFIER_CKPT
                            Classifier checkpoint (.pkl).
      --config CONFIG       Path to YAML config file.
      --output-dir OUTPUT_DIR
      --save-masks          Write binary masks to disk (warning: large).
      --every-n EVERY_N     Process every N-th frame (overrides config).
      --max-frames MAX_FRAMES
                            Stop after this many frames (overrides config).
      --device DEVICE       Compute device (overrides config).
    ```

---

### `celldform-predict-single-frame`

Quick sanity check — run a trained U-Net checkpoint on one raw frame and draw two outlines on top of it: a solid line around each predicted region (the actual, possibly irregular, area the network segmented) and a dotted line around the hand-drawn ground truth, when one exists for that frame.

Script: `scripts/inference/predict_single_frame.py`

```bash
celldform-predict-single-frame \
 \
    --checkpoint checkpoints/preswap/stage_1_labels_and_resolution/multiclass_experiment_256/unet_best.pt \
 \
    --frame data/frames/02_unet_holdout/high_cell9_005000.jpg
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-predict-single-frame [-h] --checkpoint CHECKPOINT --frame FRAME
                                   [--config CONFIG] [--min-area MIN_AREA]
                                   [--trapped-only] [--gt-mask GT_MASK]
                                   [--output OUTPUT] [--device DEVICE]

    Quick sanity check — run a trained U-Net checkpoint on one raw frame and draw
    two outlines on top of it: a solid line around each predicted region (the
    actual, possibly irregular, area the network segmented) and a dotted line
    around the hand-drawn ground truth, when one exists for that frame.

    Not part of the pipeline proper; a fast way to eyeball whether a checkpoint
    is fitting the trapped cell before committing to a full holdout evaluation.

    Reads architecture (in_channels, base_features, out_channels) and
    preprocessing settings (target_size, CLAHE/morphology params) from the
    config.yaml snapshot saved next to the checkpoint, so binary and multiclass
    checkpoints both load correctly without manual flags.

    Usage
    -----
    python scripts/inference/celldform-predict-single-frame \
        --checkpoint checkpoints/preswap/stage_1_labels_and_resolution/multiclass_experiment_256/unet_best.pt \
        --frame data/frames/02_unet_holdout/high_cell9_005000.jpg

    options:
      -h, --help            show this help message and exit
      --checkpoint CHECKPOINT
                            Path to a unet_best.pt / unet_last.pt checkpoint.
      --frame FRAME         Path to a single raw frame image.
      --config CONFIG       Config YAML to read architecture/preprocessing from
                            (default: config.yaml next to the checkpoint).
      --min-area MIN_AREA   Ignore predicted regions smaller than this (px^2) as
                            noise.
      --trapped-only        Draw only the trapped-cell class (label 1); suppress
                            other/decoy (label 2). No effect on binary
                            checkpoints, which only ever predict the trapped cell.
      --gt-mask GT_MASK     Hand-drawn mask to compare against (default: auto-
                            detected from
                            data/masks/<pool>[_multiclass]/<frame_stem>.png, if
                            one exists).
      --output OUTPUT       Where to save the figure (default:
                            results/<frame_stem>_<ckpt>_overlay.png).
      --device DEVICE
    ```

---

### `celldform-harvest-factory`

Segment every 03_mask_factory frame with the ensemble and extract classifier features.

Script: `scripts/inference/harvest_factory.py`

```bash
celldform-harvest-factory
celldform-harvest-factory --members s42 s43   # a subset
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-harvest-factory [-h] [--members MEMBERS [MEMBERS ...]] [--out OUT]
                              [--results RESULTS] [--min-area-px MIN_AREA_PX]
                              [--include-trained]

    Segment every 03_mask_factory frame with the ensemble and extract classifier
    features.

    options:
      -h, --help            show this help message and exit
      --members MEMBERS [MEMBERS ...]
      --out OUT
      --results RESULTS
      --min-area-px MIN_AREA_PX
                            in NATIVE pixels; applied as-is to the 256 run for
                            comparison
      --include-trained     keep factory frames the U-Net trained on (skipped by
                            default)
    ```

---

## Report figures and tables

### `celldform-build-report-assets`

Regenerate every figure and table in the current progress report.

Script: `scripts/report/build_report_assets.py`

```bash
celldform-build-report-assets                    # build everything
celldform-build-report-assets --list             # show jobs, run nothing
celldform-build-report-assets --only sweep pools # build a subset
celldform-build-report-assets --dry-run          # print commands only
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-build-report-assets [-h] [--report-root REPORT_ROOT]
                                  [--only ONLY [ONLY ...]] [--list] [--dry-run]
                                  [--python PYTHON] [--compile] [--tex TEX]
                                  [--feature-model {nnunet1000_loocv}]

    Rebuild every report figure and table

    options:
      -h, --help            show this help message and exit
      --report-root REPORT_ROOT
      --only ONLY [ONLY ...]
                            Job names to run (see --list)
      --list                List job names and exit
      --dry-run             Print commands, run nothing
      --python PYTHON
      --compile             After building assets, run pdflatex the required
                            number of times and delete the temp files
      --tex TEX             Report .tex to compile with --compile
      --feature-model {nnunet1000_loocv}
                            Segmentation run the feature-extraction figures and
                            tables are built from (jobs fe-dataset and features;
                            see celldform/features/segmenters.py)
    ```

---

### `celldform-make-report-tables`

Generate the progress report's LaTeX tables from the data.

Script: `scripts/report/make_report_tables.py`

```bash
celldform-make-report-tables percell --output ../masters_thesis/main_work/tables/percell.tex
celldform-make-report-tables headline --output ../masters_thesis/main_work/tables/headline.tex \
 \
    --run multiclass_experiment_256_bf64_base_cos="No augmentation (noise only)" \
 \
    --run multiclass_experiment_256_bf64_scale_cos="Scale augmentation" \
 \
    --run multiclass_experiment_256_bf64_allaug="Combined augmentation" \
 \
    --baseline multiclass_experiment_256_bf64_base_cos \
 \
    --emph multiclass_experiment_256_bf64_scale_cos
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-make-report-tables [-h] [--data-root DATA_ROOT] [--log LOG]
                                 [--run KEY[=LABEL]] [--extra-run KEY[=LABEL]]
                                 [--baseline BASELINE] [--emph EMPH]
                                 [--checkpoint CHECKPOINT] [--cell KEY[=LABEL]]
                                 [--base-features BASE_FEATURES]
                                 [--caption CAPTION] [--label LABEL]
                                 [--delta-header DELTA_HEADER] [--output OUTPUT]
                                 {percell,headline,baseline,decoy}

    Emit the report's LaTeX tables from the data

    positional arguments:
      {percell,headline,baseline,decoy}

    options:
      -h, --help            show this help message and exit
      --data-root DATA_ROOT
      --log LOG
      --run KEY[=LABEL]
      --extra-run KEY[=LABEL]
                            Row rendered in italics, below a rule — for runs that
                            are not comparable
      --baseline BASELINE   Run the delta column is measured against
      --emph EMPH           Run to bold, repeatable
      --checkpoint CHECKPOINT
                            Checkpoint dir name (baseline/decoy)
      --cell KEY[=LABEL]    Development cell, repeatable (baseline/decoy)
      --base-features BASE_FEATURES
      --caption CAPTION     Override the table caption (headline)
      --label LABEL         LaTeX label suffix, i.e. tab:<label>
      --delta-header DELTA_HEADER
                            Header for the delta column
      --output OUTPUT       Default: stdout
    ```

---

### `celldform-make-cell-gif`

Social-media GIF: one trapped cell across the laser-power ramp, raw beside the model's outline.

Script: `scripts/report/make_cell_gif.py`

```bash
celldform-make-cell-gif --cell high_cell2 --output results/social/high_cell2.gif
celldform-make-cell-gif --survey high_cell2 low_cell2 ...   # deformation per cell, no GIF
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-make-cell-gif [-h] [--cell CELL] [--survey SURVEY [SURVEY ...]]
                            [--frames FRAMES] [--output OUTPUT] [--half HALF]
                            [--round ROUND_]

    Social-media GIF: one trapped cell across the laser-power ramp, raw beside the
    model's outline.

    options:
      -h, --help            show this help message and exit
      --cell CELL
      --survey SURVEY [SURVEY ...]
      --frames FRAMES
      --output OUTPUT
      --half HALF           half-width of the view, native px
      --round ROUND_        use CV round k's model (the cell must be one of its
                            test cells)
    ```

---

### `celldform-diagnose-false-positives`

Show what a checkpoint's false-positive region actually *is*, on the raw frame.

Script: `scripts/report/diagnose_false_positives.py`

```bash
celldform-diagnose-false-positives --cell high_cell9 \
 \
    --checkpoint checkpoints/preswap/stage_4_augmentation/multiclass_experiment_256_bf64_scale_cos/unet_best.pt \
 \
    --output results/fp_diagnosis_high_cell9.png
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-diagnose-false-positives [-h] --cell CELL [--pool POOL] --checkpoint
                                       CHECKPOINT [--base-features BASE_FEATURES]
                                       [--n-frames N_FRAMES]
                                       [--half-size HALF_SIZE] [--output OUTPUT]

    Render a checkpoint's false-positive region on raw frames

    options:
      -h, --help            show this help message and exit
      --cell CELL
      --pool POOL
      --checkpoint CHECKPOINT
      --base-features BASE_FEATURES
      --n-frames N_FRAMES
      --half-size HALF_SIZE
                            Crop half-width/height in px
      --output OUTPUT
    ```

---

### `celldform-render-holdout-comparison`

Render the four-figure holdout comparison used in the thesis progress reports: a strip of N frames from one holdout cell, shown as (1) raw unannotated frames, (2) the multiclass ground-truth annotation (rendered via scripts/annotation/preview_masks.py's own color convention — not a re-drawn approximation), (3) ground truth vs. U-Net prediction on a black background (mask-only, no frame texture), and (4) the same comparison overlaid on the original frame.

Script: `scripts/report/render_holdout_comparison.py`

```bash
celldform-render-holdout-comparison \
 \
    --pool 02_unet_holdout --cell high_cell9 \
 \
    --checkpoint checkpoints/preswap/stage_3_regularisation/multiclass_experiment_256_bf32_reg2_cos/unet_best.pt
celldform-render-holdout-comparison \
 \
    --pool 02_unet_holdout --cell low_cell6 \
 \
    --checkpoint checkpoints/preswap/stage_1_labels_and_resolution/multiclass_experiment_256/unet_best.pt \
 \
    --base-features 64 --n-frames 5 --output-dir results/holdout_comparisons/low_cell6
celldform-render-holdout-comparison --pool 02_unet_holdout --cell high_cell9 \
 \
    --checkpoint checkpoints/preswap/stage_3_regularisation/multiclass_experiment_256_bf32_reg2_cos/unet_best.pt \
 \
    --gt-color 0,60,255 --half-width 140 --half-height 140   # pin crop + palette explicitly
celldform-render-holdout-comparison --pool 02_unet_holdout --cell high_cell1 \
 \
    --checkpoint checkpoints/postswap/experiment_0a_seed_baseline/postswap_ftl_cw142_s4{2,3,4,5}/unet_best.pt \
 \
    --power-labels 200 --n-frames 6
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-render-holdout-comparison [-h] [--pool POOL] --cell CELL
                                        [--mask-dir MASK_DIR]
                                        [--frame-dir FRAME_DIR]
                                        [--preprocessed-dir PREPROCESSED_DIR]
                                        --checkpoint CHECKPOINT [CHECKPOINT ...]
                                        [--base-features BASE_FEATURES]
                                        [--out-channels OUT_CHANNELS]
                                        [--in-channels IN_CHANNELS]
                                        [--n-frames N_FRAMES]
                                        [--frame-indices FRAME_INDICES]
                                        [--half-width HALF_WIDTH]
                                        [--half-height HALF_HEIGHT] [--pad PAD]
                                        [--decoy-radius DECOY_RADIUS]
                                        [--frame-labels] [--no-frame-labels]
                                        [--power-labels]
                                        [--label-color LABEL_COLOR]
                                        [--label-bg LABEL_BG] [--alpha ALPHA]
                                        [--gt-color GT_COLOR]
                                        [--pred-color PRED_COLOR]
                                        [--overlap-color OVERLAP_COLOR]
                                        [--border-color BORDER_COLOR]
                                        [--output-dir OUTPUT_DIR]
                                        [--output-prefix OUTPUT_PREFIX]

    Render the raw/labelled-mask/masks-only/overlay holdout comparison strip

    options:
      -h, --help            show this help message and exit
      --pool POOL           Pool folder name under data/frames/ and
                            data/preprocessed/evaluate/
      --cell CELL           Cell filename prefix, e.g. high_cell9
      --mask-dir MASK_DIR   Override GT mask directory (default:
                            data/masks/<pool>_multiclass)
      --frame-dir FRAME_DIR
                            Override raw frame directory (default:
                            data/frames/<pool>)
      --preprocessed-dir PREPROCESSED_DIR
                            Override preprocessed input directory (default:
                            data/preprocessed/evaluate/<pool>/inputs)
      --checkpoint CHECKPOINT [CHECKPOINT ...]
                            One checkpoint, or several to run them as an ensemble
                            (EnsembleUNet: averaged softmax, the deployed
                            configuration)
      --base-features BASE_FEATURES
                            Overrides the checkpoint dir's own config.yaml when
                            given
      --out-channels OUT_CHANNELS
      --in-channels IN_CHANNELS
      --n-frames N_FRAMES   Evenly-spaced frames to render (ignored if --frame-
                            indices given)
      --frame-indices FRAME_INDICES
                            Comma-separated 6-digit frame indices, e.g.
                            000000,002000,004000
      --half-width HALF_WIDTH
                            Pin crop half-width in px
      --half-height HALF_HEIGHT
                            Pin crop half-height in px
      --pad PAD             Padding (px) added around the auto-computed union bbox
      --decoy-radius DECOY_RADIUS
                            Max distance (px) from the trapped-cell centroid for a
                            decoy blob to count toward the auto-crop (excludes
                            far-flung debris)
      --frame-labels        Print each panel's frame index in a band above it
                            (default)
      --no-frame-labels     Render bare panels with no frame-index band
      --power-labels        Label panels by laser increment instead of frame
                            index. The acquisition ramps power in ~50 increments
                            and frames are sampled one per increment, so a frame's
                            rank in the cell's full sorted frame list IS its
                            increment: 'step 20/50'. Derived from the rank, not
                            from a fixed frames-per-step divisor -- the videos are
                            not all the same length, and cell 1's last frame
                            (010200) would come out as step 51 of 50 under a fixed
                            200.
      --label-color LABEL_COLOR
                            R,G,B for the frame-index text
      --label-bg LABEL_BG   R,G,B for the frame-index band background
      --alpha ALPHA         Overlay blend strength (0-1)
      --gt-color GT_COLOR   R,G,B for ground truth
      --pred-color PRED_COLOR
                            R,G,B for prediction
      --overlap-color OVERLAP_COLOR
                            R,G,B for overlap
      --border-color BORDER_COLOR
                            R,G,B panel border
      --output-dir OUTPUT_DIR
                            Default: results/holdout_comparisons/<pool>_<cell>
      --output-prefix OUTPUT_PREFIX
                            Default: <cell>
    ```

---

### `celldform-view-soft-target`

Look at a teacher target (.npy) from the distillation data: the frame, its label, and the teacher's per-class probabilities as heat maps.

Script: `scripts/report/view_soft_target.py`

```bash
celldform-view-soft-target --round 4                       # a random training frame
celldform-view-soft-target --round 4 --stem high_cell7_002525
celldform-view-soft-target --round 4 --n 6 --output soft.png   # a grid of 6
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-view-soft-target [-h] --round ROUND [--stem [STEM ...]] [--n N]
                               [--output OUTPUT]

    Look at a teacher target (.npy) from the distillation data: the frame, its
    label, and the teacher's per-class probabilities as heat maps.

    options:
      -h, --help         show this help message and exit
      --round ROUND
      --stem [STEM ...]
      --n N
      --output OUTPUT
    ```

---

### `celldform-plot-cell-sizes`

Per-cell trapped-cell size, measured from the annotation masks.

Script: `scripts/report/plot_cell_sizes.py`

```bash
celldform-plot-cell-sizes --output results/cell_sizes.png
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-cell-sizes [-h] [--data-root DATA_ROOT] [--output OUTPUT]
                              [--dpi DPI]

    Per-cell trapped-cell size

    options:
      -h, --help            show this help message and exit
      --data-root DATA_ROOT
      --output OUTPUT
      --dpi DPI
    ```

---

### `celldform-plot-cv`

Figure and tables for the randswap cross-validation (1 October report).

Script: `scripts/report/plot_cv.py`

```bash
celldform-plot-cv --exp randswap/experiment_0_cv_baseline --run cv_baseline \
    --out ../masters_thesis/main_work/images --tables ../masters_thesis/main_work/tables
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-cv [-h] --exp EXP --run RUN --out OUT --tables TABLES
                      [--compare-exp COMPARE_EXP] [--compare-run COMPARE_RUN]
                      [--distill-exp DISTILL_EXP] [--distill-run DISTILL_RUN]

    Figure and tables for the randswap cross-validation (1 October report).

    options:
      -h, --help            show this help message and exit
      --exp EXP
      --run RUN
      --out OUT
      --tables TABLES
      --compare-exp COMPARE_EXP
                            a second CV run to benchmark against (nnU-Net)
      --compare-run COMPARE_RUN
      --distill-exp DISTILL_EXP
                            student CV to set against ours and nnU-Net
      --distill-run DISTILL_RUN
    ```

---

### `celldform-plot-data-map`

Dataset Organization figure (randswap): where every cell and frame goes.

Script: `scripts/report/plot_data_map.py`

```bash
celldform-plot-data-map --exp randswap/experiment_0_cv_baseline \
    --run cv_baseline --output ../masters_thesis/main_work/images/data_map.png
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-data-map [-h] --exp EXP --run RUN [--factory FACTORY] --output
                            OUTPUT

    Dataset Organization figure (randswap): where every cell and frame goes.

    options:
      -h, --help         show this help message and exit
      --exp EXP
      --run RUN
      --factory FACTORY
      --output OUTPUT
    ```

---

### `celldform-plot-experiments`

Figures for the experiments chapter of the 1 October report.

Script: `scripts/report/plot_experiments.py`

```bash
celldform-plot-experiments --out ../masters_thesis/main_work/images
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-experiments [-h] --out OUT [--tables TABLES]

    Figures for the experiments chapter of the 1 October report.

    options:
      -h, --help       show this help message and exit
      --out OUT
      --tables TABLES  where the LaTeX tables go
    ```

---

### `celldform-plot-model-comparison`

The four candidate segmenters side by side, on the same 15 cross-validated cells: one small panel per measure, every value labelled. Used by the report to choose the deployment models.

Script: `scripts/report/plot_model_comparison.py`

```bash
celldform-plot-model-comparison --out ../masters_thesis/main_work/images \
    --tables ../masters_thesis/main_work/tables
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-model-comparison [-h] --out OUT --tables TABLES

    The four candidate segmenters side by side, on the same 15 cross-validated
    cells: one small panel per measure, every value labelled. Used by the report
    to choose the deployment models.

    options:
      -h, --help       show this help message and exit
      --out OUT
      --tables TABLES
    ```

---

### `celldform-plot-nnunet-epochs`

nnU-Net at 100 against 1000 epochs on the same randswap rounds (8 October report). Reads both runs' native eval logs (the latest entry per round and cell) and nnU-Net's own training logs; nothing is typed in. Writes `nnunet_epochs.png` — (a) every cell's Test DSC at 100 and 1000 epochs, (b) each round's internal validation Dice over the 1000 epochs — and `nnunet_epochs.tex`.

Script: `scripts/report/plot_nnunet_epochs.py` · build job: `celldform-build-report-assets --only nnunet-epochs`

```bash
celldform-plot-nnunet-epochs \
    --out ../masters_thesis/main_work/images --tables ../masters_thesis/main_work/tables
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-nnunet-epochs [-h] --out OUT --tables TABLES

    nnU-Net at 100 against 1000 epochs on the same randswap rounds (8 October
    report).

    options:
      -h, --help       show this help message and exit
      --out OUT        folder for nnunet_epochs.png
      --tables TABLES  folder for nnunet_epochs.tex
    ```

---

### `celldform-plot-percell-metrics`

Grouped bar chart of per-cell holdout metrics, read straight out of results/preswap/preswap_eval_log.jsonl — one panel per evaluation scope (each development-set cell, plus all cells together), grouped bars per metric, one bar per checkpoint.

Script: `scripts/report/plot_percell_metrics.py`

```bash
celldform-plot-percell-metrics \
 \
    --checkpoint multiclass_experiment_256=Original \
 \
    --checkpoint multiclass_experiment_256_bf64_scale="Scale only" \
 \
    --checkpoint multiclass_experiment_256_bf64_allaug=Combined \
 \
    --scope high_cell9="high-9 (over-segments)" \
 \
    --scope low_cell6="low-6 (under-detects)" \
 \
    --scope all_cells=All\ cells \
 \
    --output results/percell_metrics.png
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-percell-metrics [-h] [--log LOG] --checkpoint KEY[=LABEL]
                                   --scope KEY[=LABEL] [--metric KEY[=LABEL]]
                                   [--class-id CLASS_ID] [--highlight KEY]
                                   [--title TITLE] [--output OUTPUT] [--dpi DPI]

    Per-cell holdout metric bar chart

    options:
      -h, --help            show this help message and exit
      --log LOG
      --checkpoint KEY[=LABEL]
                            Checkpoint name substring, repeatable. The first is
                            the baseline.
      --scope KEY[=LABEL]   `--filter` value from the eval log, or `all_cells`.
                            One panel each.
      --metric KEY[=LABEL]  Metric key(s) from the log (default: dsc, precision)
      --class-id CLASS_ID   Class key in the log's metrics dict (default: 1 =
                            trapped cell)
      --highlight KEY       Checkpoint key to draw in the accent color; every
                            other series goes muted. Without it, colors follow
                            --checkpoint order. Use this so the accent lands on
                            the result being argued for, not on whichever run
                            happens to be listed last.
      --title TITLE
      --output OUTPUT
      --dpi DPI
    ```

---

### `celldform-plot-pool-flow`

Provenance diagram of the frame pools: which cells go where, what draws from what, and where the evaluation firewall sits.

Script: `scripts/report/plot_pool_flow.py`

```bash
celldform-plot-pool-flow --output results/pool_flow.png
celldform-plot-pool-flow --data-root data --output results/pool_flow.png
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-pool-flow [-h] [--data-root DATA_ROOT] [--output OUTPUT]
                             [--dpi DPI]

    Frame-pool provenance diagram

    options:
      -h, --help            show this help message and exit
      --data-root DATA_ROOT
      --output OUTPUT
      --dpi DPI
    ```

---

### `celldform-plot-resize-vs-crop`

Experiment 3 figure: the same frames as each input variant sees them.

Script: `scripts/report/plot_resize_vs_crop.py`

```bash
celldform-plot-resize-vs-crop --output results/exp3_resize_vs_crop.jpg
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-resize-vs-crop [-h] --output OUTPUT

    Experiment 3 figure: the same frames as each input variant sees them.

    options:
      -h, --help       show this help message and exit
      --output OUTPUT
    ```

---

### `celldform-plot-split-change`

Before/after of the train/validation/test split as two stacked bars.

Script: `scripts/report/plot_split_change.py`

```bash
celldform-plot-split-change --output results/split_change.png
celldform-plot-split-change --before 293 78 54 --after 371 54 0
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-split-change [-h] [--before TRAIN VAL TEST]
                                [--after TRAIN VAL TEST] [--output OUTPUT]
                                [--dpi DPI]

    Train/val/test split, before and after

    options:
      -h, --help            show this help message and exit
      --before TRAIN VAL TEST
      --after TRAIN VAL TEST
      --output OUTPUT
      --dpi DPI
    ```

---

### `celldform-plot-sweep-dsc`

Horizontal bar chart ranking every checkpoint in results/preswap/preswap_eval_log.jsonl by its evaluated DSC on the development set.

Script: `scripts/report/plot_sweep_dsc.py`

```bash
celldform-plot-sweep-dsc --output results/sweep_dsc.png \
 \
    --highlight multiclass_experiment_256_bf64_scale_cos \
 \
    --warn multiclass_experiment_256_bf64_reg \
 \
    --label multiclass_experiment_256_bf64_scale_cos="Scale augmentation"
```

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-sweep-dsc [-h] [--log LOG] [--class-id CLASS_ID]
                             [--metric METRIC] [--highlight HIGHLIGHT]
                             [--warn WARN] [--label KEY=LABEL] [--exclude EXCLUDE]
                             [--band N=LABEL] [--band-rest-label BAND_REST_LABEL]
                             [--row-height ROW_HEIGHT] [--xlabel XLABEL]
                             [--output OUTPUT] [--dpi DPI]

    Rank all checkpoints by evaluated DSC

    options:
      -h, --help            show this help message and exit
      --log LOG
      --class-id CLASS_ID
      --metric METRIC
      --highlight HIGHLIGHT
                            Checkpoint dir name to draw in the accent color,
                            repeatable
      --warn WARN           Checkpoint dir name to draw in the warning color,
                            repeatable
      --label KEY=LABEL     Override the display name for one checkpoint,
                            repeatable
      --exclude EXCLUDE     Checkpoint dir name to leave out entirely, repeatable
      --band N=LABEL        Shade the top N ranked bars as one group and annotate
                            it. Use when the ranking spans two training setups
                            that are not comparable to each other, so a reader can
                            see which bars belong to which.
      --band-rest-label BAND_REST_LABEL
                            Annotation for the bars below the band.
      --row-height ROW_HEIGHT
                            Vertical inches per bar. Lower it when the chart has
                            to share a printed page with body text rather than
                            stranding one.
      --xlabel XLABEL
      --output OUTPUT
      --dpi DPI
    ```

---

### `celldform-plot-fe-dataset`

Dataset Organization figure for the feature-extraction reports (22 October report onward).

Script: `scripts/report/plot_fe_dataset.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-fe-dataset [-h]
                              [--model {nnunet100_5fold,nnunet1000_5fold,nnunet1000_loocv}]
                              --output OUTPUT [--rounds-output ROUNDS_OUTPUT]
                              [--table-output TABLE_OUTPUT]

    Dataset Organization figure for the feature-extraction reports (22 October
    report onward).

    options:
      -h, --help            show this help message and exit
      --model {nnunet100_5fold,nnunet1000_5fold,nnunet1000_loocv}
      --output OUTPUT
      --rounds-output ROUNDS_OUTPUT
      --table-output TABLE_OUTPUT
    ```

### `celldform-plot-features`

Feature-extraction figures and tables (22 October report).

Script: `scripts/report/plot_features.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-features [-h] [--model MODEL] --out OUT --tables TABLES

    Feature-extraction figures and tables (22 October report).

    options:
      -h, --help       show this help message and exit
      --model MODEL
      --out OUT
      --tables TABLES
    ```

### `celldform-plot-final-models`

"Final Models at a Glance" -- the closing page of the 8 October report (written 2026-10-21).

Script: `scripts/report/plot_final_models.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-final-models [-h] --out OUT --tables TABLES

    "Final Models at a Glance" -- the closing page of the 8 October report
    (written 2026-10-21).

    options:
      -h, --help       show this help message and exit
      --out OUT
      --tables TABLES
    ```

### `celldform-plot-high1-exclusion`

Why high-1 is excluded from the features (22 October report, feature extraction).

Script: `scripts/report/plot_high1_exclusion.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-high1-exclusion [-h] [--model MODEL] --output OUTPUT
                                   [--table-output TABLE_OUTPUT]
                                   [--ref-output REF_OUTPUT]
                                   [--ref-table-output REF_TABLE_OUTPUT]

    high-1, the cell drawn largest against its hand masks (feature-extraction
    report).

    options:
      -h, --help            show this help message and exit
      --model MODEL         run whose saved test-frame masks panel (a) draws
      --output OUTPUT
      --table-output TABLE_OUTPUT
                            LaTeX table of high-1's pixel counts
      --ref-output REF_OUTPUT
                            the same tile figure for the reference cell (high-9)
      --ref-table-output REF_TABLE_OUTPUT
                            LaTeX table of the reference cell's pixel counts
    ```

### `celldform-plot-high10`

Why high-10 is over-predicted in leave-one-cell-out round 14 (8 October report, added 27 October).

Script: `scripts/report/plot_high10.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-high10 [-h] [--output OUTPUT] [--table-output TABLE_OUTPUT]

    Why high-10 is over-predicted in leave-one-cell-out round 14 (29 October
    report).

    options:
      -h, --help            show this help message and exit
      --output OUTPUT
      --table-output TABLE_OUTPUT
    ```

### `celldform-plot-classification-thesis`

Classification chapter figures from results on disk: every model's AUC and F1, per-cell P(high), the best model's permutation null and its summary card.

Script: `scripts/report/plot_classification_thesis.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-classification-thesis [-h] [--model MODEL] --img IMG

    Thesis-style figures for the HER2 classification chapter (rewrite of 2026-11-03).

        python scripts/report/plot_classification_thesis.py --model nnunet1000_loocv --img DIR

    Reads only results already on disk (results/classification/<model>/): combine_cells.csv / combine_metrics.csv
    (every model's held-out P(high) per cell), benchmark_cells.csv (permutation p of the 16 machine-learning models),
    scores_cells.csv (their P(high) per cell), rank_nested.csv, seq_metrics.csv, tuning_scores.csv / tuning_cells.csv,
    combine_null.csv (the deployment model's permutation null). Nothing is refitted.

    Figures
        cls_landscape.png    every model of the study on one page: AUC with its 95 % cell-bootstrap interval and
                             F1 of each class, grouped by family; filled marker = permutation p < 0.05
        cls_percell_map.png  held-out P(high) of the finalists and the two controls for every cell, cells ordered by size
        cls_deploy_null.png  what chance gives the deployment model: 1 000 label shuffles of the whole procedure

    options:
      -h, --help     show this help message and exit
      --model MODEL
      --img IMG
    ```

### `celldform-plot-analysis-dataset`

Dataset page of the analysis reports: what each cell contributes, and the leave-one-cell-out split.

Script: `scripts/report/plot_analysis_dataset.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-analysis-dataset [-h] [--model MODEL] --img IMG

    Dataset page of the analysis reports: what each cell contributes, and the leave-one-cell-out split.

    Used by the classification and deformation reports.

        python scripts/report/plot_analysis_dataset.py --model nnunet1000_loocv --img <dir>

        analysis_dataset.png  one row per cell: its feature frames (passed / failed the frame checks), laser steps,
                              recording position (trap height) and whether it is used
        analysis_split.png    the leave-one-cell-out split: 14 folds x 14 cells, the test cell of each fold, and the
                              frames on each side

    Reads results/features/<model>/features_frames.csv and features_cells.csv only.

    options:
      -h, --help     show this help message and exit
      --model MODEL
      --img IMG
    ```

### `celldform-plot-loocv`

Leave-one-cell-out against the 5-fold cross-validation, for the U-Net and nnU-Net (15 October report).

Script: `scripts/report/plot_loocv.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-loocv [-h] --out OUT --tables TABLES

    Leave-one-cell-out against the 5-fold cross-validation, for the U-Net and nnU-
    Net (15 October report).

    options:
      -h, --help       show this help message and exit
      --out OUT        folder for loocv.png
      --tables TABLES  folder for loocv.tex
    ```

### `celldform-plot-protocols`

Dataset Organization figure, general form (15 October report): one rule, three protocols.

Script: `scripts/report/plot_protocols.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-protocols [-h] --output OUTPUT

    Dataset Organization figure, general form (15 October report): one rule, three
    protocols.

    options:
      -h, --help       show this help message and exit
      --output OUTPUT
    ```

### `celldform-plot-region-selection`

How the extraction picks the trapped cell, and why it drops a frame -- drawn on the predicted masks (22 October report, feature extraction).

Script: `scripts/report/plot_region_selection.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-plot-region-selection [-h] [--model MODEL] --output OUTPUT
                                    --dropped-output DROPPED_OUTPUT
                                    [--table-output TABLE_OUTPUT]
                                    [--checks-output CHECKS_OUTPUT]
                                    [--cross-output CROSS_OUTPUT] [--n N]

    How the extraction picks the trapped cell, and why it drops a frame -- drawn
    on the predicted masks (22 October report, feature extraction).

    options:
      -h, --help            show this help message and exit
      --model MODEL
      --output OUTPUT       figure (a): frames with more than one region
      --dropped-output DROPPED_OUTPUT
                            figure (b): dropped frames
      --table-output TABLE_OUTPUT
                            table: every rejected frame, by the check it failed
      --checks-output CHECKS_OUTPUT
                            table: how each frame check is computed, its limit,
                            good masks
      --cross-output CROSS_OUTPUT
                            table: the checks on frames that did not set their
                            limits
      --n N                 tiles per row
    ```

### `celldform-diagnose-mask-growth`

Where do a model's extra pixels go? Compare several runs' saved predictions of ONE test cell.

Script: `scripts/report/diagnose_mask_growth.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-diagnose-mask-growth [-h] --cell CELL --run LABEL=DIR [--ring RING]
                                   [--n-strip N_STRIP] [--half HALF] --out OUT

    Where do a model's extra pixels go? Compare several runs' saved predictions of
    ONE test cell.

    options:
      -h, --help         show this help message and exit
      --cell CELL        e.g. high_cell10
      --run LABEL=DIR    a run's prediction folder; repeat for each run to compare
      --ring RING        ring width around the true boundary, px
      --n-strip N_STRIP  frames in the overlay strip
      --half HALF        crop half-size around the cell, px
      --out OUT
    ```

## Features and classification

### `celldform-extract-features`

Classifier features from the feature frames -- segment, validate every segmentation, then extract.

Script: `scripts/features/extract_features.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-extract-features [-h]
                                      [--model {nnunet100_5fold,nnunet1000_5fold,nnunet1000_loocv}]
                                      [--config CONFIG]
                                      {cut,segment,calibrate,extract,all}

    Classifier features from the feature frames -- segment, validate every
    segmentation, then extract.

    positional arguments:
      {cut,segment,calibrate,extract,all}

    options:
      -h, --help            show this help message and exit
      --model {nnunet100_5fold,nnunet1000_5fold,nnunet1000_loocv}
      --config CONFIG       settings file (default configs/analysis/features.yaml)
    ```

### `celldform-benchmark-cells`

Per-cell HER2 benchmark: seven classical classifiers on the per-cell features, leave-one-cell-out.

Script: `scripts/classification/benchmark_cells.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-benchmark-cells [-h] [--model MODEL] [--config CONFIG]
                              [--perms PERMS] [--out OUT] [--replot]
                              [--only ONLY [ONLY ...]] [--selection-only]
                              [--final]

    Per-cell HER2 benchmark: seven classical classifiers on the per-cell features, leave-one-cell-out.

        python scripts/classification/benchmark_cells.py --model nnunet1000_loocv [--perms 1000]

    One row per usable cell (features_cells.csv, usable == True), label high = 1. Every model sees the same
    14 folds; scaling and missing-value filling are fitted inside each fold. Hyperparameters are FIXED in
    advance (no tuning: with 13 training cells an inner search would mostly fit noise).

    Feature sets
        selected  chosen from the 29 by a label-free selection, once, before any classifier (main):
                    1. agreement  the per-cell value from the predicted masks against the same from the hand masks,
                                  on the hand-masked frames: Spearman rho across the cells >= 0.8 (where testable)
                    2. split-half the per-cell value from the even laser steps against the same from the odd ones,
                                  on the feature frames: Spearman rho >= 0.8
                    3. groups     the candidates are grouped by the correlation matrix (average-linkage clustering on
                                  1 - |Spearman rho| across the cells, cut at |rho| = 0.7, as AutoSpearman); from each
                                  group only its highest-ranked passing feature is kept
                  ranked by the weaker of the two rho; every group's best passing feature is kept (no cap). The HER2 labels are never
                  used, so the selection cannot leak the held-out cell.  -> feature_selection.csv / .tex
        all       every per-cell shape and deformation feature, no selection                       (sensitivity)
    Baselines
        size      baseline length along the beam only
        image     per-cell medians of the whole-frame image measures (illumination, unevenness, contrast,
                  noise). These track the recording session, not the cell: a classifier that does well on
                  them is reading the session. Its score is the bar the cell features must clear.

    Scores: correct cells, balanced accuracy, AUC of the held-out P(high), and a permutation p-value
    (labels shuffled across cells, the whole leave-one-cell-out rerun, balanced accuracy).

    Outputs in results/classification/<model>/:  feature_selection.csv / .tex, benchmark_cells.csv, scores_cells.csv
    (every held-out cell's P(high) per model), benchmark_cells.png, benchmark_cells.tex.

    Final models: after the run, every classifier on each feature set (and the size baseline) is fitted once on ALL
    usable cells and saved with its leave-one-cell-out score -- the models a new cell is classified with
    (celldform.final_models). --final does only this, from the saved selection and scores.
        -> checkpoints/classification/<model>/classical/<set>__<classifier>.joblib, manifest.csv

    options:
      -h, --help            show this help message and exit
      --model MODEL         feature run in results/features/
      --config CONFIG       settings file (default
                            configs/analysis/classifiers.yaml)
      --perms PERMS         label shuffles for the permutation p (0 = skip;
                            default: the config's)
      --out OUT             output folder (default results/classification/<model>)
      --replot              redraw the figure and table from the saved CSVs only
      --only ONLY [ONLY ...]
                            run only these classifiers (config names) and merge
                            them into the saved results
      --selection-only      only run the feature selection and write
                            feature_selection.csv (no classifiers)
      --final               only fit the final models on all cells and save them
                            (uses the saved selection and scores)
    ```

### `celldform-frames-cnn`

Per-frame HER2 classifier (deep learning), judged per held-out cell.

Script: `scripts/classification/frames_cnn.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-frames-cnn [-h] [--model MODEL] [--config CONFIG] [--epochs EPOCHS]
                         [--device DEVICE] [--seed SEED]
                         [--variants {masked,mask} [{masked,mask} ...]]
                         [--seeds SEEDS [SEEDS ...]]
                         {crops,frozen,finetune,tune,final,report,all}

    Per-frame HER2 classifier (deep learning), judged per held-out cell.

        python scripts/classification/frames_cnn.py crops      --model nnunet1000_loocv   # cut the cell crops once
        python scripts/classification/frames_cnn.py frozen     --model nnunet1000_loocv   # ResNet-18 features + LR
        python scripts/classification/frames_cnn.py finetune   --model nnunet1000_loocv   # fine-tune the last block
        python scripts/classification/frames_cnn.py report     --model nnunet1000_loocv   # table + figure
        python scripts/classification/frames_cnn.py final      --model nnunet1000_loocv --seeds 0 1 2
                                         # train on ALL cells, save to checkpoints/classification/<model>/cnn/

    Data: every passing feature frame of the usable cells (~3 800 frames, ~6 per laser step).

    Input -- built so the network cannot read the recording session from the frame:
        masked   a CROP x CROP px window centred on the trapped cell (the region the extraction kept), every pixel
                 outside the (slightly grown) mask set to zero, intensity standardised over the cell's own pixels.
                 The scale is NOT normalised: the cell's size stays in the input, as in the per-cell features.
        mask     the binary mask alone, same window (shape and size only; the control).

    Models (ImageNet ResNet-18, grey crop copied to 3 channels):
        frozen    the pretrained network as a fixed feature extractor (512 numbers per frame) + logistic regression.
        finetune  the last residual block and a new head trained per fold (flips, 90-degree turns, mild intensity
                  change), the laser step joined to the image features before the head. Cells are weighted equally
                  (each cell's frames share one unit of weight), a fixed number of epochs (no peeking at the held-out
                  cell).

    Test: leave-one-cell-out, the same 14 folds as benchmark_cells.py. A held-out cell's frame probabilities are
    averaged into one P(high) for the cell -- what a live system watching the cell would do.

    Outputs in results/classification/<model>/:  cnn_frames_<variant>.csv (every frame's held-out P(high)),
    cnn_cells.csv, cnn_cells.tex, cnn_frames_seen.png (accuracy against the number of laser steps watched).

    positional arguments:
      {crops,frozen,finetune,tune,final,report,all}
                            tune: nested-CV hyperparameter tuning of the fine-
                            tuned cell-crop CNN; final: train on all cells and
                            save to checkpoints/classification/<model>/cnn/
                            (neither is part of all)

    options:
      -h, --help            show this help message and exit
      --model MODEL
      --config CONFIG       settings file (default configs/analysis/cnn.yaml)
      --epochs EPOCHS       fine-tuning epochs per fold (default: the config's)
      --device DEVICE
      --seed SEED           fine-tuning seed (default: the config's)
      --variants {masked,mask} [{masked,mask} ...]
      --seeds SEEDS [SEEDS ...]
                            final step: one saved model per seed (default: the
                            config's seed)
    ```

### `celldform-power-resolution`

How many cells would a conclusive experiment need, and how small a deformation can the system resolve?

Script: `scripts/classification/power_resolution.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-power-resolution [-h] [--model MODEL] [--tables TABLES]

    How many cells would a conclusive experiment need, and how small a deformation can the system resolve?

        python scripts/classification/power_resolution.py --model nnunet1000_loocv [--tables <report tables dir>]

    Cells needed (from the usable cells of features_cells.csv and the selected features of benchmark_cells.py)
        per feature   Cohen's d between the classes (pooled SD), and the cells per class a two-sided Mann--Whitney
                      test needs to find a difference that size at alpha 0.05 with 80 % power:
                          n = 2 (z_0.975 + z_0.8)^2 / d^2 / 0.955     (0.955: its efficiency against the t-test)
                      d from 14 cells is itself uncertain (its 95 % interval is roughly +/-1.1), so n is an order
                      of magnitude, not a design figure.
        classifier    the cells (balanced classes) a one-sided binomial test needs to show that a classifier of a
                      given true accuracy beats 50 % at alpha 0.05 with 80 % power.

    Resolution (from the passing feature frames: length along the beam, L_par)
        frame noise   median over cells and steps of SD/mean of L_par within a laser step (~6 frames)
        step noise    (a) frame noise / sqrt(frames per step) x 1.25 (the median's efficiency)
                      (b) from the step-to-step differences of each cell's step medians: SD(diff)/sqrt(2) -- an
                          upper bound, it also contains the cell's real step-to-step change
        detectable    a change between two steps: 2.77 x step noise (95 %); the strain over the ramp (last ten steps
                      against the baseline ten): 1.96 x sqrt(2/10) x step noise x 1.25; a compliance: 1.96 x the
                      standard error of the slope over 50 steps (x 1.1, the Theil--Sen efficiency)
        compared with the observed per-cell compliances and the difference between the class means.
        Random noise only: a segmentation error that grows with the laser power (high-1) is not in it.

    Outputs in results/classification/<model>/: power.csv, resolution.csv, classify_power.tex, classify_resolution.tex.

    options:
      -h, --help       show this help message and exit
      --model MODEL    feature run in results/features/
      --tables TABLES  also write the LaTeX tables here
    ```

### `celldform-tune-cells`

Hyperparameter tuning of the classical classifiers by nested leave-one-cell-out cross-validation.

Script: `scripts/classification/tune_cells.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-tune-cells [-h] [--model MODEL] [--config CONFIG] [--perms PERMS]

    Hyperparameter tuning of the classical classifiers by nested leave-one-cell-out cross-validation.

        python scripts/classification/tune_cells.py --model nnunet1000_loocv [--config <yaml>] [--perms N]

    Outer loop: each of the 14 cells is the test cell once. Inner loop, on the other 13 cells only: every point of the
    classifier's grid (configs/analysis/classifiers.yaml, `tuning.grids`) is scored by leave-one-cell-out over the 13,
    by the Brier score of their held-out P(high) (smoother than counting 13 calls; ties go to the earlier grid point).
    The best point is refitted on the 13 and the test cell is scored once. The test cell never influences the choice,
    so the outer scores are an honest estimate of a TUNED classifier -- comparable with the untuned benchmark
    (benchmark_cells.py) on the same folds and the same selected features.

    Permutation p: the cell labels shuffled and the WHOLE nested procedure rerun (`tuning.permutations`; the slow
    tree ensembles are skipped). Outputs in results/classification/<model>/: tuning_cells.csv (per classifier),
    tuning_choices.csv (the grid point chosen in every outer fold), tuning_scores.csv (held-out P(high)),
    config_tuning.yaml.

    options:
      -h, --help       show this help message and exit
      --model MODEL    feature run in results/features/
      --config CONFIG  settings (default configs/analysis/classifiers.yaml)
      --perms PERMS    label shuffles (default: the config's tuning.permutations)
    ```

### `celldform-sequence-net`

Sequence networks on the deformation curves: can HER2 be told from HOW a cell deforms over the laser ramp?

Script: `scripts/classification/sequence_net.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-sequence-net [-h] [--model MODEL] [--config CONFIG] [--perms PERMS]

    Sequence networks on the deformation curves: can HER2 be told from HOW a cell deforms over the laser ramp?

        python scripts/classification/sequence_net.py --model nnunet1000_loocv [--config <yaml>] [--perms N]

    Input, one per cell: the strain (%) of each measure (configs/analysis/sequence.yaml: length along and across the
    beam, volume) at every laser step -- the median of the step's passing frames against the cell's own baseline
    (celldform.biomechanics.rod.step_strain) -- a (measures x steps) curve. Missing steps are interpolated; a cell
    with fewer steps is padded with its last value.

    Models (deep learning, tiny on purpose: 13 training curves per fold):
        cnn1d  two 1-D convolution layers over the steps, global average pooling, a linear output
        gru    one recurrent (GRU) layer, its last state, a linear output

    Training: leave-one-cell-out, the same 14 folds as every other model; each channel scaled by its SD over the
    training cells; class-balanced loss; a random window of `crop` steps, noise and amplitude scaling per epoch; a
    fixed number of epochs (no validation set). The held-out cell is scored on its whole curve; P(high) is averaged
    over the seeds. Permutation p: labels shuffled and the whole leave-one-cell-out rerun (first seed only).

    Outputs in results/classification/<model>/: seq_cells.csv (every cell's P(high), per model and seed),
    seq_metrics.csv, config_sequence.yaml.

    options:
      -h, --help       show this help message and exit
      --model MODEL    feature run in results/features/
      --config CONFIG  settings (default configs/analysis/sequence.yaml)
      --perms PERMS    label shuffles (default: the config's)
    ```

### `celldform-feature-ranking`

Feature rankings x k x classifiers, best-of-grid and nested choice, with permutation tests: does choosing features with the labels help?

Script: `scripts/classification/feature_ranking.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-feature-ranking [-h] [--model MODEL] [--perms PERMS]
                              [--nested-perms NESTED_PERMS]

    Which features, chosen how? Six feature rankings x seven feature counts x the eight classifiers, leave-one-cell-out.

        python scripts/classification/feature_ranking.py --model nnunet1000_loocv [--perms N] [--nested-perms N]

    The label-free selection (benchmark_cells.py) fixes 7 features before any classifier. This asks whether other
    choices -- rankings that DO use the labels, and other feature counts -- do better. A ranking that uses the labels is
    refitted inside every fold on the training cells only, so the held-out cell never influences which features it gets.

    Candidates: the 29 per-cell features (15 shape medians + 14 deformation summaries; benchmark_cells.SHAPE + DEFORMATION).
    Missing values are filled with the training cells' median and the features standardised before ranking.

    Rankings (each orders the 29 features; the top k are used)
        label-free   the fixed label-free selection's order (feature_selection.csv), then the rest by its score -- the
                     same in every fold
        univariate   |AUC - 0.5| of each feature alone on the training cells (Mann-Whitney)
        mRMR         minimum redundancy, maximum relevance: greedily the feature with the highest univariate relevance
                     minus its mean |Spearman rho| with the features already chosen
        L1 path      L1 logistic regression on a falling penalty: the order in which coefficients become non-zero
        forest       random-forest impurity importance (300 trees)
        RFE          recursive feature elimination with logistic regression: drop the smallest |coefficient|, refit
    Feature counts k: 1, 2, 3, 5, 7, 10, 29 (29 = every feature; the same for every ranking).
    Classifiers: the eight of configs/analysis/classifiers.yaml, settings fixed (benchmark_cells.models()).

    Three readings, from least to most honest
        grid         every (ranking, k, classifier): its leave-one-cell-out score. Exploratory: the best of 42 settings
                     per classifier is optimistic by construction.
        best of grid p   the permutation null of the BEST of the 42 settings (labels shuffled across cells, every ranking
                     and every fit rerun, the maximum AUC / balanced accuracy over the grid kept): the chance of a grid
                     maximum this high. Fast classifiers only (the linear models, RBF SVM, k-NN).
        nested       the ranking and k chosen INSIDE each fold, by leave-one-out over its training cells (lowest Brier
                     score), then the test cell scored once: an honest estimate of "pick the best features" as a method.
                     Logistic regression and k-NN, with a permutation p of the whole nested procedure.

    Outputs in results/classification/<model>/: rank_grid.csv (every setting), rank_cells.csv (held-out P(high) of every
    setting and the nested runs), rank_freq.csv (how often each feature is in the top 5, per ranking), rank_nested.csv,
    rank_null.npz. The thesis figures are drawn by scripts/report/plot_classification_thesis.py.

    options:
      -h, --help            show this help message and exit
      --model MODEL
      --perms PERMS         shuffles for the best-of-grid null (0 = skip)
      --nested-perms NESTED_PERMS
                            shuffles of the nested procedure (0 = skip)
    ```

### `celldform-combine-models`

Frame- and step-level logistic regression and the soft-voting ensembles (equal-weight mean of P(high)), with permutation tests.

Script: `scripts/classification/combine_models.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-combine-models [-h] [--model MODEL] [--perms PERMS]

    Frame- and step-level logistic regression, and combinations of the deep and classical models, leave-one-cell-out.

        python scripts/classification/combine_models.py --model nnunet1000_loocv [--perms N]

    Run after benchmark_cells.py, frames_cnn.py finetune (3 seeds), sequence_net.py and feature_ranking.py: it reads their
    held-out P(high) per cell and fits nothing of theirs again.

    Instance-level logistic regression (fitted here)
        every frame   one row per passing feature frame, the 15 shape measures + 5 per-frame deformation measures
                      (L_par, L_perp, volume, elongation f2, roughness); a held-out cell's P(high) = mean over its frames
        every step    the same, one row per laser step (median of the step's passing frames)
        Settings as the benchmark's logistic regression; scaling inside the fold; cells weighted by their rows. Permutation
        p: labels shuffled across CELLS (every frame of a cell keeps its cell's label), the whole thing rerun.

    Combinations: the equal-weight mean of the members' held-out P(high) -- no weights are fitted, so nothing is learnt
    from the test cell and a combination needs no extra validation. Called high at >= 0.5. Members: the CNN (fine-tuned,
    cell crop, mean of 3 seeds), the eight classifiers on the label-free selection, LR and the ANN on all 29, the nested feature-ranking runs, the instance
    LRs. A combination of classical models only has a permutation p (every member rerun under the same shuffle); one with
    the CNN has none (one shuffle = 45 CNN trainings) -- it has a 2 000-resample cell-bootstrap interval on the AUC.

    Outputs in results/classification/<model>/: instance_cells.csv, combine_cells.csv (every model's and combination's
    held-out P(high)), combine_metrics.csv, combine_null.csv. The thesis figures are drawn by
    scripts/report/plot_classification_thesis.py.

    options:
      -h, --help     show this help message and exit
      --model MODEL
      --perms PERMS
    ```

### `celldform-pca-models`

PCA in place of feature selection: does compressing the features beat the selected 4?

Script: `scripts/classification/pca_models.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-pca-models [-h] [--model MODEL] [--perms PERMS]

    PCA in place of feature selection: does compressing the features beat the selected 4? Leave-one-cell-out.

        python scripts/classification/pca_models.py --model nnunet1000_loocv [--perms N]

    Run after benchmark_cells.py and combine_models.py (it reads feature_selection.csv and the step-level LR's held-out
    P(high) from instance_cells.csv).

    What is fitted
        Three starting sets -- all 29, the 9 that pass reliability (rules 1-2), the selected 4 -- each standardised and
        reduced to its first k principal components (k = 1..6, below the set size), then a logistic regression (the
        benchmark's settings). PCA, imputation and scaling are fitted on the 14 training cells of each fold only, so the
        held-out cell never shapes the components. Each setting is scored alone and as the best model's ensemble
        (equal-weight mean with the step-level LR). The raw sets (no PCA) are in the grid as k = 0.

    Statistics
        p            permutation of the whole procedure (labels shuffled across cells, step LR and every setting rerun)
        p_counted    the same shuffles, scored against the BEST balanced accuracy of the whole grid in each shuffle --
                     the p of "pick the best of the 36 settings", the choice actually made after seeing the results
        parallel analysis: the eigenvalues against those of 15 x n_features standard-normal data (5 000 draws); a
                     component is real only above the noise 95th percentile
        C sweep      the finalist and the current best model with both LRs at C = 0.1, 0.3, 1, 3, 10

    Outputs in results/classification/<model>/: pca_metrics.csv (every setting, per-class metrics, AUC interval, p,
    p_counted), pca_cells.csv (held-out P(high)), pca_null.csv (every shuffle's score), pca_variance.csv, pca_loadings.csv, pca_csweep.csv.
    The figure is cls_pca.png in scripts/report/plot_classification_thesis.py.

    options:
      -h, --help     show this help message and exit
      --model MODEL
      --perms PERMS
    ```

### `celldform-deformation-regression`

Deformation regression: does the strain grow with the laser power, and differently for high and low HER2 cells?

Script: `scripts/biomechanics/deformation_regression.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-deformation-regression [-h] [--model MODEL] [--img IMG]
                                            [--tables TABLES] [--config CONFIG]

    Deformation regression: does the strain grow with the laser power, and differently for high and low HER2 cells?

        python scripts/biomechanics/deformation_regression.py --model nnunet1000_loocv [--img <dir>] [--tables <dir>]

    One value per cell and laser step: the median of the step's passing feature frames (usable cells only), as a strain
    in % against the cell's own baseline (median over steps 0-9), for three measures -- length along the beam, length
    across it, and the spheroid volume. Each is fitted with a linear mixed-effects model:

        strain = b0 + b1 step + b2 high + b3 step x high  +  (u0 + u1 step | cell)  +  e

        b1   the typical low cell's strain per step (its compliance)
        b3   how much more (or less) a high cell strains per step -- the HER2 effect on deformation
        u1   each cell's own slope: the random slope makes the CELL the unit, not the ~700 step values

    Fitted by REML (statsmodels MixedLM). A second fit adds the cell's baseline length (standardised) and its
    interaction with the step, to ask whether size accounts for any HER2 effect.

    How well it fits -- R^2 for mixed models (Nakagawa & Schielzeth 2013; random slopes as in Johnson 2014):
        marginal     variance of the fixed-effect prediction (step, HER2) / total variance
        conditional  (fixed + random-effect variance) / total variance -- the share explained once every cell has
                     its own line

    Does HER2 help predict a new cell? -- leave-one-cell-out: each cell in turn is left out, both models are fitted
    on the other 13, and the left-out cell's whole strain curve is predicted from the fixed effects alone (a new cell
    has no fitted line of its own):
        with HER2     strain ~ step * high
        without HER2  strain ~ step
        zero          strain = 0 at every step (no deformation)
    scored by the RMSE (percentage points of strain) over the cell's steps; the paired comparison is a Wilcoxon
    signed-rank test over the 14 cells. HER2 and the recording session are the
    same split in this dataset, so b3 cannot be told apart from a session effect.

    Outputs in results/biomechanics/<model>/: deformation_regression.csv / .tex / .png (coefficients, fitted lines),
    deformation_fit.csv / .tex (R^2 and leave-one-cell-out errors), deformation_loco.csv / .png (every cell's errors).

    options:
      -h, --help       show this help message and exit
      --model MODEL    feature run in results/features/
      --img IMG        also write the figure here
      --tables TABLES  also write the LaTeX table here
      --config CONFIG  settings file (default configs/analysis/regression.yaml)
    ```

### `celldform-rate-of-deformation`

Rate of deformation (RoD) per cell: measured over the full ramp, in real time, and predicted early.

Script: `scripts/biomechanics/rate_of_deformation.py`

??? note "All options (`--help`)"
    ```text
    usage: celldform-rate-of-deformation [-h] [--model MODEL] [--img IMG]
                                         [--tables TABLES] [--config CONFIG]

    Rate of deformation (RoD) per cell: measured over the full ramp, in real time, and predicted early.

        python scripts/biomechanics/rate_of_deformation.py --model nnunet1000_loocv [--img <dir>] [--tables <dir>]

    RoD = slope of the strain (%) of the cell's length against the laser step (celldform.biomechanics.rod), for the
    length along the beam (main) and across it, with a Newey-West 95 % interval. Usable cells, passing frames only.

        per cell     the full-ramp RoD of every cell, with its interval
        real time    the RoD from the first k steps only, k = 5, 10, ..., 50: how wide is the interval after k steps,
                     and how far is the estimate from the full-ramp RoD?
        early        leave-one-cell-out: the left-out cell's RoD after k steps, estimated four ways, against its full-ramp
                     RoD (the target):
                       own           its own k steps only
                       shrunk        own, combined with the prior from the other 13 cells (pooled)
                       shrunk+class  the same with a class-specific prior (the cell's HER2 class -- what a classifier
                                     would supply; the true class, so an upper bound on what classifying adds)
                       prior only    the other cells' mean RoD, no own data

    Outputs in results/biomechanics/<model>/: rod_cells.csv, rod_realtime.csv, rod_early.csv, rod_cells.tex,
    rod_realtime.tex, rod_cells.png, rod_realtime.png.

    Final model: the settings and the priors from ALL usable cells (pooled, and per HER2 class), for estimating a new
    cell's RoD (celldform.biomechanics.rod.estimate) -> checkpoints/biomechanics/<model>/rod_model.json

    options:
      -h, --help       show this help message and exit
      --model MODEL    feature run in results/features/
      --img IMG        also write the figures here
      --tables TABLES  also write the LaTeX tables here
      --config CONFIG  settings file (default configs/analysis/regression.yaml)
    ```

## HPC

### `celldform-hpc-submit`

Generate the SLURM job for `scripts/training/train_unet.py` on the NSU cluster and submit it with `sbatch`
(the same job as `scripts/hpc/training/train_unet.sh`, set up through `scripts/hpc/_common.sh`: conda env,
`logs/hpc/celldform-unet_<jobid>.out`, an `.err` only on failure). Run from the repo root.

```bash
celldform-hpc-submit --config configs/default.yaml
celldform-hpc-submit --config configs/default.yaml --time 24:00:00 --dry-run   # print the job, don't submit
```

| Option | Default | Description |
|---|---|---|
| `--config` | `configs/default.yaml` | Config passed to `train_unet.py` |
| `--partition` | `gpu` | SLURM partition |
| `--account` / `--qos` | none | Only if NSU starts requiring them |
| `--time` | `12:00:00` | Wall-clock limit |
| `--cpus` | `8` | CPUs per task |
| `--gpus` | `1` | V100s (the node has 3) |
| `--conda-env` | `celldform` | Conda env with the project installed |
| `--mem` | `32G` | Memory |
| `--dry-run` | off | Print the generated script instead of submitting |

For every other script on the cluster, use its job script under `scripts/hpc/` — see [HPC Guide](hpc.md).
