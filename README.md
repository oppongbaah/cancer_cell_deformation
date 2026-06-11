# celldform

**Real-Time Deep Learning Framework for Quantitative Cancer Cell Deformation Analysis in Dual-Beam Optical Tweezer Systems**

*Isaac Oppong-Baah — M.S. Electronics Engineering, Norfolk State University, 2026*
*Advisor: Dr. Patricia Mead*

---

## Overview

`celldform` is a Python library that provides a complete, modular pipeline for analysing the mechanical deformation of cancer cells imaged in a dual-beam optical tweezer setup. The pipeline covers every stage shown in the system architecture:

| Stage | Module |
|---|---|
| 1. Image acquisition (video → frames) | `celldform.acquisition` |
| 2. Data preprocessing (denoise, enhance, morphology, normalize) | `celldform.preprocessing` |
| 3. U-Net segmentation (train / infer / evaluate) | `celldform.segmentation` |
| 4. Morphological feature extraction from masks | `celldform.features` |
| 5. Cell classification (SVM, RF, DT — low / high HER2 expression) | `celldform.classification` |
| 6. Biomechanical analysis (deformation rate vs laser power / current) | `celldform.biomechanics` |
| 7. Real-time C++ integration bridge | `celldform.integration` |

Supporting components: `celldform.models` (MegaNet CNN + checkpoint manager) and `celldform.utils` (metrics, visualization).

---

## Installation

**From PyPI** (once published):

```bash
pip install celldform
```

**From source (recommended for development):**

```bash
git clone https://github.com/oppongbaah/celldform.git
cd celldform

# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install the package with all dev dependencies
pip install -e ".[dev]"
```

The editable install (`-e`) means changes to the source are reflected immediately without reinstalling. A `celldform.egg-info/` directory will be created — this is auto-generated metadata used by the import machinery and can be ignored.

For HPC environments (Anvil / SLURM):

```bash
pip install "celldform[hpc]"
```

---

## Quick Start

```python
from celldform.acquisition import FrameExtractor
from celldform.preprocessing import PreprocessingPipeline
from celldform.segmentation import UNet, SegmentationTrainer
from celldform.features import MorphologyExtractor
from celldform.classification import CellClassifier
from celldform.biomechanics import DeformationAnalyzer

# 1 — extract frames from optical tweezer video
frames = FrameExtractor("experiment_01.avi").extract(every_n=5)

# 2 — preprocess
pipeline = PreprocessingPipeline()
processed = [pipeline(f) for f in frames]

# 3 — segment with U-Net
model = UNet.from_checkpoint("checkpoints/unet_best.pt")
masks = [model.predict(img) for img in processed]

# 4 — extract features
extractor = MorphologyExtractor()
features_df = extractor.batch(masks)

# 5 — classify (low / high HER2 expression)
clf = CellClassifier.load("checkpoints/svm_classifier.pkl")
labels = clf.predict(features_df)

# 6 — biomechanical analysis
analyzer = DeformationAnalyzer(laser_power_mW=100, trap_current_mA=250)
rate = analyzer.deformation_rate(features_df["aspect_ratio"].values,
                                  timestamps=features_df["timestamp"].values)
```

---

## CLI reference

All commands are available after `pip install -e ".[dev]"`. Every command also
has an equivalent script under `scripts/` for cases where you need more control.

---

### `celldform-organize-dataset` — build pipeline-ready frame folders from raw data

Cleans `data/frames/`, copies legacy images, and extracts sampled frames from
all labeled high/low videos into four folders.

```bash
# Default run — reads data/raw/, writes data/frames/, samples every 200th frame
celldform-organize-dataset

# Custom raw data path and denser sampling
celldform-organize-dataset --raw-dir /path/to/raw --every-n 100

# More domain-adaptation frames in the annotation pool
celldform-organize-dataset --domain-adapt-n 10

# Re-run without wiping existing frames (partial update)
celldform-organize-dataset --no-clean
```

| Option | Default | Description |
|--------|---------|-------------|
| `--raw-dir` | `data/raw` | Root of raw data (`high/`, `low/`, `legacy/` expected inside) |
| `--frames-dir` | `data/frames` | Output root for organised frame folders |
| `--every-n` | `200` | Sample every N-th frame per video (~50 frames per 10 000-frame video) |
| `--domain-adapt-n` | `5` | Frames per 2025-camera cell copied into `01_annotate_pool` |
| `--no-clean` | off | Skip wiping `--frames-dir` before running |

Equivalent script: `python scripts/organize_dataset.py [same options]`

---

### `celldform-extract-frames` — extract frames from a single video

```bash
# Basic — writes every frame to the output directory
celldform-extract-frames --video data/raw/high/"cell 1 high 50 increments.mp4"

# Keep every 5th frame, resize to 640×480
celldform-extract-frames --video experiment.mp4 --every-n 5 --target-size 640 480

# Resample to a fixed frame rate instead
celldform-extract-frames --video experiment.mp4 --target-fps 5.0

# Stop after 200 frames (quick sanity check)
celldform-extract-frames --video experiment.mp4 --max-frames 200

# Write to a custom directory in TIFF format
celldform-extract-frames --video experiment.mp4 --output-dir out/frames --format tiff

# Enable trapped-cell change detection (skip near-duplicate frames)
celldform-extract-frames --video experiment.mp4 --change-threshold 5.0 \
    --channel-width 0.30 --dark-percentile 20 --min-circularity 0.60
```

| Option | Default | Description |
|--------|---------|-------------|
| `--video` | required | Input video path (.mp4, .avi, …) |
| `--config` | `configs/default.yaml` | YAML config; CLI args override config values |
| `--output-dir` | config `data.frames_dir` | Directory to write extracted frames |
| `--every-n` | config value | Keep every N-th frame |
| `--target-fps` | config value | Resample to this rate (mutually exclusive with `--every-n`) |
| `--max-frames` | config value | Hard upper limit on saved frames |
| `--target-size W H` | config value | Resize each frame to W × H pixels |
| `--format` | `png` | Output format: `png` or `tiff` |
| `--change-threshold` | disabled | Min centroid displacement (px) to save a frame |
| `--channel-width` | `0.30` | Channel strip width as fraction of image width |
| `--dark-percentile` | `20.0` | Bottom X% of channel intensities treated as cell candidates |
| `--min-circularity` | `0.60` | Minimum blob circularity to accept as a cell |

Equivalent script: `python scripts/extract_frames.py [same options]`

---

### `celldform-train-unet` — train the U-Net segmentation model

```bash
# Train with default config
celldform-train-unet --config configs/default.yaml

# Override epochs and device
celldform-train-unet --config configs/default.yaml --epochs 100 --device cuda

# Custom checkpoint directory
celldform-train-unet --config configs/default.yaml --checkpoint-dir checkpoints/run2
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `configs/default.yaml` | YAML config file |
| `--epochs` | config `training.epochs` | Number of training epochs |
| `--device` | auto-detect | `cuda` or `cpu` |
| `--checkpoint-dir` | config value | Where to save model checkpoints |

Checkpoints written: `unet_best.pt` (best validation DSC), `unet_last.pt`, and
a `config.yaml` snapshot at epoch 1.

For a complete training example including `CellDataset` and `DataLoader` setup:

```bash
python scripts/train_unet.py --config configs/default.yaml
```

---

### `celldform-infer` — run the full inference pipeline on a video or image folder

```bash
# Segmentation only — predict masks for every frame
celldform-infer data/frames/03_mask_factory \
    --unet-ckpt checkpoints/unet_best.pt

# Segmentation + HER2 classification
celldform-infer data/raw/high/"cell 1 high 50 increments.mp4" \
    --unet-ckpt checkpoints/unet_best.pt \
    --classifier-ckpt checkpoints/svm_classifier.pkl

# Write results to a specific directory, process every 5th frame
celldform-infer experiment.mp4 \
    --unet-ckpt checkpoints/unet_best.pt \
    --output-dir results/exp01 \
    --every-n 5
```

| Option | Default | Description |
|--------|---------|-------------|
| `input` | required | Video file (.mp4/.avi) or directory of images |
| `--unet-ckpt` | required | Path to trained U-Net checkpoint (`.pt`) |
| `--classifier-ckpt` | none | Path to trained classifier (`.pkl`); omit for segmentation only |
| `--output-dir` | `results/` | Directory for inference outputs |
| `--device` | auto-detect | `cuda` or `cpu` |
| `--every-n` | `1` | Process every N-th frame |

Equivalent script: `python scripts/infer.py [same options]`

---

## Project Structure

```
celldform/
├── celldform/
│   ├── acquisition/     # video → frames
│   ├── preprocessing/   # image enhancement pipeline
│   ├── segmentation/    # U-Net model + trainer
│   ├── features/        # morphological feature extraction
│   ├── classification/  # SVM / RF / DT classifiers
│   ├── biomechanics/    # deformation rate & laser analysis
│   ├── integration/     # real-time C++ bridge
│   ├── models/          # MegaNet CNN + checkpoint I/O
│   └── utils/           # metrics & visualization helpers
├── configs/
│   └── default.yaml
├── scripts/
│   ├── train_unet.py
│   ├── train_cnn.py
│   └── infer.py
└── tests/
```

---

## Citation

```bibtex
@mastersthesis{oppongbaah2026celldform,
  author  = {Isaac Oppong-Baah},
  title   = {Real-Time Deep Learning Framework for Quantitative Cancer Cell
             Deformation Analysis in Dual-Beam Optical Tweezer Systems},
  school  = {Norfolk State University},
  year    = {2026},
  advisor = {Dr. Patricia Mead}
}
```

---

## License

MIT © 2026 Isaac Oppong-Baah
