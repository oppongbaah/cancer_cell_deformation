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

## Training the U-Net

```bash
celldform-train-unet --config configs/default.yaml
# or
python scripts/train_unet.py --config configs/default.yaml
```

Checkpoints are saved to the path specified in `configs/default.yaml` under `training.checkpoint_dir`.

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
