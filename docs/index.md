# celldform

**Real-Time Machine Learning Framework for Quantitative SKBR3 Cell Deformation Analysis in Dual-Beam Optical Tweezer Systems**

*Isaac Oppong-Baah — M.S. Electronics Engineering, Norfolk State University, 2026*
*Advisor: Dr. Patricia Mead*

---

## What is celldform?

`celldform` is a Python library that provides a complete, reproducible pipeline for analysing the mechanical deformation of SKBR3 cells imaged in a dual-beam optical tweezer setup. It quantifies how HER2 expression level affects cell deformability by combining machine learning segmentation with classical morphological feature extraction and biomechanical modelling.

## Pipeline at a glance

| Stage | What it does | Module |
|-------|-------------|--------|
| 1 — Acquisition | Video → grayscale frames | `celldform.acquisition` |
| 2 — Preprocessing | Denoise → CLAHE → morphology → resize → normalise | `celldform.preprocessing` |
| 3 — Segmentation | U-Net: frame → binary cell mask | `celldform.segmentation` |
| 4 — Features | Binary mask → 15 morphological descriptors | `celldform.features` |
| 5 — Classification | Features → low / high HER2 label | `celldform.classification` |
| 6 — Biomechanics | Deformation rate + Kelvin–Voigt fitting | `celldform.biomechanics` |
| 7 — Integration | Real-time C++ bridge | `celldform.integration` |

See [Pipeline Overview](pipeline.md) for the full architecture.

---

## Installation

```bash
git clone https://github.com/oppongbaah/cancer_cell_deformation.git
cd celldform
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + dev tools
pip install -e ".[annotation]"   # adds napari for mask annotation
pip install -e ".[docs]"         # adds MkDocs for building these docs
```

---

## Quick start

```python
from celldform.acquisition import FrameExtractor
from celldform.preprocessing import PreprocessingPipeline
from celldform.segmentation import UNet
from celldform.features import MorphologyExtractor
from celldform.classification import CellClassifier

# Extract frames
frames = FrameExtractor("experiment.mp4").extract(every_n=5)

# Preprocess
pipeline = PreprocessingPipeline()
processed = [pipeline(f) for f in frames]

# Segment
model = UNet.from_checkpoint("checkpoints/unet_best.pt")
masks = [model.predict(img) for img in processed]

# Extract features and classify
features_df = MorphologyExtractor().batch(masks)
labels = CellClassifier.load("checkpoints/svm.pkl").predict(features_df)
```

---

## Navigating these docs

| Page | Contents |
|------|----------|
| [Pipeline Overview](pipeline.md) | Architecture diagram and stage-by-stage description |
| [Workflow Guide](workflow.md) | End-to-end steps: organise → annotate → preprocess → train → evaluate → infer |
| [Data Layout](data.md) | Folder structure, pool descriptions, contamination firewall |
| [Scripts Reference](scripts.md) | Every script in `scripts/` with options and examples |
| [CLI Reference](cli.md) | All `celldform-*` console commands |
| [GPU & Hardware](gpu.md) | Which stages use the GPU and how to configure it |
| [HPC Guide](hpc.md) | Running on Anvil / SLURM |
| [API Reference](api/acquisition.md) | Auto-generated class and method documentation |
