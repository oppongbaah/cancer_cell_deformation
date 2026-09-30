# celldform

> **Current model:** `checkpoints/postswap/postswap_ftl_cw142_s42/unet_best.pt` —
> development-set evaluated DSC **0.809 ± 0.049** single model (4 seeds), **0.859 as a 4-model ensemble** (4 cells, 203 frames).
> Config: `configs/postswap/postswap_ftl_cw142_s42.yaml`.
> See [docs/status.md](docs/status.md) for where the project stands, and
> `configs/README.md` / `checkpoints/README.md` for what every other file is.
>
> ⚠️ **Two of four seeds collapse on a single cell — a different cell each time**, and
> neither cell size nor internal validation predicts which. Ensembling removes it.
> A single-run difference below ~0.14 is not a result; use
> `scripts/training/compare_postswap.py` and `check_convergence.py`.

**Real-Time Machine Learning Framework for Quantitative SKBR3 Cell Deformation Analysis in Dual-Beam Optical Tweezer Systems**

*Isaac Oppong-Baah — M.S. Electronics Engineering, Norfolk State University, 2026*
*Advisor: Dr. Patricia Mead*

---

## Overview

`celldform` is a Python library for analysing the mechanical deformation of SKBR3 cells in a dual-beam optical tweezer setup. It quantifies how HER2 expression affects cell deformability through a 7-stage machine learning pipeline: frame acquisition → preprocessing → U-Net segmentation → feature extraction → classification → biomechanical analysis → real-time integration.

**Full documentation:** [oppongbaah.github.io/cancer_cell_deformation](https://oppongbaah.github.io/cancer_cell_deformation/) — auto-deployed on every push to `main`.

---

## Installation

```bash
git clone https://github.com/oppongbaah/cancer_cell_deformation.git
cd celldform
python3 -m venv .venv && source .venv/bin/activate

pip install -e ".[dev]"          # core + dev tools
pip install -e ".[annotation]"   # adds napari for mask annotation
pip install -e ".[hpc]"          # adds mpi4py for Anvil / SLURM
```

One-time, after cloning — notebook outputs are stripped on commit by a git filter whose
command lives in `.git/config` (not committed), so it has to be registered locally:

```bash
nbstripout --install --attributes .gitattributes
```

---

## Quick start

```python
from celldform.acquisition import FrameExtractor
from celldform.preprocessing import PreprocessingPipeline
from celldform.segmentation import UNet
from celldform.features import MorphologyExtractor
from celldform.classification import CellClassifier

frames    = FrameExtractor("experiment.mp4").extract(every_n=5)
processed = [PreprocessingPipeline()(f) for f in frames]
masks     = [UNet.from_checkpoint("checkpoints/unet_best.pt").predict(img) for img in processed]
labels    = CellClassifier.load("checkpoints/svm.pkl").predict(
                MorphologyExtractor().batch(masks))
```

---

## Workflow (brief)

```bash
celldform-organize-dataset                              # Stage 0 — organise raw data
python scripts/annotation/annotate.py                             # annotate training images
python scripts/annotation/validate_masks.py                       # validate masks
python scripts/data/preprocess_frames.py --masks            # Stage 2 — preprocess
celldform-train-unet --config configs/default.yaml     # Stage 3 — train U-Net
python scripts/training/evaluate_unet.py --checkpoint ...       # evaluate on holdout set
celldform-infer <input> --unet-ckpt ... --classifier-ckpt ...  # Stages 1–6 inference
```

See [docs/workflow.md](docs/workflow.md) for the full step-by-step guide.

`configs/default.yaml` is the binary, 256×256 baseline. `configs/binary_experiment_*.yaml` and `configs/multiclass_experiment*.yaml` hold a resolution × label-scheme ablation — see [docs/training.md](docs/training.md) for results.

---

## HPC Training (Anvil)

U-Net training runs **12× faster** on Anvil's A100 GPUs (~0.3 s/epoch) than on a local GPU.

```bash
# On Anvil — request a GPU session via OnDemand, then in the VSCode terminal:
cd $SCRATCH/thesis
git clone https://github.com/oppongbaah/cancer_cell_deformation.git
cd cancer_cell_deformation
module load anaconda
conda create -n celldform python=3.10 -y && conda activate celldform
pip install -e ".[hpc]"
celldform-train-unet --config configs/default.yaml
```

Upload `data/preprocessed/` via the OnDemand file browser before training.
Download `checkpoints/unet_best.pt` via the same browser after training.

See the full guide: [docs/hpc.md](docs/hpc.md)

---

## Citation

```bibtex
@mastersthesis{oppongbaah2026celldform,
  author  = {Isaac Oppong-Baah},
  title   = {Real-Time Machine Learning Framework for Quantitative Cancer Cell
             Deformation Analysis in Dual-Beam Optical Tweezer Systems},
  school  = {Norfolk State University},
  year    = {2026},
  advisor = {Dr. Patricia Mead}
}
```

---

## License

MIT © 2026 Isaac Oppong-Baah
