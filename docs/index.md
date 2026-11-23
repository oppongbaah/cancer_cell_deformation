# celldform

!!! tip "Start here"
    **[Project status](status.md)**: what exists, what has been measured and what has not.
    **[Installing on Windows](install_windows.md)**: the desktop app, no Python needed.

Machine learning for measuring how SKBR3 breast-cancer cells deform in a dual-beam optical tweezer, and for classifying their
**HER2 expression (high or low)** from that deformation.

*Isaac Oppong-Baah — M.S. Electronics Engineering, Norfolk State University, 2026*

!!! warning "Research software"
    Developed for an M.S. thesis and validated on 15 cells from one setup. It is not a medical device and must not be used for
    clinical decisions.

---

## What is celldform?

`celldform` is a Python library, a set of command-line tools and a **desktop app**. It segments the trapped cell in each frame,
measures its shape and deformation across a ramp of laser power, rejects frames that cannot be trusted, and classifies the
cell's HER2 expression from how it deforms. The desktop app does that for a **recorded** video and shows the evidence behind
the answer. The software is not real-time and is not connected to the instrument.

## Which way should I use it?

| I want to... | Use | You need |
|---|---|---|
| analyze a recording on Windows | the [Windows installer](install_windows.md) | nothing else: Python and the libraries are inside |
| analyze a recording with Python | `pip install "celldform[app]"`, then `celldform-demo` | Python 3.10–3.12 |
| use the library or repeat the research | clone the repository, `pip install -e ".[dev]"` | Python 3.10 or newer, a GPU helps |

**The trained models are not included.** Before the app can analyze anything you load three files (U-Net weights, their
settings, the HER2 classifier) or one model bundle: [Required files](demo.md#required-files-first-run-on-a-new-machine).

## Pipeline at a glance

| Stage | What it does | Module |
|-------|-------------|--------|
| 1 — Acquisition | Video → grayscale frames | `celldform.acquisition` |
| 2 — Preprocessing | Denoise → CLAHE → morphology → resize → normalise | `celldform.preprocessing` |
| 3 — Segmentation | U-Net: frame → cell mask | `celldform.segmentation` |
| 4 — Features | Mask → shape and deformation measures, with frame checks | `celldform.features` |
| 5 — Classification | Features → low / high HER2 | `celldform.classification`, `celldform.demo.framework` |
| 6 — Biomechanics | Deformation rate and Kelvin–Voigt fitting | `celldform.biomechanics` |
| 7 — Desktop app | Recording → HER2 class with evidence | `celldform.demo` |

See [Pipeline Overview](pipeline.md) for the architecture.

---

## Installation

```bash
git clone https://github.com/oppongbaah/cancer_cell_deformation.git
cd cancer_cell_deformation
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[app]"          # library + desktop app
pip install -e ".[dev]"          # test and lint tools
pip install -e ".[annotation]"   # napari, for mask annotation
pip install -e ".[docs]"         # MkDocs, for building these docs
```

On Windows without Python, use the [installer](install_windows.md) instead.

---

## Navigating these docs

| Page | Contents |
|------|----------|
| [Project status](status.md) | Results, limits and what is next |
| [Demonstration app](demo.md) | Required files, using the app, the verdict rules, privacy |
| [Installing on Windows](install_windows.md) | The installer, or pip; first start |
| [Privacy](privacy.md) | What stays on your computer |
| [Pipeline Overview](pipeline.md) | Architecture and stage-by-stage description |
| [Workflow Guide](workflow.md) | End-to-end steps: organise → annotate → preprocess → train → evaluate |
| [Data Layout](data.md) | Folder structure and the contamination firewall |
| [Scripts](scripts.md), [CLI](cli.md) | Every script and `celldform-*` command |
| [GPU & Hardware](gpu.md), [HPC Guide](hpc.md) | Which stages use the GPU; training on SLURM |
| [Research log](research_log_2026-09.md), [Training results](training.md) | How the segmentation work developed (historical) |
| [API Reference](api/acquisition.md) | Class and method documentation |
