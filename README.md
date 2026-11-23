# celldform

Machine learning for measuring how SKBR3 breast-cancer cells deform in a dual-beam optical tweezer, and for classifying
their **HER2 expression (high or low)** from that deformation. It is a Python library, a set of command-line tools, and a
**desktop app** that analyzes a recorded video.

> **Research software.** It was developed for an M.S. thesis and validated on 15 cells from one setup. It is not a
> medical device and must not be used for clinical decisions. See [Project status](docs/status.md) for exactly what has been
> shown and what has not.

*Isaac Oppong-Baah — M.S. Electronics Engineering, Norfolk State University, 2026*

**Documentation:** [oppongbaah.github.io/cancer_cell_deformation](https://oppongbaah.github.io/cancer_cell_deformation/)

---

## What is in it

| | |
|---|---|
| **Segmentation** | a U-Net finds the trapped cell in each frame (an nnU-Net is the accuracy benchmark) |
| **Features** | shape and deformation measures of the cell, with five checks that reject unusable frames |
| **HER2 classifier** | a step-level and a cell-level model, averaged; 14 of 15 cells correct in leave-one-cell-out testing |
| **Desktop app** | opens a recording, runs the models, shows the HER2 class with the evidence behind it |
| **Research tools** | training, evaluation and figure scripts, each also a `celldform-*` command |

## Which way should I use it?

| I want to... | Use | You need |
|---|---|---|
| **analyze a recording on Windows**, without installing Python | the Windows installer | nothing else: Python and every library are inside the installer |
| analyze a recording with Python | `pip install "celldform[app]"` then `celldform-demo` | Python 3.10–3.12 |
| use the library or repeat the research | clone the repository and `pip install -e ".[dev]"` | Python 3.10 or newer, a GPU helps |

**Trained models are not included.** The app (either way) starts without any model and cannot analyze until you load
files: a segmentation model (the U-Net weights with their settings file, **or** an nnU-Net checkpoint) and the HER2
classifier, or one **model bundle** that carries all of them. You get them from whoever trained them; see [the app's guide](docs/demo.md#required-files-first-run-on-a-new-machine).

## Data, models and logs

The repository holds the code, configurations, tests and documentation only. **The recordings, annotated frames and masks,
trained models (the U-Net weights, the nnU-Net checkpoint and the HER2 classifier), training and job logs, and the result
tables are not included.** They are large, and the recordings are not ours to publish through this repository. Anyone who
needs any of them can request them by email to **[oppongbaahisaac7@gmail.com](mailto:oppongbaahisaac7@gmail.com)**: say which you need and what for, and requests are answered individually.

The code that produced them is here, so the layout they take is documented ([Data layout](docs/data.md),
[the experiment index](EXPERIMENTS.md)), and the file names, settings and numbers in the thesis can be traced to a script
and a configuration. Without the trained models the desktop app starts but cannot analyze a recording until you load them
([required files](docs/demo.md#required-files-first-run-on-a-new-machine)).

## Install

**Windows app.** Run `celldform-setup-<version>.exe`; you do not need to install Python or this library first. Step by
step, including the desktop shortcut and the first start: [Installing on Windows](docs/install_windows.md).

**Python.** From a clone of this repository (works today):

```bash
git clone https://github.com/oppongbaah/cancer_cell_deformation.git
cd cancer_cell_deformation
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[app]"          # the library and the desktop app (PySide6)
pip install -e ".[dev]"          # ...plus the test and lint tools
```

Once the package is on PyPI the first line becomes `pip install "celldform[app]"`. Optional extras: `annotation` (napari, for
drawing masks), `hpc` (MPI), `docs` (MkDocs). The default install includes nnU-Net v2 and PyTorch, which are large.

## Quick start

**The app.** `celldform-demo` opens the window (the installer puts a shortcut on the desktop). Load the model files, open a
recording (a video file, several images or a recording folder), press **Analyze**. Nothing is sent anywhere: the app works
offline ([Privacy](docs/privacy.md)).

**From Python or the command line**, with a model bundle:

```bash
python -m celldform.demo.run my_recording --out results/run1 --bundle models.zip
```

```python
from celldform.demo.framework import load as load_framework
from celldform.demo.models import open_bundle
from celldform.demo.run import run
from celldform.demo.segment import UNetSegmenter

files = open_bundle("models.zip")                              # checks every file's checksum
segmenter = UNetSegmenter.from_file(files["unet"], files["unet_config"])
framework = load_framework(files["classifier"])
verdict = run("my_recording", "results/run1", segmenter, framework, segmenter_name="U-Net")
print(verdict.headline())      # HER2 high/low with P(high), or why there is no verdict
```

A verdict is given only if the recording has at least 3 usable laser steps and 80 % of the measured frames pass the checks
the classifier was validated on; otherwise the result says why. With 10 or more steps it is the full validated classifier;
with 3 to 9 it is labelled a short-recording result and uses the step model alone.

## For researchers: the workflow

The scripts that produced the results are commands too (`celldform` lists them all; see [docs/cli.md](docs/cli.md)):

```bash
celldform-organize-dataset                                  # organise raw data into frame pools
celldform-annotate                                          # annotate training images (needs .[annotation])
celldform-preprocess-frames --masks                         # preprocess
celldform-train-unet --config configs/default.yaml          # train the U-Net
celldform-evaluate-unet --checkpoint ...                    # evaluate on held-out cells
celldform-infer --video <video> --unet-ckpt ... --classifier-ckpt ...   # older per-frame inference
```

Step by step: [docs/workflow.md](docs/workflow.md). Training on a SLURM cluster: [docs/hpc.md](docs/hpc.md). Folder layout:
[docs/data.md](docs/data.md). nnU-Net runs through `scripts/training/train_nnunet_cv.py`.

## Results and limits

Holding out whole cells (15 cells, one setup): U-Net Dice 0.86, nnU-Net 0.94; HER2 class 14 of 15 correct. A recording from a
different setup may not segment at all. The full table and the limits are on [Project status](docs/status.md).

## Citation

```bibtex
@mastersthesis{oppongbaah2026celldform,
  author = {Isaac Oppong-Baah},
  title  = {Real-Time Machine Learning Framework for Quantitative Cancer Cell
            Deformation Analysis in Dual-Beam Optical Tweezer Systems},
  school = {Norfolk State University},
  year   = {2026}
}
```

## License

MIT © 2026 Isaac Oppong-Baah. The desktop app uses PySide6 and Qt (LGPL) and other open-source libraries under their own
licences; the Windows installer lists them in `THIRD_PARTY_NOTICES.txt`.
