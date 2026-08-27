# Data Layout

Full documentation of the `data/` directory — see also `data/README.md` for per-cell tables and the contamination firewall details.

---

## Directory structure

```
data/
├── raw/                            Source recordings — never modified
│   ├── high/                       High HER2-expression cells (9 videos)
│   ├── low/                        Low HER2-expression cells (5 videos)
│   ├── cell 1 low 50 increments.mp4
│   └── legacy/20210825_cancer/     220 unlabelled JPGs from 2021 microscope
│
├── frames/                         Raw extracted frames (1228×922 greyscale JPEG)
│   ├── 01_annotate_pool/           U-Net training source (428 images: 200 legacy + 228 domain-adapt)
│   ├── 02_unet_holdout/            U-Net test source
│   │   └── legacy_holdout/         20 reserved legacy frames — separate diagnostic DSC
│   ├── 03_mask_factory/            Inference → classifier training
│   └── 04_clf_arena/               Final classifier evaluation
│
├── masks/                          Hand-drawn masks (1228×922 PNG)
│   ├── 01_annotate_pool/           U-Net training labels (binary, 0/255)
│   ├── 01_annotate_pool_multiclass/  Experimental 3-class labels (0/1/2), same 40 frames
│   ├── 02_unet_holdout/            U-Net test ground truth
│   └── 02_unet_holdout/legacy_holdout/  Ground truth for the legacy holdout frames
│
└── preprocessed/                   256×256 pipeline-ready data
    ├── train/01_annotate_pool/
    │   ├── inputs/                 Preprocessed frames → U-Net training inputs
    │   └── labels/                 Resized masks       → U-Net training labels
    ├── evaluate/02_unet_holdout/
    │   ├── inputs/                 Preprocessed frames → U-Net evaluation inputs
    │   └── labels/                 Resized masks       → U-Net evaluation labels
    ├── evaluate/02_unet_holdout/legacy_holdout/
    │   ├── inputs/                 Preprocessed legacy holdout frames → separate DSC
    │   └── labels/                 Resized legacy holdout masks
    ├── inference/
    │   ├── 03_mask_factory/inputs/ Preprocessed frames → classifier training
    │   └── 04_clf_arena/inputs/    Preprocessed frames → final evaluation
    └── comparisons/                Pipeline stage figures (reporting)
```

---

## Pipeline role of each folder

| Folder | ML role | HER2 label | Masks | Touch when |
|--------|---------|-----------|-------|-----------|
| `raw/` | Source — never modified | In filename | None | Never |
| `frames/01_annotate_pool/` | U-Net training source | No | Hand-drawn (required) | Annotate now |
| `frames/02_unet_holdout/` | U-Net test source | No | Hand-drawn (required) | After U-Net training |
| `frames/02_unet_holdout/legacy_holdout/` | Legacy-domain diagnostic source | No | Hand-drawn (required) | After U-Net training |
| `frames/03_mask_factory/` | Classifier training source | Yes | U-Net predicted | After U-Net training |
| `frames/04_clf_arena/` | Classifier test source | Yes | U-Net predicted | Final evaluation only |
| `masks/01_annotate_pool/` | U-Net training labels | No | Binary, hand-drawn | During annotation |
| `masks/02_unet_holdout/` | U-Net test labels | No | Binary, hand-drawn | After U-Net training |
| `masks/02_unet_holdout/legacy_holdout/` | Legacy-domain diagnostic labels | No | Binary, hand-drawn | After U-Net training |
| `preprocessed/train/.../inputs/` | U-Net training inputs (256×256) | No | — | After preprocessing |
| `preprocessed/train/.../labels/` | U-Net training labels (256×256) | No | Resized binary | After preprocessing |
| `preprocessed/evaluate/.../inputs/` | U-Net evaluation inputs (256×256) | No | — | After holdout annotation |
| `preprocessed/evaluate/.../labels/` | U-Net evaluation labels (256×256) | No | Resized binary | After holdout annotation |
| `preprocessed/inference/03_.../inputs/` | Classifier training inputs | Yes | — | After U-Net training |
| `preprocessed/inference/04_.../inputs/` | Classifier test inputs | Yes | — | Final evaluation only |
| `preprocessed/comparisons/` | Reporting figures | — | — | On demand (`--compare`) |

---

## Pool descriptions

### `01_annotate_pool` — 428 images (U-Net training)

Source for U-Net training masks. **220/220 fully annotated as of 2026-08-10** (multiclass scheme — see below); grown to 428 on 2026-08-19 to address the domain gap (see warning below) — **208 new domain-adapt frames are currently unannotated and are the annotation priority** (see [Training Results](training.md#next-steps)).

| Subset | Count | Notes |
|--------|-------|-------|
| Legacy cancer images (2021) | 200 | Filename: `YYYYMMDDHHMMSS.jpg` — no HER2 label. 20 of the original 220 legacy JPGs were reserved into `02_unet_holdout/legacy_holdout/` before annotation, which is why this subset is 200, not 220. |
| Domain-adapt samples | 228 | Prefix: `domain_<her2>_<cell_id>_<frame>.jpg`. ~20 frames each from all 11 `03_mask_factory` cells (up from 5 frames × 4 cells) — the 4 originally-covered cells landed at 24 frames each rather than exactly 20 since a different N samples a different evenly-spaced grid, so the earlier 5 didn't line up with the new 20 and both sets stayed on disk. 220 of the 228 are already annotated (the original 4-cell batch); 208 are new. |

Domain-adapt frames come from cells also in `03_mask_factory`. The U-Net trains on them, then runs inference on the full set from those cells — same video, different sampled frame indices, so no contamination.

!!! warning "Domain gap between legacy and real-cell frames — mitigation in progress"
    With only 20 real-cell frames from 4 cells, segmenting held-out real-cell frames scored far worse than held-out legacy frames (DSC 0.538 vs. 0.944) — see [Training Results](training.md#domain-gap) for the full finding. **As of 2026-08-19:** `celldform/acquisition/organiser.py`'s `domain_adapt` cell list was extended to all 11 `03_mask_factory` cells and `--domain-adapt-n` increased 5→20/cell (`celldform-organize-dataset --no-clean --domain-adapt-n 20`), rather than waiting on the `02_unet_holdout` DSC to pick a value — see [Training Results — Next Steps](training.md#next-steps) for why `02_unet_holdout` annotation is deliberately being deferred instead.

!!! danger "Contamination bug found and fixed (2026-08-19)"
    The legacy-image copy step in `organise()` had no exclusion for the 20 frames permanently reserved in `02_unet_holdout/legacy_holdout/` (see below) — running `--no-clean` after raising `--domain-adapt-n` silently recopied all 220 legacy JPGs, holdout included, back into `01_annotate_pool`. Caught before annotation touched them (no masks existed yet for those 20 filenames). `organise()` now reads `02_unet_holdout/legacy_holdout/` and permanently excludes those filenames from the legacy copy step, so re-running `--no-clean` can no longer recontaminate the pool.

!!! note "Label scheme: multiclass (frozen)"
    The 220 originally-annotated frames use the 3-class scheme (`data/masks/01_annotate_pool_multiclass/`: 0=background, 1=trapped cell, 2=other/decoy object) — this is the scheme the pipeline trains on going forward (decided 2026-08-10; binary annotation stalled at 43/220 and isn't being continued). `scripts/annotate.py --multiclass` seeds label 1 from the existing binary mask where one exists, so only decoy objects need to be painted from scratch; it resumes on the first unannotated image, so it lands directly on the 208 new domain frames.

### `02_unet_holdout` — 101 images (U-Net test)

Held out from all training. Annotate **after** U-Net training is complete, and **after** the domain-adapt re-annotation above shows a solid validation-split domain DSC (see [Training Results](training.md#next-steps)) — this holdout is the one-shot, thesis-reportable number, so it's deliberately not being spent evaluating an intermediate checkpoint.

| Cell | HER2 | Frames |
|------|------|--------|
| cell9 | high | 50 |
| cell6 | low | 51 |

The DSC score from evaluating on this pool is the thesis-reportable segmentation metric.

!!! note "`02_unet_holdout/legacy_holdout/` — 20 images, evaluated separately"
    A diagnostic subset nested under `02_unet_holdout/` but **never merged** into the DSC above. Reserved unannotated from `01_annotate_pool` (evenly sampled across the legacy frames unannotated at the time) before annotated training data included any real-cell frames. Since training was legacy-only while `02_unet_holdout` is real-cell-only, one blended DSC couldn't separate "weak segmentation" from "domain gap." Evaluating this subset on its own (own `--image-dir`/`--mask-dir` pair; `evaluate_unet.py --pool 02_unet_holdout` doesn't recurse into it) gives a second, comparable DSC — high legacy-holdout DSC + low real-cell-holdout DSC points to the domain gap specifically.

### `03_mask_factory` — 557 images (classifier training)

U-Net inference pool. Each predicted mask → 15 morphological features → one classifier training sample.

| HER2 | Cells | Frames |
|------|-------|--------|
| high | 1, 2, 3, 4, 6, 7, 8 | 355 |
| low | 1, 2, 3, 4 | 202 |
| **Total** | | **557** |

!!! note "Class imbalance"
    Training set is ~64 % high / 36 % low. Use `class_weight='balanced'` on all classifiers.

### `04_clf_arena` — 101 images (classifier test)

Completely held out. These cells were never annotated and never seen by any model during training.

| Cell | HER2 | Frames |
|------|------|--------|
| cell10 | high | 50 |
| cell5 | low | 51 |

---

## Contamination firewall

The data split ensures no training signal leaks into final evaluation:

- **Legacy images** have no HER2 label — safe for U-Net training; cannot contaminate the classifier.
- **`02_unet_holdout`** cells were never used during U-Net training. Annotated masks give an unbiased segmentation DSC.
- **`04_clf_arena`** cells were never annotated, never used to generate classifier training features, and never included in cross-validation. They are the single honest test of classifier generalisation.

---

## Regenerating frames

```bash
celldform-organize-dataset                                    # defaults
celldform-organize-dataset --every-n 100 --domain-adapt-n 10 # denser sampling
celldform-organize-dataset --no-clean                         # skip wiping data/frames/
celldform-organize-dataset --no-clean --domain-adapt-n 20     # expand domain-adapt frames only — run 2026-08-19
                                                                # (use --no-clean so 02_unet_holdout isn't wiped
                                                                # mid-annotation; legacy_holdout frames are always
                                                                # excluded from the legacy copy regardless of N;
                                                                # existing domain-adapt frames/masks whose exact
                                                                # frame index doesn't fall on the new N's sampling
                                                                # grid are kept on disk, not deleted, so no
                                                                # annotation work is lost across re-runs)
```
