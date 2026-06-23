# Pipeline Overview

The celldform pipeline is **strictly linear** — each stage's output is the next stage's input. Stages 2 and 3 are independent: preprocessing has no knowledge of the U-Net architecture, and the U-Net receives already-preprocessed images.

---

## Architecture

```
Stage 1           Stage 2              Stage 3          Stage 4              Stage 5             Stage 6           Stage 7
──────────────    ─────────────────    ─────────────    ─────────────────    ───────────────     ─────────────     ─────────────────
Acquisition   →  Preprocessing    →   U-Net        →  Feature          →   Classification  →   Biomechanics  →   Real-Time
                 (separate stage)     Segmentation     Extraction                               Analysis          Integration

FrameExtractor   PreprocessingPipeline UNet             MorphologyExtractor  CellClassifier      DeformationAnalyzer RealTimePipeline
acquisition/     preprocessing/        segmentation/    features/            classification/     biomechanics/     integration/
```

---

## Stage 1 — Acquisition

**Module:** `celldform/acquisition/extractor.py`

`FrameExtractor` reads an optical tweezer video and writes greyscale frames as PNGs to disk. Two temporal sampling modes:

- `every_n` — keep every N-th frame
- `target_fps` — resample to a fixed rate

**Trapped-cell detection** uses three physics-based cues applied in sequence:

1. **Spatial channel filter** — only blobs in the central `channel_width` fraction of the image
2. **Circularity filter** — `4π·area/perimeter² ≥ min_circularity`
3. **Darkest blob wins** — trapped cells appear dark against the bright channel background

Each video produces a `<stem>_manifest.csv` recording frame index, timestamp, centroid, and displacement per saved frame.

!!! note "change_threshold for optical tweezers"
    Set `change_threshold=None` for all optical tweezer experiments. The cell is trapped and does not translate — deformation is shape change, not position change. A non-null threshold will filter out all frames.

---

## Stage 2 — Preprocessing

**Module:** `celldform/preprocessing/pipeline.py`

`PreprocessingPipeline` is a callable that converts a raw greyscale uint8 frame to a float32 array in `[0, 1]` at `(256, 256)`.

**Fixed operation order:**

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | Median filter (`median_kernel=3`) | Remove salt-and-pepper noise |
| 2 | CLAHE (`clip_limit=2.0`, `tile_grid=8×8`) | Equalise contrast under uneven illumination |
| 3 | Morphological open → close | Remove bright specks; fill dark gaps |
| 4 | Bicubic resize to `(256, 256)` | Match U-Net input size |
| 5 | Min-max normalise to `[0, 1]` float32 | Stable gradient flow |

Each step can be toggled individually via the `enabled` dict for ablation experiments.

!!! warning "Preprocessing is manual"
    Preprocessing is **not** applied inside the training loop. Run `scripts/preprocess_frames.py --masks` before training. See [Workflow Guide](workflow.md).

---

## Stage 3 — U-Net Segmentation

**Module:** `celldform/segmentation/unet.py`, `celldform/segmentation/trainer.py`

`UNet` implements the Ronneberger et al. (2015) encoder–decoder with BatchNorm + ReLU.

| Component | Detail |
|-----------|--------|
| Input | `(B, 1, 256, 256)` float32 preprocessed frame |
| Encoder | 4 blocks — 64 → 128 → 256 → 512 channels |
| Bottleneck | 1024 channels |
| Decoder | 4 blocks with skip connections |
| Output | `(B, 1, 256, 256)` raw logit map |

!!! note
    `forward()` returns raw logits — sigmoid is **not** applied internally. Use `model.predict()` for thresholded binary masks. Use `BCEWithLogitsLoss` in the trainer.

**Training** uses combined BCE + Dice loss:

```
Loss = α × BCE(pos_weight) + (1 − α) × Dice
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `loss_alpha` | `0.3` | BCE weight — Dice carries 70% of the signal |
| `loss_pos_weight` | `100.0` | Upweights cell pixels in BCE to counter ~550:1 background/cell imbalance |

Both are set in `configs/default.yaml` under `training`. The best checkpoint is saved by validation DSC. Supports `ReduceLROnPlateau`, cosine, or no scheduler.

!!! note "Class imbalance"
    Cancer cells occupy roughly 0.18% of a 256×256 frame (~118 pixels out of 65,536). Without `pos_weight`, the BCE loss is dominated by background pixels and the model learns to ignore the cell entirely. `pos_weight=100` tells BCE to treat each cell pixel as if it were 100 background pixels. Increase it if recall is low; decrease it if precision is low and recall is near 1.0.

**GPU:** The U-Net is the only stage that runs on the GPU. See [GPU & Hardware](gpu.md).

---

## Stage 4 — Feature Extraction

**Module:** `celldform/features/extractor.py`

`MorphologyExtractor` takes a binary mask and returns **15 morphological descriptors** per cell. The largest connected region above `min_area_px=50` is used.

| Feature | Description |
|---------|-------------|
| `area` | Cell area (px² or μm² if calibrated) |
| `perimeter` | Boundary length |
| `major_axis`, `minor_axis` | Ellipse fit axes |
| `aspect_ratio` | `major / minor` — primary deformation signal |
| `eccentricity` | Ellipse eccentricity |
| `circularity` | `4π·area / perimeter²` |
| `solidity` | `area / convex-hull area` |
| `hu_0` – `hu_6` | Log-scaled Hu moment invariants |

!!! note "Feature ordering"
    Features fed to the classifier must be sorted **alphabetically by name** — the same order `sorted(features.keys())` produces. Do not change this without updating `RealTimePipeline` and `scripts/infer.py`.

---

## Stage 5 — Classification

**Module:** `celldform/classification/classifiers.py`

`CellClassifier` wraps an RBF-kernel SVM inside `CalibratedClassifierCV` for probability output.

| Label | Meaning |
|-------|---------|
| `0` | Low HER2 expression (stiffer cell) |
| `1` | High HER2 expression (more deformable) |

The thesis requires comparative analysis of **SVM, RF, and DT**. All classifiers implement the same interface (`fit`, `predict`, `predict_proba`, `evaluate`, `save`, `load`) so they can be swapped without changes to inference scripts.

!!! warning "Class imbalance"
    Training set is ~64 % high / 36 % low. Always use `class_weight='balanced'` and **cell-level cross-validation** (leave-one-cell-out), not frame-level.

---

## Stage 6 — Biomechanics

**Module:** `celldform/biomechanics/analysis.py`

`DeformationAnalyzer` computes the **rate of deformation (RoD)** — the linear slope of a chosen morphological metric (default: `aspect_ratio`) over the experiment time axis.

Key methods:

- `sliding_rate()` — windowed local RoD estimate
- `analyze_batch()` — aggregate mean RoD and peak deformation across multiple experiments
- `optical_correlation()` — Pearson and Spearman correlations between laser power / trap current and deformation outcome (directly answers the thesis research question)
- `fit_kelvin_voigt()` — fits `σ = E·ε + η·dε/dt` to estimate Young's modulus E and viscosity η

---

## Stage 7 — Real-Time Integration

**Module:** `celldform/integration/realtime.py`

`RealTimePipeline` composes stages 2–5 into a single `process_frame(frame, frame_index, timestamp)` call for live experiments. The C++ application passes raw camera frames; the pipeline returns:

```python
{
    "mask":         np.ndarray,   # binary cell mask
    "features":     dict,         # 15 morphological features
    "label":        int,          # 0 = low HER2, 1 = high HER2
    "label_proba":  float,        # calibrated probability
    "latency_ms":   float,        # end-to-end processing time
}
```

The pipeline is **stateless across frames** — safe to share across processes without locking.
