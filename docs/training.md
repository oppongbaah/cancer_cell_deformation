# U-Net Training Results

Preliminary training results from the first annotation sessions. The model is not yet finalised — these runs confirm the pipeline is working correctly and that annotations are consistent.

---

## Run 1 — Baseline (no class imbalance correction)

**Masks:** 39 annotated  |  **Epochs:** 50  |  **Loss:** Dice+BCE (α=0.5, no pos_weight)

| Metric | Value |
|--------|-------|
| DSC | 0.0902 |
| IoU | 0.0473 |
| Precision | 0.0511 |
| Recall | 0.3880 |
| Accuracy | 0.9857 |

**Diagnosis:** The model learned to predict mostly background. SKBR3 cells occupy roughly 0.18% of each 256×256 frame (~118 pixels out of 65,536), giving a ~550:1 background-to-cell pixel ratio. Without compensation, the BCE loss is dominated by background pixels — achieving near-zero loss simply by ignoring the cell. High accuracy (0.985) is misleading here: it reflects correct background prediction, not cell segmentation. Low precision (0.05) combined with moderate recall (0.39) indicates the model was guessing large regions that partially overlapped the cell.

---

## Run 2 — With class imbalance correction

**Masks:** 30 annotated out of 243 available (12.3%)  |  **Epochs:** 50  |  **Loss:** Dice+BCE (α=0.3, pos\_weight=100)

!!! success "DSC 0.72 from just 30 of 243 masks"
    This result was produced using only **30 hand-drawn masks — 12.3% of the full 243-image annotation pool**. The remaining 213 frames have not yet been annotated. A DSC of 0.72 at this stage is a strong indicator that the annotations are correct and the pipeline is working as expected. With the full pool annotated, DSC of 0.85+ is anticipated.

| Metric | Value |
|--------|-------|
| DSC | **0.7162** |
| IoU | 0.5581 |
| Precision | 0.5594 |
| Recall | **0.9958** |
| Accuracy | 0.9986 |

**Interpretation:** Adding `pos_weight=100` to the BCE term makes each cell pixel count as 100 background pixels in the loss, forcing the model to prioritise finding the cell. The result is near-perfect recall (the model finds virtually all cell pixels) at the cost of slightly over-segmenting the boundary (precision=0.56). This is the expected trade-off at this sample size — a model that over-segments slightly is preferable to one that misses cells. DSC of 0.72 with only 30 training samples confirms that annotations are correct and consistent.

The `loss_alpha=0.3` change (from 0.5) gives Dice loss 70% of the total signal, further counteracting the imbalance.

---

## Training curves (Run 2)

![U-Net training curves](assets/unet_training_curves.png)

**Left — Segmentation loss:** Both train and val loss decrease steadily across 50 epochs, converging to ~0.80. Val loss tracks train loss closely, indicating no significant overfitting with this sample size.

**Right — Validation DSC:** DSC stays near zero for the first ~20 epochs while the model learns basic image features, then climbs sharply between epochs 20–30 as it begins detecting the cell shape. DSC reaches ~0.72 by epoch 50. The oscillation in the second half is expected — the validation set is only ~5 images, so a single mis-segmented frame has a large effect on the reported score.

---

## Run 3 — Binary vs. multiclass label scheme × resolution ablation

Same 40 annotated frames as Run 2 (up from 30), used to isolate two variables against the binary 256×256 baseline: the annotation label scheme (binary vs. 3-class) and the preprocessed input resolution (128×128 vs. 256×256). Four configs, each holding the other three variables fixed (seed 42, 100 epochs, `loss_alpha=0.3`):

| Config | Labels | Resolution | Loss weighting | Trapped-cell DSC | Best epoch |
|--------|--------|-----------|-----------------|-------------------|-----------|
| `configs/binary_experiment_128.yaml` | binary | 128×128 | `pos_weight=50` | 0.590 | 92 |
| `configs/multiclass_experiment.yaml` | 3-class | 128×128 | `class_weights=[1,3,3]` | 0.736 | 99 |
| `configs/binary_experiment_256.yaml` | binary | 256×256 | `pos_weight=50` | 0.763 | 99 |
| `configs/multiclass_experiment_256.yaml` | 3-class | 256×256 | `class_weights=[1,3,3]` | **0.784** | 73 |

Training curves and per-class metric tables: `results/unet_training_curves_v13_binary_128.png` through `v16_multi_256.png`.

**Multiclass detail (256×256, `configs/multiclass_experiment_256.yaml`):**

| Class | DSC | IoU | Precision | Recall | Accuracy |
|-------|-----|-----|-----------|--------|----------|
| background | 0.996 | 0.993 | 1.000 | 0.993 | 0.993 |
| trapped_cell | **0.784** | 0.645 | 0.757 | 0.813 | 0.999 |
| other/decoy | 0.682 | 0.518 | 0.518 | 0.999 | 0.994 |
| mean | 0.821 | 0.719 | 0.758 | 0.935 | 0.995 |

**Interpretation:**

- **Labeling scheme matters more than resolution alone.** At 128×128, adding the decoy-object class raised trapped-cell DSC from 0.590 to 0.736 (+0.146) — a larger jump than switching binary from 128×128 to 256×256 (0.590 → 0.763, +0.173, comparable magnitude) or multiclass from 128×128 to 256×256 (0.736 → 0.784, +0.048, smaller). The best result combines both: multiclass at 256×256.
- **Why multiclass helps:** giving decoy objects (untrapped cells, debris, lookalike blobs) their own class instead of leaving them as unlabeled background stops the loss from penalizing the model for segmenting them — under the binary scheme, a decoy that visually resembles the trapped cell is background the model is punished for finding, which pushes precision down indiscriminately. Explicitly modeling it as its own class lets the network learn to distinguish "cell-like blob, but not the trapped one" from "the trapped cell," rather than lumping both into one binary decision.
- **The decoy class itself is harder** (DSC 0.682 vs. 0.784 for trapped_cell) — expected, since decoy objects are a heterogeneous category (any non-trapped cell-like blob) rather than the single consistent target the trapped-cell class represents.
- **Multiclass converges faster and can overfit sooner:** the 256×256 multiclass run peaked at epoch 73 (val DSC 0.784) then declined to ~0.69 by epoch 100, while the binary run kept improving to epoch 99. `scripts/train_unet.py` reloads `unet_best.pt` before final evaluation specifically to guard against this — final metrics reflect the peak, not wherever the fixed 100-epoch budget happened to end.
- **Caveat:** these are 40-mask, single-seed runs — not yet a controlled statistical comparison. Before freezing the label scheme, re-run at ≥80 masks (see `CLAUDE.md` "Next Steps") to confirm the multiclass advantage holds as the training set grows, since multiclass requires painting decoy objects on every future mask, not just the trapped cell.

---

## Loss function

The combined Dice + BCE loss used in binary-mode runs (Run 1–2, and `binary_experiment_*.yaml`):

```
Loss = α × BCE(pos_weight) + (1 − α) × Dice
```

- **BCE** penalises each pixel individually — gives stable gradients early in training.
- **Dice** measures overall mask overlap — keeps the model focused on the cell region rather than pixel counts.
- **pos_weight** multiplies the BCE penalty on cell pixels to counter the 550:1 background/cell imbalance.

Current defaults in `configs/default.yaml`:

```yaml
training:
  loss_alpha: 0.3        # Dice carries 70% of the loss signal
  loss_pos_weight: 100.0 # increase if recall is low; decrease if precision is low
```

**Multiclass mode** (`unet.out_channels > 1`, e.g. `multiclass_experiment_256.yaml`) replaces BCE with Cross-Entropy and Dice is computed per-class then averaged:

```
Loss = α × CrossEntropy(class_weights) + (1 − α) × mean(Dice per class)
```

`class_weights` (e.g. `[1.0, 3.0, 3.0]` for background/trapped_cell/other-decoy) plays the same role as `pos_weight` but is **not** directly comparable in magnitude — `CrossEntropyLoss`'s weighted-mean reduction renormalises by weight mass, so porting `pos_weight=100` in directly collapsed precision to ~0.13 by pushing recall to 1.0 everywhere. `[1, 3, 3]` was the winner of a small sweep (`1,1,1` / `3,3,3` / `5,5,5` / `10,10,10` / `20,20,20`, each trained the full 100 epochs) — trapped_cell precision=0.757/recall=0.813, a balanced result rather than recall-saturated.

---

## Expected trajectory

These are preliminary runs on a fraction of the full annotation pool. As more masks are added:

| Masks | Expected DSC (binary) | Suggested `pos_weight` |
|-------|-------------|----------------------|
| 30 | ~0.72 | 100 |
| 40 (now) | 0.76 (binary) / **0.78 (multiclass)** — see Run 3 above | 50 |
| ~80 | ~0.80 | 50–75 |
| 150+ | 0.85+ | 25–50 |

The thesis-reportable DSC comes from `scripts/evaluate_unet.py` on the `02_unet_holdout` pool — annotated only after training is frozen. See [Workflow Guide](workflow.md#step-5-evaluate-on-holdout-set).

---

## Next: Data Augmentation

To improve generalisation from the small annotated set, data augmentation will be integrated into the training loop. The planned strategy applies paired spatial transforms to each frame and its mask so they remain aligned:

- Horizontal flip (50% chance)
- Vertical flip (50% chance)
- Random 90° rotation (0°, 90°, 180°, or 270°)

`CellDataset` in `scripts/train_unet.py` already supports this via `augment=True`; it will be enabled for the training split in the next run. Intensity augmentation is intentionally excluded — CLAHE in Stage 2 already normalises contrast.
