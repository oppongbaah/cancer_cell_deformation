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

## Loss function

The combined Dice + BCE loss used in all runs:

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

---

## Expected trajectory

These are preliminary runs on a fraction of the full annotation pool. As more masks are added:

| Masks | Expected DSC | Suggested `pos_weight` |
|-------|-------------|----------------------|
| 30 (now) | ~0.72 | 100 |
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
