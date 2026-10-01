# Experiments

Every segmentation training run in this project, grouped by the question it was
asking. Each experiment has **one folder name, mirrored under all four roots**:

```
configs/<era>/<experiment>/       what was run          (tracked)
checkpoints/<era>/<experiment>/   the weights            (git-ignored)
logs/<era>/<experiment>/          console output         (tracked)
results/<era>/<experiment>/       curves + eval_log.jsonl
```

Two eras, split by the 17 September roster exchange (`high-10` into training, `high-1`
out). **Scores from different eras are not comparable** — the testing set changed.
Preswap numbers were also measured on a 2-cell set early on and a 4-cell set later;
each row says which.

Run a new experiment with
`scripts/training/run_experiment.sh <era>/<experiment> <run>`; it trains, scores the
run on the testing set and files everything above. The move into this layout is
recorded, path by path, in `scripts/maintenance/reorg_experiments_20260930.py`.

---

## Preswap (August – 17 September)

| Stage | Question | Runs | What it found |
|---|---|---|---|
| `stage_1_labels_and_resolution` | Binary or 3-class labels? 128 or 256 px? | 4 | 3-class at 256 wins (0.784 internal, 40 masks). The plain `multiclass_experiment_256` later scored **0.525** on the 2-cell set and stayed the best for a week. Also the determinism investigation (`cudnn_benchmark`). |
| `stage_2_domain_gap` | Does oversampling real-cell frames close the legacy→real gap? | 1 | No (real-cell DSC 0.538 → 0.455). Only new annotated cells helped. |
| `stage_3_regularisation` | Width, dropout, cutout, weight decay, SWA, schedule, split | 12 | Every lever improved internal validation and **hurt** real cells (best internal 0.862 → 0.408 on the 2-cell set). |
| `stage_4_augmentation` | Brightness, scale, combined — and a controlled re-run | 5 | Scale augmentation alone: **0.671**, the settled recipe. The "combined" win was the schedule and split, not augmentation (retracted). `…_445` is invalid (trained on `low-5`). |
| `stage_5_loss` | Class weights, background in Dice, Focal-Tversky | 4 | 0.676 / 0.675 / 0.691 vs 0.668 — all inside noise; seed alone moved FTL 0.691 → 0.619. FTL kept on mechanism. |

All preswap evaluations are in one ledger, `results/preswap/preswap_eval_log.jsonl`.

## Postswap (17 September –), 4-cell testing set, 203 frames

| Experiment | Question | Runs | What it found |
|---|---|---|---|
| `experiment_0a_seed_baseline` | How much does the seed alone move the score? | 4 seeds + ensemble | **0.809 ± 0.049** single model; two of four seeds lose one cell. **Ensemble of four 0.859**, no collapse. Deployment test across the laser ramp in `results/…/deployment_ensemble/`. |
| `experiment_0b_distillation` | Can one student match the ensemble? | 1 | **No — 0.537**, below every seed (24 Sep, found unreported 30 Sep). |
| `experiment_1a_optimizer` | Does SGD (Nesterov 0.99, lr 0.01; nnU-Net's optimizer) beat AdamW? vs s43 (0.864) | 1 | **0.852 — not an improvement**: high-1 partly collapsed (0.642, recall 0.51), unstable early (val-loss spikes), decoy class barely learned (internal decoy recall 0.21). Kept on record, not pursued. |
| `experiment_1b_loss` | Does adding a Lovász-softmax (IoU) term help? vs the baseline seeds | 4 seeds + ensemble | **0.857 ± 0.026 vs baseline 0.809 ± 0.049**, better on 4/4 paired seeds (mean +0.049; paired t p = 0.18, Wilcoxon p = 0.125), worst cell 0.507 → 0.700. **Ensemble of four 0.875.** Adopted as the working loss; not statistically confirmed. |
| `experiment_2_resolution` | Does a 512×512 input beat 256×256? | 2 (+ ref) | **512 better on every cell at native 1228×922: 0.789 → 0.877 pooled, per-cell mean 0.766 → 0.868**, precision and recall both up (one seed per arm; Lovász recipe, s42, micro-batch 4 × 4 accumulation in both arms). Own-grid 0.846 → 0.874. Stopped early (best epoch 18, stop 38) — longer training untested. **Costs real time**: 62 ms/frame (16 fps) vs 26 ms; fp16 inference cuts the forward pass 47.7 → 21.3 ms with 1 px in 5.5 M changed — not yet in the pipeline. |
| `experiment_3_cropping` | Does a native-resolution crop beat resizing the whole frame? | 2 | Every trapped cell of all 15 cells lies in x 489–671, y 432–631 (fixed by the optics); crop centre (579, 510) from TRAINING cells only, legacy (648, 373). Native scores: **crop 384 0.897 pooled / 0.887 per cell, best precision 0.884, 37.8 ms/frame (26.5 fps, real time without fp16)**; crop 512 0.855 (high-9 0.703, more decoys in the window). Both vs resize 512 0.877 and resize 256 0.789. One seed each: high-res ≫ 256 is clear, the ranking among high-res inputs is not. Crop centre is a per-setup calibration. |

Baseline for experiment 1 is `postswap_ftl_cw142_s43` — **0.864, the best of the four
seeds**. One run resolves only ±0.139, so a delta against it is a screen, not a result.

## Frozen recipe (2026-10-01)

**Lovász-softmax loss + native 384×384 crop** — `configs/postswap/experiment_3_cropping/e3b_crop384_b4.yaml`
(Focal-Tversky + CE + 0.5 × Lovász, width 64, scale augmentation, cosine, micro-batch 4 × 4-step
accumulation, seed 42). This is the recipe the randomised `randswap` split will train and test,
with the crop centre derived per fold from that fold's training cells only.

