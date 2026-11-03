# Checkpoints

> # ⭐ CURRENT / BEST
>
> ## **`checkpoints/postswap/experiment_0a_seed_baseline/postswap_ftl_cw142_s42/unet_best.pt`**
>
> **This is the checkpoint to load** for inference, feature extraction, or as the
> comparison baseline for any new run.
>
> Focal–Tversky · class weights [1, 4, 2] · width 64 · 451-frame post-swap pool ·
> best epoch 87.
>
> **Development set (4 cells, 203 frames — fully annotated 23 Sept):**
>
> | | evaluated DSC | spread |
> |---|---|---|
> | single model, 4-seed mean (s42/43/44/45) | 0.809 | sd 0.049 |
> | **ensemble of all four** | **0.859** | — |
>
> ### ⚠ Two of four runs collapse on a single cell — a different cell each time
>
> | run | worst cell | its DSC | internal val | convergence check |
> |---|---|---|---|---|
> | s42 | low-6 | 0.718 | 0.9327 | passed |
> | s43 | low-6 | 0.804 | 0.9398 | passed |
> | s44 | **high-1** | **0.507** | 0.9050 | **FLAGGED** |
> | s45 | **high-9** | **0.630** | 0.9357 | **passed** ✗ |
>
> Size does not predict it (`high-1` 678 px, `high-9` 704 px). Internal validation does
> not either — it caught s44, which was behind everywhere, and missed s45, which trained
> normally and then lost one cell. **Passing `check_convergence.py` is not evidence.**
>
> **Ensembling is what fixes it.** No per-cell collapse survives averaging: `high-1`
> never below 0.823, `high-9` never below 0.822.
>
> | | high-1 | high-9 | low-5 | low-6 | **All cells** |
> |---|---|---|---|---|---|
> | single model, 4-seed mean | 0.742 | 0.792 | 0.890 | 0.770 | 0.809 |
> | **ensemble of 4** | **0.823** | **0.848** | **0.920** | **0.818** | **0.859** |
>
> An ensemble is an **inference-time construct, not a model** — no checkpoint, 4× GPU,
> and `RealTimePipeline` takes one `UNet`. Averaging the *weights* instead gives
> **DSC 0.000** (independently seeded nets sit in different loss basins). Distillation
> is the route to one deployable checkpoint at ensemble quality — not yet done.

---

> ### ⚠️ Read before comparing any two numbers here
>
> **Pre-swap and post-swap checkpoints are scored on different development sets** and
> cannot be compared. The 17 September exchange moved `high-10` into training and
> `high-1` into evaluation.
>
> **A single-run difference below ~0.14 is not a result** (4-seed sd 0.049). Measured seed-only spread on
> the complete 203-frame development set is **sd 0.057** (was 0.047 on 172 frames) —
> and a pre-swap config re-run at a different seed moved 0.691 → 0.619.
> `scripts/training/compare_postswap.py` prints this verdict for you.

## Layout

```
checkpoints/
  postswap/<experiment>/   the active series, 451-frame pool (17 Sept onward)
      experiment_0a_seed_baseline/   s42 s43 s44 s45
      experiment_0b_distillation/    postswap_distilled (failed, 0.537)
      experiment_1a_optimizer/          e1a_sgd_nesterov (not an improvement)
      experiment_1b_loss/               e1b_lovasz_s42..s45 (the working loss)
      experiment_2_resolution/          e2a_res256_b4, e2b_res512_b4
  preswap/<stage>/         26 historical runs in stage_1 ... stage_5, 101/202-frame sets
  _retired/                weights deliberately deleted; kept for the record
```

Grouped by experiment on 2026-09-30; `EXPERIMENTS.md` at the repo root lists every
experiment and what it found.

Checkpoint weights are **not** in git (`*.pt`, 7 GB). This README is. The durable
record of what each one scored is `results/preswap/preswap_eval_log.jsonl` and
`results/postswap/experiment_0a_seed_baseline/eval_log.jsonl`, which **are** tracked.

## `classification/`, `biomechanics/` — the final analysis models (since 2026-10-29)

Each method fitted **once on all 14 usable cells** of a feature run and saved, so a new cell can be classified
and its rate of deformation estimated without retraining. The leave-one-cell-out runs keep no model; their score
is stored with each final model as its expected accuracy. **A final model's score on the 14 cells it was trained
on is optimistic** (logistic regression 13/14, CNN 14/14) — quote the stored `loocv_*` numbers instead.

| folder | what | made by |
|---|---|---|
| `classification/<run>/classical/<set>__<classifier>.joblib` | 8 classifiers × {selected, all} + the size baseline (17); imputer + scaler + model, feature list, `info` | `benchmark_cells.py` (every full run; `--final` = only this) |
| `classification/<run>/cnn/finetune_masked_seed{0,1,2}.pt` | fine-tuned ResNet-18 on the cell crop, full weights (~45 MB each); a cell = mean P(high) of its frames, averaged over the 3 seeds | `frames_cnn.py final --seeds 0 1 2` |
| `biomechanics/<run>/rod_model.json` | RoD settings + priors (mean, between-cell SD) pooled and per HER2 class, per measure | `rate_of_deformation.py` (every run) |

Each folder has a `manifest.csv`. Load with `celldform.final_models.load_classifier(path).p_high(cells)`,
`frames_cnn.load_final(path)` + `frames_cnn.predict(...)` on crops from `frames_cnn.cut_crop(...)`, and
`celldform.biomechanics.rod.estimate(frames, rod.load_model(path), her2)`. Settings are the configs' (untuned:
nested tuning did not help). Not saved: the sequence networks (below chance) and the mixed-effects regression
(its coefficients are the result, in `results/biomechanics/`).

## `postswap/` — active

| Checkpoint | Seed | Evaluated DSC |
|---|---|---|
| `postswap_ftl_cw142_s42` | 42 | 0.830 |
| **`postswap_ftl_cw142_s43`** | 43 | **0.864** (best single) |
| `postswap_ftl_cw142_s44` | 44 | 0.753 — collapsed on high-1 |
| `postswap_ftl_cw142_s45` | 45 | 0.787 — collapsed on high-9 |

Scored on the complete 203-frame development set. Mean 0.809, sd 0.049; ensemble 0.859.
**Do not adopt s43 because its number is highest** — picking a seed on the development
set is selection on the evaluation data.

Identical configs; only the seed differs. The spread is the error bar, not a ranking —
do **not** adopt s43 because its number is highest.

## `preswap/` — historical, best first

| Checkpoint | Evaluated DSC | What it was |
|---|---|---|
| `..._bf64_scale_cos_ftl` | 0.691 | Focal–Tversky; best pre-swap |
| `..._bf64_scale_cos_445` | 0.680 | 445-frame pool — ⚠️ **invalid**, trained on `low_cell5` which is a development cell |
| `..._bf64_scale_cos_cw133` | 0.676 | class weights [1,3,3] |
| `..._bf64_scale_cos_nobg` | 0.675 | background excluded from the Dice term |
| `..._bf64_scale_cos` | 0.668 | the settled pre-swap recipe |
| `..._bf64_scale_cos_ftl_seed43` | 0.619 | **seed repeat of the 0.691 run** — the error bar |
| `..._bf64_base_cos` | 0.605 | no augmentation |
| `..._bf64_allaug` | 0.572 | all augmentations combined |
| `multiclass_experiment_256` | 0.525 | original, older schedule and split |

Also present: the `bf32_*` regularisation sweep (dropout / cutout / weight decay /
scheduler / split ratio — all scored on the *internal validation* split, which the
10 September controlled re-run showed carries no signal for development-set questions),
`bf64_reg`, `bf64_bc`, `domain_oversample`, and the `binary_experiment_*` /
`multiclass_experiment` label-scheme and resolution ablations.

## Notes worth keeping

- **`multiclass_experiment_256`'s epoch-69 / 0.9034 checkpoint no longer exists.** It
  was overwritten in place by successive retrains before the non-determinism cause was
  found. The domain-gap-closed result measured on it (0.9104 DSC) is a historical
  record that cannot be reproduced from anything on disk.
- Every directory holds `unet_best.pt` plus a `config.yaml` snapshot of the exact
  settings it was trained with — that snapshot, not the file in `configs/`, is what the
  run actually used.
