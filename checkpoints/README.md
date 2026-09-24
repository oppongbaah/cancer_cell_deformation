# Checkpoints

> # ⭐ CURRENT / BEST
>
> ## **`checkpoints/postswap/postswap_ftl_cw142_s42/unet_best.pt`**
>
> **This is the checkpoint to load** for inference, feature extraction, or as the
> comparison baseline for any new run.
>
> Focal–Tversky · class weights [1, 4, 2] · width 64 · 451-frame post-swap pool ·
> best epoch 87.
>
> **Development set (4 cells, 203 frames — fully annotated 23 Sept):**
>
> | | pooled DSC | spread |
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
> | | high-1 | high-9 | low-5 | low-6 | **Pooled** |
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
  postswap/     the active series, 451-frame pool (17 Sept onward)
  preswap/      26 historical runs, 101/202-frame development sets
  _retired/     weights deliberately deleted; kept for the record
```

Checkpoint weights are **not** in git (`*.pt`, 7 GB). This README is. The durable
record of what each one scored is `results/preswap/preswap_eval_log.jsonl` and
`results/postswap/postswap_eval_log.jsonl`, which **are** tracked.

## `postswap/` — active

| Checkpoint | Seed | Pooled DSC |
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

| Checkpoint | Pooled DSC | What it was |
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
