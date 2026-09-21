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
> **Development set (4 cells, 172 frames): pooled trapped-cell DSC 0.800**
> — high-1 0.604 · high-9 0.825 · low-5 0.922 · low-6 0.718
>
> 3-seed mean 0.800, sd 0.047. Seed 42 is the median draw and equals the mean.

---

> ### ⚠️ Read before comparing any two numbers here
>
> **Pre-swap and post-swap checkpoints are scored on different development sets** and
> cannot be compared. The 17 September exchange moved `high-10` into training and
> `high-1` into evaluation.
>
> **A single-run difference below ~0.13 is not a result.** Measured seed-only spread is
> sd 0.047 — one config re-run at a different seed moved 0.691 → 0.619.

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
| **`postswap_ftl_cw142_s42`** | 42 | **0.800** ⭐ |
| `postswap_ftl_cw142_s43` | 43 | 0.847 |
| `postswap_ftl_cw142_s44` | 44 | 0.754 |

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
