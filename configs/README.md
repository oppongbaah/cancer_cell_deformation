# Configs

> # ⭐ CURRENT / BEST
>
> ## **`configs/postswap/experiment_0a_seed_baseline/postswap_ftl_cw142_s42.yaml`**
>
> **Every new experiment branches from this file.** Copy it, change **one** field,
> rename it, and put it in its experiment folder, `configs/postswap/<experiment>/`
> (see `EXPERIMENTS.md` at the repo root).
>
> Focal–Tversky loss (α 0.7, β 0.3, γ 0.75) · class weights **[1, 4, 2]** ·
> width 64 · no dropout · cosine · scale augmentation only · seed 42 ·
> **split_seed pinned at 42** · 451-frame post-swap pool.
>
> **Development-set result (complete 203-frame set): single model 0.809 ± 0.049**
> across four seeds, **ensemble of four 0.859**.
> Seed 42 remains the working seed for iteration. Do not adopt whichever seed scores
> highest — that is selection on the evaluation data.

---

## Layout

```
configs/
  default.yaml        base template — binary, 256×256. NOT what is being trained.
  postswap/<experiment>/   the active series (after the 17 Sept cell exchange):
                           experiment_0a_seed_baseline, experiment_0b_distillation, experiment_1a_optimizer, experiment_1b_loss,
                           experiment_2_resolution
  preswap/<stage>/         historical, grouped into stage_1 ... stage_5. The record, not for new runs.
```

Since 2026-09-30 every experiment has the same folder name under `configs/`,
`checkpoints/`, `logs/` and `results/`. `EXPERIMENTS.md` at the repo root lists them all.

**A config's results are only comparable to others trained on the same pool.** The
17 September exchange moved `high-10` into training and `high-1` into the development
set, so nothing in `preswap/` can be compared against anything in `postswap/`.

## `postswap/` — the active series

| Config | Seed | Evaluated DSC |
|---|---|---|
| **`postswap_ftl_cw142_s42.yaml`** | 42 | 0.830 ⭐ working seed |
| `postswap_ftl_cw142_s43.yaml` | 43 | 0.864 |
| `postswap_ftl_cw142_s44.yaml` | 44 | 0.753 — collapsed on high-1 |
| `postswap_ftl_cw142_s45.yaml` | 45 | 0.787 — collapsed on high-9 |

All four are identical except `seed`. **Two of the four collapse on one cell, and which
cell is unpredictable** — see `checkpoints/README.md`. The spread across them **is the
error bar** — see the rules in `configs/postswap/README.md`.

## `preswap/` — historical, by era

**The 17 Sept loss experiments** (four-cell development set as it stood that morning):

| Config | Evaluated DSC | Outcome |
|---|---|---|
| `multiclass_experiment_256_scale_cos_ftl.yaml` | 0.691 | Focal–Tversky; best pre-swap |
| `multiclass_experiment_256_scale_cos_cw133.yaml` | 0.676 | class weights → [1,3,3]; no real effect |
| `multiclass_experiment_256_scale_cos_nobg.yaml` | 0.675 | background out of the Dice term; no real effect |
| `multiclass_experiment_256_scale_cos.yaml` | 0.668 | the settled pre-swap recipe |
| `multiclass_experiment_256_scale_cos_ftl_seed43.yaml` | 0.619 | **seed repeat of the 0.691 run** |

⚠️ That last row is the important one: **the same config at a different seed scored
0.619 against 0.691.** Every delta above it is smaller than that swing, so none of the
pre-swap loss results are established.

**Earlier sweeps and ablations** — `base_cos` (no augmentation, 0.605), `allaug`
(0.572), `oversample` (domain oversampling, tried and rejected),
`multiclass_experiment_256.yaml` (original), `binary_experiment_{128,256}.yaml` and
`multiclass_experiment.yaml` (the 128-vs-256 × binary-vs-multiclass ablation that
settled the label scheme and resolution). `scale_cos_445.yaml` is invalid — it trained
on `low_cell5`, which became a development cell.

## Adding a config

Name it `postswap_<loss>_<classweights>[_<newvar>]_s<seed>.yaml` and keep the same
order, e.g. `postswap_ftl_cw142_res384_s42.yaml`. The name is reused verbatim for the
checkpoint directory and log file, so one string finds everything for a run. Then:

```bash
python scripts/training/train_experiment.py postswap/experiment_0a_seed_baseline postswap_ftl_cw142_res384_s42
```

Never rename an existing run — `results/postswap/experiment_0a_seed_baseline/eval_log.jsonl` keys results
by the checkpoint's parent directory name, so a rename orphans its results.
