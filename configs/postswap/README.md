# `postswap` — the active experiment series (from 2026-09-17)

Everything after the 17 September cell exchange lives here. Runs from before it are
**not comparable** and stay where they were.

## Layout — one folder per stage, same `postswap/` name in each

```
configs/postswap/        this folder: one YAML per run
checkpoints/postswap/    <run-name>/unet_best.pt + config.yaml snapshot
logs/postswap/           <run-name>.log  (training stdout, incl. the split assignment)
results/postswap/        postswap_eval_log.jsonl + training-curve PNGs
data/preprocessed/train/postswap_multiclass_256/     451 frames, 22 groups
data/preprocessed/evaluate/postswap_devset/          172 annotated dev frames
```

## Naming

`postswap_<loss>_<classweights>_s<seed>` — e.g. `postswap_ftl_cw142_s42`.
Extend it for new variables, keeping the same order, e.g. `postswap_ftl_cw142_res384_s42`.
The name is reused verbatim as the config filename, the checkpoint directory and the
log filename, so one string finds everything belonging to a run.

Do **not** rename an existing run: `results/postswap/postswap_eval_log.jsonl` keys
results by the checkpoint's parent directory name, so a rename orphans its results.

## Baseline

**`postswap_ftl_cw142_s42` is the baseline every subsequent run is measured against.**
Focal-Tversky (alpha 0.7, beta 0.3, gamma 0.75), class weights [1,4,2], width 64,
cosine, scale augmentation only, 451-frame pool.

Development-set results, 3 seeds (42/43/44), split pinned at 42:

| Cell     | s42   | s43   | s44   | mean  | sd    |
|----------|-------|-------|-------|-------|-------|
| high-1   | 0.604 | 0.702 | 0.381 | 0.562 | 0.164 |
| high-9   | 0.825 | 0.895 | 0.816 | 0.845 | 0.043 |
| low-5    | 0.922 | 0.913 | 0.814 | 0.883 | 0.060 |
| low-6    | 0.718 | 0.804 | 0.777 | 0.766 | 0.044 |
| **Pooled** | **0.800** | 0.847 | 0.754 | **0.800** | 0.046 |

Seed 42 is the working seed because it is the median draw and lands exactly on the
3-seed mean; seed 43 is the high draw and would anchor everything optimistically.

## Rules for this series

1. **One seed (42) while iterating.** `split_seed` stays pinned at 42 too.
2. **A single-seed delta below ~0.16 is NOT a result.** Measured spread is sd 0.057,
   so one run resolves only differences above roughly 2*sd*sqrt(2) ~ 0.16. Anything
   smaller is logged as "screened, inconclusive" -- never as a finding. This is the
   mistake that cost this project the `scale_cos` recipe (+0.066, unverifiable) and
   the Focal-Tversky "+0.023" (its repeat came back -0.049).
3. **Three seeds on both arms before anything final**, not just the new one. A 3-seed
   candidate against a 1-seed baseline is still a single-vs-single comparison.
4. **Change one variable per run**, and if a non-experimental setting moves (schedule,
   split, width, pool), re-run the baseline under it before quoting a delta.
5. **Check the split log line** at the top of every run. `high_cell10` is 50 frames
   against a ~45-frame validation target and lands in validation on 13% of seeds; if
   it does, that run trained without its only small cell and is void.
