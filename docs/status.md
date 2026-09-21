# Project Status

Current state of the segmentation work, as of **17 September 2026**. This page is the
short answer to "where are we?"; the other pages are reference.

## Current model

**`checkpoints/postswap/postswap_ftl_cw142_s42/unet_best.pt`**
· config `configs/postswap/postswap_ftl_cw142_s42.yaml`

Focal–Tversky loss (α 0.7, β 0.3, γ 0.75, [Abraham & Khan 2019](https://arxiv.org/abs/1810.07842))
· class weights [1, 4, 2] · width 64 · cosine · scale augmentation only · 451-frame pool.

**Development set (4 cells, 172 annotated frames): pooled trapped-cell DSC 0.800 ± 0.047**
(3-seed mean, train/val split pinned).

| Cell | DSC | | Cell | DSC |
|---|---|---|---|---|
| high-1 | 0.604 | | low-5 | 0.922 |
| high-9 | 0.825 | | low-6 | 0.718 |

## What changed on 17 September

**The evaluation roster was restructured.** `high-10` moved into training and `high-1`
into the development set. The reason was a measured distribution gap: every development
cell had a smaller trapped cell (688–1033 px) than every training cell (824–1549 px), so
the model was judged on cells smaller than anything it had trained on. Training now
spans 688–1236 and the development set 704–1549.

The effect on the two cells that did **not** move, and so remain comparable:

| Cell | before | after | |
|---|---|---|---|
| high-9 | 0.600 | **0.845 ± 0.043** | +0.245 — well outside the noise |
| low-6 | 0.749 | 0.766 ± 0.044 | inside the noise |

!!! note "One caveat on that result"
    `high_cell10` (688 px) and `high_cell9` (704 px) are near-twins — same laser regime,
    nearly the same size. Moving `high-10` into training was close to adding a duplicate
    of `high-9`, so 0.845 is likely optimistic as an estimate for a genuinely new
    high-power cell. The mechanism is real; the effect size will not transfer intact.

**Run-to-run variance was measured for the first time**, and it is large — see the
[Workflow Guide](workflow.md) rules and the banner on [Training Results](training.md).

**The loss function was ruled out as the bottleneck.** Three interventions —
class-weight rebalancing, excluding background from the Dice term, and replacing Dice
with Focal–Tversky — produced 0.676, 0.675 and 0.691 against a 0.668 baseline. A
structural finding along the way: the objective is `0.3·CE + 0.7·Dice`, and
`class_weights` reaches **only the CE term**, so that knob was never touching 70% of
the loss.

## Where the evidence points

Object scale, not hyperparameters. At 256×256 the trapped cell occupies **41–90 px** —
roughly a 6×6 to 9×9 blob — and the decoy class on `high-10` about 51 px. That is very
likely why no loss function recovered the decoy class: the information may not survive
the 17× downsample. Resolution has not been revisited since the 128→256 ablation
(+0.048, on 40 masks); the pool is now 451.

## Next

1. Annotate `high-1`'s remaining 31 frames — it is currently scored on 20, with sd 0.164
   across seeds, and is the largest single contributor to the pooled error bar.
2. Test resolution (384, then 512 if the trend holds), with a matched-batch-size control.
3. The deferred optimizer and validation-split experiments, gated behind a multi-seed
   baseline.
4. nnU-Net benchmark if the above fall short. Note its default loss is Dice+CE — i.e.
   this project's baseline — so any margin will come from architecture and schedule.
