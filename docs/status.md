# Project Status

Current state of the segmentation work, as of **17 September 2026**. This page is the
short answer to "where are we?"; the other pages are reference.

## Current model

**`checkpoints/postswap/postswap_ftl_cw142_s42/unet_best.pt`**
· config `configs/postswap/postswap_ftl_cw142_s42.yaml`

Focal–Tversky loss (α 0.7, β 0.3, γ 0.75, [Abraham & Khan 2019](https://arxiv.org/abs/1810.07842))
· class weights [1, 4, 2] · width 64 · cosine · scale augmentation only · 451-frame pool.

**Development set (4 cells, 203 annotated frames, fully annotated 23 September):**

| | evaluated DSC | spread |
|---|---|---|
| single model, 4-seed mean | 0.809 | sd 0.049 |
| **ensemble of four** | **0.859** | — |

!!! danger "Two of four runs collapse on a single cell — a different cell each time"

    | run | worst cell | its DSC | internal val | convergence check |
    |---|---|---|---|---|
    | s42 | low-6 | 0.718 | 0.9327 | passed |
    | s43 | low-6 | 0.804 | 0.9398 | passed |
    | s44 | **high-1** | **0.507** | 0.9050 | **FLAGGED** |
    | s45 | **high-9** | **0.630** | 0.9357 | **passed** ✗ |

    Neither cell size (`high-1` 678 px, `high-9` 704 px) nor internal validation predicts
    which cell goes. `check_convergence.py` caught s44 — behind on three of four cells —
    and missed s45, which trained normally and then lost one cell. **Passing it is not
    evidence.**

    **Ensembling is what fixes this.** No per-cell collapse survives averaging: `high-1`
    never falls below 0.823, `high-9` never below 0.822.

    | | high-1 | high-9 | low-5 | low-6 | All cells |
    |---|---|---|---|---|---|
    | single model, 4-seed mean | 0.742 | 0.792 | 0.890 | 0.770 | 0.809 |
    | **ensemble of 4** | **0.823** | **0.848** | **0.920** | **0.818** | **0.859** |

    But an ensemble is an **inference-time construct, not a model** — no checkpoint,
    4× GPU, and `RealTimePipeline` takes one `UNet`. Averaging the *weights* instead
    gives **DSC 0.000**: independently seeded networks sit in different loss basins and
    their hidden units are not aligned. **Distillation** — training one model on the
    ensemble's soft outputs — is the route to a single deployable checkpoint, and is not
    yet done.

## What changed on 17 September

**The evaluation roster was restructured.** `high-10` moved into training and `high-1`
into the development set, because the development cells were measurably smaller than the
training cells and the model was being judged on cells smaller than anything it had
trained on.

Measured now, training spans **688–1449 px** and the development set **678–1033 px**,
and the two interleave rather than separating: `high-10` (688) sits between `high-1`
(678) and `high-9` (704) — the intended effect.

!!! note "Correction, 23 September"
    `high-1` was re-annotated when its remaining frames were completed, and now measures
    **678 px**, not the 1549 px recorded while it was a training cell. It is the
    *smallest* cell in the study, not the largest. The earlier masks no longer exist, so
    the annotations cannot be compared directly.

    This weakens the premise above: the pre-exchange claim that every development cell
    was smaller than every training cell assumed a training minimum of 824 px, which
    depended on `high-1` being large. **The size gap was real but smaller than reported,
    and part of it was an annotation artifact.** The effect on the two cells that did not
    move is measured directly and does not depend on it.

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

1. **Decide what the deployed model is** — ensemble (0.859, no checkpoint, 4× GPU) or a
   distilled single model. This determines which number the thesis reports as "the model".
2. Test resolution (384, then 512 if the trend holds), with a matched-batch-size control.
   The development set is now complete, so the baseline it is measured against is final.
3. **Understand the per-cell collapse** — the largest single source of uncertainty in
   every number above, and nothing currently predicts it.
4. The deferred optimizer and validation-split experiments, gated behind a multi-seed
   baseline.
5. nnU-Net benchmark if the above fall short. Note its default loss is Dice+CE — i.e.
   this project's baseline — so any margin will come from architecture and schedule.
