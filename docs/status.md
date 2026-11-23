# Project status

*November 2026.* What exists, what has been measured, and what has not. Earlier work is kept in the
[research log](research_log_2026-09.md).

## What the project delivers

| Part | State |
|---|---|
| **Segmentation** of the trapped cell: a U-Net (384 px native crop) and an nnU-Net benchmark | trained and tested at the level of whole cells |
| **Feature extraction** with five per-frame quality checks (a frame that fails is not used) | working |
| **HER2 classifier**: step-level and cell-level logistic regression, averaged | validated by leave-one-cell-out on 15 cells |
| **Desktop app** that analyzes a recorded video and reports the HER2 class with its evidence | working; ships no models, see [the app](demo.md) |
| Real-time operation, connection to the instrument's control software | **not done**: the app analyses recordings (see below) |

## Results

Every number below holds out **whole cells**: frames of one cell are never in both training and test, because frames of the
same cell are almost identical. 15 cells (9 high HER2, 6 low HER2), one cell line (SKBR3), one setup.

**Segmentation: Dice score of the trapped cell, mean over cells ± standard deviation**

| Model | Test protocol | Mean over cells | High HER2 | Low HER2 | Worst cell |
|---|---|---|---|---|---|
| U-Net, 384 px crop | 5 rounds, random cells held out | 0.855 ± 0.114 | 0.802 | 0.934 | 0.625 |
| U-Net, 384 px crop | leave-one-cell-out, 15 rounds | 0.865 ± 0.092 | 0.834 | 0.910 | 0.710 |
| nnU-Net (2D) | leave-one-cell-out, 15 rounds | **0.940 ± 0.069** | 0.919 | 0.972 | 0.725 |

High-HER2 cells are harder to segment than low-HER2 cells. The U-Net takes about 40 ms per frame on a modern GPU, the
nnU-Net about 0.4–0.5 s. A cell-level result varies a lot between cells, so a difference of a few hundredths between
two single runs is not evidence of anything (see the [research log](research_log_2026-09.md) for how large the noise is).

**Classification: HER2 class of a cell**

Using masks from the nnU-Net, the classifier called **14 of 15** cells correctly in leave-one-cell-out testing (AUC 0.94,
permutation p ≈ 0.002). It was then fitted on all 15 cells; that final model is what the app uses.

## Limits you should know about

- **One setup, 15 cells.** The result says how well the method works on cells recorded the way these were (a laser
  ramp from 500 to 1200 mA in 50 mA steps, 30 s per level). It does not show that it transfers to other instruments,
  cell lines or protocols.
- **A recording from a different setup may not work.** On four recordings from a newer setup (colour camera, a
  298–1000 mA ramp in steps of about 9 s) the U-Net found the cell in only 2–13 % of the measured frames, so the app
  correctly gave no verdict. Using it there needs new annotated frames and fine-tuning.
- **The app is cautious on purpose.** It gives a verdict only when at least 3 laser steps are usable and 80 % of the
  measured frames pass the checks. The classifier was validated on whole recordings (10 or more steps), so a recording of 3
  to 9 steps gets a result labelled *short recording* from the step model alone, which is weaker. An optional
  *provisional* verdict exists for recordings that fail the checks; it is always labelled not validated and is not a result.
- **Not real-time.** The U-Net is faster than the camera's 45.5 ms frame interval, but the delivered software analyses
  recordings and is not connected to the instrument.
- **Research software.** It is not a medical device and not for clinical decisions.

## What would come next

More cells and a second setup for external validation; fine-tuning for the newer recording protocol; and, if wanted,
a version that runs alongside the instrument.
