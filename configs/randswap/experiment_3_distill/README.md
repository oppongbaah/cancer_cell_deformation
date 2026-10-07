# Experiment 6 — distillation: nnU-Net (teacher) → our architecture (student)

Teacher: each randswap round's nnU-Net (`experiment_1_nnunet`, fold k-1), which never saw that round's
test cells. Student: our U-Net + native 384 crop, the `cv_baseline` recipe, plus a distillation term
(weight 1.0, temperature 2, fixed in advance) and ~2,200 nnU-Net-labelled extra frames of the round's
training cells (every 50th frame, offset 25). Same samples per epoch as the CV.

- `pilot/` — round 4 only. `student` (early-stopped on the validation cell, epoch 13): 0.587.
  `student_fixed28` (fixed 28 epochs, last weights): 0.833 vs ours 0.757 — the check that passed.
  `soft_targets.png`: teacher targets next to hand-drawn labels (`scripts/report/view_soft_target.py`).
- `cv/` — all 5 rounds, each at the fixed epoch our CV model reached (39/30/29/28/26), last weights.
  **Failed the pre-set rule**: per-cell mean 0.828 vs 0.855, Pooled DSC 0.805 vs 0.861.
  Round 4 here scored 0.617 — vs 0.833 in the pilot with the same settings minus 2 duplicate frames:
  single runs are not evidence on this data.

Training data (shared by pilot and cv): `data/preprocessed/randswap/experiment_3_distill/fold_<k>/`
(`scripts/data/build_distill_data.py`). Round 4's data was built for the pilot; 2 frames that were also
annotated frames (low-2 2475, 7425) were removed before the CV.
