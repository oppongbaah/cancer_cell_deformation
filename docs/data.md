# Data Layout

Full documentation of the `data/` directory. The data itself is not in the repository (see [Data, models and logs](https://github.com/oppongbaah/cancer_cell_deformation#data-models-and-logs)); this page describes its layout so that the code and the scripts can be followed.

!!! note "Snapshot of 2026-09-04"
    The folder structure and the contamination-firewall principle below still hold. The **pool counts and which cells play
    which role changed afterwards**: the final experiments split whole cells at random (5-fold and leave-one-cell-out, see
    [Project status](status.md)), and the machine that holds the data is the reference, not this page.

---

## Directory structure

```
data/
├── raw/                                Source recordings — never modified
│   ├── high/                           High HER2-expression cells (9 videos)
│   ├── low/                            Low HER2-expression cells (5 videos)
│   ├── cell 1 low 50 increments.mp4
│   └── legacy/20210825_cancer/         220 unlabelled JPGs from 2021 microscope
│
├── frames/                             Raw extracted frames (1228×922 greyscale JPEG)
│   ├── 01_annotate_pool/               U-Net training source (445 images: 200 legacy + 245 domain-adapt)
│   ├── 02_unet_holdout/                Development set — high_cell9 (50) + low_cell6 (51)
│   │   ├── legacy_holdout/             20 reserved legacy frames — separate diagnostic DSC
│   │   └── final_test/                 Blind test — high_cell10 (50), reserved, unannotated
│   └── 03_mask_factory/                Inference → classifier training (608 images)
│
├── masks/                              Hand-drawn masks (1228×922 PNG)
│   ├── 01_annotate_pool_multiclass/    U-Net training labels (0=bg, 1=trapped cell, 2=decoy)
│   └── 02_unet_holdout_multiclass/     Development-set ground truth (cell9 + cell6 done; cell10 pending, deliberately)
│
└── preprocessed/                       256×256 pipeline-ready data
    ├── train/01_annotate_pool_multiclass_256/
    │   ├── inputs/                     Preprocessed frames → U-Net training inputs
    │   └── labels/                     Resized masks       → U-Net training labels
    ├── evaluate/02_unet_holdout/
    │   ├── inputs/ · labels/           Development-set inputs/labels (cell9 + cell6 only)
    │   └── legacy_holdout/             Legacy-domain diagnostic inputs/labels
    ├── inference/03_mask_factory/inputs/  Preprocessed frames → classifier training
    └── comparisons/                    Pipeline stage figures (reporting)
```

There is **no `04_clf_arena` anymore** — it was collapsed into `03_mask_factory` on 2026-09-03 (its two cells, `high_cell10` and `low_cell5`, became new U-Net training identities instead of a separate classifier-only firewall) and `high_cell10` was then reassigned again on 2026-09-04 (see [Cell roles](#cell-roles) below). If you're looking at an older doc, script comment, or memory that still mentions it, treat it as historical.

---

## Pool summary

| Pool | Images | Role | Masks |
|---|---|---|---|
| `01_annotate_pool` | 445 | U-Net **training** source | Hand-drawn, multiclass — **445/445 complete as of 2026-09-10** (`low_cell5` was the last cell) |
| `02_unet_holdout` | 101 | **Development set** — U-Net + classifier, evaluated repeatedly to guide tuning | Hand-drawn, multiclass (done) |
| `02_unet_holdout/legacy_holdout/` | 20 | Legacy-domain diagnostic — isolates domain gap from segmentation quality | Hand-drawn, multiclass |
| `02_unet_holdout/final_test/` | 50 | **Blind test** — `high_cell10`, reserved, one-shot | None yet — deliberately deferred |
| `03_mask_factory` | 608 | U-Net inference → classifier **training** features (12 cells) | U-Net-predicted (not hand-drawn) |

---

## Three-tier evaluation terminology

Three genuinely different things get called "validation" or "test" informally — worth being precise, since an evaluated score from the wrong tier is easy to over-trust:

| Tier | What it is | Where | Used for |
|---|---|---|---|
| **Internal validation split** | Frames of cell identities the model *has* trained on (just held-out frames of them) | `data.val_split` inside `01_annotate_pool`, drawn fresh each training run | Checkpoint selection and early stopping *within one run*. Not evaluated against real generalization — see [the finding below](#why-internal-validation-was-misleading). |
| **Development set** | Genuinely novel cells, never trained on, evaluated repeatedly across many recipes | `02_unet_holdout` (`high_cell9` + `low_cell6`) | Guides which hyperparameters/augmentations to keep. Not a blind test — every score here has shaped a decision. |
| **Blind test** | A reserved cell, evaluated exactly once, uninvolved in any decision | `02_unet_holdout/final_test/` (`high_cell10`) | The one honest, thesis-reportable generalization number — see [Cell roles](#cell-roles) for the protocol that keeps it clean. |

A **smoke test** (does the code run at all — shapes, dtypes, no crash) is a different concept entirely from any of the three above; it says nothing about model quality, only about basic correctness, and is cheap enough to run constantly.

---

## Cell roles

| Role | High cells | Low cells |
|---|---|---|
| U-Net training (`01_annotate_pool`, domain-adapt) | 1, 2, 3, 4, 6, 7, 8 | 1, 2, 3, 4, 5 |
| Development set (`02_unet_holdout`, top level) | 9 | 6 |
| Blind test (`02_unet_holdout/final_test/`) | 10 | — |

**Changed 2026-09-04.** `high_cell10` and `low_cell5` were both folded out of the retired `04_clf_arena` into training on 2026-09-03 — but by 2026-09-04, annotated training cells were imbalanced (7 high vs. 4 low), and per-cell evaluation had revealed `high_cell9` and `low_cell6` needed genuinely different treatment (see below). So `low_cell5` stayed a training identity (grows the scarcer low-HER2 category), while `high_cell10` moved to the holdout instead — as the reserved **blind test**, not a third development cell, since `high_cell9`/`low_cell6` were already "spent" as a development signal.

`high_cell10`'s frames are extracted the same way as `high_cell9`/`low_cell6` (`_extract_group()`, `every_n=200` across the full video — 50 frames, matching `cell9`'s count exactly), but physically nested under `02_unet_holdout/final_test/` rather than sitting at the top level. This isn't just organizational: `annotate.py`, `validate_masks.py`, `preprocess_frames.py`, and `evaluate_unet.py` are all non-recursive (`iterdir()`/glob-based), so a routine `--pool 02_unet_holdout` command **physically cannot** see into `final_test/` — mirroring the existing `legacy_holdout/` pattern. Combined with `high_cell10` having no masks yet at all (so it can't be scored even if pointed at directly), this is a hard, structural guarantee against the blind test leaking into development, not just a convention to remember.

**Deliberately deferred — do not annotate `high_cell10` yet.** Full protocol:

1. Keep tuning using the development set (`high_cell9` + `low_cell6`), as normal.
2. Once the recipe is genuinely settled, fold `high_cell9` + `low_cell6` into the training pool.
3. Train the final/production checkpoint once, same recipe, on the enlarged pool.
4. *Only then* annotate `high_cell10` (`python scripts/annotation/annotate.py --image-dir data/frames/02_unet_holdout/final_test --mask-dir data/masks/02_unet_holdout_multiclass --multiclass` — note the explicit `--image-dir`/`--mask-dir` pair; a bare `--pool 02_unet_holdout/final_test` does not resolve the mask directory correctly).
5. Evaluate the final checkpoint on `high_cell10` exactly once. That's the reported blind-test score.

If asked to annotate or evaluate `high_cell10` before step 2 has actually happened, that's very likely premature — confirm first.

**Resolved 2026-09-10.** `high_cell10` was removed from `03_mask_factory` on **2026-09-10** — it is the U-Net's blind test and must not also feed classifier training. The pool is now 608 frames across 12 cells, and those 12 are exactly the cells that feed `01_annotate_pool`'s domain-adapt frames. Its frames had been sitting there since the 2026-09-03 merge, from before its role changed.

---

## Train/val split (`01_annotate_pool`)

| Split | Frames | % | Role |
|---|---|---|---|
| Train | 371 | ~90% | U-Net training |
| Val | 54 | ~10% | Checkpoint selection / early stopping only — **not** the reportable score |

**There is no third "test" split inside `01_annotate_pool` anymore** (removed 2026-09-04, `celldform.config.DataConfig`/`scripts/training/train_unet.py`) — it was computed every run and printed in the split summary, but never actually built into a loader or evaluated by anything; the external `02_unet_holdout` development/blind-test evaluation already fulfills the role a third split would have, so keeping one was pure redundancy that shrank the training pool for nothing. The current 90/10 split reuses what used to be a separate, unused 10% "test" allocation as `val_split` instead of the old dedicated 15% — net effect: the training pool grew from ~69% to ~90% of the annotated data with the same monitoring mechanics, just on a smaller slice.

### How the grouping works

Splitting is **group-aware** (`scripts/training/train_unet.py::_build_groups`/`_group_split`) — frames from the same cell or the same legacy acquisition burst never split across train/val, since they're correlated (same biology, same laser session). Two group types:

- **Legacy frames** (`YYYYMMDDHHMMSS.jpg`) are clustered into acquisition **bursts** — a gap of more than 5 seconds between sorted timestamps starts a new burst.
- **Domain-adapt frames** (`domain_<her2>_<cell_id>_<frame>.jpg`) are grouped by cell identity.

At the current pool size (445 frames), this produces **22 groups**: 11 legacy bursts (~17-20 frames each) and 11 domain cells (~19-24 frames each). The split algorithm shuffles group order (seeded, reproducible), then fills the val bucket with whole groups until it reaches the target frame count, with the remainder going to train — so split sizes *approximate* the configured ratio rather than hitting it exactly, since group sizes vary and a bucket only checks "have I reached target" before adding the next whole group (it can overshoot, never undershoot).

---

## Why internal validation was misleading

The single most consequential finding from the 2026-09 evaluation sweep, worth keeping visible here since it shapes why the three-tier terminology above exists at all:

The internal validation split tests generalization to **unseen frames of already-trained-on cell identities** — not to a genuinely novel cell. A checkpoint tuned to look excellent on it (`bf32_reg2_cos`, internal val DSC 0.862) scored only **DSC 0.163** on the first genuinely held-out cell ever evaluated (`high_cell9`), while a "worse-looking" checkpoint (higher internal train/val gap, informally called "overfitting") scored **0.428** — roughly 2.6× better on real generalization despite scoring worse on the metric everything had been tuned against.

**Root cause, once the full development set was evaluated:** both `high_cell9` and `low_cell6`'s true trapped-cell size (704 / 886 px²) falls at or below every one of the 11 training cells' (825-1549 px²) — a real train/development-set size mismatch that the internal split, drawn from the same size distribution as training, could never surface. See [Training Results](training.md#the-2026-09-full-evaluation-sweep) for the full sweep, the per-cell breakdown, and what fixed it (partially).

!!! warning "Superseded figure (23 September 2026)"
    The `1549 px²` upper bound above is `high_cell1` **as annotated at the time**.
    That cell was re-annotated on 23 September and now measures **678 px²** — the
    smallest cell in the study, not the largest. The old masks no longer exist.
    The size mismatch described here was real but **smaller than stated**, and part
    of the gap was an annotation artifact. Current ranges: training 688–1449 px²,
    development set 678–1033 px², interleaved. See `docs/status.md`.


---

## Contamination firewall

- **Legacy images** carry no HER2 label — safe for U-Net training, cannot contaminate the classifier.
- **`02_unet_holdout`'s top level** (`high_cell9`, `low_cell6`) was never used during U-Net training. It *is* used repeatedly during development, though — see the terminology table above; this is not the same guarantee as a blind test.
- **`02_unet_holdout/final_test/`** (`high_cell10`) has never been evaluated against anything, and physically cannot be reached by any routine pool command until it's pointed at explicitly (see [Cell roles](#cell-roles)). This is the one genuinely untouched pool.
- **`03_mask_factory`** cells were never part of `02_unet_holdout` at any point and never will be. `high_cell10` briefly sat in both after the 2026-09-03→09-04 role change; that was resolved on 2026-09-10 (see [Cell roles](#cell-roles)) and the two pools now share no cell.

---

## Regenerating frames

```bash
celldform-organize-dataset                                    # defaults
celldform-organize-dataset --every-n 100 --domain-adapt-n 10  # denser sampling
celldform-organize-dataset --no-clean                          # skip wiping data/frames/
```

!!! warning "The full `celldform-organize-dataset` pipeline does not currently reproduce the 2026-09-04 cell reallocation"
    `celldform/acquisition/organiser.py`'s `_video_lists()` has been updated so a **fresh** run would put `high_cell10` in `unet_test` and `low_cell5` in `domain_adapt`/`inference` correctly — but the *current* data on disk was fixed by hand (moving files, not rerunning the organiser) specifically to avoid re-touching `02_unet_holdout` mid-annotation and re-decoding videos unnecessarily. Don't run `celldform-organize-dataset --no-clean` casually right now without checking it won't fight the manual `final_test/` nesting — a full `--clean` rerun would rebuild everything correctly from scratch, but would also wipe existing annotations, which almost certainly isn't what you want mid-project.
