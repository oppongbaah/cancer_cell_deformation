# Scripts Reference

!!! info "Layout — scripts are grouped by pipeline stage (reorganised 2026-09-17)"

    ```
    scripts/
      data/        organize_dataset · extract_frames · preprocess_frames
      annotation/  annotate · annotate_frame · preview_masks · validate_masks
      training/    train_unet · evaluate_unet · compare_postswap · run_postswap.sh
      inference/   infer · predict_single_frame
      report/      build_report_assets and the figure/table generators it drives
    ```

    Scripts import `celldform` via `Path(__file__).parent.parent.parent`, so moving one
    between folders means fixing that hop. `report/render_holdout_comparison.py`
    additionally reaches into `annotation/` for `preview_masks`.


All scripts live under `scripts/<stage>/` and are thin wrappers or standalone tools. Run them directly with `python scripts/<stage>/<name>.py`. The installed CLI (`celldform-*`, see [CLI Reference](cli.md)) targets `celldform.cli` and is unaffected by this layout.

---

## `annotation/annotate.py` — napari annotation tool

Opens every image in an annotation pool, in order, in napari — including already-annotated ones, so `N`/`B` can freely browse and correct earlier work rather than only stepping through what's left. The viewer opens on the first not-yet-annotated image (not the start of the pool); revisiting an annotated frame loads its saved mask instead of a blank one, and the window title shows `[saved]` for frames that already have a mask on disk. Paint the cell mask on the Labels layer, then press `S` to save and advance.

```bash
python scripts/annotation/annotate.py                             # 01_annotate_pool (default)
python scripts/annotation/annotate.py --pool 02_unet_holdout      # holdout set
python scripts/annotation/annotate.py --multiclass                # experimental 3-class labels (same 40 images)
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pool` | `01_annotate_pool` | Pool folder name under `data/frames/` |
| `--image-dir` | `data/frames/<pool>` | Override source image directory |
| `--mask-dir` | `data/masks/<pool>` | Override mask output directory |
| `--multiclass` | off | Annotate 3 classes (0=background, 1=trapped cell, 2=other/decoy object) instead of binary. Masks default to `data/masks/<pool>_multiclass/` and are seeded from the existing binary mask (as label 1) when one exists, so you only need to paint decoy objects |

**Keybindings:**

| Key | Action |
|-----|--------|
| `2` | Paint brush |
| `3` | Eraser |
| `[` / `]` | Smaller / larger brush |
| `Ctrl+Z` | Undo last stroke |
| `S` | Save mask → next image |
| `N` | Skip (no save) → next image |
| `B` | Previous image |
| `P` | Toggle paint mode (napari default — left alone) |
| `-` / `=` | Step the active paint label down/up (multiclass mode) |

!!! note "P vs. B"
    `P` used to mean "previous image" but napari's Labels layer binds it by default to "toggle preserve labels," which shadows any viewer-level binding while the mask layer is active. `B` is reclaimed for "previous" instead; `P` is left as napari's default paint-mode toggle.

Binary mode: masks are saved as uint8 PNGs with values 0 (background) / 255 (cell). Multiclass mode: masks are saved with raw label values 0/1/2 (not binarized) — use `scripts/annotation/preview_masks.py` to view them, since label values 1–2 are visually indistinguishable from background in a plain image viewer.

---

## `annotation/preview_masks.py` — colorized mask preview

Renders saved masks as real color PNGs, mask alone against a black background — no napari needed, and no raw frame loaded or blended in. Multiclass masks (label ids 0/1/2) are nearly invisible in a plain image viewer; this renders background as black, label 1 (trapped cell) as green, and label 2 (other/decoy) as red. Legacy binary masks (0/255) also work — foreground renders green.

```bash
python scripts/annotation/preview_masks.py                                    # 01_annotate_pool_multiclass (default)
python scripts/annotation/preview_masks.py --pool 01_annotate_pool            # binary masks
python scripts/annotation/preview_masks.py --limit 10
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pool` | `01_annotate_pool_multiclass` | Pool folder name under `data/masks/` |
| `--mask-dir` | `data/masks/<pool>` | Override mask directory |
| `--output-dir` | `data/masks/<pool>_preview` | Override output directory |
| `--limit` | none | Only preview the first N masks |

---

## `report/build_report_assets.py` — rebuild every progress-report figure and table

Single entry point for the progress report's assets. It pins the arguments each figure was generated with and drives the individual plot scripts, so nothing in the report depends on remembering a command line.

```bash
python scripts/report/build_report_assets.py                     # build everything
python scripts/report/build_report_assets.py --list              # show job names, run nothing
python scripts/report/build_report_assets.py --dry-run           # print the commands, run nothing
python scripts/report/build_report_assets.py --only sweep pools  # rebuild a subset
python scripts/report/build_report_assets.py --report-root ../masters_thesis
```

Figures are written to `<report-root>/images/`, LaTeX tables to `<report-root>/main_work/tables/` (the report `\input{}`s those). Intermediates from the comparison-strip renderer stay in `results/_report_strips/` rather than cluttering the report repo.

| Option | Default | Description |
|--------|---------|-------------|
| `--report-root` | `../masters_thesis` | Where to write figures and tables |
| `--only` | all | Space-separated job names (see `--list`) |
| `--list` | off | Print job names and exit |
| `--dry-run` | off | Print each command without running it |
| `--compile` | off | Also run `pdflatex` the required number of times and delete the temp files |
| `--tex` | `progress_report_9_10_26.tex` | Report to compile with `--compile` |

```bash
python scripts/report/build_report_assets.py --compile   # assets, then compile, then clean up
```

!!! warning "The report must be compiled at least twice"
    Cross-references and `\captionof` numbers resolve through the `.aux` file, and the temp files are deleted after every build — so a **single** `pdflatex` pass always starts cold and renders every reference as `??` (e.g. `§??`). `--compile` runs the passes needed and then fails loudly if anything is still unresolved, rather than shipping a PDF with `??` in it.

**Run it after any change to the frame pools, the masks, or `results/preswap/preswap_eval_log.jsonl`**, then recompile the report. Every number in the document comes from one of the scripts it drives, so a stale figure means this wasn't run — not that a number needs editing by hand.

---

## `report/make_report_tables.py` — emit the report's LaTeX tables from the data

Generates `.tex` fragments the report `\input{}`s, so the tables can't drift from the pools the way hand-typed ones can.

```bash
python scripts/report/make_report_tables.py percell --output ../masters_thesis/main_work/tables/percell.tex
python scripts/report/make_report_tables.py headline --output ../masters_thesis/main_work/tables/headline.tex \
    --run multiclass_experiment_256_bf64_base_cos="No augmentation (noise only)" \
    --run multiclass_experiment_256_bf64_scale_cos="Scale augmentation" \
    --baseline multiclass_experiment_256_bf64_base_cos \
    --emph multiclass_experiment_256_bf64_scale_cos
```

`percell` counts frames per cell across `01_annotate_pool`, `03_mask_factory` and `02_unet_holdout` and checks each frame for a mask. `headline` looks each run up in the eval log by (checkpoint, filter) and computes the delta column against `--baseline`; `--extra-run` adds an italic row below a rule for runs that are *not* comparable. Both write a `% GENERATED` header — edit the script, not the output.

---

## `report/plot_pool_flow.py` — frame-pool provenance diagram

Replaces a pool-summary table: shows which cells go where, that `01_annotate_pool` is *drawn from* `03_mask_factory` at different frame indices, and where the evaluation firewall sits. All counts are read off disk.

```bash
python scripts/report/plot_pool_flow.py --output results/pool_flow.png
```

---

## `report/plot_cell_sizes.py` — per-cell trapped-cell size

Measures mean label-1 area per cell from the annotation masks and plots every cell on one axis, with the training range as a band. Makes "the development cells are smaller than the training cells" checkable rather than asserted.

```bash
python scripts/report/plot_cell_sizes.py --output results/cell_sizes.png
```

---

## `report/plot_split_change.py` — train/val split before and after

Two stacked bars showing the training pool growing by reclaiming an unused test split. **Counts are arguments, not derived** — the old split is historical and not reproducible from disk — and are printed on every run.

```bash
python scripts/report/plot_split_change.py --before 293 78 54 --after 371 54 0
```

---

## `report/render_holdout_comparison.py` — raw/labelled/masks-only/overlay comparison strip

Renders the four-figure holdout comparison used in the thesis progress reports: N evenly-spaced frames from one holdout cell, shown as (1) raw unannotated frames, (2) the multiclass ground-truth annotation (via `preview_masks.py`'s own `_colorize()`, not a re-drawn approximation), (3) ground truth vs. U-Net prediction on a black background (mask-only), and (4) the same comparison overlaid on the original frame. All four share one auto-computed crop window and panel size, so the cell renders at the same scale in every figure. Every panel carries its own frame index in a label band above the image, so a figure printed in a report can be traced back to the exact files on disk — pass `--no-frame-labels` for bare panels.

```bash
python scripts/report/render_holdout_comparison.py \
    --pool 02_unet_holdout --cell high_cell9 \
    --checkpoint checkpoints/multiclass_experiment_256_bf32_reg2_cos/unet_best.pt

python scripts/report/render_holdout_comparison.py \
    --pool 02_unet_holdout --cell low_cell6 \
    --checkpoint checkpoints/multiclass_experiment_256/unet_best.pt --base-features 64
```

`unet.base_features`/`out_channels`/`in_channels` are read from the checkpoint directory's own saved `config.yaml` when present; pass `--base-features` etc. to override or when no sibling config exists. The crop auto-sizes to contain the trapped cell, nearby decoy objects (within `--decoy-radius`, default 150px — excludes far-flung debris elsewhere in the frame), and the largest predicted region across the selected frames, padded by `--pad` (default 45px); pass `--half-width`/`--half-height` to pin it explicitly instead. `--gt-color`/`--pred-color`/`--overlap-color` take `R,G,B` and only affect figures (3)/(4) — the ground-truth-annotation figure (2) always uses `preview_masks.py`'s own green/red convention.

| Option | Default | Description |
|--------|---------|-------------|
| `--pool` | `02_unet_holdout` | Pool folder under `data/frames/` and `data/preprocessed/evaluate/` |
| `--cell` | *(required)* | Cell filename prefix, e.g. `high_cell9` |
| `--checkpoint` | *(required)* | Path to a trained `.pt` checkpoint |
| `--n-frames` | `5` | Evenly-spaced frames to render |
| `--frame-indices` | none | Comma-separated 6-digit indices instead of auto-selecting |
| `--half-width` / `--half-height` | auto | Pin the crop size instead of computing it |
| `--pad` | `45` | Padding (px) around the auto-computed crop |
| `--decoy-radius` | `150` | Max distance (px) from the trapped cell for a decoy to count toward the auto-crop |
| `--alpha` | `0.55` | Overlay blend strength for figure (4) |
| `--no-frame-labels` | off | Render bare panels, with no frame-index band above each one |
| `--label-color` / `--label-bg` | `255,255,255` / `20,20,20` | `R,G,B` for the frame-index text and its band |
| `--output-dir` | `results/holdout_comparisons/<pool>_<cell>` | Output directory |

---

## `annotation/validate_masks.py` — mask validation gate

Validates the masks that are present in a pool, and (by default) that every frame in the pool has one. Use as a pre-training gate.

```bash
python scripts/annotation/validate_masks.py                          # 01_annotate_pool
python scripts/annotation/validate_masks.py --pool 02_unet_holdout
python scripts/annotation/validate_masks.py --multiclass              # validates 01_annotate_pool_multiclass
python scripts/annotation/validate_masks.py --no-completeness         # skip the "every frame has a mask" check
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pool` | `01_annotate_pool` | Pool folder name (used for both `data/frames/<pool>` and `data/masks/<pool>`) |
| `--mask-dir` | `data/masks/<pool>` (or `<pool>_multiclass` with `--multiclass`) | Override mask directory |
| `--image-dir` | `data/frames/<pool>` | Override frame directory (completeness check) |
| `--no-completeness` | off | Skip the "every pool image has a mask" check entirely |
| `--multiclass` | off | Validate 3-class masks (0/1/2) instead of binary (0/255) |

**Checks performed:**

| Check | Fail condition (binary) | Fail condition (`--multiclass`) |
|-------|---------------|---------------|
| Present | Frame in the pool has no corresponding mask file (skippable with `--no-completeness`) | same |
| Readable | File cannot be decoded by OpenCV | same |
| Binary/label-valid | Pixel values other than 0 or 255 | Pixel values other than 0, 1, or 2 |
| Non-empty | No foreground pixels (255) at all | No foreground pixels of **any** class (neither label 1 nor label 2) — a mask with only a decoy object (label 2) and no trapped cell is a valid deliberate negative and passes |

Exits with code `0` if all checks pass, `1` if any issues are found. Reports `Valid: N/N (100%)` against the full pool count (or the mask count with `--no-completeness`).

---

## `data/preprocess_frames.py` — manual preprocessing

Applies `PreprocessingPipeline` to frames and saves PNGs (256×256 by default). **Required before training.** Optionally resizes masks to match.

!!! note "Mask-gated processing"
    When `--masks` is passed, only frames that have a corresponding mask in `data/masks/<pool>/` are processed. Frames without a mask are skipped. The summary line reports how many were skipped.

```bash
python scripts/data/preprocess_frames.py --masks                        # train pool
python scripts/data/preprocess_frames.py --masks --pool 02_unet_holdout # evaluate pool
python scripts/data/preprocess_frames.py --pool 03_mask_factory          # inference pool
python scripts/data/preprocess_frames.py --masks --compare --n-compare 10
python scripts/data/preprocess_frames.py --masks --target-size 128 128   # override output resolution
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pool` | `01_annotate_pool` | Pool folder name |
| `--stage` | auto-detected | `train`, `evaluate`, or `inference` |
| `--masks` | off | Also resize and save masks (binary or label, nearest-neighbour) |
| `--compare` | off | Save pipeline stage comparison figures |
| `--n-compare` | `5` | Number of images to generate figures for |
| `--image-dir` | `data/frames/<pool>` | Override source directory |
| `--output-dir` | `data/preprocessed/<stage>/<pool>/inputs` | Override frame output |
| `--mask-dir` | `data/masks/<pool>` | Override source mask directory |
| `--mask-output-dir` | `data/preprocessed/<stage>/<pool>/labels` | Override mask output |
| `--target-size` | `256 256` | Output spatial size `H W`. Both dims should stay divisible by 16 (U-Net's 4 downsampling steps) — the U-Net is fully convolutional, so no model code changes are needed, but whatever size is chosen here is what `CellDataset` will load during training |

**Stage auto-detection:**

| Pool | Stage |
|------|-------|
| `01_annotate_pool` | `train` |
| `02_unet_holdout` | `evaluate` |
| `03_mask_factory` | `inference` |

---

## `training/train_unet.py` — U-Net training

Trains the U-Net on preprocessed data. Expects already-preprocessed PNGs at whatever size was used for preprocessing (256×256 by default) — no preprocessing is applied internally. Reads `unet.out_channels` from the config: `1` (default) trains binary segmentation; `>1` trains multiclass (`CellDataset` loads raw integer label masks instead of binarising, and `_DiceCELoss` replaces `_DiceBCELoss`).

```bash
python scripts/training/train_unet.py --config configs/multiclass_experiment_256.yaml   # frozen scheme
python scripts/training/train_unet.py --config configs/default.yaml --seed 123
python scripts/training/train_unet.py --config configs/multiclass_experiment_256.yaml --tag auto
python scripts/training/train_unet.py --config configs/multiclass_experiment_256.yaml --domain-oversample-weight 8.0
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `configs/default.yaml` | YAML config file |
| `--seed` | config value | Random seed override |
| `--domain-oversample-weight` | config value (`1.0` if unset) | Sampling weight for `domain_`-prefixed (real-cell) train frames vs. legacy frames — builds a `WeightedRandomSampler`. `1.0` = off/uniform; pass `1.0` explicitly to force it off even if the config enables it. See [Training Results — Domain Gap](training.md#domain-gap) (tested at weight=8.0: no improvement on the small real-cell test sample available so far). |
| `--tag` | none | Suffix appended to the saved training-curve filename (e.g. `--tag auto` → `unet_training_curves_v6_auto.png`), for distinguishing runs at a glance |

The script reads `data.frames_dir` and `data.masks_dir` from the config. Pairs frames and masks by filename stem. Saves best and last checkpoints to `checkpoint_dir`. After training, it reloads `unet_best.pt` (not necessarily the last epoch's weights, which can be past the peak validation DSC on small datasets) before running final evaluation and plotting the training curves.

---

## `training/evaluate_unet.py` — holdout evaluation

Evaluates a trained U-Net on `02_unet_holdout` and reports the thesis-reportable segmentation metrics.

```bash
python scripts/training/evaluate_unet.py --checkpoint checkpoints/multiclass_experiment_256/unet_best.pt \
    --config configs/multiclass_experiment_256.yaml   # frozen scheme — --config must match out_channels
python scripts/training/evaluate_unet.py --checkpoint checkpoints/unet_best.pt --pool 02_unet_holdout  # binary baseline
```

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | required | Path to trained `.pt` checkpoint |
| `--pool` | `02_unet_holdout` | Pool to evaluate |
| `--config` | `configs/default.yaml` | YAML config |
| `--batch-size` | `8` | Inference batch size |
| `--results-log` | `results/preswap/preswap_eval_log.jsonl` | JSONL file each run's metrics get appended to |
| `--no-save-results` | off | Skip the append (stdout only) |

**Metrics reported:** DSC, IoU, precision, recall, accuracy. When the checkpoint's config has `unet.out_channels > 1`, results are reported per class (`background`, `trapped_cell`, ...) instead of as a single flat table.

**Every run is also logged to disk** (unless `--no-save-results`) as one JSON line appended to `results/preswap/preswap_eval_log.jsonl` — full-precision metrics per class, plus `timestamp`/`checkpoint`/`config`/`pool`/`n_images`/`multiclass`. Printing to stdout alone doesn't survive the terminal scrolling past it; this is the durable record to check a reported number against, or to re-derive a summary table from later (`json.loads` each line).

---

## `data/organize_dataset.py` — dataset organiser

Thin wrapper around `celldform.acquisition.organiser.organise()`.

```bash
python scripts/data/organize_dataset.py
python scripts/data/organize_dataset.py --every-n 100 --domain-adapt-n 10
python scripts/data/organize_dataset.py --no-clean --domain-adapt-n 20   # expand domain-adapt only, don't wipe data/frames/ — run 2026-08-19
```

Equivalent to `celldform-organize-dataset`. See [CLI Reference](cli.md). The `domain_adapt` cell list in `celldform/acquisition/organiser.py` covers all 11 `03_mask_factory` cells (as of 2026-08-10, extended from 4); `--domain-adapt-n` was raised 5→20/cell on 2026-08-19 — see [Training Results — Domain Gap](training.md#domain-gap) for why. Re-running with a larger N doesn't lose frames/masks from an earlier N — every `domain_*` file already on disk is kept and re-enumerated into the manifest, and legacy frames reserved in `02_unet_holdout/legacy_holdout/` are always excluded from the copy regardless of N (a contamination bug where this wasn't true was found and fixed the same day).

---

## `data/extract_frames.py` — single video extraction

Thin wrapper around `FrameExtractor` for a single video.

```bash
python scripts/data/extract_frames.py --video data/raw/high/"cell 1 high 50 increments.mp4"
```

Equivalent to `celldform-extract-frames`. See [CLI Reference](cli.md).

---

## `inference/infer.py` — full inference pipeline

Runs stages 1–6 on a video or directory of preprocessed images.

```bash
python scripts/inference/infer.py \
    --video data/raw/high/"cell 1 high 50 increments.mp4" \
    --unet-ckpt checkpoints/unet_best.pt \
    --classifier-ckpt checkpoints/svm.pkl
```

Equivalent to `celldform-infer`. See [CLI Reference](cli.md).

---

## `training/compare_postswap.py` — enforce the noise floor on comparisons

Reads `results/postswap/postswap_eval_log.jsonl` and compares runs against the series
baseline, per development cell and pooled.

```bash
python scripts/training/compare_postswap.py                      # every run vs. baseline
python scripts/training/compare_postswap.py --runs A B           # just these
python scripts/training/compare_postswap.py --seeds A B C        # aggregate seeds: mean ± sd
```

It exists to stop a number that looks exciting being recorded as a finding. Measured
seed-only spread on this pipeline is **sd 0.047 pooled**, so a difference between two
single runs has SE ≈ 0.066 and needs roughly twice that to be believable. Anything
below **0.133** prints as `INCONCLUSIVE`:

```
  => INCONCLUSIVE. Pooled delta +0.048 is inside the +/-0.133 noise floor.
     Log as screened, not as a result. Three seeds on BOTH arms to decide it.
```

!!! warning "This is not a formality"
    Two results in this project were recorded as findings and later retracted for
    exactly this reason — the scale-augmentation recipe (+0.066) and Focal–Tversky
    (+0.023, whose seed repeat came back −0.049). Both are smaller than the noise.

---

## `training/run_postswap.sh` — train + evaluate one run, filed correctly

```bash
scripts/training/run_postswap.sh postswap_ftl_cw142_res384_s42
```

Expects `configs/postswap/<run-name>.yaml`. Trains, evaluates on all four development
cells plus pooled, writes the log to `logs/postswap/`, moves the training curve into
`results/postswap/`, echoes the split assignment, and finishes by running
`compare_postswap.py` against the baseline.

Use it rather than calling `train_unet.py` directly, so artefacts land in the right
folders and the comparison happens automatically.


---

## `notebooks/pipeline_walkthrough.ipynb` — the pipeline end to end

A runnable tour of all seven stages against the data actually on disk: pool inspection
and the firewall check, frame extraction, mask validation, preprocessing, the train/val
split, evaluation with the noise floor applied, features, classification, biomechanics
and real-time inference.

```bash
jupyter lab notebooks/pipeline_walkthrough.ipynb
```

Most cells only read and display. The four that write are each gated:

```python
RUN_ORGANISE = False      # rebuilds data/frames/ from raw  ← destructive
RUN_PREPROCESS = False    # rewrites data/preprocessed/**
RUN_TRAINING = False      # ~15 min on an 8 GB GPU
RUN_EVAL = False          # runs the model over the development set
```

They print what they *would* do while the flag is `False`.

### Notebook outputs are stripped automatically

`nbstripout` is installed as a git **clean filter**, so outputs and `execution_count`
never reach a commit — the working file keeps them, git stores only the source. Without
it a single re-run produces a multi-megabyte diff of base64 PNGs in which nothing
meaningful changed.

`.gitattributes` is committed, but the filter command lives in `.git/config`, which is
not. **After cloning, run it once:**

```bash
pip install -e ".[dev]"
nbstripout --install --attributes .gitattributes
git config --local --get filter.nbstripout.clean    # should print a command
```

If you skip this, git silently does nothing — an undefined filter is not an error — and
outputs go into history as normal.

### The notebook sandbox

Runs launched from the notebook write to `configs/notebook/`, `checkpoints/notebook/`,
`logs/notebook/` and `results/notebook/`, never into `preswap/` or `postswap/`. That
keeps exploratory output out of the experimental record that `compare_postswap.py` and
the report tables are generated from. All four are gitignored.

`promote()` moves a sandbox run up into `configs/postswap/` when it turns out to matter.
It verifies the run against the config snapshot saved inside its checkpoint directory —
so a run whose config was edited afterwards is refused rather than silently promoted.

!!! tip "Appendix"
    The last section lists the traps this pipeline has actually fallen into — stale
    preprocessing silently breaking the firewall, single-run deltas below the noise
    floor, an oversized validation group voiding a run. Worth reading before a first
    experiment.
