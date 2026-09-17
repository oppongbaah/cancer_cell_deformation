# `scripts/report/` — progress-report asset generation

Everything that writes into the report repo (`../masters_thesis/`) lives here.
Nothing in this folder is part of the pipeline; removing it would not affect
training, evaluation or inference.

## The entry point

```bash
python scripts/report/build_report_assets.py            # rebuild every figure and table
python scripts/report/build_report_assets.py --list     # job names
python scripts/report/build_report_assets.py --dry-run  # print commands, run nothing
python scripts/report/build_report_assets.py --compile  # assets, then pdflatex, then clean
```

It pins the exact arguments each figure was made with, so no report asset depends on
anyone remembering a command line. Run it after any change to the pools, the masks, or
an eval log — then recompile.

**Always use `--compile` rather than a single `pdflatex`.** The report resolves its
cross-references through the `.aux`, which is deleted after every build, so one pass
renders every reference as `??`. `--compile` runs the passes and fails loudly if
anything is left unresolved.

## What it drives

| script | produces |
|---|---|
| `plot_pool_flow.py` | pool provenance diagram |
| `plot_split_change.py` | train/val split, before and after |
| `plot_sweep_dsc.py` | every checkpoint ranked by pooled DSC |
| `plot_percell_metrics.py` | per-cell DSC/precision bars |
| `render_holdout_comparison.py` | raw / labelled-mask / masks-only / overlay strips |
| `plot_cell_sizes.py` | per-cell trapped-cell size |
| `diagnose_false_positives.py` | predicted false-positive region on the **raw** frame |
| `make_report_tables.py` | the LaTeX tables the report `\input{}`s |

Every one of them reads from disk or from an eval log. **None takes hand-typed
numbers**, so a figure cannot drift from the run that produced it.

## Two things to know before editing

**Path depth.** These scripts sit two levels below the repo root, so the hop to it is
`Path(__file__).parent.parent.parent`, and `render_holdout_comparison.py` reaches
`scripts/` (for `preview_masks`) via `Path(__file__).parent.parent`. Moving a file in
or out of this folder means fixing those.

**Colours are pinned to the report's legend.** `render_holdout_comparison.py`'s
defaults are *not* what the report uses — the `strip-fp` job passes
`--gt-color 0,200,0 --pred-color 170,95,40` to match the `\masklegend` swatches in the
`.tex` preamble. Rendering with the defaults produces a figure whose caption names the
wrong colours, and nothing errors.
