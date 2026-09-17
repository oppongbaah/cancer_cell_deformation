#!/usr/bin/env python3
"""
Regenerate every figure and table in the current progress report.

One command rebuilds the whole report's assets from the data currently on
disk. Each job below records the exact arguments a figure was made with, so
nothing about the report depends on remembering a command line — which is what
this script exists to fix.

    python scripts/report/build_report_assets.py                    # build everything
    python scripts/report/build_report_assets.py --list             # show jobs, run nothing
    python scripts/report/build_report_assets.py --only sweep pools # build a subset
    python scripts/report/build_report_assets.py --dry-run          # print commands only

Outputs land in the report repo, not here:
    <report-root>/images/*.png              figures
    <report-root>/main_work/tables/*.tex    \\input{}-able LaTeX tables

After running, recompile the report:
    cd <report-root>/main_work && pdflatex -interaction=nonstopmode progress_report_9_17_26.tex

Run it after ANY change to the pools, the masks, or results/holdout_eval_log.jsonl.
The comparison-strip jobs additionally need the two checkpoints they name to be
present; if a checkpoint has been retired, drop or repoint that job rather than
letting the report keep a stale image.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_REPORT_ROOT = REPO.parent / "masters_thesis"

# Checkpoints the report's figures compare. Change these in one place.
BASE = "multiclass_experiment_256_bf64_base_cos"
SCALE = "multiclass_experiment_256_bf64_scale_cos"
ALLAUG = "multiclass_experiment_256_bf64_allaug"
ORIGINAL = "multiclass_experiment_256"
RETIRED = "multiclass_experiment_256_bf64_scale"      # weights deleted 2026-09-10

# Experiment 1 (2026-09-17): three loss interventions + a seed repeat of the best one.
CW133 = "multiclass_experiment_256_bf64_scale_cos_cw133"   # class_weights [1,4,2] -> [1,3,3]
NOBG  = "multiclass_experiment_256_bf64_scale_cos_nobg"    # Dice term excludes background
FTL   = "multiclass_experiment_256_bf64_scale_cos_ftl"     # Dice -> Focal-Tversky
FTL43 = "multiclass_experiment_256_bf64_scale_cos_ftl_seed43"  # FTL, seed 43 (variance probe)

# render_holdout_comparison.py emits four variants per run; only two reach the
# report, so the rest are intermediates and stay out of the report repo.
STRIP_BUILD = REPO / "results" / "_report_strips"

# Bar labels for the sweep chart: one line per run saying what that run changed.
SWEEP_LABELS = [
    f"{SCALE}=scale augmentation — width 64, cosine",
    f"{BASE}=no augmentation — width 64, cosine",
    f"{ALLAUG}=all four augmentations — width 64, cosine",
    f"{ORIGINAL}=no augmentation — width 64, plateau",
    "multiclass_experiment_256_bf64_bc=brightness/contrast aug — width 64, plateau",
    "multiclass_experiment_256_bf64_reg=dropout 0.4 + cutout — width 64, cosine",
    "multiclass_experiment_256_bf32=no augmentation — width 32, plateau",
    "multiclass_experiment_256_bf32_reg=dropout 0.3 + cutout — width 32, plateau",
    "multiclass_experiment_256_bf32_reg2=dropout 0.4 + cutout — width 32, plateau",
    "multiclass_experiment_256_bf32_reg2_det=dropout 0.4 + cutout, determinism fix — width 32, plateau",
    "multiclass_experiment_256_bf32_reg2_swa=dropout 0.4 + cutout + SWA — width 32, plateau",
    "multiclass_experiment_256_bf32_reg2_wd=dropout 0.4 + cutout + wd 1e-3 — width 32, plateau",
    "multiclass_experiment_256_bf32_reg2_wd2=dropout 0.4 + cutout + wd 1e-3 — width 32, cosine",
    "multiclass_experiment_256_bf32_reg2_cos=dropout 0.4 + cutout — width 32, cosine",
    "multiclass_experiment_256_bf32_reg3=dropout 0.5 + cutout — width 32, plateau",
    "multiclass_experiment_256_bf32_reg2_ablate=dropout 0.4 + cutout, noise off — width 32, plateau",
    "multiclass_experiment_256_bf32_reg2_split=dropout 0.4 + cutout, 65% train — width 32, plateau",
    "multiclass_experiment_256_domain_oversample=domain oversampling x8 — width 64, plateau",
]

# Crop windows are pinned per cell so the four rows of each comparison figure stay
# pixel-aligned; auto-cropping would drift between checkpoints and break that.
STRIP_COLORS = ["--gt-color", "0,200,0", "--pred-color", "170,95,40",
                "--overlap-color", "220,220,0"]


def jobs(img: Path, tab: Path) -> dict[str, list[str]]:
    """Assets for progress_report_9_17_26.tex.

    The 2026-09-10 report is closed; its assets are frozen in place and are not
    rebuilt here. Jobs are added to this dict as each experiment in that report
    produces a result.
    """
    CELLS = ["high_cell9=high-9", "high_cell10=high-10", "low_cell6=low-6", "low_cell5=low-5"]
    return {
        # ── Dataset Organization ────────────────────────────────────────────
        "pools": ["scripts/report/plot_pool_flow.py", "--output", str(img / "pool_flow_9_17.png")],
        "table-percell": ["scripts/report/make_report_tables.py", "percell",
                          "--output", str(tab / "percell.tex")],

        # ── Baseline on the restructured four-cell development set ──────────
        "table-baseline": ["scripts/report/make_report_tables.py", "baseline",
                           "--checkpoint", SCALE,
                           "--cell", "high_cell9=high-9", "--cell", "low_cell6=low-6",
                           "--cell", "low_cell5=low-5", "--cell", "high_cell10=high-10",
                           "--output", str(tab / "baseline.tex")],

        # ── high-9 false-positive diagnosis ─────────────────────────────────
        "fp-diagnosis": ["scripts/report/diagnose_false_positives.py", "--cell", "high_cell9",
                         "--checkpoint", f"checkpoints/{SCALE}/unet_best.pt",
                         "--output", str(img / "fp_diagnosis_9_17.png")],
        # ── Experiment 1: the three loss interventions, + the seed repeat ───
        "table-loss": ["scripts/report/make_report_tables.py", "headline",
                       "--run", f"{SCALE}=Baseline — Dice+CE, class weights [1,4,2]",
                       "--run", f"{CW133}=1a  class weights [1,3,3]",
                       "--run", f"{NOBG}=1b  Dice excludes background",
                       "--run", f"{FTL}=1c  Focal-Tversky (a=0.7, b=0.3, g=0.75)",
                       "--run", f"{FTL43}=1c repeated, seed 43",
                       "--baseline", SCALE, "--emph", FTL,
                       "--label", "loss", "--delta-header", r"vs.\ baseline",
                       "--caption", ("Experiment 1: three interventions on the loss, all measured on the "
                                     "four-cell development set as it stood before the 17 September "
                                     "reshuffle. Every run is identical to the baseline except the field "
                                     "named. The final row repeats run 1c changing nothing but the random "
                                     "seed, and is the error bar for the whole column."),
                       "--output", str(tab / "loss.tex")],

        # ── high-9 FP diagnosis, rendered as masks on a black background ────
        # Same renderer and visual convention as the 2026-09-10 report's comparison
        # strips, so the figure reads consistently across the report series.
        # Colours are NOT the script defaults: ground truth is green and prediction
        # brown, deliberately swapped so ground truth matches the trapped-cell green
        # of the labelled-mask convention. These must stay in step with \masklegend's
        # swatches in the report preamble (gtgreen 0,200,0 / predbrown 170,95,40) —
        # rendering with the defaults silently mislabels every colour in the caption.
        "strip-fp": ["scripts/report/render_holdout_comparison.py",
                     "--pool", "02_unet_holdout", "--cell", "high_cell9",
                     "--checkpoint", f"checkpoints/{SCALE}/unet_best.pt",
                     "--base-features", "64", "--n-frames", "6",
                     "--gt-color", "0,200,0", "--pred-color", "170,95,40",
                     "--output-dir", str(STRIP_BUILD / "high_cell9_fp"),
                     "--output-prefix", "high_cell9_fp"],

        "table-decoy": ["scripts/report/make_report_tables.py", "decoy", "--checkpoint", SCALE,
                        *sum([["--cell", c] for c in CELLS], []),
                        "--output", str(tab / "decoy.tex")],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild every report figure and table")
    p.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--only", nargs="+", default=None, help="Job names to run (see --list)")
    p.add_argument("--list", action="store_true", help="List job names and exit")
    p.add_argument("--dry-run", action="store_true", help="Print commands, run nothing")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--compile", action="store_true",
                   help="After building assets, run pdflatex the required number of times "
                        "and delete the temp files")
    p.add_argument("--tex", default="progress_report_9_17_26.tex",
                   help="Report .tex to compile with --compile")
    args = p.parse_args()

    img = (args.report_root / "images").resolve()
    tab = (args.report_root / "main_work" / "tables").resolve()
    all_jobs = jobs(img, tab)

    if args.list:
        for name in all_jobs:
            print(f"  {name}")
        return
    if not args.report_root.exists():
        sys.exit(f"Report root not found: {args.report_root} (pass --report-root)")

    names = args.only or list(all_jobs)
    unknown = [n for n in names if n not in all_jobs]
    if unknown:
        sys.exit(f"Unknown job(s): {', '.join(unknown)}. Use --list to see the names.")

    img.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)

    failed = []
    for i, name in enumerate(names, 1):
        cmd = [args.python, *all_jobs[name]]
        print(f"\n[{i}/{len(names)}] {name}")
        print("    " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        if args.dry_run:
            continue
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode != 0:
            failed.append(name)
            print(f"    FAILED (exit {r.returncode})")

    # The comparison strips write four variants each; the report uses masks_only.
    if not args.dry_run and any(n.startswith("strip-") for n in names):
        for cell in ("low_cell6", "high_cell9"):
            for tag in ("test1", "test2"):
                src = STRIP_BUILD / f"{cell}_{tag}" / f"{cell}_{tag}_masks_only.png"
                if src.exists():
                    short = "cell6" if cell == "low_cell6" else "cell9"
                    (img / f"cmp_{short}_{tag}.png").write_bytes(src.read_bytes())
            src = STRIP_BUILD / f"{cell}_test1" / f"{cell}_test1_raw.png"
            gt = STRIP_BUILD / f"{cell}_test1" / f"{cell}_test1_labelled_mask.png"
            short = "cell6" if cell == "low_cell6" else "cell9"
            if src.exists():
                (img / f"cmp_{short}_raw.png").write_bytes(src.read_bytes())
            if gt.exists():
                (img / f"cmp_{short}_gt.png").write_bytes(gt.read_bytes())
        fp = STRIP_BUILD / "high_cell9_fp" / "high_cell9_fp_masks_only.png"
        if fp.exists():
            (img / "fp_masks_9_17.png").write_bytes(fp.read_bytes())
        print("\n  comparison strips copied to their report filenames")

    print(f"\n{'DRY RUN — nothing written' if args.dry_run else 'Done'}: "
          f"{len(names) - len(failed)}/{len(names)} jobs succeeded")
    if failed:
        print("  failed: " + ", ".join(failed))
        sys.exit(1)
    if args.dry_run:
        return
    if not args.compile:
        print(f"\nNext: cd {args.report_root / 'main_work'} && "
              f"python {Path(__file__).name} --compile   (or pdflatex twice — see below)")
        return

    # The report must be compiled at least TWICE. Cross-references and \captionof
    # numbers resolve through the .aux, and the temp files are deleted after every
    # build, so a single pass always starts cold and renders every reference as
    # "??" — which is exactly what a one-shot `pdflatex` produces here.
    work = (args.report_root / "main_work").resolve()
    tex = work / args.tex
    if not tex.exists():
        sys.exit(f"Report not found: {tex}")
    print(f"\nCompiling {tex.name}")
    for i in range(3):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex.name],
                       cwd=work, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = tex.with_suffix(".log")
    undefined = log.read_text(errors="ignore").count("undefined") if log.exists() else -1
    for ext in (".aux", ".log", ".out", ".toc", ".lof", ".lot"):
        tex.with_suffix(ext).unlink(missing_ok=True)
    if undefined:
        print(f"    WARNING: {undefined} unresolved reference(s) — they will print as '??'")
        sys.exit(1)
    print(f"    -> {tex.with_suffix('.pdf')}  (all references resolved, temp files removed)")


if __name__ == "__main__":
    main()
