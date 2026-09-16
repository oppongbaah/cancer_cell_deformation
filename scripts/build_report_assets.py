#!/usr/bin/env python3
"""
Regenerate every figure and table in the current progress report.

One command rebuilds the whole report's assets from the data currently on
disk. Each job below records the exact arguments a figure was made with, so
nothing about the report depends on remembering a command line — which is what
this script exists to fix.

    python scripts/build_report_assets.py                    # build everything
    python scripts/build_report_assets.py --list             # show jobs, run nothing
    python scripts/build_report_assets.py --only sweep pools # build a subset
    python scripts/build_report_assets.py --dry-run          # print commands only

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

REPO = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_ROOT = REPO.parent / "masters_thesis"

# Checkpoints the report's figures compare. Change these in one place.
BASE = "multiclass_experiment_256_bf64_base_cos"
SCALE = "multiclass_experiment_256_bf64_scale_cos"
ALLAUG = "multiclass_experiment_256_bf64_allaug"
ORIGINAL = "multiclass_experiment_256"
RETIRED = "multiclass_experiment_256_bf64_scale"      # weights deleted 2026-09-10

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
    CELLS = ["high_cell9=high-9", "low_cell6=low-6", "low_cell5=low-5"]
    return {
        # ── Dataset Organization ────────────────────────────────────────────
        "pools": ["scripts/plot_pool_flow.py", "--output", str(img / "pool_flow_9_17.png")],
        "table-percell": ["scripts/make_report_tables.py", "percell",
                          "--output", str(tab / "percell.tex")],

        # ── Baseline on the restructured four-cell development set ──────────
        "table-baseline": ["scripts/make_report_tables.py", "baseline",
                           "--checkpoint", SCALE,
                           "--cell", "high_cell9=high-9", "--cell", "low_cell6=low-6",
                           "--cell", "low_cell5=low-5 (20 of 51 annotated)",
                           "--cell", "high_cell10=high-10 (not yet annotated)",
                           "--output", str(tab / "baseline.tex")],

        # ── high-9 false-positive diagnosis ─────────────────────────────────
        "fp-diagnosis": ["scripts/diagnose_false_positives.py", "--cell", "high_cell9",
                         "--checkpoint", f"checkpoints/{SCALE}/unet_best.pt",
                         "--output", str(img / "fp_diagnosis_9_17.png")],
        "table-decoy": ["scripts/make_report_tables.py", "decoy", "--checkpoint", SCALE,
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
