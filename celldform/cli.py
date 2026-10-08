"""
Console entry points registered in pyproject.toml.

Every non-HPC script under scripts/ is a command named after it:
``scripts/training/train_experiment.py`` -> ``celldform-train-experiment``. Each command runs its
script with the current python and passes every argument through unchanged, so a command can
never drift from the script it wraps. ``celldform`` on its own lists them all, and
``celldform <command> [args]`` runs one.

Two commands are implemented here instead of in a script:

  celldform-organize-dataset  -> celldform.acquisition.organiser.main
  celldform-hpc-submit        -> hpc_submit() (generates + submits the SLURM training job)

scripts/hpc/ (SLURM job scripts, run with sbatch) and scripts/maintenance/ (one-off migrations,
not meant to be re-run) have no commands.

The scripts are not part of the installed package, so the script commands need the editable
install from the repo (``pip install -e .``). Like ``python scripts/...``, they resolve relative
paths against the current directory, so run them from the repo root.

Run each with --help for usage. docs/cli.md documents them all.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

# command (without the "celldform-" prefix) -> script under scripts/, grouped by pipeline stage.
# tests/test_cli.py checks this against scripts/ and against pyproject.toml's [project.scripts].
SCRIPT_COMMANDS: dict[str, dict[str, str]] = {
    "data": {
        "extract-frames":       "data/extract_frames.py",
        "preprocess-frames":    "data/preprocess_frames.py",
        "build-randswap":       "data/build_randswap.py",
        "build-nnunet-dataset": "data/build_nnunet_dataset.py",
        "build-distill-data":   "data/build_distill_data.py",
    },
    "annotation": {
        "annotate":             "annotation/annotate.py",
        "annotate-frame":       "annotation/annotate_frame.py",
        "preview-masks":        "annotation/preview_masks.py",
        "validate-masks":       "annotation/validate_masks.py",
    },
    "training": {
        "train-unet":           "training/train_unet.py",
        "train-experiment":     "training/train_experiment.py",
        "train-cv-fold":        "training/train_cv_fold.py",
        "train-deploy-model":   "training/train_deploy_model.py",
        "train-nnunet-cv":      "training/train_nnunet_cv.py",
        "train-student-cv":     "training/train_student_cv.py",
        "distill-ensemble":     "training/distill_ensemble.py",
    },
    "evaluation": {
        "evaluate-unet":        "evaluation/evaluate_unet.py",
        "eval-native":          "evaluation/eval_native.py",
        "eval-testing-set":     "evaluation/eval_testing_set.py",
        "eval-nnunet":          "evaluation/eval_nnunet.py",
        "ensemble-eval":        "evaluation/ensemble_eval.py",
        "eval-ensemble-cv":     "evaluation/eval_ensemble_cv.py",
        "summarize-cv":         "evaluation/summarize_cv.py",
        "compare-postswap":     "evaluation/compare_postswap.py",
        "check-convergence":    "evaluation/check_convergence.py",
        "latency-benchmark":    "evaluation/latency_benchmark.py",
    },
    "inference": {
        "infer":                "inference/infer.py",
        "predict-single-frame": "inference/predict_single_frame.py",
        "harvest-factory":      "inference/harvest_factory.py",
    },
    "report": {
        "build-report-assets":        "report/build_report_assets.py",
        "make-report-tables":         "report/make_report_tables.py",
        "make-cell-gif":              "report/make_cell_gif.py",
        "diagnose-false-positives":   "report/diagnose_false_positives.py",
        "render-holdout-comparison":  "report/render_holdout_comparison.py",
        "view-soft-target":           "report/view_soft_target.py",
        "plot-cell-sizes":            "report/plot_cell_sizes.py",
        "plot-cv":                    "report/plot_cv.py",
        "plot-data-map":              "report/plot_data_map.py",
        "plot-experiments":           "report/plot_experiments.py",
        "plot-model-comparison":      "report/plot_model_comparison.py",
        "plot-nnunet-epochs":         "report/plot_nnunet_epochs.py",
        "plot-percell-metrics":       "report/plot_percell_metrics.py",
        "plot-pool-flow":             "report/plot_pool_flow.py",
        "plot-resize-vs-crop":        "report/plot_resize_vs_crop.py",
        "plot-split-change":          "report/plot_split_change.py",
        "plot-sweep-dsc":             "report/plot_sweep_dsc.py",
    },
}
# Implemented in this module / the package, not as scripts.
BUILTIN_COMMANDS = {"organize-dataset": "data", "hpc-submit": "hpc"}


def _script_for(command: str) -> str | None:
    for group in SCRIPT_COMMANDS.values():
        if command in group:
            return group[command]
    return None


def run_script(rel: str, argv: list[str] | None = None) -> int:
    """Run scripts/<rel> with the current python and the given arguments; return its exit code."""
    script = SCRIPTS / rel
    if not script.is_file():
        print(f"[celldform] {script} not found. The script commands run the repo's scripts/, so "
              "celldform must be installed editable from the repo (pip install -e .).", file=sys.stderr)
        return 1
    try:
        return subprocess.call([sys.executable, str(script), *(sys.argv[1:] if argv is None else argv)])
    except KeyboardInterrupt:
        return 130


def _command(rel: str):
    def command() -> None:
        sys.exit(run_script(rel))
    command.__doc__ = f"CLI entry point: python scripts/{rel} [args]"
    return command


# One module-level function per script command, e.g. train_experiment() for celldform-train-experiment,
# which is what pyproject.toml's entry points name.
for _group in SCRIPT_COMMANDS.values():
    for _name, _rel in _group.items():
        globals()[_name.replace("-", "_")] = _command(_rel)
del _group, _name, _rel


def main() -> None:
    """``celldform``: list every command, or run one (``celldform <command> [args]``)."""
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "list"):
        print("usage: celldform <command> [args]      (or run celldform-<command> directly)\n")
        groups: dict[str, list[tuple[str, str]]] = {g: [(c, f"scripts/{r}") for c, r in cmds.items()]
                                                     for g, cmds in SCRIPT_COMMANDS.items()}
        groups["data"].insert(0, ("organize-dataset", "celldform.acquisition.organiser"))
        groups["hpc"] = [("hpc-submit", "generate + submit the SLURM training job")]
        for g, cmds in groups.items():
            print(f"{g}:")
            for c, what in cmds:
                print(f"  {c:<28} {what}")
        print("\n<command> --help shows that command's options. Full reference: docs/cli.md")
        return
    command, rest = argv[0].removeprefix("celldform-"), argv[1:]
    if command in BUILTIN_COMMANDS:
        sys.argv = [f"celldform-{command}", *rest]
        {"organize-dataset": organize_dataset, "hpc-submit": hpc_submit}[command]()
        return
    rel = _script_for(command)
    if rel is None:
        sys.exit(f"celldform: unknown command {argv[0]!r} -- run `celldform` for the list")
    sys.exit(run_script(rel, rest))


# ---------------------------------------------------------------------------
# organize_dataset
# ---------------------------------------------------------------------------

def organize_dataset() -> None:
    """CLI entry point: organise raw data into pipeline-ready frame folders."""
    from celldform.acquisition.organiser import main
    main()


def hpc_submit() -> None:
    """CLI entry point: generate and submit a SLURM training job on the NSU cluster."""
    import subprocess
    import textwrap

    parser = argparse.ArgumentParser(
        prog="celldform-hpc-submit",
        description="Submit a SLURM U-Net training job on the NSU cluster "
                    "(same job as scripts/hpc/training/train_unet.sh).",
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config YAML passed to scripts/training/train_unet.py (default: configs/default.yaml).")
    parser.add_argument("--partition", default="gpu",
                        help="SLURM partition (default: gpu).")
    parser.add_argument("--account", default=None,
                        help="SLURM account -- NSU does not require one (default: none).")
    parser.add_argument("--qos", default=None,
                        help="SLURM QOS -- NSU does not require one (default: none).")
    parser.add_argument("--time", default="12:00:00",
                        help="Wall-clock time limit HH:MM:SS (default: 12:00:00).")
    parser.add_argument("--cpus", type=int, default=8,
                        help="CPUs per task (default: 8).")
    parser.add_argument("--gpus", type=int, default=1,
                        help="V100 GPUs (default: 1; the node has 3).")
    parser.add_argument("--conda-env", default="celldform",
                        help="Conda env with the project installed (default: celldform).")
    parser.add_argument("--mem", default="32G",
                        help="Memory per node (default: 32G).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the script without submitting.")
    args = parser.parse_args()

    Path("logs/hpc").mkdir(parents=True, exist_ok=True)

    optional = "".join(f"#SBATCH --{k}={v}\n" for k, v in (("account", args.account), ("qos", args.qos)) if v)
    script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=celldform-unet
        #SBATCH --partition={args.partition}
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task={args.cpus}
        #SBATCH --gres=gpu:v100:{args.gpus}
        #SBATCH --mem={args.mem}
        #SBATCH --time={args.time}
        #SBATCH --output=logs/hpc/%x_%j.out
        @@OPTIONAL@@
        # Generated by celldform-hpc-submit; same job as scripts/hpc/training/train_unet.sh.
        # scripts/hpc/_common.sh activates the conda env and keeps logs/hpc/%x_<jobid>.err only on failure.
        set -euo pipefail
        cd "$SLURM_SUBMIT_DIR"
        CONDA_ENV={args.conda_env}
        source scripts/hpc/_common.sh

        run_py scripts/training/train_unet.py --config {args.config}
    """).replace("@@OPTIONAL@@\n", optional)

    if args.dry_run:
        print(script)
        return

    result = subprocess.run(["sbatch"], input=script, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] sbatch failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    job_id = result.stdout.strip().split()[-1]
    print(f"[celldform] Submitted job {job_id}")
    print(f"[celldform] Monitor : squeue -u $USER")
    print(f"[celldform] Log     : tail -f logs/hpc/celldform-unet_{job_id}.out  (a .err appears only on failure)")
    print(f"[celldform] Cancel  : scancel {job_id}")
