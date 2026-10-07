#!/usr/bin/env python3
"""
Train + evaluate one run of an experiment, keeping every artefact inside that
experiment's folder (2026-09-30 layout: one folder per experiment, mirrored under
configs/ checkpoints/ logs/ results/).

    python scripts/training/train_experiment.py <experiment> <run> [extra train_unet.py args...]
    e.g. python scripts/training/train_experiment.py postswap/experiment_1b_loss e1b_lovasz_s42

    config      configs/<experiment>/<run>.yaml   (its checkpoint_dir must be
                                                   checkpoints/<experiment>/<run>)
    log         logs/<experiment>/<run>.log
    curves      results/<experiment>/unet_training_curves_v*_<run>.png
    eval log    results/<experiment>/eval_log.jsonl          (per cell + all cells, testing set, own size)
                results/<experiment>/eval_log_native.jsonl   (same, at native 1228x922)

Runs with whichever python launched it.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from celldform.config import load as load_config  # noqa: E402


def now() -> str:
    return datetime.now().strftime("%H:%M")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("experiment", help="e.g. postswap/experiment_1b_loss")
    ap.add_argument("run", help="config name inside configs/<experiment>/, without .yaml")
    ap.add_argument("train_args", nargs=argparse.REMAINDER, help="passed on to train_unet.py")
    args = ap.parse_args()
    os.chdir(ROOT)
    exp, run = args.experiment, args.run

    cfg_path = Path(f"configs/{exp}/{run}.yaml")
    ck = Path(f"checkpoints/{exp}/{run}/unet_best.pt")
    if not cfg_path.is_file():
        sys.exit(f"no such config: {cfg_path}")
    cfg = load_config(str(cfg_path))
    if cfg.checkpoint_dir != f"checkpoints/{exp}/{run}":
        sys.exit(f"{cfg_path}: checkpoint_dir must be checkpoints/{exp}/{run}")
    log = Path(f"logs/{exp}/{run}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    Path(f"results/{exp}").mkdir(parents=True, exist_ok=True)

    print(f"[{exp}] training {run}  ({now()})", flush=True)
    with log.open("w") as fh:
        rc = subprocess.run([sys.executable, "-u", "scripts/training/train_unet.py", "--config", str(cfg_path),
                             "--tag", run, "--results-dir", f"results/{exp}", *args.train_args],
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    lines = log.read_text(errors="replace").splitlines()
    if rc != 0:
        print("\n".join(lines[-20:]), file=sys.stderr)
        sys.exit(f"[{exp}] training {run} failed (exit {rc}); full log: {log}")
    for line in lines:
        if re.search(r"Early stopping|Reloaded best", line):
            print(line)

    # Score on the testing set at the config's own input size, then at native resolution
    # (one grid for every model, so runs at different sizes compare like with like).
    # The testing set matching the training data: postswap_multiclass<_v> -> postswap_devset<_v>
    # (_256 -> plain postswap_devset, which predates the suffixes).
    variant = cfg.data.frames_dir.split("postswap_multiclass")[1].split("/")[0]
    pool = "postswap_devset" if variant == "_256" else f"postswap_devset{variant}"
    if not Path(f"data/preprocessed/evaluate/{pool}").is_dir():
        sys.exit(f"no testing set data/preprocessed/evaluate/{pool}")
    subprocess.run([sys.executable, "scripts/evaluation/eval_testing_set.py", str(ck), str(cfg_path),
                    f"results/{exp}/eval_log.jsonl", pool], check=True)
    native = subprocess.run([sys.executable, "scripts/evaluation/eval_native.py", "--checkpoint", str(ck),
                             "--config", str(cfg_path), "--inputs", f"data/preprocessed/evaluate/{pool}/inputs",
                             "--log", f"results/{exp}/eval_log_native.jsonl"],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    print("\n".join(native.stdout.splitlines()[-2:]))
    print(f"[{exp}] {run} done  ({now()})")


if __name__ == "__main__":
    main()
