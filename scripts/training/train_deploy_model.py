#!/usr/bin/env python3
"""
The randswap DEPLOYMENT model -- the one RealTimePipeline runs on the camera: every cell
trains, fixed length = the CV median best epoch.

    python scripts/training/train_deploy_model.py <era>/<experiment> <run>
    e.g. python scripts/training/train_deploy_model.py randswap/experiment_0_cv_baseline cv_baseline

The epoch count is read from results/<experiment>/<run>_cv_summary.json (summarize_cv.py),
never typed in. No cell is held out, so there is no score here -- the CV rounds are the
estimate of how this model does on a new cell.

    checkpoint  <checkpoint_dir>/fold_deploy/unet_deploy.pt
    log         logs/<experiment>/<run>_fold_deploy.log

Runs with whichever python launched it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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
    ap.add_argument("experiment", help="e.g. randswap/experiment_0_cv_baseline")
    ap.add_argument("run", help="config name inside configs/<experiment>/, without .yaml")
    args = ap.parse_args()
    os.chdir(ROOT)
    exp, run = args.experiment, args.run

    cfg_path = f"configs/{exp}/{run}.yaml"
    log = Path(f"logs/{exp}/{run}_fold_deploy.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    summary = Path(f"results/{exp}/{run}_cv_summary.json")
    if not summary.is_file():
        sys.exit(f"no CV summary {summary} -- run the CV rounds and summarize_cv.py first")
    epochs = int(json.loads(summary.read_text())["median_best_epoch"])
    data_root = Path(load_config(cfg_path).split.data_root)

    print(f"[{exp}] deploy: building data for all cells  ({now()})", flush=True)
    with log.open("w") as fh, subprocess.Popen(
            [sys.executable, "scripts/data/build_randswap.py", "--config", cfg_path, "--fold", "deploy"],
            stdout=subprocess.PIPE, text=True) as p:
        for line in p.stdout:                       # | tee
            sys.stdout.write(line)
            fh.write(line)
    if p.returncode != 0:
        sys.exit(f"[{exp}] deploy: build_randswap.py failed (exit {p.returncode})")
    shutil.copy(data_root / "fold_deploy" / "split.json", f"results/{exp}/{run}_fold_deploy_split.json")

    print(f"[{exp}] deploy: training {epochs} epochs (CV median best epoch)  ({now()})", flush=True)
    with log.open("a") as fh:
        rc = subprocess.run([sys.executable, "-u", "scripts/training/train_unet.py", "--config", cfg_path,
                             "--fold", "deploy", "--epochs", str(epochs), "--tag", f"{run}_fold_deploy",
                             "--results-dir", f"results/{exp}"],
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    print("\n".join(log.read_text(errors="replace").splitlines()[-3:]))
    if rc != 0:
        sys.exit(f"[{exp}] deploy: training failed (exit {rc}); full log: {log}")
    print(f"[{exp}] deploy done  ({now()})")


if __name__ == "__main__":
    main()
