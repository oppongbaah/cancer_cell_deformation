#!/usr/bin/env python3
"""
One randswap cross-validation round: draw + build the fold's data, train, score.

    python scripts/training/train_cv_fold.py <era>/<experiment> <run> <fold>
    e.g. python scripts/training/train_cv_fold.py randswap/experiment_0_cv_baseline cv_baseline 1

    config      configs/<experiment>/<run>.yaml          (must be COMMITTED -- the draw)
    data        <split.data_root>/fold_<k>/              (+ split.json, the record)
    checkpoint  <checkpoint_dir>/fold_<k>/unet_best.pt
    log         logs/<experiment>/<run>_fold_<k>.log
    results     results/<experiment>/: curves, fold_<k>_split.json, eval_log_native.jsonl

Runs with whichever python launched it.
"""
from __future__ import annotations

import argparse
import os
import re
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


def tee(cmd: list[str], log: Path, mode: str, stderr=None) -> None:
    """Run cmd, echoing its stdout to the console and to log (like `| tee`); fail on non-zero exit."""
    with log.open(mode) as fh, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr, text=True) as p:
        for line in p.stdout:
            sys.stdout.write(line)
            fh.write(line)
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("experiment", help="e.g. randswap/experiment_0_cv_baseline")
    ap.add_argument("run", help="config name inside configs/<experiment>/, without .yaml")
    ap.add_argument("fold", help="CV round, 1..5")
    args = ap.parse_args()
    os.chdir(ROOT)
    exp, run, fold = args.experiment, args.run, args.fold

    cfg_path = f"configs/{exp}/{run}.yaml"
    Path(f"logs/{exp}").mkdir(parents=True, exist_ok=True)
    Path(f"results/{exp}").mkdir(parents=True, exist_ok=True)
    log = Path(f"logs/{exp}/{run}_fold_{fold}.log")

    print(f"[{exp}] fold {fold}: drawing the split and building data  ({now()})", flush=True)
    tee([sys.executable, "scripts/data/build_randswap.py", "--config", cfg_path, "--fold", fold], log, "w")
    cfg = load_config(cfg_path)
    data_root, ck_root = Path(cfg.split.data_root), Path(cfg.checkpoint_dir)
    shutil.copy(data_root / f"fold_{fold}" / "split.json", f"results/{exp}/{run}_fold_{fold}_split.json")

    print(f"[{exp}] fold {fold}: training  ({now()})", flush=True)
    with log.open("a") as fh:
        subprocess.run([sys.executable, "-u", "scripts/training/train_unet.py", "--config", cfg_path, "--fold", fold,
                        "--tag", f"{run}_fold_{fold}", "--results-dir", f"results/{exp}"],
                       stdout=fh, stderr=subprocess.STDOUT, check=True)
    for line in log.read_text(errors="replace").splitlines():
        if re.search(r"randswap fold|per-cell balanced|Early stopping|Reloaded best", line):
            print(line)

    print(f"[{exp}] fold {fold}: scoring the test cells at native resolution", flush=True)
    tee([sys.executable, "scripts/evaluation/eval_native.py",
         "--checkpoint", str(ck_root / f"fold_{fold}" / "unet_best.pt"), "--config", cfg_path,
         "--fold-dir", str(data_root / f"fold_{fold}"), "--log", f"results/{exp}/eval_log_native.jsonl"],
        log, "a", stderr=subprocess.DEVNULL)
    print(f"[{exp}] fold {fold} done  ({now()})")


if __name__ == "__main__":
    main()
