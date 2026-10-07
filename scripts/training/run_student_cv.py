#!/usr/bin/env python3
"""
Distillation student CV (Experiment 6): every randswap round, against our CV model at the same epoch.

    python scripts/training/run_student_cv.py                 # rounds 1..5
    python scripts/training/run_student_cv.py --rounds 2 3

Per round k: build the distillation data if missing (build_distill_data.py: nnU-Net fold k-1 labels
the round's training cells), train the student for the FIXED epoch our CV model reached in round k
(cv_baseline_cv_rounds.csv: best_epoch), score its last-epoch weights on round k's test frames with
eval_native.py, and finally summarize_cv.py -> results/randswap/experiment_3_distill/cv/student_cv_*.
Pass rule (fixed before the run): beat our CV model on per-cell mean DSC AND Pooled DSC.
Runs with whichever python launched it.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXP, CFG = "randswap/experiment_3_distill/cv", "configs/randswap/experiment_3_distill/cv/student_cv.yaml"
DATA, CKPT = Path("data/preprocessed/randswap/experiment_3_distill"), Path("checkpoints/randswap/experiment_3_distill/cv/student_cv")


def run(cmd, log=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    with (open(log, "a") if log else open("/dev/null", "w")) as fh:
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=fh if log else None, stderr=subprocess.STDOUT if log else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rounds", nargs="+", default=["1", "2", "3", "4", "5"])
    a = ap.parse_args()
    epochs = {r["fold"]: r["best_epoch"] for r in
              csv.DictReader((ROOT / "results/randswap/experiment_0_cv_baseline/cv_baseline_cv_rounds.csv").open())}
    (ROOT / "results" / EXP).mkdir(parents=True, exist_ok=True)
    (ROOT / "logs" / EXP).mkdir(parents=True, exist_ok=True)
    py = sys.executable
    for k in a.rounds:
        log = ROOT / "logs" / EXP / f"student_cv_fold_{k}.log"
        if not (ROOT / DATA / f"fold_{k}" / "split.json").exists():
            run([py, "-u", "scripts/data/build_distill_data.py", "--round", k], log)
        shutil.copy(ROOT / DATA / f"fold_{k}" / "split.json", ROOT / "results" / EXP / f"student_cv_fold_{k}_split.json")
        run([py, "-u", "scripts/training/train_unet.py", "--config", CFG, "--fold", k, "--epochs", epochs[k],
             "--save-last", "--tag", f"student_cv_fold_{k}", "--results-dir", f"results/{EXP}"], log)
        run([py, "scripts/evaluation/eval_native.py", "--checkpoint", str(CKPT / f"fold_{k}" / "unet_last.pt"),
             "--name", f"student_cv_fold_{k}", "--fold", k, "--config", CFG, "--fold-dir", str(DATA / f"fold_{k}"),
             "--log", f"results/{EXP}/eval_log_native.jsonl"], log)
        print(f"round {k} done ({epochs[k]} epochs)", flush=True)
    run([py, "scripts/evaluation/summarize_cv.py", "--exp", EXP, "--run", "student_cv"])


if __name__ == "__main__":
    main()
