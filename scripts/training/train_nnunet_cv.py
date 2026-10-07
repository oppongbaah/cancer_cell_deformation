#!/usr/bin/env python3
"""
nnU-Net benchmark: the randswap cross-validation, round for round, with nnU-Net in place of our U-Net.

    python scripts/training/train_nnunet_cv.py --setup          # dataset + plan/preprocess, once
    python scripts/training/train_nnunet_cv.py                  # train + predict + score rounds 1..5
    python scripts/training/train_nnunet_cv.py --rounds 3 4     # some rounds only
    python scripts/training/train_nnunet_cv.py --rounds 2 --continue-training   # resume an interrupted round

    # nnU-Net's standard 1000-epoch schedule, filed as its own experiment (rounds can run as
    # parallel jobs with --no-summary, then summarize_cv.py once they have all finished):
    python scripts/training/train_nnunet_cv.py --trainer nnUNetTrainer \
        --exp randswap/experiment_1b_nnunet_1000ep --rounds 1

Pure Python: nnU-Net is IMPORTED (nnunetv2 is a project dependency), never called through
`nnUNetv2_*` executables at a fixed path, so this runs from any environment that has the
project installed -- the local .venv, a conda env on NSU, a SLURM job -- with whatever
`python` launched it. The three nnU-Net folders are set here (unless already exported) before
nnU-Net reads them; nnU-Net looks them up lazily, at use, not at import.

Same cells, same test frames, same native scoring as our CV:
  * data      scripts/data/build_nnunet_dataset.py (raw 1228x922 frames, the 5 fold records
              as splits_final.json; nnU-Net fold k-1 = randswap round k)
  * trainer   nnUNetTrainer_100epochs by default (user decision 2026-10-01; the default 1000 epochs
              is ~1-2 days per round on the RTX 4060). --trainer nnUNetTrainer runs nnU-Net's
              standard 1000 epochs (2026-10-07, on the NSU V100s) -- give it its own --exp so it
              never touches the 100-epoch benchmark's files. nnU-Net keeps its LAST epoch
              (checkpoint_final.pth) -- it does not select an epoch on the validation cell.
  * scoring   eval_nnunet.score(): trapped cell (label 1) at native resolution, pixel counts
              per testing cell, logged in eval_log_native.jsonl's schema
  * summary   summarize_cv.py -> results/randswap/experiment_1_nnunet/nnunet_cv_*.{csv,json}

Outputs (<exp> = --exp, default randswap/experiment_1_nnunet)
  checkpoints/<exp>/Dataset501_*/<trainer>__nnUNetPlans__2d/fold_<k-1>/
  data/nnunet/predictions/round<k>/            (default --exp)
  data/nnunet/predictions/<exp name>/round<k>/  (any other --exp)
  results/<exp>/{eval_log_native.jsonl, nnunet_fold_<k>_split.json, nnunet_cv_*}
The dataset and its preprocessing (data/nnunet/raw, data/nnunet/preprocessed) are shared by every --exp.
nnU-Net's own per-epoch log is training_log_*.txt inside the fold folder (line-buffered; watch
that one, not a redirected stdout).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))   # eval_nnunet

DATASET_ID, DATASET = 501, "Dataset501_CellTrapRandswap"
CONFIGURATION, PLANS = "2d", "nnUNetPlans"
EXP = "randswap/experiment_1_nnunet"
SRC_EXP, SRC_RUN = "randswap/experiment_0_cv_baseline", "cv_baseline"
CV_CONFIG = f"configs/{SRC_EXP}/{SRC_RUN}.yaml"


def set_paths(exp: str) -> None:
    for var, sub in (("nnUNet_raw", "data/nnunet/raw"), ("nnUNet_preprocessed", "data/nnunet/preprocessed"),
                     ("nnUNet_results", f"checkpoints/{exp}")):
        os.environ.setdefault(var, str(ROOT / sub))
        Path(os.environ[var]).mkdir(parents=True, exist_ok=True)


def setup(num_processes: int) -> None:
    """Dataset from the fold records, then nnU-Net's fingerprint + plan + preprocess (2d only)."""
    subprocess.run([sys.executable, "scripts/data/build_nnunet_dataset.py", "--config", CV_CONFIG,
                    "--splits", f"results/{SRC_EXP}", "--run", SRC_RUN], check=True, cwd=ROOT)
    splits = Path(os.environ["nnUNet_preprocessed"]) / DATASET / "splits_final.json"
    ours = splits.read_text()
    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints, plan_experiments, preprocess)
    extract_fingerprints([DATASET_ID], num_processes=num_processes, check_dataset_integrity=True)
    plans = plan_experiments([DATASET_ID])
    preprocess([DATASET_ID], plans, configurations=(CONFIGURATION,), num_processes=(num_processes,))
    if splits.read_text() != ours:      # nnU-Net writes its own random folds only if none exist
        raise SystemExit(f"{splits} no longer holds the randswap rounds")


def split_record(k: int) -> Path:
    """Round k's split record: the CV's results/ copy, or the original in the fold's data folder
    (identical -- train_cv_fold.py copies one to the other; results/ may not be synced to every machine)."""
    copy = ROOT / "results" / SRC_EXP / f"{SRC_RUN}_fold_{k}_split.json"
    if copy.is_file():
        return copy
    from celldform.config import load as load_config
    original = ROOT / load_config(str(ROOT / CV_CONFIG)).split.data_root / f"fold_{k}" / "split.json"
    if not original.is_file():
        raise SystemExit(f"no split record for round {k}: neither {copy} nor {original}")
    return original


def run_round(k: int, trainer: str, continue_training: bool, exp: str = EXP) -> dict:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.run.run_training import run_training
    import torch
    from eval_nnunet import score

    fold = k - 1
    res = ROOT / "results" / exp
    res.mkdir(parents=True, exist_ok=True)
    shutil.copy(split_record(k), res / f"nnunet_fold_{k}_split.json")

    print(f"[{exp}] round {k} (nnU-Net fold {fold}): training with {trainer}", flush=True)
    # strings, exactly as nnUNetv2_train passes them (run_training calls .startswith on the dataset)
    run_training(str(DATASET_ID), CONFIGURATION, str(fold), trainer, PLANS, continue_training=continue_training,
                 device=torch.device("cuda"))

    model_dir = Path(os.environ["nnUNet_results"]) / DATASET / f"{trainer}__{PLANS}__{CONFIGURATION}"
    pred = ROOT / "data" / "nnunet" / "predictions"      # the 100-epoch benchmark's layout, kept as is
    pred = pred / f"round{k}" if exp == EXP else pred / Path(exp).name / f"round{k}"
    shutil.rmtree(pred, ignore_errors=True)
    print(f"[{exp}] round {k}: predicting its test frames", flush=True)
    p = nnUNetPredictor(device=torch.device("cuda"), allow_tqdm=False)       # nnU-Net defaults (mirroring TTA on)
    p.initialize_from_trained_model_folder(str(model_dir), use_folds=(fold,), checkpoint_name="checkpoint_final.pth")
    p.predict_from_files(str(Path(os.environ["nnUNet_raw"]) / DATASET / f"imagesTs_round{k}"), str(pred),
                         num_processes_preprocessing=2, num_processes_segmentation_export=2)
    return score(k, pred, model_dir / f"fold_{fold}" / "checkpoint_final.pth",
                 Path(os.environ["nnUNet_raw"]) / DATASET / "test_frames.json", res / "eval_log_native.jsonl",
                 trainer=trainer)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--setup", action="store_true", help="build the dataset and plan/preprocess, then stop")
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--trainer", default="nnUNetTrainer_100epochs")
    ap.add_argument("--continue-training", action="store_true")
    ap.add_argument("--exp", default=EXP,
                    help=f"experiment folder for checkpoints/ and results/ (default {EXP}); "
                         "use a new one for any other trainer")
    ap.add_argument("--no-summary", action="store_true",
                    help="skip summarize_cv.py at the end (rounds running as parallel jobs)")
    ap.add_argument("-np", "--num-processes", type=int, default=6)
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    os.chdir(ROOT)
    set_paths(a.exp)
    if a.setup:
        setup(a.num_processes)
        return
    if a.exp == EXP and a.trainer != "nnUNetTrainer_100epochs":
        raise SystemExit(f"--trainer {a.trainer} would write into the 100-epoch benchmark ({EXP}); pass a new --exp")
    for k in a.rounds:
        run_round(k, a.trainer, a.continue_training, a.exp)
    if not a.no_summary:
        subprocess.run([sys.executable, "scripts/evaluation/summarize_cv.py", "--exp", a.exp, "--run", "nnunet"],
                       check=True)


if __name__ == "__main__":
    main()
