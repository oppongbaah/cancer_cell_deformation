#!/usr/bin/env python3
"""
Evaluate an ensemble of checkpoints on the development set.

Scores :class:`celldform.segmentation.ensemble.EnsembleUNet` -- the same class
``RealTimePipeline`` and ``infer.py`` run -- so the number reported here is the
number the deployed model produces. The combination rule (average the per-class
**softmax probabilities**, then argmax once) lives in that class and nowhere else;
changing it there changes what this script measures, by construction.

Averaging probabilities rather than voting on argmaxed masks keeps each model's
confidence, so a model that is unsure about a pixel does not get the same say as
one that is certain.

The per-member columns re-run each member through its own ``UNet.predict``, which
costs a second set of forward passes per frame (~2x). That is deliberate: both the
ensemble row and the member rows then come from shipped inference code rather than
from a copy of it written for this script.

Why this exists: on the 2026-09-23 three-seed baseline, `high-1` scored 0.818 /
0.833 / 0.507 across seeds (sd 0.184) while `high-9` sat at 0.043. That spread
is a property of the training run, not of the cell — seed 44 is behind on three
of four cells and on internal validation. An ensemble averages that variance out
instead of reporting it.

Costs 3x inference. Only Stage 3 uses the GPU, so this is the cheap part of the
pipeline to triple.

Usage
-----
python scripts/training/ensemble_eval.py                      # all postswap seeds
python scripts/training/ensemble_eval.py --runs A B C
python scripts/training/ensemble_eval.py --save-log           # append to the eval log
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from celldform.segmentation.ensemble import EnsembleUNet

DEV_IN = Path("data/preprocessed/evaluate/postswap_devset/inputs")
DEV_LB = Path("data/preprocessed/evaluate/postswap_devset/labels")
CKPT_ROOT = Path("checkpoints/postswap/experiment_0a_seed_baseline")
EVAL_LOG = Path("results/postswap/experiment_0a_seed_baseline/eval_log.jsonl")

# Named explicitly, not globbed. The headline number in the report must not depend
# on what happens to be sitting in checkpoints/postswap/ -- globbing silently swept
# in postswap_distilled (an invalid run: its student saw no augmentation) and moved
# ALL CELLS from 0.859 to 0.856. Pass --runs to score anything else.
DEFAULT_RUNS = [f"postswap_ftl_cw142_s{s}" for s in (42, 43, 44, 45)]


def load_ensemble(names, device, root=CKPT_ROOT):
    """Wrap every named run's best checkpoint in one EnsembleUNet."""
    paths = [root / n / "unet_best.pt" for n in names]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing checkpoint(s): {missing}")
    return EnsembleUNet.from_checkpoints(paths, in_channels=1, out_channels=3,
                                         base_features=64, device=device)


def dice(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    return 2.0 * inter / denom if denom else 1.0


def metrics(pred_bin, gt_bin):
    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()
    return {"dsc": (2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 1.0,
            "iou": (tp/(tp+fp+fn)) if (tp+fp+fn) else 1.0,
            "precision": (tp/(tp+fp)) if (tp+fp) else 1.0,
            "recall": (tp/(tp+fn)) if (tp+fn) else 1.0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=None,
                   help=f"Run names under {CKPT_ROOT}. Default: {' '.join(DEFAULT_RUNS)}")
    p.add_argument("--inputs", type=Path, default=DEV_IN)
    p.add_argument("--labels", type=Path, default=DEV_LB)
    p.add_argument("--ckpt-root", type=Path, default=CKPT_ROOT,
                   help="Experiment folder holding the runs (default: the seed baseline)")
    p.add_argument("--log", type=Path, default=EVAL_LOG, help="Eval log --save-log appends to")
    p.add_argument("--save-log", action="store_true",
                   help="Append the ensemble's metrics to the postswap eval log")
    p.add_argument("--name", default=None, help="Name recorded in the log")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = args.runs or DEFAULT_RUNS
    ens = load_ensemble(names, device, args.ckpt_root)
    name = args.name or f"ENSEMBLE({len(ens)})[{'+'.join(n.split('_')[-1] for n in names)}]"
    print(f"ensemble of {len(ens)}: {', '.join(names)}\n")

    frames = sorted(args.inputs.glob("*.png"))
    # accumulate confusion counts per cell so pooled == concatenate-then-score,
    # matching evaluate_unet.py rather than averaging per-frame Dice
    acc = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0})
    per_model_acc = {n: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}) for n in names}

    for f in frames:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(args.labels / f.name), cv2.IMREAD_GRAYSCALE) == 1
        cell = re.sub(r"_\d+$", "", f.stem)

        # EnsembleUNet.predict expects float in [0, 1], the same contract
        # UNet.predict documents; the PNGs on disk are uint8.
        norm = img.astype(np.float32) / 255.0

        pred = ens.predict(norm, binary_output=True).astype(bool)
        for scope in (cell, "ALL CELLS"):
            a = acc[scope]
            a["tp"] += int(np.logical_and(pred, gt).sum())
            a["fp"] += int(np.logical_and(pred, ~gt).sum())
            a["fn"] += int(np.logical_and(~pred, gt).sum())
            a["n"] += 1
        for n, m in zip(names, ens.models):
            pm = m.predict(norm, binary_output=True).astype(bool)
            for scope in (cell, "ALL CELLS"):
                b = per_model_acc[n][scope]
                b["tp"] += int(np.logical_and(pm, gt).sum())
                b["fp"] += int(np.logical_and(pm, ~gt).sum())
                b["fn"] += int(np.logical_and(~pm, gt).sum())

    def score(a):
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        return {"dsc": 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 1.0,
                "iou": tp/(tp+fp+fn) if (tp+fp+fn) else 1.0,
                "precision": tp/(tp+fp) if (tp+fp) else 1.0,
                "recall": tp/(tp+fn) if (tp+fn) else 1.0}

    scopes = [s for s in acc if s != "ALL CELLS"] + ["ALL CELLS"]
    hdr = f"  {'cell':<11}{'n':>4}" + "".join(f"{n.split('_')[-1]:>9}" for n in names) \
          + f"{'mean':>9}{'sd':>8}{'ENSEMBLE':>11}{'vs mean':>10}{'vs best':>10}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for s in scopes:
        singles = [score(per_model_acc[n][s])["dsc"] for n in names]
        e = score(acc[s])["dsc"]
        lbl = s.replace("_cell", "-") if s != "ALL CELLS" else "ALL CELLS"
        print(f"  {lbl:<11}{acc[s]['n']:>4}" + "".join(f"{x:>9.3f}" for x in singles)
              + f"{st.mean(singles):>9.3f}{st.stdev(singles):>8.3f}"
              + f"{e:>11.3f}{e-st.mean(singles):>+10.3f}{e-max(singles):>+10.3f}")

    em = score(acc["ALL CELLS"])
    print(f"\n  ensemble, all cells: DSC {em['dsc']:.3f}  IoU {em['iou']:.3f}  "
          f"prec {em['precision']:.3f}  rec {em['recall']:.3f}")

    if args.save_log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("a") as fh:
            for s in scopes:
                m = score(acc[s])
                fh.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "checkpoint": f"{args.ckpt_root}/{name}/unet_best.pt",
                    "config": "ensemble", "pool": "postswap_devset",
                    "filter": None if s == "ALL CELLS" else s,
                    "n_images": acc[s]["n"], "multiclass": True,
                    "ensemble_members": names,
                    "metrics": {"1": m}}) + "\n")
        print(f"\n  appended to {args.log} as '{name}'")


if __name__ == "__main__":
    main()
