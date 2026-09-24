#!/usr/bin/env python3
"""
Evaluate an ensemble of checkpoints on the development set.

Averages the per-class **softmax probabilities** across models, then takes the
argmax once. Averaging probabilities rather than voting on argmaxed masks keeps
each model's confidence, so a model that is unsure about a pixel does not get the
same say as one that is certain.

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
from celldform.segmentation.unet import UNet

DEV_IN = Path("data/preprocessed/evaluate/postswap_devset/inputs")
DEV_LB = Path("data/preprocessed/evaluate/postswap_devset/labels")
CKPT_ROOT = Path("checkpoints/postswap")
EVAL_LOG = Path("results/postswap/postswap_eval_log.jsonl")


def load_models(names, device):
    out = []
    for n in names:
        ck = CKPT_ROOT / n / "unet_best.pt"
        if not ck.exists():
            raise SystemExit(f"missing checkpoint: {ck}")
        m = UNet.from_checkpoint(ck, in_channels=1, out_channels=3,
                                 base_features=64, device=device)
        m.eval()
        out.append((n, m))
    return out


@torch.no_grad()
def probs(model, img, device):
    """Softmax probabilities for one preprocessed frame, shape (C, H, W)."""
    x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(device)
    return torch.softmax(model(x), dim=1)[0]


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
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--inputs", type=Path, default=DEV_IN)
    p.add_argument("--labels", type=Path, default=DEV_LB)
    p.add_argument("--save-log", action="store_true",
                   help="Append the ensemble's metrics to the postswap eval log")
    p.add_argument("--name", default=None, help="Name recorded in the log")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    names = args.runs or sorted(d.name for d in CKPT_ROOT.iterdir()
                                if d.is_dir() and (d / "unet_best.pt").exists())
    models = load_models(names, device)
    name = args.name or f"ENSEMBLE({len(models)})[{'+'.join(n.split('_')[-1] for n in names)}]"
    print(f"ensemble of {len(models)}: {', '.join(names)}\n")

    frames = sorted(args.inputs.glob("*.png"))
    # accumulate confusion counts per cell so pooled == concatenate-then-score,
    # matching evaluate_unet.py rather than averaging per-frame Dice
    acc = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0})
    per_model_acc = {n: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}) for n, _ in models}

    for f in frames:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(args.labels / f.name), cv2.IMREAD_GRAYSCALE) == 1
        cell = re.sub(r"_\d+$", "", f.stem)

        stack = torch.stack([probs(m, img, device) for _, m in models])   # (M, C, H, W)
        ens = stack.mean(0).argmax(0).cpu().numpy() == 1
        for scope in (cell, "POOLED"):
            a = acc[scope]
            a["tp"] += int(np.logical_and(ens, gt).sum())
            a["fp"] += int(np.logical_and(ens, ~gt).sum())
            a["fn"] += int(np.logical_and(~ens, gt).sum())
            a["n"] += 1
        for i, (n, _) in enumerate(models):
            pm = stack[i].argmax(0).cpu().numpy() == 1
            for scope in (cell, "POOLED"):
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

    scopes = [s for s in acc if s != "POOLED"] + ["POOLED"]
    hdr = f"  {'cell':<11}{'n':>4}" + "".join(f"{n.split('_')[-1]:>9}" for n, _ in models) \
          + f"{'mean':>9}{'sd':>8}{'ENSEMBLE':>11}{'vs mean':>10}{'vs best':>10}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for s in scopes:
        singles = [score(per_model_acc[n][s])["dsc"] for n, _ in models]
        e = score(acc[s])["dsc"]
        lbl = s.replace("_cell", "-") if s != "POOLED" else "POOLED"
        print(f"  {lbl:<11}{acc[s]['n']:>4}" + "".join(f"{x:>9.3f}" for x in singles)
              + f"{st.mean(singles):>9.3f}{st.stdev(singles):>8.3f}"
              + f"{e:>11.3f}{e-st.mean(singles):>+10.3f}{e-max(singles):>+10.3f}")

    em = score(acc["POOLED"])
    print(f"\n  ensemble pooled: DSC {em['dsc']:.3f}  IoU {em['iou']:.3f}  "
          f"prec {em['precision']:.3f}  rec {em['recall']:.3f}")

    if args.save_log:
        EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with EVAL_LOG.open("a") as fh:
            for s in scopes:
                m = score(acc[s])
                fh.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "checkpoint": f"checkpoints/postswap/{name}/unet_best.pt",
                    "config": "ensemble", "pool": "postswap_devset",
                    "filter": None if s == "POOLED" else s,
                    "n_images": acc[s]["n"], "multiclass": True,
                    "ensemble_members": names,
                    "metrics": {"1": m}}) + "\n")
        print(f"\n  appended to {EVAL_LOG} as '{name}'")


if __name__ == "__main__":
    main()
