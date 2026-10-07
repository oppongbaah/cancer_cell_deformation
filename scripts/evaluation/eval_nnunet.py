#!/usr/bin/env python3
"""
Score one round of the nnU-Net benchmark exactly as eval_native.py scores our U-Net.

    python scripts/evaluation/eval_nnunet.py --round 1 \
        --pred-dir data/nnunet/predictions/round1 \
        --checkpoint checkpoints/randswap/experiment_1_nnunet/Dataset501_.../fold_0/checkpoint_final.pth \
        --log results/randswap/experiment_1_nnunet/eval_log_native.jsonl

nnU-Net predicts at the frames' native 1228x922, so there is nothing to resample: the
predicted label map is compared with the native hand-drawn mask, trapped cell (label 1)
only, pixel counts summed per testing cell and over the round. Same test frames (the
round's split record, via test_frames.json), same counting, same log schema -- plus an
explicit "fold" field, because nnU-Net's own fold numbers are 0-based.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from celldform.splits import parse_stem

TEST_JSON = Path("data/nnunet/raw/Dataset501_CellTrapRandswap/test_frames.json")


def _scores(tp: int, fp: int, fn: int) -> dict:
    return {"dsc": 2 * tp / max(2 * tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn}


def score(round_: str | int, pred_dir: Path | str, checkpoint: Path | str,
          test_json: Path | str = TEST_JSON, log: Path | str | None = None,
          trainer: str = "nnUNetTrainer_100epochs") -> dict:
    """Score one round's predictions; append to ``log`` (eval_native.py's schema + "fold").
    ``trainer`` is recorded in the log's "config" field."""
    t = json.loads(Path(test_json).read_text())[str(round_)]
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for s in t["stems"]:
        pred = cv2.imread(str(Path(pred_dir) / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(t["masks"][s], cv2.IMREAD_GRAYSCALE)
        if pred is None:
            raise SystemExit(f"no prediction for {s} in {Path(pred_dir)}")
        if pred.shape != gt.shape:
            raise SystemExit(f"{s}: prediction {pred.shape} vs mask {gt.shape}")
        p, g = pred == 1, gt == 1
        for key in (parse_stem(s)[0], "ALL CELLS"):
            c = acc[key]
            c[0] += int((p & g).sum()); c[1] += int((p & ~g).sum()); c[2] += int((~p & g).sum())

    cells = sorted(k for k in acc if k != "ALL CELLS")
    per_cell = {c: _scores(*acc[c]) for c in cells}
    pooled = _scores(*acc["ALL CELLS"])
    mean_cell = st.mean(m["dsc"] for m in per_cell.values())
    print(f"nnU-Net round {round_}  (native {gt.shape[1]}x{gt.shape[0]}, {len(t['stems'])} frames)")
    for c in cells:
        m = per_cell[c]
        print(f"  {c.replace('_cell', '-'):<9} DSC {m['dsc']:.3f}  prec {m['precision']:.3f}  rec {m['recall']:.3f}")
    print(f"  pooled    DSC {pooled['dsc']:.3f}  prec {pooled['precision']:.3f}  rec {pooled['recall']:.3f}")

    if log:
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with log.open("a") as fh:
            for scope, m in [*per_cell.items(), ("ALL CELLS", pooled)]:
                fh.write(json.dumps({
                    "timestamp": stamp, "checkpoint": str(checkpoint), "fold": str(round_),
                    "members": None, "config": f"{trainer} / nnUNetPlans / 2d",
                    "resolution": "native", "input": f"{gt.shape[1]}x{gt.shape[0]}",
                    "filter": None if scope == "ALL CELLS" else scope, "metrics": {"1": m},
                    "per_cell_mean_dsc": mean_cell if scope == "ALL CELLS" else None}) + "\n")

    return {"cells": per_cell, "pooled": pooled}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--round", required=True, help="randswap round 1..k")
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True, help="recorded in the log")
    ap.add_argument("--test-json", type=Path,
                    default=TEST_JSON)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--trainer", default="nnUNetTrainer_100epochs", help="nnU-Net trainer, recorded in the log")
    a = ap.parse_args()

    score(a.round, a.pred_dir, a.checkpoint, a.test_json, a.log, a.trainer)


if __name__ == "__main__":
    main()
