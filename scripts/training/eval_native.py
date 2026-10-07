#!/usr/bin/env python3
"""
Score a checkpoint on the testing set at NATIVE resolution (1228x922).

Models trained at different input sizes (256, 512, ...) predict on different label
grids, so scoring each on its own grid does not compare like with like. Here every
model's class probabilities are resampled to the frame's full size with
``feature_mask()`` -- exactly what the feature stage does since 2026-09-30 -- and
compared with the ORIGINAL hand-drawn masks. One grid for every model, and the one
the downstream features are measured on.

Reports trapped-cell Dice per cell, pooled over all frames, and the per-cell mean.

    python scripts/training/eval_native.py --checkpoint CK --config CFG \\
        --inputs data/preprocessed/evaluate/postswap_devset_512/inputs \\
        --log results/postswap/<experiment>/eval_log_native.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from celldform.config import load as load_config
from celldform.integration.realtime import feature_mask
from celldform.preprocessing.pipeline import crop_box
from celldform.segmentation.ensemble import EnsembleUNet
from celldform.segmentation.unet import UNet
from celldform.splits import parse_stem

GT_DIR = Path("data/masks/02_unet_holdout_multiclass")


def _cell(stem: str) -> str:
    cell = parse_stem(stem)[0]
    if cell is None:
        raise SystemExit(f"{stem} is not a real-cell frame")
    return cell


def _scores(tp: int, fp: int, fn: int) -> dict:
    # Raw pixel counts are logged too, so scores can be pooled across CV folds exactly.
    return {"dsc": 2 * tp / max(2 * tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, nargs="+", required=True,
                    help="one checkpoint, or several to score as an ensemble (averaged softmax)")
    ap.add_argument("--name", default=None, help="run name in the log (default: checkpoint folder)")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, default=None,
                    help="preprocessed testing-set inputs at the MODEL's resolution")
    ap.add_argument("--gt-dir", type=Path, default=GT_DIR)
    ap.add_argument("--fold-dir", type=Path, default=None,
                    help="randswap fold folder: test inputs, native masks and this fold's crop "
                         "centre all come from its split.json (overrides --inputs/--gt-dir)")
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--fold", default=None, help="CV round to record in the log (summarize_cv.py reads it); "
                    "needed when the checkpoint is not <...>/fold_<k>/unet_best.pt, e.g. unet_last.pt")
    args = ap.parse_args()

    conf = load_config(str(args.config))
    gt_of = {}
    if args.fold_dir:
        rec = json.loads((args.fold_dir / "split.json").read_text())
        args.inputs = args.fold_dir / "test" / "inputs"
        gt_of = {k: Path(v) for k, v in rec["test_masks_native"].items()}
        if rec.get("crop_center"):
            conf.preprocessing.crop_center = tuple(rec["crop_center"])
    if args.inputs is None:
        raise SystemExit("give --inputs or --fold-dir")
    kw = dict(in_channels=1, out_channels=conf.unet.out_channels, base_features=conf.unet.base_features)
    if len(args.checkpoint) == 1:
        model = UNet.from_checkpoint(args.checkpoint[0], **kw)
    else:
        model = EnsembleUNet.from_checkpoints(args.checkpoint, **kw)
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    frames = sorted(args.inputs.glob("*.png"))
    for f in frames:
        gt_raw = cv2.imread(str(gt_of.get(f.stem, args.gt_dir / f.name)), cv2.IMREAD_GRAYSCALE)
        if gt_raw is None:
            raise SystemExit(f"no native ground truth for {f.name} in {args.gt_dir}")
        x = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        box = (crop_box(conf.preprocessing.crop_center, conf.preprocessing.crop_size, *gt_raw.shape)
               if conf.preprocessing.crop_size else None)
        pred = feature_mask(model, x, (gt_raw.shape[1], gt_raw.shape[0]), box).astype(bool)
        gt = gt_raw == 1
        for key in (_cell(f.stem), "ALL CELLS"):
            a = acc[key]
            a[0] += int((pred & gt).sum()); a[1] += int((pred & ~gt).sum()); a[2] += int((~pred & gt).sum())

    cells = sorted(k for k in acc if k != "ALL CELLS")
    per_cell = {c: _scores(*acc[c]) for c in cells}
    pooled = _scores(*acc["ALL CELLS"])
    mean_cell = st.mean(m["dsc"] for m in per_cell.values())
    run = args.name or args.checkpoint[0].parent.name
    ck_record = (str(args.checkpoint[0]) if len(args.checkpoint) == 1
                 else str(args.checkpoint[0].parent.parent / run / "unet_best.pt"))
    print(f"{run}  (native {gt_raw.shape[1]}x{gt_raw.shape[0]}, {len(frames)} frames, "
          f"input {x.shape[1]}x{x.shape[0]})")
    for c in cells:
        m = per_cell[c]
        print(f"  {c.replace('_cell', '-'):<9} DSC {m['dsc']:.3f}  prec {m['precision']:.3f}  rec {m['recall']:.3f}")
    print(f"  pooled    DSC {pooled['dsc']:.3f}  prec {pooled['precision']:.3f}  rec {pooled['recall']:.3f}")
    print(f"  per-cell mean DSC {mean_cell:.3f}")

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with args.log.open("a") as fh:
            for scope, m in [*per_cell.items(), ("ALL CELLS", pooled)]:
                fh.write(json.dumps({
                    "timestamp": stamp, "checkpoint": ck_record, **({"fold": str(args.fold)} if args.fold else {}),
                    "members": [str(c) for c in args.checkpoint] if len(args.checkpoint) > 1 else None,
                    "config": str(args.config),
                    "resolution": "native", "input": f"{x.shape[1]}x{x.shape[0]}",
                    "filter": None if scope == "ALL CELLS" else scope, "metrics": {"1": m},
                    "per_cell_mean_dsc": mean_cell if scope == "ALL CELLS" else None}) + "\n")


if __name__ == "__main__":
    main()
