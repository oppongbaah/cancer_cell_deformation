#!/usr/bin/env python3
"""
Show what a checkpoint's false-positive region actually *is*, on the raw frame.

`high_cell9` loses precision to a large predicted region that is not the cell.
Two augmentation fixes have been aimed at that failure without anyone first
establishing what the region contains. This renders it directly: the predicted
false-positive area outlined on the **raw** frame — the extracted JPEG, before
CLAHE — so the question "does it track a visible optical halo?" can be answered
by looking, not inferred.

CLAHE matters here. Contrast enhancement is applied per-tile before the model
sees anything, so it can amplify or invent local structure. Drawing on the raw
frame is the only way to tell whether the model is chasing something physically
in the image or something preprocessing created.

Alongside the strip it prints intensity statistics per frame — false-positive
region vs. true cell vs. background — and their trend against laser power
(frame index is a proxy: these videos ramp power over 50 increments).

Usage
-----
python scripts/report/diagnose_false_positives.py --cell high_cell9 \\
    --checkpoint checkpoints/preswap/multiclass_experiment_256_bf64_scale_cos/unet_best.pt \\
    --output results/fp_diagnosis_high_cell9.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root
from celldform.segmentation.unet import UNet

GT_GREEN = (0, 200, 0)      # BGR — ground-truth trapped cell
FP_RED = (40, 40, 210)      # BGR — predicted but not cell
_EXTS = (".jpg", ".jpeg", ".png")


def _find(d: Path, stem: str) -> Path | None:
    for e in _EXTS:
        if (d / f"{stem}{e}").exists():
            return d / f"{stem}{e}"
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Render a checkpoint's false-positive region on raw frames")
    p.add_argument("--cell", required=True)
    p.add_argument("--pool", default="02_unet_holdout")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--base-features", type=int, default=64)
    p.add_argument("--n-frames", type=int, default=6)
    p.add_argument("--half-size", type=int, default=210, help="Crop half-width/height in px")
    p.add_argument("--output", type=Path, default=Path("results/fp_diagnosis.png"))
    args = p.parse_args()

    fr = Path("data/frames") / args.pool
    mk = Path("data/masks") / f"{args.pool}_multiclass"
    pre = Path("data/preprocessed/evaluate") / args.pool / "inputs"
    model = UNet.from_checkpoint(args.checkpoint, in_channels=1, base_features=args.base_features,
                                 out_channels=3)

    stems = sorted(q.stem for q in fr.iterdir()
                   if q.suffix.lower() in _EXTS and q.stem.startswith(f"{args.cell}_")
                   and (mk / f"{q.stem}.png").exists())
    if not stems:
        sys.exit(f"No annotated frames for {args.cell} in {fr}")
    pick = [stems[i] for i in np.linspace(0, len(stems) - 1, args.n_frames).round().astype(int)]

    panels, stats = [], []
    for stem in pick:
        raw = cv2.imread(str(_find(fr, stem)), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(mk / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        pin = _find(pre, stem)
        if pin is None:
            continue
        small = cv2.imread(str(pin), cv2.IMREAD_GRAYSCALE)
        pred = cv2.resize(model.predict(small.astype(np.float32) / 255.0, binary_output=True),
                          (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)

        cell = (gt == 1)
        fp = (pred > 0) & ~cell
        bg = (gt == 0) & (pred == 0)
        if fp.sum() == 0 or cell.sum() == 0:
            continue

        idx = int(stem.rsplit("_", 1)[1])
        stats.append((idx, float(raw[fp].mean()), float(raw[cell].mean()),
                      float(raw[bg].mean()), int(fp.sum())))

        # crop around the cell so the artifact and the cell are both in view
        ys, xs = np.where(cell)
        cx, cy = int(xs.mean()), int(ys.mean())
        h = args.half_size
        x0, y0 = max(0, cx - h), max(0, cy - h)
        x1, y1 = min(raw.shape[1], cx + h), min(raw.shape[0], cy + h)

        vis = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        for mask, color in ((fp, FP_RED), (cell, GT_GREEN)):
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, color, 2)
        crop = vis[y0:y1, x0:x1]
        crop = cv2.copyMakeBorder(crop, 26, 2, 2, 2, cv2.BORDER_CONSTANT, value=(250, 250, 250))
        cv2.putText(crop, f"{idx}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1,
                    cv2.LINE_AA)
        panels.append(crop)

    if not panels:
        sys.exit("No frame had both a ground-truth cell and a false-positive region.")
    hmax = max(q.shape[0] for q in panels)
    panels = [cv2.copyMakeBorder(q, 0, hmax - q.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                 value=(250, 250, 250)) for q in panels]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), np.hstack(panels))

    print(f"-> {args.output}")
    print("   green = ground-truth cell, red = predicted but NOT cell (raw frame, pre-CLAHE)\n")
    print(f"   {'frame':>7}{'FP px':>8}{'FP int':>9}{'cell int':>10}{'bkgd int':>10}   FP vs bkgd")
    for idx, f_i, c_i, b_i, n in stats:
        print(f"   {idx:>7}{n:>8}{f_i:>9.1f}{c_i:>10.1f}{b_i:>10.1f}   {f_i - b_i:>+6.1f}")
    if len(stats) > 2:
        xs = np.array([s[0] for s in stats], float)
        print(f"\n   FP area vs laser power (frame index): r = "
              f"{np.corrcoef(xs, [s[4] for s in stats])[0,1]:+.2f}")
        print(f"   FP brightness vs background:            mean "
              f"{np.mean([s[1]-s[3] for s in stats]):+.1f} grey levels")


if __name__ == "__main__":
    main()
