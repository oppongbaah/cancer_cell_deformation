#!/usr/bin/env python3
"""
Where do a model's extra pixels go? Compare several runs' saved predictions of ONE test cell.

Written 2026-10-20 for the high-10 question: on the same 20 test frames, nnU-Net's 5-fold
round 4 over-predicts high-10 by ~70 px/frame and the leave-one-cell-out round 14 by ~415.
This sorts every false-positive pixel into one of three kinds, because each points at a
different cause:

  ring    within --ring px of the true cell boundary -> the mask is drawn FATTER (boundary /
          halo; what a smaller-than-usual cell would suffer from)
  decoy   on an object annotated as decoy (label 2) -> a neighbouring object taken for the cell
  other   anywhere else -> a separate spurious blob

and reports each per frame against laser step (frame index; the videos ramp power over 50
steps), plus an overlay strip of the raw frames. Reads only files on disk: the ground-truth
multiclass masks, the raw frames, and each run's prediction PNGs (nnU-Net's saved
segmentations, 0 = background, 1 = trapped cell).

    python scripts/report/diagnose_mask_growth.py --cell high_cell10 \\
        --run "5-fold R4=data/nnunet/predictions/<exp>/round4" \\
        --run "LOOCV R14=data/nnunet/predictions/<exp>/round14" \\
        --out results/randswap/diagnosis/high_cell10
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
MASK_DIRS = ["data/masks/01_annotate_pool_multiclass", "data/masks/02_unet_holdout_multiclass"]
FRAME_DIRS = ["data/frames/01_annotate_pool", "data/frames/02_unet_holdout"]
# BGR, the report's mask legend: ground truth green, prediction-only brown, overlap yellow
GT_ONLY, PRED_ONLY, BOTH, DECOY = (0, 200, 0), (40, 95, 170), (0, 220, 220), (60, 60, 220)


def find(stem: str, dirs: list[str], ext: str) -> Path:
    for d in dirs:
        p = REPO / d / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{stem}{ext} not in {dirs}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cell", required=True, help="e.g. high_cell10")
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                    help="a run's prediction folder; repeat for each run to compare")
    ap.add_argument("--ring", type=int, default=6, help="ring width around the true boundary, px")
    ap.add_argument("--n-strip", type=int, default=5, help="frames in the overlay strip")
    ap.add_argument("--half", type=int, default=90, help="crop half-size around the cell, px")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    runs = [(lab, REPO / d) for lab, d in (r.split("=", 1) for r in a.run)]
    stems = sorted({p.stem for _, d in runs for p in d.glob(f"*{a.cell}_*.png")})
    for lab, d in runs:
        missing = [s for s in stems if not (d / f"{s}.png").exists()]
        if missing:
            raise SystemExit(f"{lab}: {len(missing)} of {len(stems)} frames missing in {d}")
    if not stems:
        raise SystemExit(f"no {a.cell} predictions in any run folder")
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.ring + 1, 2 * a.ring + 1))

    a.out.mkdir(parents=True, exist_ok=True)
    rows, strips = [], {lab: [] for lab, _ in runs}
    pick = set(np.linspace(0, len(stems) - 1, a.n_strip).round().astype(int))
    for i, s in enumerate(stems):
        gt = cv2.imread(str(find(s, MASK_DIRS, ".png")), cv2.IMREAD_UNCHANGED)
        cell, decoy = gt == 1, gt == 2
        ring = (cv2.dilate(cell.astype(np.uint8), kern) > 0) & ~cell
        raw = cv2.imread(str(find(s, FRAME_DIRS, ".jpg")), cv2.IMREAD_GRAYSCALE)
        step = int(re.search(r"_(\d+)$", s).group(1))
        for lab, d in runs:
            pred = cv2.imread(str(d / f"{s}.png"), cv2.IMREAD_UNCHANGED) == 1
            fp = pred & ~cell
            rows.append({"run": lab, "frame": s, "frame_index": step, "gt_px": int(cell.sum()),
                         "pred_px": int(pred.sum()), "fn_px": int((cell & ~pred).sum()),
                         "fp_ring": int((fp & ring).sum()), "fp_decoy": int((fp & decoy & ~ring).sum()),
                         "fp_other": int((fp & ~ring & ~decoy).sum())})
            if i in pick:
                ys, xs = np.nonzero(cell | pred)
                cy, cx = int(ys.mean()), int(xs.mean())
                y0, x0 = max(cy - a.half, 0), max(cx - a.half, 0)
                win = (slice(y0, y0 + 2 * a.half), slice(x0, x0 + 2 * a.half))
                img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                ov = img.copy()
                ov[decoy & ~pred] = DECOY
                ov[cell & ~pred], ov[pred & ~cell], ov[pred & cell] = GT_ONLY, PRED_ONLY, BOTH
                tile = cv2.addWeighted(img, 0.45, ov, 0.55, 0)[win]
                tile = cv2.resize(tile, (2 * a.half * 2, 2 * a.half * 2), interpolation=cv2.INTER_NEAREST)
                cv2.putText(tile, f"{lab}  #{step}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                strips[lab].append(tile)

    with (a.out / f"{a.cell}_fp_breakdown.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sheet = np.vstack([np.hstack(t) for t in strips.values()])
    cv2.imwrite(str(a.out / f"{a.cell}_overlays.png"), sheet)

    n = len(stems)
    print(f"{a.cell}: {n} frames, ring = {a.ring} px.  Per frame (mean):")
    print(f"{'run':>14} {'true':>6} {'pred':>6} {'pred/true':>9} {'missed':>7} {'extra: ring':>12} {'decoy':>6} {'other':>6}")
    for lab, _ in runs:
        r = [x for x in rows if x["run"] == lab]
        m = {k: sum(x[k] for x in r) / n for k in ("gt_px", "pred_px", "fn_px", "fp_ring", "fp_decoy", "fp_other")}
        print(f"{lab:>14} {m['gt_px']:6.0f} {m['pred_px']:6.0f} {m['pred_px'] / m['gt_px']:9.2f} {m['fn_px']:7.0f}"
              f" {m['fp_ring']:12.0f} {m['fp_decoy']:6.0f} {m['fp_other']:6.0f}")
        extra = [x["fp_ring"] + x["fp_decoy"] + x["fp_other"] for x in r]
        rho = np.corrcoef([x["frame_index"] for x in r], extra)[0, 1] if np.std(extra) else float("nan")
        print(f"{'':>14} extra px vs laser step: r = {rho:+.2f}")
    print(f"-> {a.out / f'{a.cell}_fp_breakdown.csv'}, {a.out / f'{a.cell}_overlays.png'}")


if __name__ == "__main__":
    main()
