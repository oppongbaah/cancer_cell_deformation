#!/usr/bin/env python3
"""
Segment every 03_mask_factory frame with the ensemble and extract classifier features.

The mask factory holds the frames of the 11 *training* cells (no development cell
is in it -- the firewall rule), and its only job is to supply morphology features
for the HER2 classifier. This script produces them.

Features are extracted at NATIVE frame resolution (1228x922), not at the 256x256
the network sees, for two reasons measured on 2026-09-24/30:

1. ``MorphologyExtractor.min_area_px = 50`` is a raw-resolution threshold. At
   256x256 the mean true cell is 48 px, so it rejected the hand-drawn ground truth
   on 114 of 203 development frames.
2. ``PreprocessingPipeline`` STRETCHES 1228x922 to 256x256 -- x shrinks 4.80x, y
   only 3.60x -- so every shape feature taken at 256 (aspect ratio, eccentricity,
   circularity, Hu moments) describes a cell squashed ~1.33:1 horizontally.

So the ensemble's averaged per-class probabilities are resampled (bilinear) back to
the frame's own size and argmaxed there. Resampling the probabilities rather than
the argmaxed mask avoids a 4.8x3.6-px staircase on the boundary, which would bias
perimeter and circularity.

For comparison the same features are also written at 256x256, the way the
pipeline computed them before this fix.

Frames the U-Net TRAINED on are skipped (byte-identical to a file in
01_annotate_pool; 2026-09-30, user decision). The two pools were sampled in parallel
on different frame grids and share 57 frames: frame 0 of seven cells and all 50 of
high-10's original factory frames, which became its training frames when it moved
into training on 2026-09-17. Those 57 were then REMOVED from the factory the same
day (undo record: 03_mask_factory/00_dedupe_UNDO_20260930.json), and high-10 refilled
with 50 fresh frames at the grid midpoints (100, 300, ..., 9900) -- so the skip now
finds nothing and stays as a guard. Skipping
trained frames does not make the rest independent of training -- every factory frame
is ~6 s from a trained frame of the same cell -- so the classifier must still be
judged per cell, with its final test on the development cells.

Outputs (under --out, default data/factory/ensemble4/)
-------
probs_256/<stem>.npy      float16 (3, 256, 256) averaged softmax -- also usable as
                          soft distillation targets on frames no teacher labelled
masks_256/<stem>.png      uint8 class ids 0/1/2 at 256x256
masks_native/<stem>.png   uint8 0/1 trapped cell at the frame's own size
and under results/factory/:
features_ensemble4_native.csv, features_ensemble4_256.csv
   one row per frame: stem, cell, her2, frame_index, cell_found, 15 features.
   cell_found=False rows carry NaN features and must be dropped, not imputed.

Usage
-----
python scripts/inference/harvest_factory.py
python scripts/inference/harvest_factory.py --members s42 s43   # a subset
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

from celldform.features.extractor import MorphologyExtractor
from celldform.preprocessing.pipeline import PreprocessingPipeline
from celldform.segmentation.ensemble import EnsembleUNet

REPO = Path(__file__).parent.parent.parent
FACTORY = REPO / "data/frames/03_mask_factory"
CKPT = "checkpoints/postswap/experiment_0a_seed_baseline/postswap_ftl_cw142_{}/unet_best.pt"
TRAIN_POOL = REPO / "data/frames/01_annotate_pool"
STEM = re.compile(r"^(high|low)_(cell\d+)_(\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--members", nargs="+", default=["s42", "s43", "s44", "s45"])
    ap.add_argument("--out", type=Path, default=REPO / "data/factory/ensemble4")
    ap.add_argument("--results", type=Path, default=REPO / "results/factory")
    ap.add_argument("--min-area-px", type=int, default=50,
                    help="in NATIVE pixels; applied as-is to the 256 run for comparison")
    ap.add_argument("--include-trained", action="store_true",
                    help="keep factory frames the U-Net trained on (skipped by default)")
    args = ap.parse_args()

    frames = sorted(FACTORY.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {FACTORY}")
    if not args.include_trained:
        md5 = lambda q: hashlib.md5(q.read_bytes()).hexdigest()
        trained = {md5(q) for q in TRAIN_POOL.iterdir() if q.is_file()}
        skipped = [f for f in frames if md5(f) in trained]
        frames = [f for f in frames if f not in set(skipped)]
        per_cell = pd.Series(["_".join(STEM.match(f.stem).group(1, 2)) for f in skipped]).value_counts()
        print(f"[harvest] skipping {len(skipped)} frames the U-Net trained on: "
              + ", ".join(f"{c} {n}" for c, n in per_cell.items()))
    for d in ("probs_256", "masks_256", "masks_native"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)

    ens = EnsembleUNet.from_checkpoints([REPO / CKPT.format(m) for m in args.members])
    pre = PreprocessingPipeline()
    ext = MorphologyExtractor(min_area_px=args.min_area_px)
    print(f"[harvest] {ens!r} on {len(frames)} frames -> {args.out.relative_to(REPO)}")

    rows_native, rows_256 = [], []
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        m = STEM.match(f.stem)
        if not m:
            raise SystemExit(f"unexpected frame name: {f.name}")
        her2, cell, idx = m.group(1), m.group(2), int(m.group(3))
        raw = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        h, w = raw.shape

        # Quantise exactly as preprocess_frames.py does before training/evaluation, so
        # the network sees the same input distribution it was scored on.
        x = (pre(raw) * 255).astype(np.uint8).astype(np.float32) / 255.0
        probs = ens.predict_proba(x)                                   # (3, 256, 256)
        classes_256 = probs.argmax(0).astype(np.uint8)
        probs_native = np.stack([cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
                                 for p in probs])
        mask_native = (probs_native.argmax(0) == 1).astype(np.uint8)

        np.save(args.out / "probs_256" / f"{f.stem}.npy", probs.astype(np.float16))
        cv2.imwrite(str(args.out / "masks_256" / f"{f.stem}.png"), classes_256)
        cv2.imwrite(str(args.out / "masks_native" / f"{f.stem}.png"), mask_native)

        meta = {"stem": f.stem, "cell": f"{her2}_{cell}", "her2": her2, "frame_index": idx}
        for rows, mask in ((rows_native, mask_native),
                           (rows_256, (classes_256 == 1).astype(np.uint8))):
            feats = ext.extract(mask)
            rows.append({**meta, "cell_found": not np.isnan(feats["area"]), **feats})

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(frames)}  ({(time.perf_counter() - t0) / (i + 1) * 1e3:.0f} ms/frame)")

    nat, low = pd.DataFrame(rows_native), pd.DataFrame(rows_256)
    nat.to_csv(args.results / "features_ensemble4_native.csv", index=False)
    low.to_csv(args.results / "features_ensemble4_256.csv", index=False)

    print(f"\n[harvest] done in {time.perf_counter() - t0:.0f} s")
    summary = pd.DataFrame({
        "frames": nat.groupby("cell").size(),
        "found_native": nat.groupby("cell").cell_found.sum(),
        "found_256": low.groupby("cell").cell_found.sum(),
        "area_native_med": nat.groupby("cell").area.median().round(0),
        "AR_native_med": nat.groupby("cell").aspect_ratio.median().round(3),
        "AR_256_med": low.groupby("cell").aspect_ratio.median().round(3),
    })
    print(summary.to_string())
    print(f"\ncell found: native {nat.cell_found.sum()}/{len(nat)}, "
          f"256 {low.cell_found.sum()}/{len(low)}")


if __name__ == "__main__":
    main()
