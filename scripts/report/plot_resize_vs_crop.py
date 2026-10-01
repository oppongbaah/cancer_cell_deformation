#!/usr/bin/env python3
"""
Experiment 3 figure: the same frames as each input variant sees them.

Columns: whole frame resized to 512, native 512 crop, native 384 crop. Rows: testing
cells high-9 and low-6 at laser step 25, and a legacy 2021 frame (cropped around its
own trap). Outlines are the hand-drawn labels at that variant's grid: green = trapped
cell, red = decoy. Reads the preprocessed sets on disk, so it shows exactly what the
networks were trained and scored on.

    python scripts/report/plot_resize_vs_crop.py --output results/exp3_resize_vs_crop.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROWS = [("evaluate", "postswap_devset", "high_cell9_005000"),
        ("evaluate", "postswap_devset", "low_cell6_005000"),
        ("train", "postswap_multiclass", "20210825164050")]
COLS = [("_512", "resize 512"), ("_crop512", "crop 512"), ("_crop384", "crop 384")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=300)
    args = ap.parse_args()
    rows = []
    for split, pool, name in ROWS:
        tiles = []
        for suffix, label in COLS:
            base = Path("data/preprocessed") / split / f"{pool}{suffix}"
            im = cv2.cvtColor(cv2.imread(str(base / "inputs" / f"{name}.png"), 0), cv2.COLOR_GRAY2BGR)
            lb = cv2.imread(str(base / "labels" / f"{name}.png"), 0)
            for cls, colour in ((1, (0, 200, 0)), (2, (0, 0, 220))):
                cs, _ = cv2.findContours((lb == cls).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cs, -1, colour, 1)
            im = cv2.resize(im, (args.tile, args.tile), interpolation=cv2.INTER_AREA)
            cv2.putText(im, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
            tiles.append(im)
        rows.append(np.hstack(tiles))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), np.vstack(rows))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
