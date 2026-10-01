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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROWS = [("evaluate", "postswap_devset", "high_cell9_005000"),
        ("evaluate", "postswap_devset", "low_cell6_005000"),
        ("train", "postswap_multiclass", "20210825164050")]
COLS = [("_512", "resize 512"), ("_crop512", "crop 512"), ("_crop384", "crop 384")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    # Names are set OUTSIDE the images, dark on white, with white space around every
    # tile (2026-10-01: labels printed on the grey frames were unreadable in print).
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(9.6, 9.9), facecolor="white",
                             gridspec_kw={"wspace": 0.08, "hspace": 0.16})
    row_names = ["high-9, step 25", "low-6, step 25", "legacy 2021 frame"]
    for r, (split, pool, name) in enumerate(ROWS):
        for k, (suffix, label) in enumerate(COLS):
            base = Path("data/preprocessed") / split / f"{pool}{suffix}"
            im = cv2.cvtColor(cv2.imread(str(base / "inputs" / f"{name}.png"), 0), cv2.COLOR_GRAY2RGB)
            lb = cv2.imread(str(base / "labels" / f"{name}.png"), 0)
            t = max(1, im.shape[0] // 256)
            for cls, colour in ((1, (0, 200, 0)), (2, (220, 0, 0))):
                cs, _ = cv2.findContours((lb == cls).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cs, -1, colour, t)
            ax = axes[r, k]
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#c9c8c3"); sp.set_linewidth(0.8)
            if r == 0:
                ax.set_title(label, fontsize=12, color="#0b0b0b", pad=8)
            if k == 0:
                ax.set_ylabel(row_names[r], fontsize=11, color="#0b0b0b", labelpad=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
