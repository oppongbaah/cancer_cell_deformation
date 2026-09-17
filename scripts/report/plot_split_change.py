#!/usr/bin/env python3
"""
Before/after of the train/validation/test split as two stacked bars.

The table this replaces listed only the *new* split and left the old one in
prose, so the reader had to hold three numbers in their head to see the point:
the training pool grew purely by reclaiming a test split that was computed
every run and never evaluated. Two bars make that a length comparison.

Unlike the other plot scripts here, these counts are **arguments, not derived** —
the old split is historical and no longer reproducible from disk without
re-running the old config. They default to the values recorded in CLAUDE.md and
are printed on every run so a figure can always be traced back to what it was
told.

Usage
-----
python scripts/report/plot_split_change.py --output results/split_change.png
python scripts/report/plot_split_change.py --before 293 78 54 --after 371 54 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NSU_GREEN = "#226644"
NSU_GOLD = "#D6B02C"
NEUTRAL = "#B9BDC2"
INK = "#2B2B2B"
MUTED = "#6B7075"


def main() -> None:
    p = argparse.ArgumentParser(description="Train/val/test split, before and after")
    p.add_argument("--before", nargs=3, type=int, default=[293, 78, 54], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--after", nargs=3, type=int, default=[371, 54, 0], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--output", type=Path, default=Path("results/split_change.png"))
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    rows = [("after  (every run in this report)", args.after),
            ("before", args.before)]
    names = ["train", "validation (monitoring only)", "test — computed, never evaluated"]
    colors = [NSU_GREEN, NSU_GOLD, NEUTRAL]

    fig, ax = plt.subplots(figsize=(9.2, 2.35))
    for i, (label, vals) in enumerate(rows):
        left = 0
        for v, c in zip(vals, colors):
            if v == 0:
                continue
            ax.barh(i, v, left=left, color=c, height=0.52, zorder=3,
                    edgecolor="white", linewidth=2)
            ax.text(left + v / 2, i, f"{v}", ha="center", va="center", fontsize=11.5,
                    color="white", fontweight="bold", zorder=5)
            left += v
        ax.text(left + 6, i, f"{sum(vals)} frames", ha="left", va="center",
                fontsize=10.5, color=MUTED, zorder=5)

    # what actually changed
    # Placed above both bars: between them it collides with the segment labels.
    ax.annotate("", xy=(args.after[0], 1.46), xytext=(args.before[0], 1.46),
                arrowprops=dict(arrowstyle="-|>", color=NSU_GREEN, linewidth=1.5))
    ax.text((args.before[0] + args.after[0]) / 2, 1.72,
            f"+{args.after[0] - args.before[0]} frames into training, no new annotation",
            ha="center", va="center", fontsize=10.5, color=NSU_GREEN, zorder=5)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=11, color=INK)
    ax.set_xlim(0, sum(args.before) * 1.16)
    ax.set_ylim(-0.5, 2.05)
    ax.set_xlabel("Frames in 01_annotate_pool", fontsize=10.5, color=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D9D9D9")
    ax.tick_params(axis="both", length=0, colors=INK, labelsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, -0.34),
              ncol=3, frameon=False, fontsize=10)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"-> {args.output}")
    print(f"   before: train {args.before[0]}  val {args.before[1]}  test {args.before[2]}")
    print(f"   after:  train {args.after[0]}  val {args.after[1]}  test {args.after[2]}")
    print("   (counts supplied as arguments — not read from disk)")


if __name__ == "__main__":
    main()
