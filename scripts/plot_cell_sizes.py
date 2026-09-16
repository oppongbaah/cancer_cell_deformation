#!/usr/bin/env python3
"""
Per-cell trapped-cell size, measured from the annotation masks.

Replaces the summary table that reported only a training mean and range. The
claim that section makes — *both development cells are smaller than every
training cell* — is a statement about where two points sit relative to a
distribution, which a table of three summary rows cannot show. Here every cell
is a dot on one axis and the reader checks the claim directly.

Areas are counted off the masks (label 1 = trapped cell) at annotation
resolution, so the numbers are re-derived rather than transcribed.

Usage
-----
python scripts/plot_cell_sizes.py --output results/cell_sizes.png
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NSU_GREEN = "#226644"
NSU_GOLD = "#D6B02C"
WARN_RED = "#B02A2A"
NEUTRAL = "#B9BDC2"
GRID = "#D9D9D9"
INK = "#2B2B2B"
MUTED = "#6B7075"


def _cell_of(stem: str) -> str | None:
    s = stem[len("domain_"):] if stem.startswith("domain_") else stem
    parts = s.split("_")
    if len(parts) >= 2 and parts[1].startswith("cell"):
        return f"{parts[0]}-{parts[1].replace('cell', '')}"
    return None


def _areas(mask_dir: Path) -> dict[str, list[int]]:
    """Mean label-1 (trapped cell) pixel count per cell id."""
    out: dict[str, list[int]] = defaultdict(list)
    if not mask_dir.exists():
        return out
    for p in sorted(mask_dir.glob("*.png")):
        cell = _cell_of(p.stem)
        if cell is None:
            continue
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        out[cell].append(int((m == 1).sum()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Per-cell trapped-cell size")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("results/cell_sizes.png"))
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    train = _areas(args.data_root / "masks" / "01_annotate_pool_multiclass")
    dev = _areas(args.data_root / "masks" / "02_unet_holdout_multiclass")
    if not train or not dev:
        sys.exit("No masks found — run from the repo root.")

    tr = sorted(((c, float(np.mean(v))) for c, v in train.items()), key=lambda t: t[1])
    dv = sorted(((c, float(np.mean(v))) for c, v in dev.items()), key=lambda t: t[1])
    tr_vals = [v for _, v in tr]
    lo, hi, mean = min(tr_vals), max(tr_vals), float(np.mean(tr_vals))

    rows = dv + tr                                    # development cells at the bottom
    fig, ax = plt.subplots(figsize=(9.0, 0.30 * len(rows) + 1.5))

    # training range as a band, so "outside the training distribution" is visible
    ax.axvspan(lo, hi, color=NSU_GREEN, alpha=0.07, zorder=0)
    ax.axvline(mean, color=NSU_GREEN, linewidth=1.2, linestyle=(0, (5, 3)), zorder=2)
    ax.text(mean, len(rows) - 0.25, f"  training mean {mean:.0f}", fontsize=10,
            color=NSU_GREEN, ha="left", va="center", zorder=5)
    ax.text((lo + hi) / 2, -1.15, f"training range  {lo:.0f} – {hi:.0f} px$^2$",
            fontsize=10, color=NSU_GREEN, ha="center", va="center", zorder=5)

    for i, (cell, v) in enumerate(rows):
        is_dev = i < len(dv)
        color = WARN_RED if is_dev else NEUTRAL
        ax.plot([0, v], [i, i], color=color, alpha=0.35 if not is_dev else 0.55,
                linewidth=1.1, zorder=2)
        ax.scatter([v], [i], s=95 if is_dev else 55, color=color, zorder=4,
                   edgecolor="white", linewidth=1.1)
        ax.text(v + hi * 0.022, i, f"{v:.0f}", fontsize=10.5, va="center", ha="left",
                color=WARN_RED if is_dev else INK,
                fontweight="bold" if is_dev else "normal", zorder=5)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{c}  (development)" if i < len(dv) else c
                        for i, (c, _) in enumerate(rows)], fontsize=11)
    for tick, (_, _) in zip(ax.get_yticklabels(), rows):
        pass
    for i, lbl in enumerate(ax.get_yticklabels()):
        lbl.set_color(WARN_RED if i < len(dv) else INK)
        if i < len(dv):
            lbl.set_fontweight("bold")

    ax.set_xlim(0, hi * 1.18)
    ax.set_ylim(-1.6, len(rows) - 0.3)
    ax.set_xlabel("Mean trapped-cell area per frame (px$^2$, annotation resolution)",
                  fontsize=11, color=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=INK, labelsize=10)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"-> {args.output}")
    print(f"   training: n={len(tr)}  mean {mean:.0f}  range {lo:.0f}-{hi:.0f}")
    for c, v in dv:
        print(f"   development {c}: {v:.0f}  ({'below' if v < lo else 'inside'} the training range)")


if __name__ == "__main__":
    main()
