"""
Dataset page of the analysis reports: what each cell contributes, and the leave-one-cell-out split.

Used by the classification and deformation reports.

    python scripts/report/plot_analysis_dataset.py --model nnunet1000_loocv --img <dir>

    analysis_dataset.png  one row per cell: its feature frames (passed / failed the frame checks), laser steps,
                          recording position (trap height) and whether it is used
    analysis_split.png    the leave-one-cell-out split: 14 folds x 14 cells, the test cell of each fold, and the
                          frames on each side

Reads results/features/<model>/features_frames.csv and features_cells.csv only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HIGH, LOW = "#2a78d6", "#eb6834"
HIGH_L, LOW_L = "#cfe0f5", "#fadccb"
INK, INK2, GRID, FAINT = "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c4"


def short(c: str) -> str:
    return c.replace("_cell", "-")


def load(model: str) -> pd.DataFrame:
    res = ROOT / "results" / "features" / model
    f = pd.read_csv(res / "features_frames.csv")
    c = pd.read_csv(res / "features_cells.csv").set_index("cell")
    g = f.groupby("cell").agg(frames=("stem", "size"), passed=("passed", "sum"), steps=("laser_step", "nunique"),
                              trap_y=("trap_y", "median"))
    g = g.join(c[["her2", "usable", "cell_validated"]]).reset_index()
    g["_n"] = g.cell.str.extract(r"(\d+)$")[0].astype(int)
    return g.sort_values(["her2", "_n"]).drop(columns="_n").reset_index(drop=True)


def dataset_fig(g: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6), sharey=True, gridspec_kw={"width_ratios": [3, 1, 1.3]})
    ax, sx, px = axes
    y = np.arange(len(g))[::-1]
    split = (g.trap_y.max() + g.trap_y.min()) / 2
    for yi, (_, r) in zip(y, g.iterrows()):
        dark, light = (HIGH, HIGH_L) if r.her2 == "high" else (LOW, LOW_L)
        used = bool(r.usable)
        ax.barh(yi, r.passed, color=dark if used else FAINT, height=0.62)
        ax.barh(yi, r.frames - r.passed, left=r.passed, color=light if used else "#eeeeec", height=0.62,
                hatch="////", edgecolor="white", lw=0)
        ax.text(r.frames + 4, yi, f"{int(r.passed)}" + ("" if r.frames == r.passed else f" / {int(r.frames)}"),
                va="center", fontsize=7, color=INK2)
        if not used:
            ax.text(150, yi, "excluded: mask 1.22$\\times$ too large", va="center", ha="center", fontsize=7,
                    color=INK2, style="italic")
        sx.scatter(r.steps, yi, s=26, color=dark if used else FAINT, edgecolor="white", lw=1, zorder=3)
        upper = r.trap_y > split
        px.scatter(r.trap_y, yi, s=34, marker="o" if upper else "s", color=dark if used else FAINT,
                   edgecolor="white", lw=1, zorder=3)
    ax.set_yticks(y, [short(c) for c in g.cell], fontsize=8)
    for lab, h, u in zip(ax.get_yticklabels(), g.her2, g.usable):
        lab.set_color((HIGH if h == "high" else LOW) if u else FAINT)
        lab.set_fontweight("bold")
    ax.set_xlim(0, 360)
    ax.set_xlabel("feature frames (solid: passed the frame checks)", fontsize=8, color=INK2)
    ax.set_title("(a) Feature frames per cell", fontsize=9, loc="left", color=INK)
    sx.set_xlim(46, 54)
    sx.set_xlabel("laser steps", fontsize=8, color=INK2)
    sx.set_title("(b) Steps", fontsize=9, loc="left", color=INK)
    px.axvspan(g.trap_y.min() - 8, split - 30, color=GRID, alpha=0.5, lw=0)
    px.axvspan(split + 30, g.trap_y.max() + 8, color=GRID, alpha=0.5, lw=0)
    px.set_xlabel("trap height $y$ (px)", fontsize=8, color=INK2)
    px.set_title("(c) Recording position", fontsize=9, loc="left", color=INK)
    px.text(0.03, -0.2, "two positions = two sessions;\nlow-1 recorded with the high cells", transform=px.transAxes,
            fontsize=6.8, color=INK2, va="top")
    for a in axes:
        a.grid(axis="x", color=GRID, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(axis="x", labelsize=7.5, colors=INK2)
        a.tick_params(axis="y", length=0)
        a.axhline(y[(g.her2 == "high").sum() - 1] - 0.5, color=INK2, lw=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def split_fig(g: pd.DataFrame, out: Path) -> None:
    u = g[g.usable].reset_index(drop=True)
    n = len(u)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.9), gridspec_kw={"width_ratios": [2.1, 1]})
    for i in range(n):                      # fold i holds out cell i
        for j in range(n):
            high = u.her2[j] == "high"
            test = i == j
            col = (HIGH if high else LOW) if test else (HIGH_L if high else LOW_L)
            ax.add_patch(Rectangle((j, n - 1 - i), 0.92, 0.92, color=col, lw=0))
            if test:
                ax.text(j + 0.46, n - 1 - i + 0.46, "test", ha="center", va="center", fontsize=5.6, color="white",
                        fontweight="bold")
    ax.set_xlim(-0.1, n); ax.set_ylim(-0.1, n)
    ax.set_xticks(np.arange(n) + 0.46, [short(c) for c in u.cell], rotation=60, fontsize=7.5)
    for lab, h in zip(ax.get_xticklabels(), u.her2):
        lab.set_color(HIGH if h == "high" else LOW)
    ax.set_yticks(np.arange(n) + 0.46, [f"fold {n - k}" for k in range(n)], fontsize=7)
    ax.set_title("(a) Leave-one-cell-out: 14 folds, each tests one unseen cell", fontsize=9, loc="left", color=INK)
    ax.tick_params(length=0, colors=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0, -2.6, "light = training cells (13)    dark = the test cell (1)    no validation set: every setting fixed "
                     "in advance", fontsize=7, color=INK2, transform=ax.transData)
    tot = u.passed.sum()
    yy = np.arange(n)[::-1]
    bx.barh(yy, tot - u.passed, color=FAINT, height=0.62, label="training frames")
    bx.barh(yy, u.passed, left=tot - u.passed, color=[HIGH if h == "high" else LOW for h in u.her2], height=0.62,
            label="test frames")
    bx.set_yticks(yy, [f"fold {k + 1}: {short(c)}" for k, c in enumerate(u.cell)], fontsize=7)
    bx.set_xlim(0, tot * 1.02)
    bx.set_xlabel(f"passing frames (all {tot:,})".replace(",", " "), fontsize=8, color=INK2)
    bx.set_title("(b) Frames per fold", fontsize=9, loc="left", color=INK)
    bx.legend(fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, -0.2), ncol=2)
    bx.spines[["top", "right"]].set_visible(False)
    bx.tick_params(axis="x", labelsize=7.5, colors=INK2)
    bx.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nnunet1000_loocv")
    ap.add_argument("--img", type=Path, required=True)
    a = ap.parse_args()
    g = load(a.model)
    print(g.to_string(index=False))
    dataset_fig(g, a.img / "analysis_dataset.png")
    split_fig(g, a.img / "analysis_split.png")
    print(f"wrote {a.img}/analysis_{{dataset,split}}.png")


if __name__ == "__main__":
    main()
