#!/usr/bin/env python3
"""
Why high-10 is over-predicted in leave-one-cell-out round 14 (8 October report, added 27 October).

Everything from disk -- the two 1000-epoch nnU-Net cross-validations' saved test predictions and the
hand-drawn masks; no model is run:
  5-fold  round 4   data/nnunet/predictions/experiment_1b_nnunet_1000ep/round4
  LOOCV   round 14  data/nnunet/predictions/experiment_4_loocv_nnunet/round14

Figure (two panels):
  (a) three of high-10's 20 hand-masked frames (laser step on the left): the raw frame, then each round's
      prediction -- pixels it calls "cell" filled red, pixels it calls "decoy" filled blue -- over the hand-drawn
      trapped cell (green outline) and untrapped neighbour (orange outline). Cropped around both hand-drawn objects.
  (b) every frame: the round's "cell" pixels that lie on the hand-drawn neighbour, and the neighbour's size
      relative to the trapped cell's.
Table: every cell's Dice in both cross-validations, as logged (every pixel the model calls "cell") and after
the feature pipeline's selection (only the "cell" region nearest the trap, select_trapped_region; trap = the
median centroid of the largest predicted "cell" region over the cell's own frames).

    python scripts/report/plot_high10.py --output ../masters_thesis/images/high10_round14.png \\
        --table-output ../masters_thesis/main_work/tables/high10_selected.tex
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from celldform.features.quality import select_trapped_region  # noqa: E402

RUNS = {  # label: (dataset, prediction folder)
    "5-fold": ("Dataset501_CellTrapRandswap", "experiment_1b_nnunet_1000ep"),
    "LOOCV": ("Dataset502_CellTrapLOOCV", "experiment_4_loocv_nnunet"),
}
ROUND = {"5-fold": "4", "LOOCV": "14"}            # the rounds that tested high-10
SHOW = ["000600", "004200", "006800"]               # wrong early, right, wrong late (in LOOCV round 14)
HALF_X, HALF_Y = 105, 55                           # crop half-size, px (around both cells)
CELL_C, NEIGH_C = "#2ca02c", "#ff7f0e"


def tests(label: str) -> dict:
    return json.loads((REPO / "data/nnunet/raw" / RUNS[label][0] / "test_frames.json").read_text())


def pred(label: str, rnd: str, stem: str) -> np.ndarray:
    return cv2.imread(str(REPO / "data/nnunet/predictions" / RUNS[label][1] / f"round{rnd}" / f"{stem}.png"),
                      cv2.IMREAD_GRAYSCALE)


def cell_of(stem: str) -> str:
    m = re.search(r"(high|low)_cell(\d+)", stem)
    return f"{m[1]}-{m[2]}"


def neighbour(gt: np.ndarray) -> np.ndarray:
    """The hand-drawn decoy region closest to the trapped cell (other decoys in the frame left out)."""
    dist = cv2.distanceTransform((gt != 1).astype(np.uint8), cv2.DIST_L2, 5)
    n, lab = cv2.connectedComponents((gt == 2).astype(np.uint8))
    return lab == 1 + min(range(n - 1), key=lambda j: dist[lab == j + 1].min())


def dice(tp: float, fp: float, fn: float) -> float:
    return 2 * tp / (2 * tp + fp + fn)


def selected_table() -> list[dict]:
    """Per cell and cross-validation: Dice of every 'cell' pixel, and of the region nearest the trap."""
    rows = []
    for label in RUNS:
        for rnd, t in tests(label).items():
            by = defaultdict(list)
            for s in t["stems"]:
                by[cell_of(s)].append(s)
            for cell, stems in by.items():
                P = {s: pred(label, rnd, s) == 1 for s in stems}
                G = {s: cv2.imread(str(REPO / t["masks"][s]), cv2.IMREAD_GRAYSCALE) == 1 for s in stems}
                cs = []
                for p in P.values():
                    n, _, st, cen = cv2.connectedComponentsWithStats(p.astype(np.uint8))
                    if n > 1:
                        cs.append(cen[1 + np.argmax(st[1:, 4])])
                trap = tuple(np.median(cs, 0))
                a = np.zeros(6)
                for s in stems:
                    p, g = P[s], G[s]
                    r, _ = select_trapped_region(p, trap)
                    q = r["mask"] if r else np.zeros_like(g)
                    a += [(p & g).sum(), (p & ~g).sum(), (g & ~p).sum(), (q & g).sum(), (q & ~g).sum(), (g & ~q).sum()]
                rows.append({"cv": label, "cell": cell, "raw": dice(*a[:3]), "sel": dice(*a[3:])})
    return rows


def write_table(rows: list[dict], out: Path) -> None:
    v = {(r["cv"], r["cell"]): r for r in rows}
    cells = sorted({r["cell"] for r in rows}, key=lambda c: (c.split("-")[0] != "high", int(c.split("-")[1])))
    mean = lambda cv, k: np.mean([v[cv, c][k] for c in cells])
    moved = [c for c in cells if abs(v["LOOCV", c]["raw"] - v["5-fold", c]["raw"]) > 0.01
             or abs(v["LOOCV", c]["sel"] - v["5-fold", c]["sel"]) > 0.01]
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r" & \multicolumn{2}{c}{every ``cell'' pixel (as logged)} & \multicolumn{2}{c}{region nearest the trap} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Cell & 5-fold & LOOCV & 5-fold & LOOCV \\", r"\midrule"]
    for c in moved:
        f = lambda cv, k: f"{v[cv, c][k]:.3f}"
        L.append(rf"\code{{{c}}} & {f('5-fold', 'raw')} & \textbf{{{f('LOOCV', 'raw')}}} & {f('5-fold', 'sel')} & "
                 rf"\textbf{{{f('LOOCV', 'sel')}}} \\")
    others = [c for c in cells if c not in moved]
    gap = max(max(abs(v["LOOCV", c][k] - v["5-fold", c][k]) for k in ("raw", "sel")) for c in others)
    L += [rf"\emph{{other {len(others)} cells}} & \multicolumn{{4}}{{c}}{{\emph{{the two cross-validations agree to within "
          rf"{gap:.3f}}}}} \\", r"\midrule",
          rf"Per-cell mean, 15 cells & {mean('5-fold', 'raw'):.3f} & {mean('LOOCV', 'raw'):.3f} & "
          rf"{mean('5-fold', 'sel'):.3f} & {mean('LOOCV', 'sel'):.3f} \\", r"\bottomrule", r"\end{tabular}"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")


def tint(gray: np.ndarray, lab: np.ndarray) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, 2).astype(float) / 255
    for cls, col in ((1, (0.85, 0.1, 0.1)), (2, (0.15, 0.35, 0.9))):
        m = lab == cls
        rgb[m] = 0.45 * rgb[m] + 0.55 * np.array(col)
    return rgb


def outline(ax, mask: np.ndarray, color: str, ls: str = "-") -> None:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        c = np.vstack([c[:, 0], c[:1, 0]])
        ax.plot(c[:, 0], c[:, 1], color=color, lw=1.3, ls=ls)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", type=Path, default=REPO / "results/report/high10_round14.png")
    ap.add_argument("--table-output", type=Path, default=None)
    a = ap.parse_args()

    t14 = tests("LOOCV")[ROUND["LOOCV"]]
    stems = t14["stems"]
    on_neigh = {k: [] for k in RUNS}
    ratio, steps = [], []
    for s in stems:
        gt = cv2.imread(str(REPO / t14["masks"][s]), cv2.IMREAD_GRAYSCALE)
        ratio.append(neighbour(gt).sum() / (gt == 1).sum())
        steps.append(int(s[-6:]) // 200)
        for k in RUNS:
            on_neigh[k].append(int(((pred(k, ROUND[k], s) == 1) & neighbour(gt)).sum()))

    fig = plt.figure(figsize=(10.4, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.15, 1], hspace=0.28)
    top = gs[0].subgridspec(len(SHOW), 3, wspace=0.04, hspace=0.06)
    for i, tail in enumerate(SHOW):
        s = next(x for x in stems if x.endswith(tail))
        gt = cv2.imread(str(REPO / t14["masks"][s]), cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(str(REPO / "data/nnunet/raw" / RUNS["LOOCV"][0] / f"imagesTs_round{ROUND['LOOCV']}" / f"{s}_0000.png"),
                         cv2.IMREAD_GRAYSCALE)
        cy, cx = np.argwhere(gt > 0).mean(0).astype(int)          # centre of the cell and its neighbour
        sl = np.s_[max(cy - HALF_Y, 0):cy + HALF_Y, max(cx - HALF_X, 0):cx + HALF_X]
        for j, (title, img) in enumerate([("frame", None), *[(f"{k} round {ROUND[k]}", pred(k, ROUND[k], s)) for k in RUNS]]):
            ax = fig.add_subplot(top[i, j])
            ax.imshow(np.repeat(raw[sl][..., None], 3, 2) if img is None else tint(raw[sl], img[sl]), interpolation="nearest")
            if img is not None:
                outline(ax, gt[sl] == 1, CELL_C)
                outline(ax, gt[sl] == 2, NEIGH_C)
                k = list(RUNS)[j - 1]
                n = on_neigh[k][stems.index(s)]
                ax.text(0.03, 0.04, f"{n} px of neighbour called cell", transform=ax.transAxes, fontsize=7.5,
                        color="white" if n < 100 else "#ffdddd", bbox=dict(fc="black", alpha=0.55, lw=0, pad=1.5))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"step {int(tail) // 200}", fontsize=9.5)
    fig.text(0.075, 0.965, "(a)", fontsize=12, fontweight="bold")
    fig.legend(handles=[Patch(color=(0.85, 0.1, 0.1), alpha=0.6, label='predicted "cell"'),
                        Patch(color=(0.15, 0.35, 0.9), alpha=0.6, label='predicted "decoy"'),
                        Line2D([], [], color=CELL_C, lw=1.5, label="hand-drawn trapped cell"),
                        Line2D([], [], color=NEIGH_C, lw=1.5, label="hand-drawn neighbour")],
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=8.5, frameon=False)

    ax = fig.add_subplot(gs[1])
    x = np.arange(len(stems)); w = 0.4
    ax.bar(x - w / 2, on_neigh["5-fold"], w, color="#9aa7b8", label="5-fold round 4")
    ax.bar(x + w / 2, on_neigh["LOOCV"], w, color="#d62728", label="LOOCV round 14")
    ax.set_xticks(x, [str(s) for s in steps], fontsize=8)
    ax.set_xlabel("laser step of the hand-masked frame", fontsize=9)
    ax.set_ylabel('neighbour pixels called "cell"', fontsize=9)
    ax.spines[["top"]].set_visible(False)
    ax2 = ax.twinx()
    ax2.plot(x, ratio, "o-", color=NEIGH_C, ms=3.5, lw=1.2, label="neighbour / trapped-cell size")
    ax2.set_ylabel("neighbour / cell size", fontsize=9, color=NEIGH_C)
    ax2.tick_params(axis="y", colors=NEIGH_C, labelsize=8)
    ax2.spines[["top"]].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13))
    ax.text(-0.075, 1.08, "(b)", transform=ax.transAxes, fontsize=12, fontweight="bold")

    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=200, bbox_inches="tight")
    print(f"wrote {a.output}")
    wrong = np.array(on_neigh["LOOCV"]) > 300
    print(f"  neighbour/cell size, median: LOOCV-wrong frames {np.median(np.array(ratio)[wrong]):.2f}, "
          f"right frames {np.median(np.array(ratio)[~wrong]):.2f}; per step {dict(zip(steps, np.round(ratio, 2)))}")
    for k in RUNS:
        print(f"  {k}: {sum(v > 300 for v in on_neigh[k])} of {len(stems)} frames call the neighbour cell, "
              f"{sum(on_neigh[k])} px")
    if a.table_output:
        write_table(selected_table(), a.table_output)
        print(f"wrote {a.table_output}")


if __name__ == "__main__":
    main()
