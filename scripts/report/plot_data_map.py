#!/usr/bin/env python3
"""
Dataset Organization figure (randswap): where every cell and frame goes.

One map, read left to right:
  * every real cell (grouped by HER2 class) and the 2021 legacy frames, one row each;
  * its role in each cross-validation round R1..R5 and in the deployment model --
    train / validation / test -- with the Test DSC printed in the round that tested it;
  * how many annotated frames it gives segmentation, and how many feature
    frames it gives the classifier.

Everything is read from disk: the per-round split records the CV wrote
(results/<exp>/<run>_fold_<k>_split.json), its per-cell scores (<run>_cv_cells.csv)
and the feature-factory folder. Nothing is typed in.

    python scripts/report/plot_data_map.py --exp randswap/experiment_0_cv_baseline \
        --run cv_baseline --output ../masters_thesis/images/data_map.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from celldform.splits import parse_stem

# Roles: validated pair (scripts/validate_palette.js, light): CVD dE 27.9, normal 34.1,
# both >= 3:1 on the surface. Train is a neutral. HER2 chips reuse categorical slots 1-2.
TRAIN, VAL, TEST = "#e3e2dd", "#4a3aa7", "#008300"
HIGH, LOW = "#2a78d6", "#eb6834"
# Bars: annotated frames dark ink, feature frames brown -- a colour used nowhere else in
# the figure (user, 2026-10-01). Validated against every colour here (--pairs all, light):
# normal-vision floor passes, >= 3:1 contrast on the surface.
ANN_BAR, FEAT_BAR = "#52514e", "#8c5a2b"
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#8a8985", "#fcfcfb"


def short(cell: str) -> str:
    return cell.replace("_cell", "-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--factory", type=Path, default=Path("data/frames/03_mask_factory"))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    R = Path("results") / a.exp

    recs = [json.loads(p.read_text()) for p in sorted(R.glob(f"{a.run}_fold_*_split.json"),
                                                      key=lambda p: p.name)]
    recs = [r for r in recs if r["fold"] != "deploy"]
    her2 = recs[0]["her2"]
    ann = recs[0]["annotated_frames_per_cell"]
    n_legacy = recs[0]["n_legacy_frames"]
    dsc = {row["cell"]: float(row["dsc"]) for row in csv.DictReader((R / f"{a.run}_cv_cells.csv").open())}
    fac = Counter(parse_stem(p.stem)[0] for p in a.factory.glob("*.jpg"))
    cells = sorted(her2, key=lambda c: (her2[c] != "high", int(c.split("cell")[1])))
    n_high = sum(her2[c] == "high" for c in cells)
    rounds = [f"R{r['fold']}" for r in recs] + ["Deploy"]

    def role(cell, k):
        if k == len(recs):
            return "train"
        r = recs[k]["cells"]
        return "test" if cell in r["test"] else "val" if cell in r["val"] else "train"

    # row positions (top to bottom), with gaps between groups
    ys, y = {}, 0.0
    for i, c in enumerate(cells):
        if i == n_high:
            y += 0.55
        ys[c] = y
        y += 1.0
    y_legacy = y + 0.55
    H = y_legacy + 1

    fig = plt.figure(figsize=(9.2, 8.7), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(-4.5, 13.0)
    ax.set_ylim(H + 2.9, -3.4)
    ax.axis("off")

    gx0, cw = -1.55, 1.32                     # grid origin and column width
    def box(x, y, w, h, fc, ec="none", lw=0, r=0.12, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw, zorder=z))

    # ── title + split line ─────────────────────────────────────────────────
    ax.text(-4.3, -2.75, "15 cells, 5 rounds — every cell is tested exactly once, by a model that never saw it",
            fontsize=13.5, color=INK, weight="bold", va="center")
    c0 = recs[0]["counts"]; pc = recs[0]["percent_cells"]
    ax.text(-4.3, -1.95, f"Each round:  {c0['train']} train  ·  {c0['val']} validation  ·  {c0['test']} test cells"
            f"   ({pc['train']:.0f} % : {pc['val']:.0f} % : {pc['test']:.0f} %)   —   drawn at random, "
            f"stratified by HER2, seed {recs[0]['cell_split_seed']}, committed before the draw",
            fontsize=9.6, color=INK2, va="center")

    # ── column headers ──────────────────────────────────────────────────────
    for k, name in enumerate(rounds):
        x = gx0 + k * cw + (0.25 if name == "Deploy" else 0)
        ax.text(x + cw / 2 - 0.06, -0.75, name, ha="center", va="center", fontsize=10.5,
                color=INK, weight="bold")
    ax.text(gx0 + 2.5 * cw, -1.3, "cross-validation rounds", ha="center", fontsize=8.6, color=MUTED)
    ax.text(gx0 + 5 * cw + 0.25 + cw / 2 - 0.06, -1.3, "all cells", ha="center", fontsize=8.6, color=MUTED)
    bx_ann, bx_fac, bw = gx0 + 6 * cw + 0.75, gx0 + 6 * cw + 3.6, 2.35
    ax.text(bx_ann + bw / 2, -0.75, "annotated frames", ha="center", fontsize=10, color=INK, weight="bold")
    ax.text(bx_ann + bw / 2, -1.3, "→ segmentation", ha="center", fontsize=8.6, color=MUTED)
    ax.text(bx_fac + bw / 2, -0.75, "feature frames", ha="center", fontsize=10, color=INK, weight="bold")
    ax.text(bx_fac + bw / 2, -1.3, "→ classifier features", ha="center", fontsize=8.6, color=MUTED)

    amax = max(ann.values())
    for c in cells:
        y = ys[c]
        col = HIGH if her2[c] == "high" else LOW
        box(-4.25, y + 0.14, 0.32, 0.72, col, r=0.08)
        ax.text(-3.75, y + 0.5, short(c), va="center", fontsize=10, color=INK)
        for k in range(len(rounds)):
            rl = role(c, k)
            x = gx0 + k * cw + (0.25 if k == len(recs) else 0)
            fc = {"train": TRAIN, "val": VAL, "test": TEST}[rl]
            box(x, y + 0.08, cw - 0.12, 0.84, fc)
            if rl == "test":
                ax.text(x + (cw - 0.12) / 2, y + 0.5, f"{dsc[c]:.2f}", ha="center", va="center",
                        fontsize=9.2, color="white", weight="bold", zorder=3)
            elif rl == "val":
                ax.text(x + (cw - 0.12) / 2, y + 0.5, "val", ha="center", va="center", fontsize=8.4,
                        color="white", zorder=3)
        n = ann[c]
        box(bx_ann, y + 0.25, bw * n / amax, 0.5, ANN_BAR, r=0.06)
        ax.text(bx_ann + bw * n / amax + 0.08, y + 0.5, str(n), va="center", fontsize=8.6, color=INK)
        f = fac.get(c, 0)
        if f:
            box(bx_fac, y + 0.25, bw * f / amax, 0.5, FEAT_BAR, r=0.06)
            ax.text(bx_fac + bw * f / amax + 0.08, y + 0.5, str(f), va="center", fontsize=8.6, color=INK)
        else:
            ax.text(bx_fac + 0.02, y + 0.5, "0 — to extract", va="center", fontsize=8.6,
                    color="#b02a2a", weight="bold")

    # group labels
    for (lo, hi, lab, col) in ((0, n_high, "high HER2", HIGH), (n_high, len(cells), "low HER2", LOW)):
        y0, y1 = ys[cells[lo]], ys[cells[hi - 1]] + 1
        ax.text(-4.42, (y0 + y1) / 2, lab, rotation=90, ha="right", va="center", fontsize=10,
                color=col, weight="bold")

    # legacy row: trains in every round, never validates or tests
    y = y_legacy
    box(-4.25, y + 0.14, 0.32, 0.72, MUTED, r=0.08)
    ax.text(-3.75, y + 0.5, "legacy 2021", va="center", fontsize=10, color=INK)
    for k in range(len(rounds)):
        x = gx0 + k * cw + (0.25 if k == len(recs) else 0)
        box(x, y + 0.08, cw - 0.12, 0.84, TRAIN)
    ax.text(gx0 + 2.9 * cw, y + 0.5, "train only — no cell, no HER2 label, other microscope",
            ha="center", va="center", fontsize=8.4, color=INK2, zorder=3)
    box(bx_ann, y + 0.25, bw, 0.5, ANN_BAR, r=0.06)
    ax.text(bx_ann + bw + 0.08, y + 0.5, f"{n_legacy}", va="center", fontsize=8.6, color=INK)
    ax.text(bx_ann + 0.35 * bw, y + 1.25, "(bar capped)", fontsize=7.4, color=MUTED, ha="center")
    ax.text(bx_fac + 0.35, y + 0.5, "not used", va="center", fontsize=8.6, color=MUTED)

    # crop centres per round, under the grid
    yb = H + 0.55
    ax.text(-3.75, yb + 0.25, "crop centre (x, y)", va="center", fontsize=8.8, color=INK2)
    for k, r in enumerate(recs):
        cx, cy = r["crop_center"]
        ax.text(gx0 + k * cw + (cw - 0.12) / 2, yb + 0.25, f"{cx},{cy}", ha="center", va="center",
                fontsize=8.4, color=INK2)
    ax.text(gx0 + 5 * cw + 0.25 + (cw - 0.12) / 2, yb + 0.25, "all 15", ha="center", va="center",
            fontsize=8.4, color=INK2)
    ax.text(-3.75, yb + 1.05, "from that round's training cells only — never the test cells",
            va="center", fontsize=8.2, color=MUTED)

    # totals
    tot_ann = sum(ann.values())
    ax.text(bx_ann + bw / 2, yb + 0.25, f"{tot_ann} cell + {n_legacy} legacy", ha="center",
            fontsize=8.6, color=INK2, va="center")
    ax.text(bx_fac + bw / 2, yb + 0.25, f"{sum(fac.values())} from {len(fac)} cells", ha="center",
            fontsize=8.6, color=INK2, va="center")

    leg = [Patch(fc=TRAIN, label="train"), Patch(fc=VAL, label="validation (picks the epoch)"),
           Patch(fc=TEST, label="test — number = that cell's Test DSC"),
           Patch(fc=HIGH, label="high HER2"), Patch(fc=LOW, label="low HER2")]
    ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.03, 0.005), ncol=5, frameon=False,
              fontsize=9, handlelength=1.4)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=190, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    print(f"-> {a.output}")


if __name__ == "__main__":
    main()
