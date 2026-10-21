#!/usr/bin/env python3
"""
Dataset Organization figure, general form (15 October report): one rule, three protocols.

Every protocol the project uses applies the same rule -- each cell is tested exactly once, by a
model that never saw it -- and differs only in how many cells a round tests:

  * 5-fold cross-validation   11 train : 1 val : 3 test, 5 rounds, stratified by HER2
  * leave-one-cell-out        13 train : 1 val : 1 test, 15 rounds
  * deployment                all 15 cells train one model (fixed epochs, no test)

Each protocol is drawn as a cells x rounds grid, rows sorted by the round that tests the cell, so
the test cells form a staircase (3-cell steps for 5-fold, a diagonal for leave-one-out). The split
is a property of the protocol, shared by every model trained under it -- no scores are drawn, so
the figure does not belong to any one run.

Read from disk: the 5-fold and deployment split records (results/<exp>/<run>_fold_*_split.json) and
the leave-one-out round table its summary job printed (logs/hpc/loocv_unet_summary_*.out; the
split files of that run stay on the cluster).

    python scripts/report/plot_protocols.py --output ../masters_thesis/images/protocols.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

REPO = Path(__file__).resolve().parent.parent.parent
KFOLD = ("randswap/experiment_0_cv_baseline", "cv_baseline")
LOOCV_LOG = "loocv_unet_summary_*.out"

# The validated role / HER2 colours of plot_data_map.py (scripts/validate_palette.js).
TRAIN, VAL, TEST = "#e3e2dd", "#4a3aa7", "#008300"
HIGH, LOW = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
LEGACY = "#efeeea"

# Which experiments ran under each protocol -- a description of the report, not a measurement.
USED_BY = {
    "kfold": ["EXP 4  U-Net", "EXP 5, 5b  nnU-Net", "EXP 6  ensemble", "EXP 7  MegaNet"],
    "loocv": ["EXP 8  U-Net", "EXP 8  nnU-Net"],
    "deploy": ["U-Net deployment model", "nnU-Net, all 15 (features)"],
}


def short(cell: str) -> str:
    return cell.replace("_cell", "-")


def kfold_rounds() -> tuple[list[dict], dict, dict]:
    R = REPO / "results" / KFOLD[0]
    recs = [json.loads(p.read_text()) for p in R.glob(f"{KFOLD[1]}_fold_*_split.json")]
    deploy = next(r for r in recs if r["fold"] == "deploy")
    recs = sorted((r for r in recs if r["fold"] != "deploy"), key=lambda r: int(r["fold"]))
    rounds = [{"test": r["cells"]["test"], "val": r["cells"]["val"]} for r in recs]
    return rounds, recs[0]["her2"], deploy


def loocv_rounds() -> list[dict]:
    log = sorted((REPO / "logs" / "hpc").glob(LOOCV_LOG))[-1]
    rows = re.findall(r"^\s+(\d+)\s+(\S+_cell\d+)\s+(\S+_cell\d+)\s+\d+", log.read_text(), re.M)
    return [{"test": [t], "val": [v]} for _, t, v in sorted(rows, key=lambda r: int(r[0]))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    kf, her2, deploy = kfold_rounds()
    lo = loocv_rounds()
    cells = sorted(her2, key=lambda c: (her2[c] != "high", int(c.split("cell")[1])))
    if sorted(c for r in kf for c in r["test"]) != sorted(cells) or sorted(c for r in lo for c in r["test"]) != sorted(cells):
        raise SystemExit("a protocol does not test every cell exactly once")
    n_legacy = json.loads(next((REPO / "results" / KFOLD[0]).glob(f"{KFOLD[1]}_fold_1_split.json")).read_text())["n_legacy_frames"]

    def order(rounds):         # rows by the round that tests the cell, high HER2 first within a round
        return [c for r in rounds for c in sorted(r["test"], key=cells.index)]

    panels = [
        ("kfold", "5-fold cross-validation", order(kf), kf,
         [f"{len(cells) - len(kf[0]['test']) - 1} train · 1 validation · {len(kf[0]['test'])} test",
          f"{len(kf)} rounds · stratified by HER2"]),
        ("loocv", "Leave-one-cell-out", order(lo), lo,
         [f"{len(cells) - 2} train · 1 validation · 1 test", f"{len(lo)} rounds · one cell per round"]),
        ("deploy", "Deployment", cells, [{"test": [], "val": []}],
         [f"all {len(deploy['cells']['train'])} cells train", "1 model · fixed epochs"]),
    ]

    cw, ch, lab = 0.25, 0.25, 0.78          # cell width / height (inch), row-label width
    gap = 0.32
    widths = [lab + len(p[3]) * cw for p in panels]
    W = sum(widths) + gap * (len(panels) - 1) + 0.3
    n_rows = len(cells) + 1
    top, body = 0.9, n_rows * ch + 0.2
    H = top + body + 1.75
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def box(x, y, w, h, fc, r=0.04, ec="none", lw=0, hatch=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw, hatch=hatch, zorder=2))

    ax.text(0.15, 0.25, "One rule: every cell is tested exactly once, by a model that never saw it",
            fontsize=10.5, weight="bold", color=INK, va="center")
    ax.text(0.15, 0.52, "Rows = cells, sorted by the round that tests them;  columns = rounds.  "
            "The same split serves every model trained under it.", fontsize=7.6, color=INK2, va="center")

    x0 = 0.15
    for (key, title, rows, rounds, spec), w in zip(panels, widths):
        gx = x0 + lab
        ax.text(x0, top - 0.02, title, fontsize=8.8, weight="bold", color=INK, va="bottom")
        y0 = top + 0.18
        for j in range(len(rounds)):     # column numbers
            if len(rounds) > 1:
                ax.text(gx + j * cw + cw / 2 - 0.02, y0 - 0.06, str(j + 1), ha="center", va="bottom",
                        fontsize=5.6 if len(rounds) > 9 else 6.4, color=MUTED)
        for i, c in enumerate(rows):
            y = y0 + i * ch
            box(x0, y + 0.05, 0.07, ch - 0.1, HIGH if her2[c] == "high" else LOW, r=0.02)
            ax.text(x0 + 0.12, y + ch / 2, short(c), fontsize=6.8, color=INK, va="center")
            for j, r in enumerate(rounds):
                fc = TEST if c in r["test"] else VAL if c in r["val"] else TRAIN
                box(gx + j * cw, y + 0.02, cw - 0.04, ch - 0.04, fc)
        y = y0 + len(rows) * ch + 0.06    # legacy row: trains in every round, of every protocol
        ax.text(x0 + 0.12, y + ch / 2, "legacy", fontsize=6.6, color=INK2, va="center", style="italic")
        for j in range(len(rounds)):
            box(gx + j * cw, y + 0.02, cw - 0.04, ch - 0.04, LEGACY, ec=MUTED, lw=0.4, hatch="////")
        ys = y + ch + 0.15
        for k, line in enumerate(spec):
            ax.text(x0, ys + k * 0.16, line, fontsize=7.0, color=INK if k == 0 else INK2, va="top",
                    weight="bold" if k == 0 else "normal")
        ax.text(x0, ys + 0.4, "used by", fontsize=6.4, color=MUTED, va="top", style="italic")
        for k, u in enumerate(USED_BY[key]):
            ax.text(x0, ys + 0.54 + k * 0.14, u, fontsize=6.6, color=INK2, va="top")
        x0 += w + gap

    leg = [Patch(fc=TRAIN, label="train"), Patch(fc=VAL, label="validation (picks the epoch only)"),
           Patch(fc=TEST, label="test (20 evenly spaced frames)"),
           Patch(fc=LEGACY, ec=MUTED, lw=0.4, hatch="////", label=f"legacy 2021, {n_legacy} frames: train only"),
           Patch(fc=HIGH, label="high HER2"), Patch(fc=LOW, label="low HER2")]
    ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=3, frameon=False,
              fontsize=7.0, handlelength=1.3, columnspacing=1.4)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=200, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    print(f"-> {a.output}")


if __name__ == "__main__":
    main()
