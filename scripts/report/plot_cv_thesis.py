"""
Thesis figure and numbers for Section 4.4 (Cross-Validation of the Segmentation Models), 2026-11-09.

    python scripts/report/plot_cv_thesis.py --img ../masters_thesis/main_report/images/png

Reads only the per-cell score tables of the cross-validations (trapped-cell Dice over each test cell's 20 test
frames, native 1228 x 922 resolution) under results/randswap/:
  experiment_0_cv_baseline      U-Net, five-fold
  experiment_1_nnunet           nnU-Net 100 epochs, five-fold
  experiment_1b_nnunet_1000ep   nnU-Net 1000 epochs, five-fold
  experiment_2_ensemble         U-Net + nnU-Net (100 epochs), five-fold
  experiment_3_distill/cv       distilled U-Net, five-fold
  experiment_4_loocv/{unet,nnunet}   leave-one-cell-out
  experiment_4_loocv/nnunet_r14_rerun   repeat of round 14 (high-10)
and the latencies in experiment_1_nnunet/latency_ms.json.
Prints the summary of every run and the paired Wilcoxon tests, and writes fig_ch4_cv.png: Dice of every cell for
both networks in both cross-validations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent.parent
R = ROOT / "results" / "randswap"
RUNS = {"U-Net, five-fold": "experiment_0_cv_baseline/cv_baseline_cv_cells.csv",
        "nnU-Net 100 epochs, five-fold": "experiment_1_nnunet/nnunet_cv_cells.csv",
        "nnU-Net, five-fold": "experiment_1b_nnunet_1000ep/nnunet_cv_cells.csv",
        "Ensemble, five-fold": "experiment_2_ensemble/ensemble_cv_cells.csv",
        "Distilled U-Net, five-fold": "experiment_3_distill/cv/student_cv_cv_cells.csv",
        "U-Net, leave-one-cell-out": "experiment_4_loocv/unet/loocv_baseline_cv_cells.csv",
        "nnU-Net, leave-one-cell-out": "experiment_4_loocv/nnunet/nnunet_cv_cells.csv"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
UNET, NNUNET = "#5546c4", "#b8336a"          # validated pair (dataviz validate_palette.js, light)

for f in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyretermes-*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})


def label(c):
    g, n = c.replace("_cell", "-").split("-")
    return f"{g}-{n}"


def order(c):
    return (not c.startswith("high"), int(c.split("cell")[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, required=True)
    a = ap.parse_args()

    d = {k: pd.read_csv(R / v).set_index("cell") for k, v in RUNS.items()}
    print(f"{'run':32s} {'per-cell mean':>15s} {'pooled':>7s} {'high':>6s} {'low':>6s} {'prec':>6s} {'rec':>6s}  lowest")
    for k, x in d.items():
        hi, lo = x[x.her2 == "high"].dsc.mean(), x[x.her2 == "low"].dsc.mean()
        pooled = 2 * x.tp.sum() / (2 * x.tp.sum() + x.fp.sum() + x.fn.sum())
        print(f"{k:32s} {x.dsc.mean():.3f} +- {x.dsc.std():.3f} {pooled:7.3f} {hi:6.3f} {lo:6.3f} "
              f"{x.precision.mean():6.3f} {x.recall.mean():6.3f}  {label(x.dsc.idxmin())} {x.dsc.min():.3f}")

    D = pd.DataFrame({k: v.dsc for k, v in d.items()})
    print("\n== Paired comparisons (Wilcoxon signed-rank, 15 cells; cells differing by more than 0.01)")
    for p, q in [("nnU-Net, five-fold", "U-Net, five-fold"),
                 ("nnU-Net, five-fold", "nnU-Net 100 epochs, five-fold"),
                 ("Ensemble, five-fold", "nnU-Net 100 epochs, five-fold"),
                 ("Distilled U-Net, five-fold", "U-Net, five-fold"),
                 ("U-Net, leave-one-cell-out", "U-Net, five-fold"),
                 ("nnU-Net, leave-one-cell-out", "nnU-Net, five-fold"),
                 ("nnU-Net, leave-one-cell-out", "U-Net, leave-one-cell-out")]:
        x = D[p] - D[q]
        print(f"  {p} vs {q}: mean {x.mean():+.3f}, higher on {(x > 0.01).sum()}, lower on {(x < -0.01).sum()}, "
              f"p = {wilcoxon(D[p], D[q]).pvalue:.4f}")
    o = D.drop("high_cell10")
    print(f"  nnU-Net without high-10: leave-one-cell-out {o['nnU-Net, leave-one-cell-out'].mean():.3f}, five-fold "
          f"{o['nnU-Net, five-fold'].mean():.3f}, largest difference "
          f"{(o['nnU-Net, leave-one-cell-out'] - o['nnU-Net, five-fold']).abs().max():.3f}")
    for line in open(R / "experiment_4_loocv" / "nnunet_r14_rerun" / "eval_log_native.jsonl"):
        r = json.loads(line)
        if r.get("filter") == "high_cell10":
            m = r["metrics"]["1"]
            print(f"  round 14 repeated: high-10 Dice {m['dsc']:.3f}, precision {m['precision']:.3f}, "
                  f"recall {m['recall']:.3f}")
    print("\n== Time per frame (ms):", json.load(open(R / "experiment_1_nnunet" / "latency_ms.json")))

    # ---------------- figure ----------------
    cells = sorted(D.index, key=order)
    y = np.arange(len(cells))[::-1].astype(float)
    y[[i for i, c in enumerate(cells) if c.startswith("low")]] -= 0.6       # gap between the HER2 groups
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    for net, col, dy in (("U-Net", UNET, 0.14), ("nnU-Net", NNUNET, -0.14)):
        f5, lo = D[f"{net}, five-fold"][cells].values, D[f"{net}, leave-one-cell-out"][cells].values
        for yy, p, q in zip(y + dy, f5, lo):
            ax.plot([p, q], [yy, yy], color=col, lw=2, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.scatter(f5, y + dy, s=26, facecolor="white", edgecolor=col, lw=1.3, zorder=3)
        ax.scatter(lo, y + dy, s=26, color=col, edgecolor="white", lw=0.6, zorder=4)
    ax.set_yticks(y, [label(c) for c in cells], fontsize=8)
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(y.min() - 0.7, y.max() + 0.7)
    ax.set_xlabel("Dice, trapped cell", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    yh = y[cells.index("high_cell10")] - 0.14
    ax.annotate("labels the neighbor as the cell\nin 9 of 20 frames", xy=(0.725, yh), xytext=(0.615, yh - 1.5),
                fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.legend(handles=[Line2D([], [], color=UNET, lw=2, label="U-Net"),
                       Line2D([], [], color=NNUNET, lw=2, label="nnU-Net"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, ms=5, label="Five-fold"),
                       Line2D([], [], marker="o", ls="", color=INK2, ms=5, label="Leave-one-cell-out")],
              loc="upper center", bbox_to_anchor=(0.45, -0.11), ncol=4, fontsize=7.5, frameon=False)
    a.img.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.img / "fig_ch4_cv.png", dpi=400, bbox_inches="tight", facecolor="white")
    print(f"\n-> {a.img / 'fig_ch4_cv.png'}")


if __name__ == "__main__":
    main()
