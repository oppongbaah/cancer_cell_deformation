"""
Thesis figure and numbers for Section 4.2 (The Dataset), 2026-11-09.

    python scripts/report/plot_dataset_thesis.py --img ../masters_thesis/main_report/images/png

Reads only what is on disk:
  data/masks/01_annotate_pool_multiclass, data/masks/02_unet_holdout_multiclass   the 654 ground-truth masks
      (label 1 = trapped cell, 2 = other object), 1228 x 922 px
  results/features/nnunet1000_loocv/features_frames.csv   image measures of every feature frame + trap height

Prints the class balance, the per-cell size of the trapped cell (ground truth) and the image measures grouped by
HER2 group and by trap position, and writes fig_ch4_dataset.png:
  (a) equivalent diameter of the trapped cell per cell (median over its ground-truth masks), by HER2 group
  (b-d) whole-frame illumination, contrast and noise per cell, grouped by trap position (A: 9 high + low-1)
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent.parent
UM_PER_PX = 0.50                          # analysed resolution (thesis 3.2.3)
HIGH, LOW, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
IQ = {"focus_frame": "Focus, whole frame", "focus_cell": "Focus, around the cell", "illum_mean": "Illumination, mean",
      "illum_gradient": "Illumination, unevenness", "contrast_rms": "Contrast, whole frame",
      "contrast_cell": "Contrast, cell against surround", "noise_sigma": "Noise"}

for f in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyretermes-*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def compare(a, b):
    """AUC (share of pairs with a > b), Mann-Whitney p, Cohen's d."""
    u = mannwhitneyu(a, b)
    return u.statistic / (len(a) * len(b)), u.pvalue, cohen_d(a, b)


def masks():
    rows = []
    for d in ("01_annotate_pool_multiclass", "02_unet_holdout_multiclass"):
        for f in sorted(os.listdir(ROOT / "data" / "masks" / d)):
            a = np.array(Image.open(ROOT / "data" / "masks" / d / f))
            m = re.search(r"(high|low)_cell(\d+)", f)
            rows.append({"cell": f"{m.group(1)}-{m.group(2)}" if m else "legacy", "n": a.size,
                         "cell_px": int((a == 1).sum()), "other_px": int((a == 2).sum())})
    return pd.DataFrame(rows)


def balance(df, name):
    n, c, o = df.n.sum(), df.cell_px.sum(), df.other_px.sum()
    print(f"{name:9s} frames={len(df):3d}  background {100 * (n - c - o) / n:.2f} %  trapped cell {100 * c / n:.2f} %  "
          f"other {100 * o / n:.2f} %  frames with another object {(df.other_px > 0).sum()} "
          f"({100 * (df.other_px > 0).mean():.0f} %)  frames without a cell {(df.cell_px == 0).sum()}  "
          f"median cell area {df.cell_px.median():.0f} px")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, required=True)
    a = ap.parse_args()

    m = masks()
    print("== Class balance of the ground-truth masks")
    balance(m, "all")
    balance(m[m.cell != "legacy"], "recorded")
    balance(m[m.cell == "legacy"], "legacy")

    rec = m[m.cell != "legacy"]
    area = rec.groupby("cell").cell_px.median()
    diam = 2 * np.sqrt(area * UM_PER_PX ** 2 / np.pi)
    hi = [c for c in diam.index if c.startswith("high")]
    lo = [c for c in diam.index if c.startswith("low")]
    auc, p, d = compare(diam[hi], diam[lo])
    print("\n== Trapped-cell size per cell (median of its ground-truth masks)")
    for c in sorted(diam.index, key=lambda s: (s[0] != "h", int(s.split("-")[1]))):
        print(f"  {c:8s} area {area[c]:7.1f} px  diameter {diam[c]:.1f} um")
    print(f"  high {diam[hi].min():.1f}-{diam[hi].max():.1f} um (median {diam[hi].median():.1f}), "
          f"low {diam[lo].min():.1f}-{diam[lo].max():.1f} um (median {diam[lo].median():.1f}); "
          f"AUC {auc:.2f}, Mann-Whitney p {p:.3f}, Cohen's d {d:.2f}")

    fr = pd.read_csv(ROOT / "results" / "features" / "nnunet1000_loocv" / "features_frames.csv")
    per = fr.groupby("cell").agg(her2=("her2", "first"), ty=("trap_y", "first"), **{k: (k, "mean") for k in IQ})
    gap = np.sort(per.ty.values)
    cut = gap[np.argmax(np.diff(gap))] + np.max(np.diff(gap)) / 2
    per["posA"] = per.ty > cut
    print(f"\n== Image measures per cell (mean over feature frames); position A = trap y > {cut:.0f} px: "
          f"{', '.join(per.index[per.posA])}")
    print(f"{'measure':34s} {'high':>8s} {'low':>8s} {'AUC HER2':>9s} {'p':>7s} {'AUC pos':>8s} {'p':>7s}")
    for k, lab in IQ.items():
        h, l = per[per.her2 == "high"][k], per[per.her2 == "low"][k]
        A, B = per[per.posA][k], per[~per.posA][k]
        a1, p1, _ = compare(h, l)
        a2, p2, _ = compare(A, B)
        print(f"{lab:34s} {h.mean():8.4g} {l.mean():8.4g} {a1:9.2f} {p1:7.3f} {a2:8.2f} {p2:7.4f}")

    # ---------------- figure ----------------
    fig, axs = plt.subplots(1, 4, figsize=(6.5, 2.35), gridspec_kw={"wspace": 0.75, "width_ratios": [1, 1, 1, 1]})
    rng = np.random.default_rng(0)

    ax = axs[0]
    for x, (cells, col) in enumerate(((hi, HIGH), (lo, LOW))):
        v = diam[cells].values
        xs = x + rng.uniform(-0.14, 0.14, len(v))
        ax.scatter(xs, v, s=16, color=col, lw=0, alpha=0.9, zorder=3)
        ax.plot([x - 0.28, x + 0.28], [np.median(v)] * 2, color=INK2, lw=1.6, zorder=2)
        if "low-1" in cells:
            j = cells.index("low-1")
            ax.scatter(xs[j], v[j], s=55, facecolor="none", edgecolor=INK, lw=0.9, zorder=4)
    ax.set_xticks([0, 1], ["High", "Low"], fontsize=8)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xlabel("HER2 group", fontsize=8)
    ax.set_ylabel("Equivalent diameter (µm)", fontsize=8)
    ax.set_title("(a) Cell size", fontsize=8.5, color=INK, loc="left")
    ax.tick_params(labelsize=7.5)

    for ax, k, t in zip(axs[1:], ("illum_mean", "contrast_rms", "noise_sigma"),
                        ("(b) Illumination", "(c) Contrast", "(d) Noise")):
        for x, grp in enumerate((per[per.posA], per[~per.posA])):
            xs = x + rng.uniform(-0.14, 0.14, len(grp))
            ax.scatter(xs, grp[k], s=16, c=[HIGH if h == "high" else LOW for h in grp.her2], lw=0, alpha=0.9,
                       zorder=3)
            ax.plot([x - 0.28, x + 0.28], [grp[k].median()] * 2, color=INK2, lw=1.6, zorder=2)
            if "low_cell1" in grp.index:
                j = list(grp.index).index("low_cell1")
                ax.scatter(xs[j], grp.loc["low_cell1", k], s=55, facecolor="none", edgecolor=INK, lw=0.9, zorder=4)
        ax.set_xticks([0, 1], ["A", "B"], fontsize=8)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xlabel("Trap position", fontsize=8)
        ax.set_title(t, fontsize=8.5, color=INK, loc="left")
        ax.tick_params(labelsize=7.5)
    axs[1].set_ylabel("Mean gray level", fontsize=8)
    axs[2].set_ylabel("Relative contrast", fontsize=8)
    axs[3].set_ylabel("Noise level", fontsize=8)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=HIGH, ms=4.5, label="High HER2 cell"),
                        Line2D([], [], marker="o", ls="", color=LOW, ms=4.5, label="Low HER2 cell"),
                        Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=6.5, label="low-1"),
                        Line2D([], [], color=INK2, lw=1.6, label="Median")],
               loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=7.5, frameon=False)
    a.img.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.img / "fig_ch4_dataset.png", dpi=400, bbox_inches="tight", facecolor="white")
    print(f"\n-> {a.img / 'fig_ch4_dataset.png'}")


if __name__ == "__main__":
    main()
