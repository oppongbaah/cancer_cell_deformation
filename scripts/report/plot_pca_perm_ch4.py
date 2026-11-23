"""
Thesis figures and numbers for Section 4.6: the permutation test and the principal components, 2026-11-09.

    python scripts/report/plot_pca_perm_ch4.py --img ../masters_thesis/main_report/images/png

Reads only results/classification/nnunet1000_loocv/:
  pca_null.csv      1,000 label shuffles x 36 PCA settings (balanced accuracy, AUC), whole procedure rerun
  pca_metrics.csv   the real scores of the 36 settings (and their p, p counted over the 36)
  pca_variance.csv  eigenvalues of the three starting sets against 5,000 draws of pure noise (parallel analysis)
  pca_loadings.csv  loadings on all 15 cells
Writes fig_ch4_permutation.png and fig_ch4_pca.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent.parent
R = ROOT / "results" / "classification" / "nnunet1000_loocv"
BEST = "Step LR + LR, selected 4, PC1"
INK, INK2, GRID, ACC = "#0b0b0b", "#52514e", "#e6e5e1", "#5546c4"
SETS = {"all 29": "All 29 features", "reliable 9": "Reliable 9", "selected 4": "Selected 4"}

for f in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyretermes-*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})


def style(ax):
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def permutation(a):
    null = pd.read_csv(R / "pca_null.csv")
    pm = pd.read_csv(R / "pca_metrics.csv").set_index("model")
    real = pm.loc[BEST, "balanced_acc"]
    nb = null[null.model == BEST].balanced_acc.to_numpy()
    mx = null.groupby("shuffle").balanced_acc.max().to_numpy()
    p1 = (1 + (nb >= real - 1e-9).sum()) / (1 + len(nb))
    p2 = (1 + (mx >= real - 1e-9).sum()) / (1 + len(mx))
    print(f"== Permutation: real balanced accuracy {real:.3f}; shuffles {len(nb)}; median {np.median(nb):.2f}, "
          f"95th pct {np.percentile(nb, 95):.2f}; at or above real {(nb >= real - 1e-9).sum()} -> p {p1:.3f} "
          f"(logged {pm.loc[BEST, 'p_balanced_acc']:.3f}); best of 36 per shuffle: median {np.median(mx):.2f}, "
          f"95th pct {np.percentile(mx, 95):.2f}, at or above real {(mx >= real - 1e-9).sum()} -> p {p2:.3f} "
          f"(logged {pm.loc[BEST, 'p_counted']:.3f})")

    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), sharey=True, gridspec_kw={"wspace": 0.12})
    bins = np.linspace(0, 1, 37)
    for ax, x, t, p in ((axs[0], nb, "(a) The final classifier", p1),
                        (axs[1], mx, "(b) Best of the 36 settings, per shuffle", p2)):
        h, e = np.histogram(x, bins)
        c = (e[:-1] + e[1:]) / 2
        ax.bar(c, h, width=e[1] - e[0] - 0.004, color=[ACC if v >= real - 1e-9 else "#b9b8b3" for v in c], lw=0,
               zorder=2)
        ax.axvline(real, color=INK, lw=1.0, zorder=3)
        ax.text(real - 0.02, h.max() * 0.95, f"real {real:.2f}\n$p$ = {p:.3f}", ha="right", va="top", fontsize=7,
                color=INK)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Balanced accuracy", fontsize=8)
        ax.set_title(t, fontsize=8.5, color=INK, loc="left")
        style(ax)
    axs[0].set_ylabel("Shuffles", fontsize=8)
    fig.savefig(a.img / "fig_ch4_permutation.png", dpi=400, bbox_inches="tight", facecolor="white")
    print(f"-> {a.img / 'fig_ch4_permutation.png'}")


def pca(a):
    var = pd.read_csv(R / "pca_variance.csv")
    pm = pd.read_csv(R / "pca_metrics.csv").set_index("model")
    load = pd.read_csv(R / "pca_loadings.csv")
    print("\n== Components above noise:", var[var.above_noise].groupby("set").component.max().to_dict())
    print(var[var.component <= 2][["set", "component", "explained"]].round(3).to_string(index=False))
    print(load[(load.set == "selected 4")].pivot(index="feature", columns="component", values="weight").round(2))

    fig = plt.figure(figsize=(6.5, 4.4))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.32)
    for j, (s, lab) in enumerate(SETS.items()):
        ax = fig.add_subplot(gs[0, j])
        v = var[var.set == s].query("component <= 6")
        ax.fill_between(v.component, v.noise_p5, v.noise_p95, color=GRID, lw=0, zorder=1)
        ax.plot(v.component, v.noise_mean, color=INK2, lw=0.8, ls="--", zorder=2)
        ax.scatter(v.component, v.eigenvalue, s=22, zorder=3,
                   c=[ACC if x else "white" for x in v.above_noise], edgecolor=ACC, lw=1)
        for _, r in v[v.above_noise].iterrows():
            ax.text(r.component + 0.25, r.eigenvalue, f"{100 * r.explained:.0f}%", fontsize=6.5, va="center",
                    color=INK)
        ax.set_xticks(range(1, int(v.component.max()) + 1))
        ax.set_xlabel("Component", fontsize=8)
        ax.set_title(f"({'abc'[j]}) {lab}", fontsize=8.5, color=INK, loc="left")
        style(ax)
    fig.axes[0].set_ylabel("Eigenvalue", fontsize=8)

    gb = gs[1, :].subgridspec(1, 2, width_ratios=[1.55, 1], wspace=0.55)
    bx = fig.add_subplot(gb[0, 0])
    for s, mk in zip(SETS, ("s", "D", "o")):
        ks, ys, sig = [], [], []
        for k in range(0, 7):
            name = f"Step LR + LR, {s}, " + ("raw" if k == 0 else "PC1" if k == 1 else f"PC1-{k}")
            if name in pm.index:
                ks.append(k), ys.append(pm.loc[name, "correct"]), sig.append(pm.loc[name, "p_balanced_acc"] < 0.05)
        bx.plot(ks, ys, color=INK2, lw=0.7, zorder=2)
        bx.scatter(ks, ys, marker=mk, s=26, zorder=3, c=[INK if q else "white" for q in sig], edgecolor=INK, lw=1,
                   label=SETS[s])
    bx.set_xticks(range(0, 7), ["none"] + [str(k) for k in range(1, 7)])
    bx.set_xlabel("Components kept (none: the features themselves)", fontsize=8)
    bx.set_ylabel("Cells right (of 15)", fontsize=8)
    bx.set_ylim(9.5, 14.5)
    bx.set_title("(d) Interval LR + LR with the components", fontsize=8.5, color=INK, loc="left")
    style(bx)
    bx.legend(loc="upper right", fontsize=7, frameon=False, ncol=1)

    cx = fig.add_subplot(gb[0, 1])
    L = load[load.set == "selected 4"].pivot(index="feature", columns="component", values="weight")
    order = ["len_par_base", "fluctuation", "volume_change", "orient_spread"]
    L = L.loc[order]
    for k in L.columns:                                      # sign so that the largest weight is positive
        if L[k].iloc[L[k].abs().argmax()] < 0:
            L[k] = -L[k]
    im = cx.imshow(L.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            cx.text(j, i, f"{L.values[i, j]:.2f}", ha="center", va="center", fontsize=6.3,
                    color="white" if abs(L.values[i, j]) > 0.6 else INK)
    cx.set_xticks(range(L.shape[1]), [f"PC{k}" for k in L.columns], fontsize=7)
    cx.set_yticks(range(4), ["Length", "Fluctuation", "Volume ch.", "Orient. spread"], fontsize=7)
    cx.set_title("(e) Loadings, selected 4", fontsize=8.5, color=INK, loc="left")
    for sp in cx.spines.values():
        sp.set_visible(False)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", mfc=ACC, mec=ACC, ms=5, label="Above noise"),
                        Line2D([], [], marker="o", ls="", mfc="white", mec=ACC, ms=5, label="Within noise"),
                        Line2D([], [], color=INK2, lw=0.8, ls="--", label="Noise mean, 5th to 95th pct."),
                        Line2D([], [], marker="o", ls="", mfc=INK, mec=INK, ms=5, label="$p$ < 0.05"),
                        Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=5, label="$p$ ≥ 0.05")],
               loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5, fontsize=7, frameon=False)
    fig.savefig(a.img / "fig_ch4_pca.png", dpi=400, bbox_inches="tight", facecolor="white")
    print(f"-> {a.img / 'fig_ch4_pca.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, required=True)
    a = ap.parse_args()
    a.img.mkdir(parents=True, exist_ok=True)
    permutation(a)
    pca(a)


if __name__ == "__main__":
    main()
