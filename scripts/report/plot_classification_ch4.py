"""
Thesis figure and numbers for Section 4.6 (HER2 Classification), 2026-11-09.

    python scripts/report/plot_classification_ch4.py --img ../masters_thesis/main_report/images/png

Reads only results/classification/nnunet1000_loocv/ (leave-one-cell-out, 15 cells):
  pca_cells.csv / pca_metrics.csv / pca_csweep.csv   the final classifier "Step LR + LR, selected 4, PC1"
  combine_cells.csv / combine_metrics.csv            the size and session controls (and every other model)
  benchmark_cells.csv                                permutation p of the single models and controls
  feature_selection.csv + features_cells.csv          PC1 loadings refitted in each fold (no labels)
Writes fig_ch4_classifier.png:
  (a) calls against the truth, (b) ROC against the size control, (c) accuracy and F1 per group against size,
  (d) classification report (the thesis version of Fig. 1 of the classification report)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
R = ROOT / "results" / "classification" / "nnunet1000_loocv"
BEST = "Step LR + LR, selected 4, PC1"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
HIGH, LOW = "#2a78d6", "#eb6834"

for f in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyretermes-*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})


def label(c):
    return c.replace("_cell", "-")


def pc1_stability():
    """PC1 of the selected 4 refitted on the 14 training cells of each fold, as the classifier does."""
    warnings.simplefilter("ignore")
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("combine", ROOT / "scripts/classification/combine_models.py")
    cm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cm)
    cells = cm.bench.load("nnunet1000_loocv")
    sel = pd.read_csv(R / "feature_selection.csv")
    f4 = sel[sel.selected].sort_values("rank").feature.tolist()
    X = cells[f4].to_numpy(float)

    def pc1(X):
        Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
        w = PCA().fit(Z).components_[0]
        return w if w.sum() > 0 else -w

    w_all = pc1(X)
    print(f"\n== PC1 loadings ({', '.join(f4)}): all 15 cells {np.round(w_all, 2)}")
    for i, c in enumerate(cells.cell):
        w = pc1(X[np.arange(len(X)) != i])
        w = w if w @ w_all > 0 else -w
        cos = w @ w_all / np.linalg.norm(w) / np.linalg.norm(w_all)
        print(f"   without {label(c):8s} {np.round(w, 2)}  angle to all-15 PC1 {np.degrees(np.arccos(min(cos, 1))):5.1f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, required=True)
    a = ap.parse_args()

    pm = pd.read_csv(R / "pca_metrics.csv").set_index("model")
    cmx = pd.read_csv(R / "combine_metrics.csv").set_index("model")
    print("== Final classifier:", pm.loc[BEST].to_dict())
    print("== C sweep:\n", pd.read_csv(R / "pca_csweep.csv").query("model == @BEST")[["C", "correct", "missed"]]
          .to_string(index=False))
    for m in ("Size only", "Image only (session)", "LR every laser step + LR, selected 4"):
        print(f"== {m}:", cmx.loc[m, ["correct", "acc_high", "acc_low", "f1_high", "f1_low", "auc", "missed"]].to_dict())
    pc1_stability()

    pb = pd.read_csv(R / "pca_cells.csv", index_col=0)[BEST]
    cc = pd.read_csv(R / "combine_cells.csv", index_col=0)
    ps, pi = cc["Size only"], cc["Image only (session)"]
    cells = sorted(pb.index, key=lambda c: (not c.startswith("high"), int(c.split("cell")[1])))
    y = np.array([c.startswith("high") for c in cells]).astype(int)
    print("\n== Held-out P(high): cell, final, size, session")
    for c in cells:
        print(f"   {label(c):8s} {pb[c]:.3f} {ps[c]:.3f} {pi[c]:.3f}")

    from sklearn.metrics import auc as auc_of, precision_recall_fscore_support
    pred = np.array([pb[c] >= 0.5 for c in cells]).astype(int)
    psz = np.array([ps[c] >= 0.5 for c in cells]).astype(int)
    fig = plt.figure(figsize=(6.5, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.62], width_ratios=[0.8, 1, 1.15], hspace=0.55, wspace=0.55)

    # (a) calls against the truth
    ax = fig.add_subplot(gs[0, 0])
    cmx = np.array([[((y == t) & (pred == q)).sum() for q in (1, 0)] for t in (1, 0)])
    n = cmx.sum(axis=1, keepdims=True)
    ax.imshow(cmx / n, cmap="Purples", vmin=0, vmax=1.25)
    for i in range(2):
        for j in range(2):
            frac = cmx[i, j] / n[i, 0]
            ax.text(j, i - 0.08, f"{cmx[i, j]}", ha="center", va="center", fontsize=13,
                    color="white" if frac > 0.6 else INK)
            ax.text(j, i + 0.25, f"of {n[i, 0]}", ha="center", va="center", fontsize=6.5,
                    color="white" if frac > 0.6 else INK2)
    ax.set_xticks([0, 1], ["High", "Low"], fontsize=7.5)
    ax.set_yticks([0, 1], ["High", "Low"], fontsize=7.5)
    ax.set_xlabel("Called", fontsize=8)
    ax.set_ylabel("True HER2 group", fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"(a) {int((pred == y).sum())} of 15 cells", fontsize=8.5, color=INK, loc="left")

    # (b) ROC
    bx = fig.add_subplot(gs[0, 1])
    for p_, lab, lw, col in ((pb, "Final classifier", 1.6, INK), (ps, "Size control", 1.0, INK2)):
        fpr, tpr, thr = roc_curve(y, [p_[c] for c in cells])
        bx.plot(fpr, tpr, lw=lw, color=col, label=f"{lab} ({auc_of(fpr, tpr):.2f})", drawstyle="steps-post")
    fp = ((pred == 1) & (y == 0)).sum() / (y == 0).sum()
    tp = ((pred == 1) & (y == 1)).sum() / (y == 1).sum()
    bx.scatter([fp], [tp], s=22, color=INK, zorder=4)
    bx.annotate("cut at 0.5", (fp, tp), xytext=(0.22, 0.62), fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    bx.plot([0, 1], [0, 1], color=GRID, lw=0.8)
    bx.set_xlim(-0.02, 1.02), bx.set_ylim(-0.02, 1.04)
    bx.set_xlabel("Low cells called high", fontsize=8)
    bx.set_ylabel("High cells called high", fontsize=8)
    bx.legend(loc="lower right", fontsize=6.8, frameon=False, title="AUC", title_fontsize=6.8)
    bx.tick_params(labelsize=7.5)
    bx.set_title("(b) Ranking", fontsize=8.5, color=INK, loc="left")

    # (c) per-class accuracy and F1 against size
    cx = fig.add_subplot(gs[0, 2])
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    prs, rcs, f1s, _ = precision_recall_fscore_support(y, psz, labels=[1, 0], zero_division=0)
    vals = [rc[0], rc[1], f1[0], f1[1]]
    size = [rcs[0], rcs[1], f1s[0], f1s[1]]
    labs = ["Acc. high", "Acc. low", "F1 high", "F1 low"]
    yy = np.arange(4)[::-1]
    cx.barh(yy, vals, height=0.6, color=[HIGH, LOW, HIGH, LOW], zorder=2)
    for yv, v, sv in zip(yy, vals, size):
        cx.plot([sv, sv], [yv - 0.36, yv + 0.36], color=INK, lw=1.6, zorder=3)
        cx.text(v + 0.03, yv, f"{v:.2f}", va="center", fontsize=7, color=INK)
    cx.axvline(0.85, color=INK2, lw=0.8, ls="--", zorder=1)
    cx.set_yticks(yy, labs, fontsize=7.5)
    cx.set_xlim(0, 1.2)
    cx.set_xticks([0, 0.5, 1.0])
    cx.tick_params(axis="x", labelsize=7.5)
    cx.set_title("(c) Per group", fontsize=8.5, color=INK, loc="left")
    cx.legend(handles=[Line2D([], [], color=INK, lw=1.6, label="Size control"),
                       Line2D([], [], color=INK2, lw=0.8, ls="--", label="Criterion 0.85")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=6.8, frameon=False, ncol=2)

    # (d) classification report
    dx = fig.add_subplot(gs[1, :])
    dx.axis("off")
    dx.set_title("(d) Classification report at the cut of 0.5 (recall = accuracy of the group)", fontsize=8.5,
                 color=INK, loc="left")
    acc = (pred == y).mean()
    w = np.array([(y == 1).sum(), (y == 0).sum()]) / len(y)
    rows = [("High HER2", f"{pr[0]:.2f}", f"{rc[0]:.2f}", f"{f1[0]:.2f}", str((y == 1).sum())),
            ("Low HER2", f"{pr[1]:.2f}", f"{rc[1]:.2f}", f"{f1[1]:.2f}", str((y == 0).sum())),
            ("Accuracy", "", "", f"{acc:.2f}", "15"),
            ("Macro average", f"{pr.mean():.2f}", f"{rc.mean():.2f}", f"{f1.mean():.2f}", "15"),
            ("Weighted average", f"{pr @ w:.2f}", f"{rc @ w:.2f}", f"{f1 @ w:.2f}", "15")]
    xs = [0.30, 0.48, 0.60, 0.72, 0.84]
    for x, h in zip(xs, ("", "Precision", "Recall", "F1", "Cells")):
        dx.text(x, 0.86, h, ha="right" if x == xs[0] else "center", fontsize=7.5, color=INK2)
    dx.plot([0.12, 0.90], [0.79, 0.79], color=INK2, lw=0.6, transform=dx.transAxes)
    for i, r in enumerate(rows):
        yv = 0.66 - i * 0.15 - (0.04 if i >= 2 else 0)
        for x, t in zip(xs, r):
            dx.text(x, yv, t, ha="right" if x == xs[0] else "center", fontsize=7.5,
                    color=(HIGH if i == 0 else LOW if i == 1 else INK) if x == xs[0] else INK)
    dx.plot([0.12, 0.90], [0.405, 0.405], color=GRID, lw=0.6, transform=dx.transAxes)
    dx.set_xlim(0, 1), dx.set_ylim(0, 1)
    print(f"report: precision {pr.round(2)}, recall {rc.round(2)}, F1 {f1.round(2)}, acc {acc:.2f}")
    a.img.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.img / "fig_ch4_classifier.png", dpi=400, bbox_inches="tight", facecolor="white")
    print(f"-> {a.img / 'fig_ch4_classifier.png'}")


if __name__ == "__main__":
    main()
