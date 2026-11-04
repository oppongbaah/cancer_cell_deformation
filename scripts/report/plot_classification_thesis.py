"""
Thesis-style figures for the HER2 classification chapter (rewrite of 2026-11-03).

    python scripts/report/plot_classification_thesis.py --model nnunet1000_loocv --img DIR

Reads only results already on disk (results/classification/<model>/): combine_cells.csv / combine_metrics.csv
(every model's held-out P(high) per cell), benchmark_cells.csv (permutation p of the 16 machine-learning models),
scores_cells.csv (their P(high) per cell), rank_nested.csv, seq_metrics.csv, tuning_scores.csv / tuning_cells.csv,
combine_null.csv (the deployment model's permutation null). Nothing is refitted.

Figures
    cls_landscape.png    every model of the study on one page: AUC with its 95 % cell-bootstrap interval and
                         F1 of each class, grouped by family; filled marker = permutation p < 0.05
    cls_percell_map.png  held-out P(high) of the finalists and the two controls for every cell, cells ordered by size
    cls_deploy_null.png  what chance gives the deployment model: 1 000 label shuffles of the whole procedure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
HIGH, LOW, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
FAM = {"Controls": "#8a8985", "Deep learning": "#b02a2a", "Machine learning, label-free 4": "#1b7a3a",
       "Machine learning, all 29": "#12805f", "Chosen with the labels": "#a56c00", "Frame / step level": "#00687f",
       "Ensembles": "#4a3aa7"}
DEPLOY = "LR every laser step + LR, label-free 4"
TUNED_SVM = "Linear SVM, tuned (nested C)"
SHORT = {"Logistic regression": "LR", "Linear SVM": "linear SVM", "Shrinkage LDA": "LDA", "RBF SVM": "RBF SVM",
         "k-NN (k=3)": "k-NN", "Random forest": "random forest", "Gradient boosting": "gradient boosting",
         "MLP (ANN)": "ANN"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def boot_auc(y, p, n=2000, seed=0):
    rng, out = np.random.default_rng(seed), []
    while len(out) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            out.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def load(res: Path):
    cc = pd.read_csv(res / "combine_cells.csv", index_col=0)
    cells = list(cc.index)
    y = np.array([1 if c.startswith("high") else 0 for c in cells])
    P, p = {}, {}

    def add(name, probs, pv=np.nan, fam=""):
        P[name] = (np.asarray(probs, float), fam)
        p[name] = pv

    cm = pd.read_csv(res / "combine_metrics.csv").set_index("model")
    bench = pd.read_csv(res / "benchmark_cells.csv")
    sc = pd.read_csv(res / "scores_cells.csv")
    seq = pd.read_csv(res / "seq_metrics.csv")
    nested = pd.read_csv(res / "rank_nested.csv").set_index("classifier")
    tun = pd.read_csv(res / "tuning_cells.csv").set_index("classifier")
    tsc = pd.read_csv(res / "tuning_scores.csv").set_index("classifier")

    add("Size only", cc["Size only"], 0.140, "Controls")
    add("Image only (session)", cc["Image only (session)"], 0.001, "Controls")
    add("CNN, cell crop (3 seeds)", cc["CNN (cell crop)"], np.nan, "Deep learning")
    for k, arch in (("1-D CNN, strain curves", "cnn1d"), ("GRU, strain curves", "gru")):
        add(k, cc[k], float(seq.loc[seq.arch == arch, "p_perm_seed0"].iloc[0]), "Deep learning")
    for feats, fam in (("selected", "Machine learning, label-free 4"), ("all", "Machine learning, all 29")):
        for clf in SHORT:
            r = sc[(sc.iloc[:, 0] == feats) & (sc["classifier"] == clf)] if "classifier" in sc else None
            b = bench[(bench.features == feats) & (bench.classifier == clf)].iloc[0]
            add(f"{SHORT[clf]}, {'label-free 4' if feats == 'selected' else 'all 29'}",
                r[cells].iloc[0].to_numpy(float), b.p_perm, fam)
    add("LR, features ranked in each fold", cc["LR, nested ranking"], nested.loc["Logistic regression", "p_balanced_acc"],
        "Chosen with the labels")
    add("k-NN, features ranked in each fold", cc["k-NN, nested ranking"], nested.loc["k-NN (k=3)", "p_balanced_acc"],
        "Chosen with the labels")
    add("LR, tuned (nested C)", tsc.loc["Logistic regression", cells], tun.loc["Logistic regression", "p_perm"],
        "Chosen with the labels")
    add(TUNED_SVM, tsc.loc["Linear SVM", cells], tun.loc["Linear SVM", "p_perm"], "Chosen with the labels")
    add("LR every frame", cc["LR every frame"], cm.loc["LR every frame", "p_auc"], "Frame / step level")
    add("LR every laser step", cc["LR every laser step"], cm.loc["LR every laser step", "p_auc"], "Frame / step level")
    for k in [DEPLOY, "LR every laser step + ANN, all 29", "CNN + ANN, all 29", "CNN + LR, all 29",
              "CNN + LR every laser step", "Mean of the 8 classifiers, label-free 4"]:
        add(k, cc[k], cm.loc[k, "p_balanced_acc"], "Ensembles")
    return cells, y, P, p


def landscape(cells, y, P, p, out: Path):
    order = list(FAM)
    rows = [k for f in order for k in P if P[k][1] == f]
    n = len(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 0.215 * n + 1.0), sharey=True,
                                 gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.05})
    ypos, yy, prev = {}, 0.0, None
    for k in rows:
        if prev is not None and P[k][1] != prev:
            yy += 0.9
        ypos[k], prev = yy, P[k][1]
        yy += 1
    for k in rows:
        pr, fam = P[k]
        col = FAM[fam]
        pred = (pr >= .5).astype(int)
        auc = roc_auc_score(y, pr)
        lo, hi = boot_auc(y, pr)
        r = ypos[k]
        if k == DEPLOY:
            for ax in (a1, a2):
                ax.axhspan(r - .5, r + .5, color=col, alpha=.12, lw=0)
        if k == TUNED_SVM:
            for ax in (a1, a2):
                ax.axhspan(r - .5, r + .5, color="#b02a2a", alpha=.08, lw=0)
        a1.plot([lo, hi], [r, r], color=col, alpha=.35, lw=2.4, solid_capstyle="butt")
        sig = p[k] < 0.05 if not np.isnan(p[k]) else None
        a1.plot(auc, r, "o", ms=5.2, mec=col, mew=1.3, mfc=col if sig else "white")
        if sig is None:
            a1.plot(auc, r, "o", ms=1.6, color=col)
        a1.text(1.005, r, f"{auc:.2f}", va="center", fontsize=7, color=INK2)
        f1h = f1_score(y, pred, pos_label=1, zero_division=0)
        f1l = f1_score(y, pred, pos_label=0, zero_division=0)
        a2.plot([min(f1h, f1l), max(f1h, f1l)], [r, r], color=GRID, lw=2.4, zorder=1)
        a2.plot(f1h, r, "o", ms=5, color=HIGH, zorder=2)
        a2.plot(f1l, r, "s", ms=4.6, color=LOW, zorder=2)
        a2.text(1.03, r, f"{(pred == y).sum()}/15", va="center", fontsize=7.5,
                color=INK if k == DEPLOY else INK2, fontweight="bold" if k == DEPLOY else "normal")
    a1.set_yticks([ypos[k] for k in rows])
    a1.set_yticklabels(rows, fontsize=7.6)
    for t, k in zip(a1.get_yticklabels(), rows):
        t.set_color("#b02a2a" if k == TUNED_SVM else FAM[P[k][1]])
        if k == DEPLOY:
            t.set_fontweight("bold")
    seen = set()
    for k in rows:
        f = P[k][1]
        if f not in seen:
            seen.add(f)
            a1.text(0.205, ypos[k] - .62, f, fontsize=7.6, fontweight="bold", color=FAM[f], va="bottom")
    a1.invert_yaxis()
    a1.axvline(.5, color=INK2, ls=":", lw=.8)
    a1.set_xlim(.2, 1.07)
    a1.set_xlabel("AUC, 95 % cell-bootstrap interval")
    a2.set_xlim(0, 1.14)
    a2.set_xticks([0, .25, .5, .75, 1])
    a2.set_xlabel("F1:  ● high HER2   ■ low HER2")
    for ax in (a1, a2):
        ax.grid(axis="x", color=GRID, lw=.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    a1.set_title("(a) Ranking: AUC", loc="left", fontsize=9, color=INK)
    a2.set_title("(b) Calls: F1 per class", loc="left", fontsize=9, color=INK)
    a2.text(1.03, -1.25, "correct", fontsize=7.2, color=INK2, va="center")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def percell(cells, y, P, out: Path):
    rows = ["Size only", "Image only (session)", "CNN, cell crop (3 seeds)", "ANN, all 29", "LR, label-free 4",
            "LR every laser step", TUNED_SVM, DEPLOY]
    labels = {"Size only": "Size only (control)", "Image only (session)": "Image only (session control)",
              "CNN, cell crop (3 seeds)": "Deep learning: CNN, cell crop", "ANN, all 29": "Machine learning: ANN, all 29",
              "LR, label-free 4": "Machine learning: LR, label-free 4", "LR every laser step": "Step level: LR every step",
              TUNED_SVM: "Tuned linear SVM (rejected)", DEPLOY: "Best model: step LR + LR, label-free 4"}
    size = P["Size only"][0]
    hi = sorted([i for i, c in enumerate(cells) if y[i] == 1], key=lambda i: size[i])
    lo = sorted([i for i, c in enumerate(cells) if y[i] == 0], key=lambda i: size[i])
    cols = lo + hi
    M = np.array([P[k][0][cols] for k in rows])
    cmap = LinearSegmentedColormap.from_list("hl", [LOW, "#fbe3d6", "#f7f7f5", "#d6e6f8", HIGH])
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for r, k in enumerate(rows):
        for c, i in enumerate(cols):
            v = P[k][0][i]
            wrong = (v >= .5) != (y[i] == 1)
            txt = f"{v:.3f}" if abs(v - .5) < .005 else f"{v:.2f}"
            ax.text(c, r, txt, ha="center", va="center", fontsize=6.3 if len(txt) == 4 else 5.6,
                    color=INK, fontweight="bold" if wrong else "normal")
            if wrong:
                ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, ec=INK, lw=1.3))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([cells[i].replace("_cell", "-") for i in cols], rotation=45, ha="right", fontsize=7.4)
    for t, i in zip(ax.get_xticklabels(), cols):
        t.set_color(HIGH if y[i] else LOW)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([labels[k] for k in rows], fontsize=7.6)
    for t, k in zip(ax.get_yticklabels(), rows):
        if k == DEPLOY:
            t.set_fontweight("bold")
        if k == TUNED_SVM:
            t.set_color("#b02a2a")
    ax.axvline(len(lo) - .5, color=INK, lw=1.2)
    ax.axhline(1.5, color=INK2, lw=.8)
    ax.axhline(len(rows) - 1.5, color=INK2, lw=.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(len(lo) / 2 - .5, -0.85, "low HER2   (larger → smaller cell)", ha="center", fontsize=7.8, color=LOW)
    ax.text(len(lo) + len(hi) / 2 - .5, -0.85, "high HER2   (larger → smaller cell)", ha="center", fontsize=7.8, color=HIGH)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def null_fig(res: Path, cells, y, P, out: Path):
    f = res / "combine_null.csv"
    if not f.exists():
        print("combine_null.csv missing: skipping", out.name)
        return
    nl = pd.read_csv(f)
    nl = nl[nl.model == DEPLOY]
    pr = P[DEPLOY][0]
    pred = (pr >= .5).astype(int)
    obs_b = 0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())
    obs_a = roc_auc_score(y, pr)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.2), sharey=True)
    for ax, col, obs, lab, bins in ((axs[0], "balanced_acc", obs_b, "balanced accuracy", np.linspace(0, 1, 37)),
                                    (axs[1], "auc", obs_a, "AUC", np.linspace(0, 1, 28))):
        v = nl[col].to_numpy()
        n = len(v)
        pv = (1 + (v >= obs - 1e-12).sum()) / (1 + n)
        h, e = np.histogram(v, bins=bins)
        w = np.diff(e)[0]
        c = np.where(e[:-1] + w / 2 >= obs - 1e-9, FAM["Ensembles"], "#b9b8b3")
        ax.bar(e[:-1] + w / 2, h / n, width=w * .82, color=c)
        ax.axvline(obs, color=FAM["Ensembles"], lw=2)
        ax.axvline(.5, color=INK2, ls=":", lw=.8)
        q = np.percentile(v, 95)
        ax.axvline(q, color=INK2, ls="--", lw=.9)
        ax.text(q - .01, ax.get_ylim()[1] * .55, f"95 % of\nshuffles\nbelow {q:.2f}", ha="right", va="center",
                fontsize=6.8, color=INK2)
        ax.text(obs + .02, ax.get_ylim()[1] * .92, f"real\nlabels\n{obs:.2f}\np = {pv:.3f}", ha="left", va="top",
                fontsize=7.4,
                color=FAM["Ensembles"])
        ax.set_xlabel(f"{lab}, {n:,} label shuffles")
        ax.grid(axis="y", color=GRID, lw=.6)
        ax.set_axisbelow(True)
        print(f"deployment null {col}: obs {obs:.3f} p {pv:.4f} median {np.median(v):.3f} 95th {np.percentile(v, 95):.3f}")
    axs[0].set_ylabel("share of shuffles")
    axs[0].set_title("(a) Calls: balanced accuracy", loc="left", fontsize=9, color=INK)
    axs[1].set_title("(b) Ranking: AUC", loc="left", fontsize=9, color=INK)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def deploy_card(cells, y, P, p, out: Path):
    """The deployment model at a glance: confusion matrix, ROC against size and chance, per-class metrics."""
    from sklearn.metrics import roc_curve
    pr, ps = P[DEPLOY][0], P["Size only"][0]
    pred, preds = (pr >= .5).astype(int), (ps >= .5).astype(int)
    V = FAM["Ensembles"]
    fig = plt.figure(figsize=(7.4, 4.25))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.85, 1, 1.15], height_ratios=[2.45, 1.3], wspace=0.75, hspace=0.42)
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    a4 = fig.add_subplot(gs[1, :])
    # summary header
    ci = boot_auc(y, pr)
    bal = ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean()) / 2
    fig.text(0.07, 1.015, "Best model: LR on every laser step + LR on the label-free 4 features",
             fontsize=10.5, fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(0.07, 0.985, f"{(pred == y).sum()} of {len(y)} cells right   ·   balanced accuracy {bal:.2f}   "
             f"·   AUC {roc_auc_score(y, pr):.2f} [{ci[0]:.2f}–{ci[1]:.2f}]   ·   permutation "
             f"p = {p[DEPLOY]:.3f}   ·   leave-one-cell-out, {int(y.sum())} high / {int((1 - y).sum())} low",
             fontsize=7.6, color=INK2, ha="left", va="bottom")
    # (a) confusion matrix, rows = true class
    cm = np.array([[(pred[y == 1] == 1).sum(), (pred[y == 1] == 0).sum()],
                   [(pred[y == 0] == 1).sum(), (pred[y == 0] == 0).sum()]])
    tot = np.array([[9], [6]])
    a1.imshow(cm / tot, cmap=LinearSegmentedColormap.from_list("v", ["#ffffff", V]), vmin=0, vmax=1.3)
    for i in range(2):
        for j in range(2):
            a1.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] / tot[i, 0] > .6 else INK)
            a1.text(j, i + .3, f"of {tot[i, 0]}", ha="center", va="center", fontsize=6.5,
                    color="white" if cm[i, j] / tot[i, 0] > .6 else INK2)
    a1.set_xticks([0, 1]); a1.set_xticklabels(["high", "low"])
    a1.set_yticks([0, 1]); a1.set_yticklabels(["high", "low"])
    for t in a1.get_yticklabels() + a1.get_xticklabels():
        t.set_color(HIGH if t.get_text() == "high" else LOW); t.set_fontweight("bold")
    a1.set_xlabel("called"); a1.set_ylabel("true HER2")
    a1.tick_params(length=0)
    for sp in a1.spines.values():
        sp.set_visible(False)
    a1.set_title("(a) 13 of 15 cells", loc="left", fontsize=9, color=INK)
    # (b) ROC
    for prob, col, lab, lw in ((ps, "#8a8985", f"size only  {roc_auc_score(y, ps):.2f}", 1.4),
                               (pr, V, f"best model  {roc_auc_score(y, pr):.2f}", 2.2)):
        fpr, tpr, _ = roc_curve(y, prob)
        a2.step(fpr, tpr, where="post", color=col, lw=lw, label=lab)
    a2.plot([0, 1], [0, 1], ls=":", color=INK2, lw=.8)
    fp_op = (pred[y == 0] == 1).mean(); tp_op = (pred[y == 1] == 1).mean()
    a2.plot(fp_op, tp_op, "o", color=V, mec="white", ms=6, zorder=5)
    a2.annotate("cut 0.5", (fp_op, tp_op), xytext=(.18, .52), fontsize=6.8, color=V,
                arrowprops=dict(arrowstyle="-", color=V, lw=.7))
    a2.set_xlim(-.02, 1.02); a2.set_ylim(-.02, 1.02)
    a2.set_xlabel("low cells called high"); a2.set_ylabel("high cells called high")
    a2.legend(title="AUC", fontsize=6.6, title_fontsize=6.8, loc="lower right", frameon=False)
    a2.set_aspect("equal")
    a2.set_title("(b) Ranking: AUC 0.89", loc="left", fontsize=9, color=INK)
    # (c) per-class metrics, deployment vs size
    def per(pd_):
        return [(pd_[y == 1] == 1).mean(), (pd_[y == 0] == 0).mean(),
                f1_score(y, pd_, pos_label=1), f1_score(y, pd_, pos_label=0)]
    m, ms = per(pred), per(preds)
    labs = ["Acc. high", "Acc. low", "F1 high", "F1 low"]
    cols = [HIGH, LOW, HIGH, LOW]
    yy = np.arange(4)[::-1]
    a3.barh(yy, m, color=cols, height=.6)
    a3.plot(ms, yy, "|", color=INK, ms=13, mew=1.6)
    for yi, v in zip(yy, m):
        a3.text(v + .02, yi, f"{v:.2f}", va="center", fontsize=7.5, color=INK, fontweight="bold")
    a3.set_yticks(yy); a3.set_yticklabels(labs)
    a3.set_xlim(0, 1.22); a3.set_xticks([0, .5, 1])
    a3.axvline(.85, color=INK2, ls="--", lw=.7)
    a3.text(.85, -.72, "0.85", fontsize=6.3, color=INK2, ha="center", va="top")
    a3.plot([], [], "|", color=INK, ms=9, mew=1.6, label="size only")
    a3.legend(fontsize=6.6, frameon=False, loc="upper right", bbox_to_anchor=(1.04, 1.02), handletextpad=.2)
    a3.set_ylim(-.75, 3.55)
    a3.grid(axis="x", color=GRID, lw=.6); a3.set_axisbelow(True)
    a3.tick_params(axis="y", length=0)
    pb = p[DEPLOY]
    a3.set_title(f"(c) Per class  (p = {pb:.3f})", loc="left", fontsize=9, color=INK)
    # (d) classification report, scikit-learn layout, threshold 0.5
    from sklearn.metrics import precision_recall_fscore_support
    pc, rc, fc, sc = precision_recall_fscore_support(y, pred, labels=[1, 0], zero_division=0)
    acc = (pred == y).mean()
    rows = [("high HER2", pc[0], rc[0], fc[0], sc[0], HIGH), ("low HER2", pc[1], rc[1], fc[1], sc[1], LOW),
            ("accuracy", None, None, acc, sc.sum(), INK),
            ("macro avg", pc.mean(), rc.mean(), fc.mean(), sc.sum(), INK),
            ("weighted avg", *(np.average(v, weights=sc) for v in (pc, rc, fc)), sc.sum(), INK)]
    a4.axis("off"); a4.set_xlim(0, 1); a4.set_ylim(len(rows) + .9, -.9)
    xs = [.30, .44, .57, .70, .83]
    for x, h in zip(xs[1:], ["precision", "recall", "F1", "cells"]):
        a4.text(x, 0, h, ha="center", va="center", fontsize=8, color=INK2, fontweight="bold")
    a4.plot([.22, .9], [.5, .5], color=INK2, lw=.6)
    a4.plot([.22, .9], [2.6, 2.6], color=GRID, lw=.6)
    for i, (lab, pv, rv, fv, n, col) in enumerate(rows, start=1):
        yi = i + (0.25 if i >= 3 else 0)
        a4.text(xs[0] + .09, yi, lab, ha="right", va="center", fontsize=8.2, color=col,
                fontweight="bold" if i <= 2 else "normal")
        for x, v in zip(xs[1:4], (pv, rv, fv)):
            if v is not None:
                a4.text(x, yi, f"{v:.2f}", ha="center", va="center", fontsize=8.2, color=INK)
        a4.text(xs[4], yi, f"{int(n)}", ha="center", va="center", fontsize=8.2, color=INK)
    a4.set_title("(d) Classification report  (held-out calls, cut P(high) = 0.5; recall = per-class accuracy)", loc="left", fontsize=9,
                 color=INK)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nnunet1000_loocv")
    ap.add_argument("--img", type=Path, required=True)
    a = ap.parse_args()
    res = ROOT / "results/classification" / a.model
    a.img.mkdir(parents=True, exist_ok=True)
    cells, y, P, p = load(res)
    landscape(cells, y, P, p, a.img / "cls_landscape.png")
    percell(cells, y, P, a.img / "cls_percell_map.png")
    null_fig(res, cells, y, P, a.img / "cls_deploy_null.png")
    deploy_card(cells, y, P, p, a.img / "cls_deploy_card.png")
    for k in P:
        pr = P[k][0]
        print(f"{k:45s} {((pr >= .5) == (y == 1)).sum():2d}/15 AUC {roc_auc_score(y, pr):.2f} p {p[k]:.3f}")


if __name__ == "__main__":
    main()
