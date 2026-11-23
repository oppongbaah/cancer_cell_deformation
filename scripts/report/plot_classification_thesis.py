"""
Thesis-style figures for the HER2 classification chapter (rewrite of 2026-11-03).

    python scripts/report/plot_classification_thesis.py --model nnunet1000_loocv --img DIR

Reads only results already on disk (results/classification/<model>/): combine_cells.csv / combine_metrics.csv
(every model's held-out P(high) per cell), benchmark_cells.csv (permutation p of the 16 machine-learning models),
scores_cells.csv (their P(high) per cell), rank_nested.csv, seq_metrics.csv, tuning_scores.csv / tuning_cells.csv,
pca_*.csv (scripts/classification/pca_models.py: the PCA settings, the best model and its permutation null). Nothing is refitted.

Figures
    cls_landscape.png    every model of the study on one page: AUC with its 95 % cell-bootstrap interval and
                         F1 of each class, grouped by family; filled marker = permutation p < 0.05
    cls_percell_map.png  held-out P(high) of the finalists and the two controls for every cell, cells ordered by size
    cls_deploy_null.png  what chance gives the best model: 1 000 label shuffles of the whole procedure
    cls_deploy_card.png  the best model at a glance; cls_selection.png  the feature selection
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
FAM = {"Controls": "#8a8985", "Deep learning": "#b02a2a", "Machine learning, selected 4": "#1b7a3a",
       "Machine learning, all 29": "#12805f", "Chosen with the labels": "#a56c00", "Frame / step level": "#00687f",
       "Principal components": "#8a3b78", "Ensembles": "#4a3aa7"}
PREV = "LR every laser step + LR, selected 4"     # the same ensemble on the raw four features
DEPLOY = "Step LR + LR, PC1 of selected 4"       # the best model: step LR + LR on PC1 of the selected 4 (pca_models.py)
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
    for feats, fam in (("selected", "Machine learning, selected 4"), ("all", "Machine learning, all 29")):
        for clf in SHORT:
            r = sc[(sc.iloc[:, 0] == feats) & (sc["classifier"] == clf)] if "classifier" in sc else None
            b = bench[(bench.features == feats) & (bench.classifier == clf)].iloc[0]
            add(f"{SHORT[clf]}, {'selected 4' if feats == 'selected' else 'all 29'}",
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
    for k in [PREV, "LR every laser step + ANN, all 29", "CNN + ANN, all 29", "CNN + LR, all 29",
              "CNN + LR every laser step", "Mean of the 8 classifiers, selected 4"]:
        add(k, cc[k], cm.loc[k, "p_balanced_acc"], "Ensembles")
    if (res / "pca_cells.csv").exists():                 # scripts/classification/pca_models.py
        pc = pd.read_csv(res / "pca_cells.csv", index_col=0).reindex(cells)
        pm = pd.read_csv(res / "pca_metrics.csv").set_index("model")
        for k, src in (("LR, PC1 of selected 4", "LR, selected 4, PC1"), ("LR, PC1 of all 29", "LR, all 29, PC1")):
            add(k, pc[src], pm.loc[src, "p_balanced_acc"], "Principal components")
        add(DEPLOY, pc["Step LR + LR, selected 4, PC1"], pm.loc["Step LR + LR, selected 4, PC1", "p_balanced_acc"],
            "Ensembles")
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
    rows = ["Size only", "Image only (session)", "CNN, cell crop (3 seeds)", "ANN, all 29", "LR, selected 4",
            "LR every laser step", TUNED_SVM, PREV, DEPLOY]
    labels = {"Size only": "Size only (control)", "Image only (session)": "Image only (session control)",
              "CNN, cell crop (3 seeds)": "Deep learning: CNN, cell crop", "ANN, all 29": "Machine learning: ANN, all 29",
              "LR, selected 4": "Machine learning: LR, selected 4", "LR every laser step": "Step level: LR every step",
              TUNED_SVM: "Tuned linear SVM (rejected)", PREV: "Step LR + LR, selected 4",
              DEPLOY: "Best model: step LR + LR, PC1 of selected 4"}
    size = P["Size only"][0]
    hi = sorted([i for i, c in enumerate(cells) if y[i] == 1], key=lambda i: size[i])
    lo = sorted([i for i, c in enumerate(cells) if y[i] == 0], key=lambda i: size[i])
    cols = lo + hi
    M = np.array([P[k][0][cols] for k in rows])
    cmap = LinearSegmentedColormap.from_list("hl", [LOW, "#fbe3d6", "#f7f7f5", "#d6e6f8", HIGH])
    fig, ax = plt.subplots(figsize=(7.2, 0.39 * len(rows) + 0.05))
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
    f = res / "pca_null.csv"
    if not f.exists():
        print("pca_null.csv missing: skipping", out.name)
        return
    nl = pd.read_csv(f)
    nl = nl[nl.model == "Step LR + LR, selected 4, PC1"]
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
    fig.text(0.07, 1.015, "Best model: LR on every laser step + LR on the first principal component of the selected 4",
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
    a1.set_title(f"(a) {(pred == y).sum()} of {len(y)} cells", loc="left", fontsize=9, color=INK)
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
    a2.set_title(f"(b) Ranking: AUC {roc_auc_score(y, pr):.2f}", loc="left", fontsize=9, color=INK)
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


SEL_NAMES = {"len_par_base": "baseline length along", "len_perp_base": "baseline length across", "area": "area",
             "perimeter": "perimeter", "minor_axis": "minor axis", "major_axis": "major axis",
             "fluctuation": "fluctuation", "volume_change": "volume change", "orient_spread": "orientation spread",
             "compliance_perp": "compliance across", "compliance_par": "compliance along", "nonlinearity": "nonlinearity",
             "circularity": "circularity", "solidity": "solidity", "aspect_ratio": "aspect ratio",
             "eccentricity": "eccentricity", "roughness_median": "roughness", "ellip_change": "elongation change",
             "poisson": "transverse ratio", "creep": "creep", "onset_step": "onset step", "creep_tau": "creep time"}


def selection_fig(res: Path, out: Path):
    """The feature selection (without the labels), one row per candidate (ranked as the selection ranks them): agreement with the hand
    masks and split-half reliability, the 0.8 bar, and the verdict."""
    t = pd.read_csv(res / "feature_selection.csv")
    t = t[t.split_half_rho > 0].sort_values("score", ascending=False).reset_index(drop=True)   # creep: split-half < 0
    G, R = FAM["Machine learning, selected 4"], "#8fc4a1"
    hu = {f"hu_{i}": f"Hu moment {i}" for i in range(7)}
    names = [SEL_NAMES.get(f, hu.get(f, f.replace("_", " "))) for f in t.feature]
    # groups, as benchmark_cells.select_features decides them: each passing feature against those kept above it
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts/classification/benchmark_cells.py")
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    vals = bench.load(res.name).set_index("cell")
    kept, red = [], {}
    for _, r in t[t.passes].iterrows():
        rh = {k: abs(bench.spearman(vals[r.feature], vals[k])) for k in kept}
        k_max = max(rh, key=rh.get) if rh else None
        red[r.feature] = (rh.get(k_max, np.nan), k_max)
        if not isinstance(r.redundant_with, str):
            kept.append(r.feature)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 4.3), sharey=True,
                                 gridspec_kw={"width_ratios": [2.6, 1], "wspace": 0.08})
    yy = np.arange(len(t))[::-1]
    ax.axvspan(.8, 1.03, color=G, alpha=.07, lw=0)
    ax.axvline(.8, color=INK2, ls="--", lw=.8)
    for yi, (_, r) in zip(yy, t.iterrows()):
        col = G if r.selected else (R if r.passes else "#b9b8b3")
        vals = [v for v in (r.agreement_rho, r.split_half_rho) if pd.notna(v)]
        ax.plot([min(vals), max(vals)], [yi, yi], color=col, lw=1.4, zorder=2)
        if pd.notna(r.agreement_rho):
            ax.plot(r.agreement_rho, yi, "o", color=col, ms=5.5, zorder=3)
        ax.plot(r.split_half_rho, yi, "D", mfc="white", mec=col, ms=4.2, mew=1.3, zorder=3)
        if r.feature in red and pd.notna(red[r.feature][0]):
            v, k = red[r.feature]
            bx.barh(yi, v, height=.62, color=G if r.selected else R)
            txt = "selected" if r.selected else "dropped: = " + SEL_NAMES.get(k, k).replace("baseline length along", "size")
            bx.text(v + .03 if v < .8 else v - .03, yi, f"{v:.2f}", va="center", ha="left" if v < .8 else "right",
                    fontsize=6.3, color=INK if v < .8 else "white")
        elif r.selected:
            bx.text(.03, yi, "first: kept", va="center", fontsize=6.3, color=G)
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=7.2)
    for tl, (_, r) in zip(ax.get_yticklabels(), t.iterrows()):
        tl.set_color(G if r.selected else (INK2 if r.passes else "#8a8985"))
        tl.set_fontweight("bold" if r.selected else "normal")
    ax.set_xlim(-.15, 1.03); ax.set_xticks([0, .2, .4, .6, .8, 1])
    ax.set_xlabel("Spearman ρ across the 15 cells")
    ax.plot([], [], "o", color=INK2, ms=5, label="agreement with the hand masks")
    ax.plot([], [], "D", mfc="white", mec=INK2, ms=4, label="split-half (even vs odd steps)")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.99))
    ax.grid(axis="x", color=GRID, lw=.5); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-.7, len(t) - .3)
    ax.set_title("(a) Reliability: both checks ≥ 0.8", loc="left", fontsize=8.5, color=INK)
    bx.axvline(.7, color=INK2, ls="--", lw=.8)
    bx.axvspan(.7, 1.0, color="#b9b8b3", alpha=.18, lw=0)
    bx.text(.68, 2.0, "same group\nabove 0.7\n(all with\nsize)", ha="right", va="center", fontsize=6.5, color=INK2)
    bx.set_xlim(0, 1); bx.set_xticks([0, .35, .7])
    bx.set_xlabel("max |ρ| with a feature\nkept above it")
    bx.set_title("(b) Groups", loc="left", fontsize=8.5, color=INK)
    bx.tick_params(axis="y", length=0); bx.grid(axis="x", color=GRID, lw=.5); bx.set_axisbelow(True)
    bx.spines["left"].set_visible(False)
    n_pass = int(t.passes.sum())
    bx.text(.42, len(t) - n_pass - 3.5, "fails (a):\nnot tested", ha="center", va="center", fontsize=6.5, color="#8a8985")
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _bench():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts/classification/benchmark_cells.py")
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    return bench


def _fname(f):
    hu = {f"hu_{i}": f"Hu moment {i}" for i in range(7)}
    img = {"illum_mean": "illumination", "illum_gradient": "illum. unevenness", "contrast_rms": "frame contrast",
           "noise_sigma": "noise"}
    return SEL_NAMES.get(f, hu.get(f, img.get(f, f.replace("_", " "))))


EXP_GROUPS = (("size", ["area", "perimeter", "major_axis", "minor_axis", "len_par_base", "len_perp_base"], "#6b6a66"),
              ("shape", ["aspect_ratio", "eccentricity", "circularity", "solidity"] + [f"hu_{i}" for i in range(7)],
               "#00687f"),
              ("deformation", ["compliance_par", "compliance_perp", "poisson", "nonlinearity", "onset_step",
                               "volume_change", "ellip_change", "roughness_median", "orient_spread", "fluctuation",
                               "creep"], "#1b7a3a"),
              ("image (session control)", ["illum_mean", "illum_gradient", "contrast_rms", "noise_sigma"], "#a56c00"))


def _ordered(vals):
    """Group order; inside a group, hierarchical clustering on 1 - |Spearman rho| so related features sit together."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    feats, groups = [], []
    for g, fs, _ in EXP_GROUPS:
        fs = [f for f in fs if f in vals]
        if len(fs) > 2:
            d = 1 - vals[fs].corr("spearman").abs().values
            np.fill_diagonal(d, 0)
            fs = [fs[i] for i in leaves_list(linkage(squareform(d, checks=False), "average"))]
        feats += fs; groups += [g] * len(fs)
    return feats, groups


def corr_fig(res: Path, out: Path):
    """Spearman correlation between every pair of the 28 testable features and the 4 image measures, across the 15
    cells (creep time left out: 12 of 15 cells have no value). Full square: colour below the diagonal, the value
    above it (bold: |rho| >= 0.7, the grouping cut-off). Drawn before any selection: every candidate looks the same."""
    vals = _bench().load(res.name).set_index("cell")
    feats, groups = _ordered(vals)
    C = vals[feats].corr("spearman").values
    n = len(feats)
    cmap = LinearSegmentedColormap.from_list("div", ["#b02a2a", "#f7f6f3", "#4a3aa7"])   # not the class colours
    M = np.ma.masked_where(~np.tril(np.ones_like(C, bool), -1), C)
    fig, ax = plt.subplots(figsize=(7.4, 7.6))
    ax.add_patch(plt.Rectangle((-.5, -.5), n, n, facecolor="white", edgecolor="none", zorder=0))
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest", zorder=1)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, facecolor="#e6e5e1", edgecolor="none", zorder=1))
        for j in range(i + 1, n):
            r = C[i, j]
            big = abs(r) >= .7
            txt = f"{r:+.2f}".replace("0.", ".").replace("+1.00", "1").replace("-1.00", "−1").replace("-", "−")
            ax.text(j, i, txt, ha="center", va="center", fontsize=3.9, zorder=2,
                    color=(cmap(.5 + r / 2) if abs(r) >= .3 else "#9a9994") if not big else INK,
                    fontweight="bold" if big else "normal")
    names = [_fname(f) for f in feats]
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=6.4)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=6.4)
    gcol = {g: c for g, _, c in EXP_GROUPS}
    for tl, g in zip(list(ax.get_yticklabels()) + list(ax.get_xticklabels()), groups + groups):
        tl.set_color(gcol[g])
    for k in [i for i in range(1, n) if groups[i] != groups[i - 1]]:
        ax.axvline(k - .5, color=INK, lw=.7, zorder=3); ax.axhline(k - .5, color=INK, lw=.7, zorder=3)
    ax.set_xlim(-.5, n - .5); ax.set_ylim(n - .5, -.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [plt.Line2D([], [], color=c, lw=6, label=g) for g, _, c in EXP_GROUPS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=4, fontsize=7, frameon=False,
              title="feature group (label colour); bold value: |ρ| ≥ 0.7 (the grouping cut-off)", title_fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.015, shrink=.6)
    cb.set_label("Spearman ρ across the 15 cells", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    cb.set_ticks([-1, -.5, 0, .5, 1])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def explore_fig(res: Path, out: Path):
    """Before any selection: cell size against frame noise. Noise splits by recording session (low-1 sits with the
    high cells), and the sessions differ in size, so size and session are entangled."""
    vals = _bench().load(res.name).set_index("cell")
    y = (vals.her2 == "high").values
    i1 = list(vals.index).index("low_cell1")
    fig, cx = plt.subplots(figsize=(4.0, 3.1))
    a_, n_ = vals["area"].values, vals["noise_sigma"].values
    for lab, col, mk in ((True, HIGH, "o"), (False, LOW, "s")):
        m = y == lab
        cx.scatter(a_[m], n_[m], s=30, color=col, marker=mk, edgecolor="white", lw=.6,
                   label="high HER2" if lab else "low HER2", zorder=3)
    cx.scatter(a_[i1], n_[i1], s=110, facecolor="none", edgecolor=INK, lw=.9, zorder=4)
    cx.annotate("low-1", (a_[i1], n_[i1]), xytext=(6, 5), textcoords="offset points", fontsize=6.8, color=INK)
    cx.tick_params(labelsize=7)
    rho = _bench().spearman(vals["area"], vals["noise_sigma"])
    cx.text(.03, .04, f"Spearman ρ = {rho:.2f} over all 15 cells", transform=cx.transAxes, ha="left", va="bottom",
            fontsize=7, color=INK2)
    cx.legend(fontsize=6.8, frameon=False, loc="center right", handletextpad=.2)
    cx.set_xlabel("cell area (median over its frames, px)", fontsize=7.5)
    cx.set_ylabel("frame noise σ", fontsize=7.5)
    cx.grid(color=GRID, lw=.5); cx.set_axisbelow(True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"area vs noise rho {rho:.2f}")


def funnel_fig(res: Path, out: Path):
    """How the 29 candidates became the selected 4: the count after each rule and the features each rule drops."""
    t = pd.read_csv(res / "feature_selection.csv")
    name = lambda f: _fname(f)   # noqa: E731
    few = t[t.score.isna()].feature.tolist()
    t1 = t[t.score.notna()]
    agree_fail = t1[t1.testable_on_hand_masks & (t1.agreement_rho < .8)].feature.tolist()
    t2 = t1[~t1.feature.isin(agree_fail)]
    split_fail = t2[t2.split_half_rho < .8].feature.tolist()
    t3 = t2[~t2.feature.isin(split_fail)]
    grp = t3[~t3.selected].feature.tolist()
    stages = [("29 candidates", len(t), None),
              ("values in ≥ 12 cells", len(t1), few),
              ("1. agree with the hand masks (ρ ≥ 0.8)", len(t2), agree_fail),
              ("2. split-half reliable (ρ ≥ 0.8)", len(t3), split_fail),
              ("3. one per group (|ρ| ≥ 0.7)", int(t.selected.sum()), grp)]
    G = FAM["Machine learning, selected 4"]
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    for i, (lab, n, drop) in enumerate(stages):
        yy = len(stages) - 1 - i
        last = i == len(stages) - 1
        ax.barh(yy, n, height=.62, color=G if last else "#b9b8b3", left=0)
        ax.text(n + .4, yy, str(n), va="center", fontsize=8, fontweight="bold", color=G if last else INK)
        ax.text(-.6, yy, lab, va="center", ha="right", fontsize=7.4, color=INK)
        if drop:
            d = ", ".join(name(f) for f in drop)
            if len(drop) > 4:
                d = ", ".join(name(f) for f in drop[:4]) + f" + {len(drop) - 4} more"
            ax.text(n + 2.6, yy, f"− {len(drop)}: {d}", va="center", fontsize=6.4, color=INK2)
    sel = t[t.selected].sort_values("rank").feature
    ax.text(0, -.95, "kept: " + ", ".join(name(f) for f in sel), fontsize=7, color=G, fontweight="bold")
    ax.set_xlim(0, 30); ax.set_ylim(-1.3, len(stages) - .5)
    ax.axis("off")
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def selected_fig(res: Path, out: Path):
    """The selected 4 up close, one row each: (i) measured right? predicted against hand masks per cell (fluctuation:
    even against odd laser steps); (ii) over the laser ramp, cell by cell; (iii) the per-cell value by class."""
    from scipy.stats import mannwhitneyu
    bench = _bench()
    vals = bench.load(res.name).set_index("cell")
    y = (vals.her2 == "high")
    sel = pd.read_csv(res / "feature_selection.csv").query("selected").sort_values("rank").feature.tolist()
    chk = pd.read_csv(ROOT / "results/features" / res.name / "cell_feature_check.csv").set_index("cell")
    fr = pd.read_csv(ROOT / "results/features" / res.name / "features_frames.csv")
    fr = fr[fr.cell.isin(vals.index)]
    even = bench.half_values(fr[fr.laser_step % 2 == 0]); odd = bench.half_values(fr[fr.laser_step % 2 == 1])
    ok = fr[fr.passed]

    def axis_mean(a):
        r = np.radians(a.dropna().values) * 2
        return np.degrees(.5 * np.arctan2(np.sin(r).mean(), np.cos(r).mean()))

    traj = {}
    for c, f in ok.groupby("cell"):
        st = f.groupby("laser_step")
        base = st.len_par.median()[st.len_par.median().index < 10].median()
        v0 = st.volume.median()[st.volume.median().index < 10].median()
        traj[c] = {"len_par_base": st.len_par.median(),
                   "volume_change": st.volume.median() / v0 - 1,
                   "orient_spread": st.orientation.apply(axis_mean).abs(),
                   "fluctuation": st.len_par.agg(lambda x: x.std() / x.mean())}
    ylab = {"len_par_base": "length along the beam (px)", "volume_change": "volume / baseline − 1",
            "orient_spread": "|long-axis angle| to the beam (°)", "fluctuation": "SD / mean of the length\nwithin the step"}
    unit = {"len_par_base": "px", "volume_change": "", "orient_spread": "°", "fluctuation": ""}
    fig, axes = plt.subplots(len(sel), 3, figsize=(7.4, 2.05 * len(sel)),
                             gridspec_kw={"width_ratios": [1, 1.7, .8], "wspace": .42, "hspace": .62})
    for k, f in enumerate(sel):
        a0, a1, a2 = axes[k]
        # (i) measured right?
        if f"pred_{f}" in chk:
            xv, yv, lx, ly, tag = chk[f"gt_{f}"], chk[f"pred_{f}"], "hand masks", "predicted masks", "agreement"
        else:
            xv, yv, lx, ly, tag = even[f].astype(float), odd[f].astype(float), "even steps", "odd steps", "split-half"
        xv, yv = xv.reindex(vals.index), yv.reindex(vals.index)
        for lab, col, mk in ((True, HIGH, "o"), (False, LOW, "s")):
            m = (y == lab).values
            a0.scatter(xv[m], yv[m], s=14, color=col, marker=mk, edgecolor="white", lw=.4, zorder=3)
        lo, hi = np.nanmin([xv.min(), yv.min()]), np.nanmax([xv.max(), yv.max()])
        pad = (hi - lo) * .08
        a0.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, lw=.6, ls="--", zorder=1)
        a0.set_xlim(lo - pad, hi + pad); a0.set_ylim(lo - pad, hi + pad)
        rho = bench.spearman(xv, yv)
        a0.text(.04, .96, f"{tag} ρ = {rho:.2f}", transform=a0.transAxes, va="top", fontsize=6.5, color=INK)
        a0.set_xlabel(lx, fontsize=6.5); a0.set_ylabel(ly, fontsize=6.5); a0.tick_params(labelsize=6)
        a0.set_title(f"{_fname(f)}", loc="left", fontsize=8, color=INK, fontweight="bold")
        # (ii) over the ramp
        for c in vals.index:
            sr = traj[c][f]
            a1.plot(sr.index, sr.values, color=HIGH if y[c] else LOW, lw=.6, alpha=.45)
        for lab, col in ((True, HIGH), (False, LOW)):
            M = pd.concat([traj[c][f] for c in vals.index[(y == lab).values]], axis=1).sort_index()
            med = M.median(axis=1)[M.notna().sum(axis=1) >= (M.shape[1] + 1) // 2]   # steps most cells reached
            a1.plot(med.index, med.values, color=col, lw=1.8)
        if f in ("len_par_base", "volume_change"):
            a1.axvspan(-.5, 9.5, color=GRID, alpha=.7, lw=0, zorder=0)
            a1.text(4.5, 1.0, "baseline", transform=a1.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=6, color=INK2)
        if f == "volume_change":
            mx = max(s.index.max() for s in (traj[c][f] for c in vals.index))
            a1.axvspan(mx - 9.5, mx + .5, color="#d6e6f8", alpha=.6, lw=0, zorder=0)
            a1.text(mx - 4.5, 1.0, "last ten", transform=a1.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=6, color=INK2)
            a1.axhline(0, color=INK2, lw=.5)
        a1.set_xlabel("laser step", fontsize=6.5); a1.set_ylabel(ylab[f], fontsize=6.5); a1.tick_params(labelsize=6)
        a1.grid(color=GRID, lw=.4); a1.set_axisbelow(True)
        # (iii) per-cell value by class
        v = vals[f].astype(float)
        rng = np.random.default_rng(k)
        for j, (lab, col, mk) in enumerate(((True, HIGH, "o"), (False, LOW, "s"))):
            m = (y == lab).values
            a2.scatter(j + rng.uniform(-.12, .12, m.sum()), v[m], s=14, color=col, marker=mk, edgecolor="white",
                       lw=.4, zorder=3)
            a2.plot([j - .25, j + .25], [v[m].median()] * 2, color=col, lw=1.8, zorder=4)
        pv = mannwhitneyu(v[y.values], v[~y.values]).pvalue
        a2.text(.5, 1.02, f"p {pv:.2f}" if pv >= .01 else f"p {pv:.3f}", transform=a2.transAxes, ha="center",
                fontsize=6.5, color=INK if pv < .05 else INK2, fontweight="bold" if pv < .05 else "normal")
        a2.set_xticks([0, 1]); a2.set_xticklabels(["high", "low"], fontsize=6.5); a2.set_xlim(-.6, 1.6)
        a2.set_ylabel(f"per cell{' (' + unit[f] + ')' if unit[f] else ''}", fontsize=6.5); a2.tick_params(labelsize=6)
    axes[0][0].annotate("(i) measured right?", (0, 1.32), xycoords="axes fraction", fontsize=7.5, color=INK2)
    axes[0][1].annotate("(ii) over the laser ramp", (0, 1.32),
                        xycoords="axes fraction", fontsize=7.5, color=INK2)
    axes[0][2].annotate("(iii) per cell", (0, 1.32), xycoords="axes fraction", fontsize=7.5, color=INK2)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def selected_corr_fig(res: Path, out: Path):
    """Spearman correlation between the selected features across the 15 cells, every cell labelled."""
    vals = _bench().load(res.name).set_index("cell")
    sel = pd.read_csv(res / "feature_selection.csv").query("selected").sort_values("rank").feature.tolist()
    C = vals[sel].corr("spearman").values
    n = len(sel)
    cmap = LinearSegmentedColormap.from_list("div", ["#b02a2a", "#f7f6f3", "#4a3aa7"])
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    im = ax.imshow(np.where(np.eye(n, dtype=bool), np.nan, C), cmap=cmap, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#e6e5e1"))
                continue
            ax.text(j, i, f"{C[i, j]:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=8,
                    color="white" if abs(C[i, j]) > .6 else INK)
    names = [_fname(f).replace("baseline length along", "baseline\nlength along")
             .replace("orientation spread", "orientation\nspread").replace("volume change", "volume\nchange")
             for f in sel]
    ax.set_xticks(range(n)); ax.set_xticklabels(names, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    cb.set_ticks([-1, 0, 1]); cb.ax.tick_params(labelsize=6.5)
    cb.set_label("Spearman ρ", fontsize=7)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)



def pca_fig(res: Path, out: Path):
    """PCA ranked: eigenvalues of all 29 and of the selected 4 against noise (parallel analysis), what the
    components of the selected 4 are made of, and the cells each setting calls right (step LR + LR)."""
    var = pd.read_csv(res / "pca_variance.csv")
    load = pd.read_csv(res / "pca_loadings.csv")
    m = pd.read_csv(res / "pca_metrics.csv").set_index("model")
    V = FAM["Principal components"]
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=.55, wspace=.42)
    for ax_i, (s, K, title) in enumerate((("all 29", 8, "(a) All 29, first 8 components"),
                                          ("selected 4", 4, "(b) The selected 4"))):
        ax = fig.add_subplot(gs[0, ax_i])
        v = var[var.set == s].sort_values("component").head(K)
        x = np.arange(len(v))
        ax.fill_between(x, v.noise_p5, v.noise_p95, color=GRID, zorder=1, label="noise, 5–95 %")
        ax.plot(x, v.noise_mean, color=INK2, ls="--", lw=1, zorder=2, label="noise, mean")
        ax.bar(x, v.eigenvalue, width=.55, zorder=3,
               color=[V if a else "#c9b3c4" for a in v.above_noise])
        for i, (e, f) in enumerate(zip(v.eigenvalue, v.explained)):
            ax.text(i, e + .03 * v.eigenvalue.max(), f"{f:.0%}", ha="center", fontsize=6.6, color=INK)
        ax.set_xticks(x, [str(c) for c in v.component]); ax.set_xlabel("component, ranked by variance")
        ax.set_ylabel("eigenvalue"); ax.set_title(title, loc="left", fontsize=8.6, color=INK)
        ax.grid(axis="y", color=GRID, lw=.6); ax.set_axisbelow(True)
        if ax_i == 0:
            ax.legend(frameon=False, fontsize=6.8, loc="upper right")
            n_real = int(v.above_noise.sum())
            ax.text(.98, .55, f"{n_real} above noise", transform=ax.transAxes, ha="right", fontsize=7, color=V)
        else:
            ax.text(.98, .9, "none above noise", transform=ax.transAxes, ha="right", fontsize=7, color=V)
    # loadings of the selected 4
    ax = fig.add_subplot(gs[1, 1])
    sel = load[load.set == "selected 4"]
    feats = list(dict.fromkeys(sel.feature))
    W = sel.pivot(index="feature", columns="component", values="weight").reindex(feats).values
    W = W * np.sign(W[np.argmax(np.abs(W), axis=0), range(W.shape[1])])     # largest weight positive
    cmap = LinearSegmentedColormap.from_list("div", ["#b02a2a", "#f7f6f3", "#4a3aa7"])
    ax.imshow(W, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ax.text(j, i, f"{W[i, j]:+.2f}".replace("-", "−"), ha="center", va="center", fontsize=7,
                    color=INK, fontweight="bold" if abs(W[i, j]) >= .5 else "normal")
    ax.set_xticks(range(W.shape[1]), [f"PC{c}" for c in range(1, W.shape[1] + 1)], fontsize=7)
    short = {"len_par_base": "length", "fluctuation": "fluctuation", "volume_change": "volume\nchange",
             "orient_spread": "orientation\nspread"}
    ax.set_yticks(range(len(feats)), [short.get(f, _fname(f)) for f in feats], fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("(d) Weights in the selected 4", loc="left", fontsize=8.6, color=INK)
    # results by k
    ax = fig.add_subplot(gs[1, 0])
    cols = {"all 29": "#12805f", "reliable 9": "#00687f", "selected 4": V}
    for s, c in cols.items():
        ks = [k for k in range(7) if f"Step LR + LR, {s}, {'raw' if k == 0 else 'PC1' if k == 1 else f'PC1-{k}'}" in m.index]
        keys = [f"Step LR + LR, {s}, {'raw' if k == 0 else 'PC1' if k == 1 else f'PC1-{k}'}" for k in ks]
        yv = [m.loc[k, "correct"] for k in keys]
        ax.plot(ks, yv, color=c, lw=1.6, zorder=2, label=s)
        for k, key, yy in zip(ks, keys, yv):
            sig = m.loc[key, "p_balanced_acc"] < .05
            ax.plot(k, yy, "o", ms=5.5, mec=c, mew=1.3, mfc=c if sig else "white", zorder=3)
    ax.set_xticks(range(7), ["none"] + [str(k) for k in range(1, 7)])
    ax.set_xlim(-.3, 6.3); ax.set_ylim(9.3, 14.9); ax.set_yticks(range(10, 15))
    ax.legend(frameon=False, fontsize=6.8, loc="upper right", title="starting set", title_fontsize=6.8)
    ax.set_xlabel("components kept"); ax.set_ylabel("cells correct of 15")
    ax.set_title("(c) Step LR + LR on the top components", loc="left", fontsize=8.6, color=INK)
    ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
    ax.text(.02, .04, "filled: p < 0.05", transform=ax.transAxes, fontsize=6.6, color=INK2)
    fig.savefig(out, dpi=260, bbox_inches="tight")
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
    selection_fig(res, a.img / "cls_selection.png")
    corr_fig(res, a.img / "cls_correlation.png")
    explore_fig(res, a.img / "cls_explore.png")
    funnel_fig(res, a.img / "cls_funnel.png")
    selected_fig(res, a.img / "cls_selected.png")
    selected_corr_fig(res, a.img / "cls_selected_corr.png")
    if (res / "pca_metrics.csv").exists():
        pca_fig(res, a.img / "cls_pca.png")
    for k in P:
        pr = P[k][0]
        print(f"{k:45s} {((pr >= .5) == (y == 1)).sum():2d}/15 AUC {roc_auc_score(y, pr):.2f} p {p[k]:.3f}")


if __name__ == "__main__":
    main()
