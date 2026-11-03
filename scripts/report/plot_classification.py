"""
Classification report figures and tables: confusion matrices, ROC curves and all metrics, deep and classical models.

For the per-frame deep model (frames_cnn.py) and the per-cell classical classifiers (benchmark_cells.py).

    python scripts/report/plot_classification.py --model nnunet1000_loocv --img <dir> --tables <dir>

Reads the saved held-out predictions only (results/classification/<model>/), so it never retrains anything. Every
number is per CELL (14 cells, leave-one-cell-out); high HER2 is the positive class:
    sensitivity  high cells called high          specificity  low cells called low
    precision    calls of high that are right    F1           harmonic mean of precision and sensitivity
    balanced acc (sensitivity + specificity) / 2 AUC          ranking of the held-out P(high)
A cell's deep-model P(high) is the mean of its frames' P(high); the call is P(high) >= 0.5. 95 % intervals on the
cells correct are Clopper-Pearson.

With 14 cells a confusion matrix only restates the counts and every AUC has a 95 % interval of about 0.5-1.0, so
the report evaluates the models on what the question needs (cls_eval.tex / .png):
    Brier skill  1 - Brier / Brier of guessing each held-out cell's class from the other 13 cells' proportions:
                 > 0 means the probabilities carry information, < 0 that they are worse than guessing
    vs size      cells the model gets right that size alone gets wrong (gained) and the reverse (lost), with an exact
                 McNemar p: does the model add anything beyond the cell's size?
    AUC [CI]     with a 2000-resample cell bootstrap interval, to show how little 14 cells pin it down
    low-1        P(high) of the low cell recorded at the high cells' position (the session check)
Intervals are 2000-resample bootstraps over cells. SVM scores are a sigmoid of the margin, not probabilities.

Outputs: cls_cnn_metrics.tex, cls_ml_metrics.tex, cls_seq_metrics.tex, cls_compare.tex, cls_eval.tex,
cls_tuning.tex, cls_cnn_roc_cm.png, cls_cnn_cells.png, cls_seq.png, cls_ml_roc.png, cls_ml_cm.png, cls_compare.png,
cls_eval.png, cls_tuning.png.
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
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

HIGH, LOW = "#2a78d6", "#eb6834"
INK, INK2, GRID, FAINT = "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c4"
ACCENT, GOOD, SESSION, SEQ = "#b02a2a", "#1b7a3a", "#7a4fd6", "#b8730f"
SEQ_NAMES = {"cnn1d": "1-D CNN on the strain curves", "gru": "GRU (recurrent) on the strain curves"}
CNN = {   # file stem -> label
    "finetune_masked_seedmean": "CNN fine-tuned, cell crop (3 seeds)",
    "finetune_masked": "  seed 0", "finetune_masked_seed1": "  seed 1", "finetune_masked_seed2": "  seed 2",
    "finetune_mask": "CNN fine-tuned, mask only",
    "frozen_masked": "CNN frozen, cell crop", "frozen_mask": "CNN frozen, mask only"}
CNN_MAIN = ["finetune_masked_seedmean", "finetune_mask", "frozen_masked", "frozen_mask"]
CNN_STYLE = {"finetune_masked_seedmean": (ACCENT, "-"), "finetune_mask": (GOOD, "--"),
             "frozen_masked": (FAINT, "-"), "frozen_mask": (INK2, ":")}


def short(c: str) -> str:
    return c.replace("_cell", "-")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if prec and np.isfinite(prec) and prec + sens else 0.0
    k, n = tp + tn, len(y)
    lo = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    auc = roc_auc_score(y, p) if len(set(p)) > 1 else np.nan
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "correct": k, "n": n, "acc_lo": lo, "acc_hi": hi,
            "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
            "balanced_acc": (sens + spec) / 2, "auc": auc}


# ── data ─────────────────────────────────────────────────────────────────────

def cnn_cells(res: Path) -> tuple[pd.DataFrame, dict]:
    """Per-cell P(high) of every deep model, and the frame tables."""
    frames = {k: pd.read_csv(res / f"cnn_frames_{k}.csv") for k in CNN}
    cells = None
    for k, f in frames.items():
        c = f.groupby(["cell", "her2"]).p_high.mean().rename(k).reset_index()
        cells = c if cells is None else cells.merge(c, on=["cell", "her2"])
    cells["_n"] = cells.cell.str.extract(r"(\d+)$")[0].astype(int)
    return cells.sort_values(["her2", "_n"]).drop(columns="_n").reset_index(drop=True), frames


def ml_cells(res: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(res / "scores_cells.csv")
    table = pd.read_csv(res / "benchmark_cells.csv")
    return scores, table


# ── tables ───────────────────────────────────────────────────────────────────

def pct(v: float) -> str:
    return "--" if not np.isfinite(v) else f"{v:.2f}"


def metric_row(name: str, m: dict, extra: str = "") -> str:
    return (f"{name} & {m['correct']}/{m['n']} & {m['acc_lo'] * 100:.0f}--{m['acc_hi'] * 100:.0f}\\,\\% & "
            f"{pct(m['sensitivity'])} & {pct(m['specificity'])} & {pct(m['balanced_acc'])} & {pct(m['f1'])} & "
            f"{pct(m['auc'])}{extra} \\\\")


HEAD = ("Model & Correct & 95\\,\\% CI & Sens. & Spec. & Bal.\\ acc. & F1 & AUC")


def cnn_table(cells: pd.DataFrame, frames: dict, out: Path) -> dict:
    y = (cells.her2 == "high").to_numpy().astype(int)
    rows, ms = [], {}
    for k, label in CNN.items():
        m = metrics(y, cells[k].to_numpy())
        f = frames[k]
        fm = metrics((f.her2 == "high").to_numpy().astype(int), f.p_high.to_numpy())
        low1 = cells.set_index("cell").loc["low_cell1", k]
        ms[k] = m | {"frame_bal_acc": fm["balanced_acc"], "low1": low1}
        name = label.replace("  ", "\\quad ")
        rows.append(metric_row(name, m, f" & {fm['balanced_acc']:.2f} & {low1:.2f}"))
    out.write_text("\\begin{tabular}{l c c c c c c c c c}\n\\toprule\n" + HEAD +
                   " & Frame bal.\\ acc. & \\code{low-1} $P$ \\\\\n\\midrule\n" + "\n".join(rows) +
                   "\n\\bottomrule\n\\end{tabular}\n")
    return ms


def ml_table(scores: pd.DataFrame, table: pd.DataFrame, cells: list[str], out: Path) -> dict:
    y = np.array([c.startswith("high") for c in cells]).astype(int)
    rows, ms = [], {}
    order = [("selected", c) for c in table[table.features == "selected"].classifier] + \
            [("size", "Logistic regression"), ("image", "Logistic regression")] + \
            [("all", c) for c in table[table.features == "all"].classifier]
    for i, (fs, clf) in enumerate(order):
        s = scores[(scores.features == fs) & (scores.classifier == clf)].iloc[0]
        p = s[cells].to_numpy(float)
        m = metrics(y, p)
        pv = table[(table.features == fs) & (table.classifier == clf)].p_perm.iloc[0]
        ms[(fs, clf)] = m | {"p_perm": pv}
        name = {"size": "Size only (LR)", "image": "Image only (LR)"}.get(fs, clf)
        if fs in ("selected", "all") and clf in ("Random forest", "Gradient boosting"):
            name += "$^\\ddagger$"
        if fs == "selected" and clf == "Logistic regression":
            name = "\\textbf{Logistic regression}"
        rows.append(metric_row(name, m, " & " + ("--" if not np.isfinite(pv) else f"{pv:.3f}")))
        n_sel = int((table.features == "selected").sum())
        if i in (n_sel - 1, n_sel + 1):                     # after the selected set, after the two baselines
            rows.append("\\midrule")
            if i == n_sel + 1:
                rows.append("\\multicolumn{9}{l}{\\emph{All 29 features (sensitivity check)}} \\\\")
    out.write_text("\\begin{tabular}{l c c c c c c c c}\n\\toprule\n" + HEAD + " & $p_{\\text{perm}}$ \\\\\n"
                   "\\midrule\n\\multicolumn{9}{l}{\\emph{Selected features (7)}} \\\\\n" + "\n".join(rows) +
                   "\n\\bottomrule\n\\end{tabular}\n")
    return ms


def seq_cells(res: Path, order: list[str]) -> tuple[dict, pd.DataFrame] | None:
    """Per-cell P(high) of each sequence network (mean of its seeds) and its metrics table (sequence_net.py)."""
    if not (res / "seq_cells.csv").exists():
        return None
    c = pd.read_csv(res / "seq_cells.csv")
    c = c[c.seed.astype(str) == "mean"].set_index("arch")[order]
    return {a: c.loc[a].to_numpy(float) for a in c.index}, pd.read_csv(res / "seq_metrics.csv")


def seq_table(seq: dict, met: pd.DataFrame, order: list[str], out: Path) -> dict:
    y = np.array([c.startswith("high") for c in order]).astype(int)
    rows, ms = [], {}
    for arch, p in seq.items():
        m = metrics(y, p)
        pv = met[(met.arch == arch) & (met.run == "mean of seeds")].p_perm_seed0.iloc[0]
        ms[arch] = m | {"p_perm": pv, "low1": p[order.index("low_cell1")]}
        rows.append(metric_row(SEQ_NAMES.get(arch, arch) + " (3 seeds)", m,
                               f" & {ms[arch]['low1']:.2f} & " + ("--" if not np.isfinite(pv) else f"{pv:.3f}")))
    out.write_text("\\begin{tabular}{l c c c c c c c c c c}\n\\toprule\n" + HEAD +
                   " & \\code{low-1} $P$ & $p_{\\text{perm}}$ \\\\\n\\midrule\n" + "\n".join(rows) +
                   "\n\\bottomrule\n\\end{tabular}\n")
    return ms


# ── figures ──────────────────────────────────────────────────────────────────

def seq_fig(model: str, seq: dict, order: list[str], out: Path):
    """(a) What the sequence networks see: every cell's strain curve along the beam, by class. (b) Their per-cell
    P(high) (mean of 3 seeds)."""
    import importlib.util
    from celldform.analysis_config import load as load_config
    spec = importlib.util.spec_from_file_location("sequence_net", ROOT / "scripts" / "classification" / "sequence_net.py")
    sn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sn)
    cfg = load_config("sequence")
    X, cells = sn.curves(model, cfg)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    j = list(cfg.measures).index("len_par") if "len_par" in cfg.measures else 0
    for i, h in enumerate(cells.her2):
        ax.plot(np.arange(X.shape[2]), X[i, j], color=HIGH if h == "high" else LOW, lw=1.1, alpha=0.75)
    ax.axhline(0, color=INK2, lw=0.7, ls=":")
    ax.plot([], [], color=HIGH, label="high HER2 (8)"); ax.plot([], [], color=LOW, label="low HER2 (6)")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_xlabel("laser step", fontsize=8, color=INK2)
    ax.set_ylabel("strain along the beam (%)", fontsize=8, color=INK2)
    ax.set_title("(a) The input: one strain curve per cell (and across the beam, volume)", fontsize=9, loc="left")
    style(ax)
    y = np.array([c.startswith("high") for c in order]).astype(int)
    w = 0.38
    for k, (arch, p) in enumerate(seq.items()):
        xs = np.arange(len(order)) + (k - 0.5) * w
        bx.bar(xs, p, width=w, color=[HIGH if h else LOW for h in y], alpha=0.45 if k else 0.85,
               edgecolor="white", label=SEQ_NAMES.get(arch, arch).split(" on ")[0])
        for x, pi, yi in zip(xs, p, y):
            if (pi >= 0.5) != yi:
                bx.text(x, min(pi, 0.92) + 0.04, "✕", ha="center", fontsize=7, color=INK)
    bx.axhline(0.5, color=INK2, lw=0.8, ls=":")
    bx.set_xticks(range(len(order)), [short(c) for c in order], fontsize=7, rotation=45)
    for lab, h in zip(bx.get_xticklabels(), y):
        lab.set_color(HIGH if h else LOW)
    bx.set_ylim(0, 1.05)
    bx.set_ylabel("held-out P(high)", fontsize=8, color=INK2)
    bx.set_title("(b) P(high) per cell: solid 1-D CNN, light GRU; ✕ = wrong", fontsize=9, loc="left")
    style(bx); bx.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def style(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=INK2)


def roc_axes(ax, title: str):
    ax.plot([0, 1], [0, 1], color=FAINT, lw=1, ls=":", zorder=0)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("1 $-$ specificity (low cells called high)", fontsize=7.5, color=INK2)
    ax.set_ylabel("sensitivity (high cells called high)", fontsize=7.5, color=INK2)
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    style(ax)


def confusion(ax, m: dict, title: str):
    M = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
    tot = np.array([[m["tp"] + m["fn"]] * 2, [m["fp"] + m["tn"]] * 2])
    frac = M / np.maximum(tot, 1)
    ax.imshow(frac, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("g", ["#ffffff", "#2f7d4f"]),
              vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]}", ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if frac[i, j] > 0.6 else INK)
    ax.set_xticks([0, 1], ["high", "low"], fontsize=7.5); ax.set_yticks([0, 1], ["high", "low"], fontsize=7.5)
    ax.set_xlabel("called", fontsize=7.5, color=INK2); ax.set_ylabel("true", fontsize=7.5, color=INK2)
    ax.set_title(title, fontsize=8, color=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, colors=INK2)


def cnn_roc_cm(cells: pd.DataFrame, ms: dict, out: Path):
    y = (cells.her2 == "high").to_numpy().astype(int)
    fig = plt.figure(figsize=(11, 3.9))
    gs = fig.add_gridspec(2, 5, width_ratios=[2.3, 0.15, 1, 1, 0.05], hspace=0.65, wspace=0.45)
    ax = fig.add_subplot(gs[:, 0])
    roc_axes(ax, "(a) ROC, per cell")
    for k in CNN_MAIN[::-1]:
        col, ls = CNN_STYLE[k]
        fpr, tpr, _ = roc_curve(y, cells[k])
        ax.plot(fpr, tpr, color=col, ls=ls, lw=2.2 if k == CNN_MAIN[0] else 1.5, drawstyle="steps-post",
                label=f"{CNN[k]}  AUC {ms[k]['auc']:.2f}")
    ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    for i, k in enumerate(CNN_MAIN):
        confusion(fig.add_subplot(gs[i // 2, 2 + i % 2]), ms[k], CNN[k].replace(" (3 seeds)", "").replace(", ", ",\n"))
    fig.text(0.53, 0.97, "(b) Confusion matrices (cells)", fontsize=9, color=INK)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def cnn_cells_fig(cells: pd.DataFrame, frames: dict, out: Path):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10.5, 6.4), gridspec_kw={"height_ratios": [1.15, 1]})
    keys = ["frozen_masked", "frozen_mask", "finetune_mask", "finetune_masked", "finetune_masked_seed1",
            "finetune_masked_seed2", "finetune_masked_seedmean"]
    P = np.array([cells[k].to_numpy() for k in keys])
    y = (cells.her2 == "high").to_numpy().astype(int)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("her2", [LOW, "#ffffff", HIGH])
    im = ax.imshow(P, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(keys)):
        for j in range(len(y)):
            if (P[i, j] >= 0.5) != y[j]:
                ax.text(j, i, "\u2715", ha="center", va="center", fontsize=9, color="#111111")
    ax.set_yticks(range(len(keys)), [("CNN fine-tuned, cell crop, " + CNN[k].strip()) if "seed" in CNN[k] and "3 seeds" not in CNN[k]
                                     else CNN[k].strip() for k in keys], fontsize=7.5)
    ax.set_xticks(range(len(y)), [short(c) for c in cells.cell], fontsize=7.5, rotation=45)
    for lab, h in zip(ax.get_xticklabels(), y):
        lab.set_color(HIGH if h else LOW)
    ax.axvline(y.sum() - 0.5, color="k", lw=1); ax.axhline(2.5, color="k", lw=0.8)
    ax.set_title("(a) Held-out P(high) per cell (mean of its frames); \u2715 = wrong call", fontsize=9, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.ax.tick_params(labelsize=7)
    f = frames["finetune_masked_seedmean"]
    for j, c in enumerate(cells.cell):
        v = f[f.cell == c].p_high.to_numpy()
        col = HIGH if c.startswith("high") else LOW
        parts = bx.violinplot(v, positions=[j], widths=0.8, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(col); b.set_alpha(0.35); b.set_edgecolor("none")
        bx.scatter(j, v.mean(), s=26, color=col, edgecolor="white", lw=1, zorder=3)
    bx.axhline(0.5, color=INK2, lw=0.8, ls=":")
    bx.set_xticks(range(len(cells)), [short(c) for c in cells.cell], fontsize=7.5, rotation=45)
    for lab, c in zip(bx.get_xticklabels(), cells.cell):
        lab.set_color(HIGH if c.startswith("high") else LOW)
    bx.set_ylabel("frame P(high)", fontsize=8, color=INK2); bx.set_ylim(-0.02, 1.02)
    bx.set_title("(b) CNN, fine-tuned on the cell crop (3 seeds): every frame's P(high), dot = cell mean", fontsize=9,
                 loc="left")
    style(bx)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def ml_roc(scores: pd.DataFrame, cells: list[str], ms: dict, out: Path):
    y = np.array([c.startswith("high") for c in cells]).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    roc_axes(axes[0], "(a) Seven classifiers, selected features")
    sel = scores[scores.features == "selected"]
    for _, s in sel.iterrows():
        fpr, tpr, _ = roc_curve(y, s[cells].to_numpy(float))
        main = s.classifier == "Logistic regression"
        axes[0].plot(fpr, tpr, color=GOOD if main else FAINT, lw=2.4 if main else 1.2, drawstyle="steps-post",
                     zorder=3 if main else 2,
                     label=f"Logistic regression (primary)  AUC {ms[('selected', s.classifier)]['auc']:.2f}" if main else None)
    axes[0].plot([], [], color=FAINT, lw=1.2, label="the other six  AUC 0.67--0.84")
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    roc_axes(axes[1], "(b) Against the baselines")
    for (fs, clf), col, ls, lab in [(("selected", "Logistic regression"), GOOD, "-", "Selected features (LR)"),
                                    (("size", "Logistic regression"), INK2, "--", "Size only (LR)"),
                                    (("image", "Logistic regression"), SESSION, "-", "Image only (LR)")]:
        s = scores[(scores.features == fs) & (scores.classifier == clf)].iloc[0]
        fpr, tpr, _ = roc_curve(y, s[cells].to_numpy(float))
        axes[1].plot(fpr, tpr, color=col, ls=ls, lw=2, drawstyle="steps-post", label=f"{lab}  AUC {ms[(fs, clf)]['auc']:.2f}")
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def ml_cm(ms: dict, table: pd.DataFrame, out: Path):
    keys = [("selected", c) for c in table[table.features == "selected"].classifier] + \
           [("size", "Logistic regression"), ("image", "Logistic regression")]
    fig, axes = plt.subplots(2, 5, figsize=(10.5, 4.6))
    for ax, k in zip(axes.flat, keys):
        title = {"size": "Size only", "image": "Image only"}.get(k[0], k[1])
        confusion(ax, ms[k], f"{title}\n{ms[k]['correct']}/{ms[k]['n']} correct")
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def compare(cnn: dict, ml: dict, out_png: Path, out_tex: Path, seq: dict | None = None):
    rows = [("Deep learning (CNN)", "CNN fine-tuned, cell crop (3 seeds)", cnn["finetune_masked_seedmean"]),
            ("Deep learning (CNN)", "CNN fine-tuned, mask only", cnn["finetune_mask"]),
            ("Deep learning (CNN)", "CNN frozen, cell crop", cnn["frozen_masked"])]
    rows += [("Deep learning (sequence)", SEQ_NAMES.get(a, a).replace(" on the strain curves", ""), m)
             for a, m in (seq or {}).items()]
    rows += [("Classical ML", "Logistic regression (primary)", ml[("selected", "Logistic regression")]),
             ("Classical ML", "k-NN (best)", ml[("selected", "k-NN (k=3)")])]
    if ("selected", "MLP (ANN)") in ml:
        rows.append(("Classical ML", "MLP (ANN)", ml[("selected", "MLP (ANN)")]))
    rows += [("Classical ML", "Gradient boosting", ml[("selected", "Gradient boosting")]),
             ("Baseline", "Size only", ml[("size", "Logistic regression")]),
             ("Baseline", "Image only (session)", ml[("image", "Logistic regression")])]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 0.9))
    colors = {"Deep learning (CNN)": ACCENT, "Deep learning (sequence)": SEQ, "Classical ML": GOOD, "Baseline": INK2}
    for i, (grp, name, m) in enumerate(rows):
        yv = len(rows) - 1 - i
        c = SESSION if "Image" in name else colors[grp]
        ax.plot([m["acc_lo"], m["acc_hi"]], [yv, yv], color=c, lw=2, solid_capstyle="round", alpha=0.45)
        ax.scatter(m["correct"] / m["n"], yv, s=46, color=c, edgecolor="white", lw=1.2, zorder=3)
        ax.text(1.03, yv, f"{m['correct']}/{m['n']}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_yticks(range(len(rows)), [r[1] for r in rows][::-1], fontsize=7.8)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("cells correct (proportion), 95 % CI", fontsize=8, color=INK2)
    groups = [r[0] for r in rows]
    for grp in dict.fromkeys(groups):
        ys = [len(rows) - 1 - i for i, g in enumerate(groups) if g == grp]
        lo, hi = min(ys) - 0.5, max(ys) + 0.5
        ax.axhspan(lo, hi, color=colors[grp], alpha=0.04, lw=0)
        ax.text(0.01, hi - 0.35, grp, fontsize=7.5, color=colors[grp], fontweight="bold")
    style(ax); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    lines = [f"{grp} & {name} & {m['correct']}/{m['n']} & {m['acc_lo'] * 100:.0f}--{m['acc_hi'] * 100:.0f}\\,\\% & "
             f"{pct(m['balanced_acc'])} & {pct(m['auc'])} \\\\" for grp, name, m in rows]
    out_tex.write_text("\\begin{tabular}{l l c c c c}\n\\toprule\nApproach & Model & Correct & 95\\,\\% CI & "
                       "Bal.\\ acc. & AUC \\\\\n\\midrule\n" + "\n".join(lines) +
                       "\n\\bottomrule\n\\end{tabular}\n")


def evaluation(cells: pd.DataFrame, scores: pd.DataFrame, out_tex: Path, out_png: Path, n_boot: int = 2000,
               seq: dict | None = None):
    from scipy.stats import binomtest
    order = cells.cell.tolist()
    y = (cells.her2 == "high").to_numpy().astype(int)
    ml = lambda fs, clf: scores[(scores.features == fs) & (scores.classifier == clf)].iloc[0][order].to_numpy(float)  # noqa: E731
    models = [("CNN", "CNN fine-tuned, cell crop (3 seeds)", cells["finetune_masked_seedmean"].to_numpy()),
              ("CNN", "CNN fine-tuned, mask only", cells["finetune_mask"].to_numpy()),
              ("CNN", "CNN frozen, mask only", cells["frozen_mask"].to_numpy()),
              ("CNN", "CNN frozen, cell crop", cells["frozen_masked"].to_numpy())]
    models += [("Seq", SEQ_NAMES.get(a, a).replace(" on the strain curves", ""), pr) for a, pr in (seq or {}).items()]
    models += [("ML", c + ("$^\\dagger$" if "SVM" in c else ""), ml("selected", c))
               for c in scores[scores.features == "selected"].classifier]
    models += [("Base", "Size only", ml("size", "Logistic regression")),
               ("Base", "Image only (session)", ml("image", "Logistic regression"))]
    size_ok = (ml("size", "Logistic regression") >= 0.5).astype(int) == y
    prev = np.where(y == 1, (y.sum() - 1) / 13, y.sum() / 13)            # other 13 cells' share of high
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    rows = []
    for grp, name, pr in models:
        ok = (pr >= 0.5).astype(int) == y
        bss = lambda i: 1 - np.mean((pr[i] - y[i]) ** 2) / np.mean((prev[i] - y[i]) ** 2)   # noqa: E731
        all_i = np.arange(len(y))
        b = [bss(i) for i in idx]
        a = [roc_auc_score(y[i], pr[i]) for i in idx if 0 < y[i].sum() < len(i)]
        gained, lost = int((ok & ~size_ok).sum()), int((~ok & size_ok).sum())
        pm = binomtest(gained, gained + lost).pvalue if gained + lost else 1.0
        rows.append({"group": grp, "model": name, "correct": int(ok.sum()), "bss": bss(all_i),
                     "bss_lo": np.percentile(b, 2.5), "bss_hi": np.percentile(b, 97.5),
                     "auc": roc_auc_score(y, pr), "auc_lo": np.percentile(a, 2.5), "auc_hi": np.percentile(a, 97.5),
                     "gained": gained, "lost": lost, "mcnemar_p": pm,
                     "low1": pr[order.index("low_cell1")]})
    t = pd.DataFrame(rows)
    t.to_csv(out_tex.with_suffix(".csv"), index=False)
    lines, prev_grp = [], None
    for _, r in t.iterrows():
        if prev_grp and r.group != prev_grp:
            lines.append("\\midrule")
        prev_grp = r.group
        name = r.model.replace("Size only", "\\emph{Size only}").replace("Image only (session)", "\\emph{Image only (session)}")
        gl = "--" if r.model == "Size only" else f"+{r.gained} / $-${r.lost} & {r.mcnemar_p:.2f}"
        gl = gl if r.model != "Size only" else "-- & --"
        lines.append(f"{name} & {r.correct}/14 & ${r.bss:+.2f}$ [${r.bss_lo:+.2f}$, ${r.bss_hi:+.2f}$] & "
                     f"{r.auc:.2f} [{r.auc_lo:.2f}, {r.auc_hi:.2f}] & {gl} & {r.low1:.2f} \\\\")
    out_tex.write_text("\\begin{tabular}{l c c c c c c}\n\\toprule\n"
                       " & & & & \\multicolumn{2}{c}{Against size alone} & \\\\\n\\cmidrule(lr){5-6}\n"
                       "Model & Correct & Brier skill [95\\,\\% CI] & AUC [95\\,\\% CI] & gained / lost & $p$ & "
                       "\\code{low-1} $P$ \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    col = {"CNN": ACCENT, "Seq": SEQ, "ML": GOOD, "Base": INK2}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 0.3 * len(t) + 0.9), sharey=True,
                                 gridspec_kw={"width_ratios": [1.4, 1]})
    yy = np.arange(len(t))[::-1]
    x_min = -1.5
    for yi, (_, r) in zip(yy, t.iterrows()):
        c = SESSION if "Image" in r.model else col[r.group]
        ax.plot([max(r.bss_lo, x_min), r.bss_hi], [yi, yi], color=c, lw=2, alpha=0.45, solid_capstyle="round")
        if r.bss >= x_min:
            ax.scatter(r.bss, yi, s=40, color=c, edgecolor="white", lw=1.1, zorder=3)
        else:                        # off the scale: an arrow at the edge, with the value
            ax.scatter(x_min, yi, s=46, color=c, marker="<", zorder=3)
            ax.text(x_min + 0.06, yi + 0.3, f"{r.bss:+.1f}", fontsize=6.5, color=c)
        if r.model != "Size only":
            bx.barh(yi, r.gained, color=GOOD, height=0.6)
            bx.barh(yi, -r.lost, color=ACCENT, height=0.6)
            bx.text(max(r.gained, 0) + 0.15, yi, f"+{r.gained}/$-${r.lost}", va="center", fontsize=7, color=INK2)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(yy, [m.replace("$^\\dagger$", "\u2020") for m in t.model], fontsize=7.8)
    ax.set_xlabel("Brier skill (0 = no better than guessing the class proportions)", fontsize=8, color=INK2)
    ax.set_title("(a) Are the probabilities informative?", fontsize=9, loc="left", color=INK)
    ax.set_xlim(max(x_min - 0.05, t.bss_lo.min() - 0.05), 1)
    bx.axvline(0, color=INK2, lw=0.9)
    bx.set_xlabel("cells: gained (right) / lost (left) against size alone", fontsize=8, color=INK2)
    bx.set_title("(b) Anything beyond size?", fontsize=9, loc="left", color=INK)
    lim = max(5, int(t[["gained", "lost"]].to_numpy().max()) + 1.5)
    bx.set_xlim(-lim, lim)
    for a in (ax, bx):
        style(a); a.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return t


def tuning(res: Path, cells: pd.DataFrame, out_tex: Path, out_png: Path) -> pd.DataFrame | None:
    """Untuned against nested-CV-tuned, for the CNN (seed 0, cell crop) and the classifiers."""
    from sklearn.metrics import balanced_accuracy_score
    if not (res / "tuning_cells.csv").exists() or not (res / "cnn_frames_tuned_masked.csv").exists():
        print("tuning results missing -- run tune_cells.py and frames_cnn.py tune first")
        return None
    order = cells.cell.tolist()
    y = (cells.her2 == "high").to_numpy().astype(int)

    def cnn_row(stem):
        f = pd.read_csv(res / f"cnn_frames_{stem}.csv").groupby("cell").p_high.mean().reindex(order).to_numpy()
        pred = (f >= 0.5).astype(int)
        return int((pred == y).sum()), balanced_accuracy_score(y, pred)
    uc, ub = cnn_row("finetune_masked")
    tc, tb = cnn_row("tuned_masked")
    ch = pd.read_csv(res / "cnn_tuning_choices.csv")
    lab = ch.apply(lambda r: f"{int(r.epochs)} ep, lr {r.lr:g}", axis=1).value_counts()
    rows = [{"group": "CNN", "model": "CNN fine-tuned, cell crop", "grid_points": len(pd.read_csv(res / "cnn_tuning_grid.csv")),
             "untuned_correct": uc, "correct": tc, "untuned_balanced_acc": ub, "balanced_acc": tb, "p_perm": np.nan,
             "untuned_p_perm": np.nan, "modal_choice": lab.index[0], "modal_share": lab.iloc[0] / len(ch),
             "distinct_choices": len(lab)}]
    t = pd.read_csv(res / "tuning_cells.csv")
    for _, r in t.iterrows():
        mc = r.modal_choice.replace("{", "").replace("}", "").replace("'", "")
        rows.append({"group": "ML", "model": r.classifier, "grid_points": r.grid_points,
                     "untuned_correct": r.untuned_correct, "correct": r.correct,
                     "untuned_balanced_acc": r.untuned_balanced_acc, "balanced_acc": r.balanced_acc,
                     "p_perm": r.p_perm, "untuned_p_perm": r.untuned_p_perm, "modal_choice": mc,
                     "modal_share": r.modal_share, "distinct_choices": r.distinct_choices})
    d = pd.DataFrame(rows)
    fmt_p = lambda v: "--" if not np.isfinite(v) else f"{v:.3f}"   # noqa: E731
    lines = []
    for i, r in d.iterrows():
        if i == 1:
            lines.append("\\midrule")
        lines.append(f"{r.model} & {int(r.grid_points)} & {int(r.untuned_correct)}/14 & \\textbf{{{int(r.correct)}/14}} & "
                     f"{r.untuned_balanced_acc:.2f} $\\to$ {r.balanced_acc:.2f} & {fmt_p(r.untuned_p_perm)} $\\to$ "
                     f"{fmt_p(r.p_perm)} & {r.modal_choice.replace('_', ' ')} & {r.modal_share * 100:.0f}\\,\\% "
                     f"({int(r.distinct_choices)}) \\\\")
    out_tex.write_text("\\begin{tabular}{l c c c c c l c}\n\\toprule\n"
                       " & Grid & \\multicolumn{2}{c}{Cells correct} & Bal.\\ acc. & $p_{\\text{perm}}$ & "
                       "\\multicolumn{2}{c}{Setting chosen} \\\\\n\\cmidrule(lr){3-4}\\cmidrule(lr){7-8}\n"
                       "Model & points & untuned & tuned & untuned $\\to$ tuned & untuned $\\to$ tuned & most often & "
                       "folds (distinct) \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True, gridspec_kw={"width_ratios": [1.3, 1]})
    yy = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(yy, d.iterrows()):
        col = ACCENT if r.group == "CNN" else GOOD
        ax.plot([r.untuned_correct, r.correct], [yi, yi], color=FAINT, lw=2, zorder=1)
        ax.scatter(r.untuned_correct, yi, s=42, color="white", edgecolor=col, lw=1.6, zorder=3)
        ax.scatter(r.correct, yi, s=42, color=col, edgecolor="white", lw=1, zorder=4)
        bx.barh(yi, r.modal_share, color=col, alpha=0.75, height=0.6)
        bx.text(r.modal_share + 0.02, yi, r.modal_choice, va="center", fontsize=6.8, color=INK2)
    ax.axvline(10, color=INK2, lw=0.8, ls=":")
    ax.text(10.05, len(d) - 0.45, "size only", fontsize=7, color=INK2)
    ax.set_yticks(yy, d.model, fontsize=7.8)
    ax.set_xlim(6.5, 12.5)
    ax.set_xlabel("cells correct of 14   (open: untuned, filled: tuned)", fontsize=8, color=INK2)
    ax.set_title("(a) Does tuning help?", fontsize=9, loc="left", color=INK)
    bx.set_xlim(0, 1.6)
    bx.set_xticks([0, 0.25, 0.5, 0.75, 1])
    bx.set_xlabel("share of the 14 folds choosing the same setting", fontsize=8, color=INK2)
    bx.set_title("(b) Is the chosen setting stable?", fontsize=9, loc="left", color=INK)
    for a in (ax, bx):
        style(a); a.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    d.to_csv(out_tex.with_suffix(".csv"), index=False)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nnunet1000_loocv")
    ap.add_argument("--img", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    a = ap.parse_args()
    res = ROOT / "results" / "classification" / a.model
    cells, frames = cnn_cells(res)
    scores, table = ml_cells(res)
    order = cells.cell.tolist()
    cnn_ms = cnn_table(cells, frames, a.tables / "cls_cnn_metrics.tex")
    ml_ms = ml_table(scores, table, order, a.tables / "cls_ml_metrics.tex")
    sq = seq_cells(res, order)
    seq_ms = None
    if sq is not None:
        seq_ms = seq_table(sq[0], sq[1], order, a.tables / "cls_seq_metrics.tex")
        seq_fig(a.model, sq[0], order, a.img / "cls_seq.png")
    cnn_roc_cm(cells, cnn_ms, a.img / "cls_cnn_roc_cm.png")
    cnn_cells_fig(cells, frames, a.img / "cls_cnn_cells.png")
    ml_roc(scores, order, ml_ms, a.img / "cls_ml_roc.png")
    ml_cm(ml_ms, table, a.img / "cls_ml_cm.png")
    compare(cnn_ms, ml_ms, a.img / "cls_compare.png", a.tables / "cls_compare.tex", seq_ms)
    ev = evaluation(cells, scores, a.tables / "cls_eval.tex", a.img / "cls_eval.png",
                    seq=sq[0] if sq is not None else None)
    tu = tuning(res, cells, a.tables / "cls_tuning.tex", a.img / "cls_tuning.png")
    if tu is not None:
        print(tu.to_string(index=False))
    print(ev.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for k, m in (seq_ms or {}).items():
        print(f"SEQ {k:28s} {m['correct']}/14 sens {m['sensitivity']:.2f} spec {m['specificity']:.2f} "
              f"bal {m['balanced_acc']:.2f} AUC {m['auc']:.2f} p {m['p_perm']:.3f}")
    for k, m in cnn_ms.items():
        print(f"CNN {k:28s} {m['correct']}/14 "
              f"sens {m['sensitivity']:.2f} spec {m['specificity']:.2f} "
              f"bal {m['balanced_acc']:.2f} F1 {m['f1']:.2f} AUC {m['auc']:.2f}")
    for k, m in ml_ms.items():
        print(f"ML  {k[0]:8s} {k[1]:20s} {m['correct']}/14 "
              f"sens {m['sensitivity']:.2f} spec {m['specificity']:.2f} "
              f"bal {m['balanced_acc']:.2f} F1 {m['f1']:.2f} AUC {m['auc']:.2f} p {m['p_perm']:.3f}")
    print(f"wrote {a.img}, {a.tables}")


if __name__ == "__main__":
    main()
