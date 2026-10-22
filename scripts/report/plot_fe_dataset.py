#!/usr/bin/env python3
"""
Dataset Organization figure for the feature-extraction reports (22 October report onward).

Segmentation is closed; what matters now is which frames the features come from and what checks them.
One row per cell (HER2 high above low), one column per laser-power step:

  * feature frame  -- one per laser step (every 200th video frame); segmented, gated, measured
  * hand-masked    -- the cell's 5-fold test frames (~20); never measured for features, used only to
                      check that the model segments THIS cell well (cell validation)

Beside each row: the cross-validation round whose model segments the cell (the round that held it out,
so no cell is segmented by a model that trained on it), and the trap's height in the frame -- the
per-video trap position, which separates the recording sessions.

A table (--table-output) lists the models used: per round its checkpoint, the cells it trained on, and
for each cell it segments the cross-validation Test DSC, precision and recall (from the run's eval log,
results/<exp>/eval_log_native.jsonl) and whether the cell's features were used; then the all-cell
summary and the time per frame (results/randswap/experiment_1_nnunet/latency_ms.json).

A second figure (--rounds-output) shows which checkpoint segments which cell (fold k-1 = round k, as
nnU-Net counts from 0). Its layout follows the protocol of --model:
  5-fold   one column per round: the 15 cells' roles, the checkpoint, its three test cells
  LOOCV    15 rounds in three blocks: round, checkpoint, the one cell it segments, its validation cell

--model picks the segmentation run (celldform/features/segmenters.py: nnunet100_5fold,
nnunet1000_5fold, nnunet1000_loocv). Read from disk: the run's round plan (segmenters.plan) and its
extraction tables (results/features/<model>/{features_frames,checks_test}.csv). No scores are drawn:
the figures describe the data, not a run's result.

    python scripts/report/plot_fe_dataset.py --model nnunet100_5fold --output ../masters_thesis/images/fe_dataset.png \
        --rounds-output ../masters_thesis/images/fe_rounds.png
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from celldform.features.segmenters import MODELS, plan, round_of  # noqa: E402

HIGH, LOW = "#2a78d6", "#eb6834"
FEAT, MASK = "#9fc9a8", "#1b5e2b"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
TRAIN, VAL, TEST = "#d6d5cf", "#4a3aa7", "#008300"     # role colours of plot_protocols.py


def short(c: str) -> str:
    return c.replace("_cell", "-")


def by_cell(cells: list[str]):
    return lambda c: cells.index(c)


def plot_rounds(model: str, cells: list[str], n_frames: dict[str, int], output: Path) -> None:
    if MODELS[model].protocol == "loocv":
        return plot_rounds_loocv(model, cells, n_frames, output)
    return plot_rounds_5fold(model, cells, n_frames, output)


def plot_rounds_5fold(model: str, cells: list[str], n_frames: dict[str, int], output: Path) -> None:
    """Which checkpoint segments which cell: one column per 5-fold round."""
    label = MODELS[model].label
    recs = {r["round"]: r for r in plan(model)}
    step, wcol = 2.25, 1.95                         # column pitch and width (data units)
    fig, ax = plt.subplots(figsize=(7.4, 3.1), facecolor="white")
    ax.set_xlim(-1.55, len(recs) * step - 0.25)
    ax.set_ylim(-3.1, 1.0)
    ax.axis("off")
    for i, (k, r) in enumerate(recs.items()):
        x0 = i * step
        xc = x0 + wcol / 2
        role = {c: "train" for c in r["train"]} | {c: "val" for c in r["val"]} | {c: "test" for c in r["test"]}
        ax.text(xc, 0.82, f"Round {k}", ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
        w = wcol / len(cells)
        for j, c in enumerate(cells):              # the 15 cells' roles in this round, high-1 ... low-6
            col = {"train": TRAIN, "val": VAL, "test": TEST}[role[c]]
            ax.add_patch(plt.Rectangle((x0 + j * w + 0.01, 0.42), w - 0.02, 0.2, color=col, lw=0))
        ax.annotate("", xy=(xc, 0.0), xytext=(xc, 0.38),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, mutation_scale=8))
        ax.text(xc + 0.07, 0.2, f"{len(r['train'])} + {len(r['val'])} val",
                ha="left", va="center", fontsize=7, color=INK2)
        ax.add_patch(FancyBboxPatch((x0, -0.62), wcol, 0.6, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor="#f1eef9", edgecolor=VAL, lw=0.9))
        ax.text(xc, -0.31, f"fold_{k - 1}", ha="center", va="center", fontsize=9, family="monospace",
                color=INK, fontweight="bold")
        tests = sorted(r["test"], key=by_cell(cells))
        ax.plot([x0 + 0.18] * 2, [-0.64, -0.98 - (len(tests) - 1) * 0.5], color=TEST, lw=1.0)
        for t, c in enumerate(tests):              # the cells it segments -- the ones it never saw
            yc = -1.15 - t * 0.5
            ax.annotate("", xy=(x0 + 0.3, yc + 0.17), xytext=(x0 + 0.18, yc + 0.17),
                        arrowprops=dict(arrowstyle="-|>", color=TEST, lw=1.0, mutation_scale=7))
            col = HIGH if c.startswith("high") else LOW
            ax.add_patch(FancyBboxPatch((x0 + 0.32, yc), wcol - 0.32, 0.34,
                                        boxstyle="round,pad=0,rounding_size=0.05",
                                        facecolor="white", edgecolor=col, lw=1.1))
            ax.text(x0 + 0.32 + (wcol - 0.32) / 2, yc + 0.17, f"{short(c)}  ({n_frames[c]})",
                    ha="center", va="center", fontsize=7.0, color=col, fontweight="bold")
    for y, t, col in ((0.52, "15 cells'\nroles", INK2), (-0.32, "checkpoint", INK2),
                      (-1.48, "segments\nthe frames\n(n) of", TEST)):
        ax.text(-1.5, y, t, ha="left", va="center", fontsize=7.0, color=col, linespacing=0.95)
    x = 0.0                                          # legend, drawn in data units so nothing overlaps
    for col, t in ((TRAIN, "train"), (VAL, "validation"), (TEST, "test (never seen)")):
        ax.add_patch(plt.Rectangle((x, -2.6), 0.16, 0.13, color=col, lw=0))
        ax.text(x + 0.22, -2.535, t, ha="left", va="center", fontsize=7.0, color=INK2)
        x += 0.5 + 0.14 * len(t)
    ax.text(0.0, -2.95, f"{label};  each checkpoint is fold_k/checkpoint_final.pth, and fold_k is round k+1",
            ha="left", va="center",
            fontsize=7, color=INK2)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {output}")


def plot_rounds_loocv(model: str, cells: list[str], n_frames: dict[str, int], output: Path) -> None:
    """Which checkpoint segments which cell: 15 leave-one-cell-out rounds, in three blocks of five rows."""
    label = MODELS[model].label
    rounds = plan(model)
    per, pitch, dy = 5, 4.45, 0.56                  # rows per block, block pitch, row pitch (data units)
    fig, ax = plt.subplots(figsize=(7.4, 2.5), facecolor="white")
    ax.set_xlim(-0.05, 3 * pitch - 0.2)
    ax.set_ylim(-per * dy - 0.85, 0.55)
    ax.axis("off")

    def chip(x, y, w, text, edge, face="white", bold=True, mono=False, color=None):
        ax.add_patch(FancyBboxPatch((x, y - 0.18), w, 0.36, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=face, edgecolor=edge, lw=1.0))
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=6.6 if not mono else 7.2,
                fontweight="bold" if bold else "normal", family="monospace" if mono else None,
                color=color or edge)
    for b in range(3):
        x0 = b * pitch
        for x, t, ha in ((1.18, "checkpoint", "center"), (2.79, "segments (n)", "center"),
                         (3.62, "val", "left")):
            ax.text(x0 + x, 0.32, t, ha=ha, va="center", fontsize=6.8, color=INK2)
        for i, r in enumerate(rounds[b * per:(b + 1) * per]):
            y = -i * dy
            c, v = r["test"][0], r["val"][0]
            ax.text(x0 + 0.02, y, f"R{r['round']}", ha="left", va="center", fontsize=8.2, fontweight="bold",
                    color=INK)
            chip(x0 + 0.62, y, 1.12, f"fold_{r['fold']}", VAL, face="#f1eef9", mono=True, color=INK)
            ax.annotate("", xy=(x0 + 2.04, y), xytext=(x0 + 1.77, y),
                        arrowprops=dict(arrowstyle="-|>", color=TEST, lw=1.0, mutation_scale=7))
            col = HIGH if c.startswith("high") else LOW
            chip(x0 + 2.08, y, 1.42, f"{short(c)} ({n_frames[c]})", col)
            ax.text(x0 + 3.62, y, short(v), ha="left", va="center", fontsize=7.2, color=VAL)
    ax.text(0.0, -per * dy - 0.05, f"Each checkpoint trains on the other {len(cells) - 1} cells "
            f"({len(cells) - 2} + 1 for validation) and segments only the cell it never saw.",
            ha="left", va="center", fontsize=7.0, color=INK2)
    ax.text(0.0, -per * dy - 0.45, f"{label};  each checkpoint is fold_k/checkpoint_final.pth, and fold_k is "
            "round k+1", ha="left", va="center", fontsize=7.0, color=INK2)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {output}")


def write_models_table(model: str, cells: list[str], res: Path, output: Path) -> None:
    """The checkpoints used, one row per cell they segment, with that cell's cross-validation scores."""
    seg = MODELS[model]
    m = {}
    for line in (REPO / "results" / seg.exp / "eval_log_native.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["filter"] and f"fold_{int(r['fold']) - 1}/" in r["checkpoint"]:
            m[r["filter"]] = r["metrics"]["1"]
    if sorted(m) != sorted(cells):
        raise SystemExit(f"{seg.exp}: eval log does not score all {len(cells)} cells")
    use = pd.read_csv(res / "features_cells.csv", index_col="cell")
    lat = json.loads((REPO / "results" / "randswap" / "experiment_1_nnunet" / "latency_ms.json").read_text())
    def her(c: str) -> str:
        return (r"\hiHER" if c.startswith("high") else r"\loHER") + f"{{{short(c)}}}"

    def used(c: str) -> str:
        if use.loc[c, "usable"]:
            return "yes"
        if not use.loc[c, "cell_validated"]:
            return rf"\textcolor{{warnred}}{{no}} ({use.loc[c, 'size_bias_annotated']:.2f}$\times$ size)"
        return rf"\textcolor{{warnred}}{{no}} ({int(use.loc[c, 'passed'])} frames)"
    lines = [r"% GENERATED by scripts/report/plot_fe_dataset.py -- do not edit by hand.",
             r"\begin{tabular}{clcl ccc c}", r"\toprule",
             r"Round & Checkpoint & Trained on & Segments & Test DSC & Precision & Recall & Used \\",
             r"\midrule"]
    worst = min(cells, key=lambda c: m[c]["dsc"])
    for i, r in enumerate(plan(model)):
        tests = sorted(r["test"], key=by_cell(cells))
        for j, c in enumerate(tests):
            first = j == 0
            dsc = f"{m[c]['dsc']:.3f}"
            lines.append(" & ".join([
                f"R{r['round']}" if first else "",
                rf"\code{{fold\_{r['fold']}}}" if first else "",
                f"{len(r['train'])} + {len(r['val'])} val" if first else "",
                her(c), rf"\textbf{{{dsc}}}" if c == worst else dsc,
                f"{m[c]['precision']:.3f}", f"{m[c]['recall']:.3f}", used(c)]) + r" \\")
        if len(tests) > 1 and i < len(plan(model)) - 1:
            lines.append(r"\addlinespace[2pt]")
    def mean(k: str) -> float:
        return st.mean(m[c][k] for c in cells)
    lines += [r"\midrule",
              rf"\multicolumn{{4}}{{l}}{{\textbf{{Per-cell mean}} $\pm$ SD}} & "
              rf"\textbf{{{mean('dsc'):.3f}}} $\pm$ {st.stdev(m[c]['dsc'] for c in cells):.3f} & "
              rf"{mean('precision'):.3f} & {mean('recall'):.3f} & {int(use.usable.sum())} of {len(cells)} \\",
              rf"\multicolumn{{4}}{{l}}{{Time per frame (TTA / no TTA)}} & "
              rf"\multicolumn{{3}}{{c}}{{{lat['nnunet_tta']:.0f} / {lat['nnunet_no_tta']:.0f}\,ms}} & \\",
              r"\bottomrule", r"\end{tabular}", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"-> {output}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=list(MODELS), default="nnunet100_5fold")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds-output", type=Path)
    ap.add_argument("--table-output", type=Path)
    a = ap.parse_args()
    res = REPO / "results" / "features" / a.model
    feat = pd.read_csv(res / "features_frames.csv")
    chk = pd.read_csv(res / "checks_test.csv")
    rnd = round_of(a.model)
    cells = sorted(feat.cell.unique(), key=lambda c: (not c.startswith("high"), int(c.split("cell")[1])))
    if sorted(rnd) != sorted(cells):
        raise SystemExit("5-fold test cells do not match the extracted cells")
    y = {c: i + (1 if c.startswith("low") else 0) for i, c in enumerate(cells)}   # gap between groups
    trap_y = feat.groupby("cell").trap_y.first()           # the video's median position (the cell moves within it:
    y_lo = feat.groupby("cell").local_trap_y.min()         # its local position over the laser steps spans these)
    y_hi = feat.groupby("cell").local_trap_y.max()

    fig = plt.figure(figsize=(10.6, 4.6), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.0], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    for c in cells:
        f = feat[feat.cell == c]
        ax.scatter(f.laser_step, [y[c]] * len(f), marker="s", s=17, color=FEAT, lw=0, zorder=2)
        k = chk[chk.cell == c]
        ax.scatter(k.laser_step, [y[c]] * len(k), marker="o", s=26, facecolor="none", edgecolor=MASK,
                   lw=1.1, zorder=3)
        col = HIGH if c.startswith("high") else LOW
        ax.text(-1.6, y[c], short(c), ha="right", va="center", fontsize=8.2, color=col, fontweight="bold")
        ax.text(54.0, y[c], f"R{rnd[c]}", ha="center", va="center", fontsize=8, color=INK2)
        ax.text(59.5, y[c], f"{len(f)}", ha="center", va="center", fontsize=8, color=INK2)
        ax.text(65.5, y[c], f"{len(k)}", ha="center", va="center", fontsize=8, color=MASK)
    top = -1.15
    for x, t in ((54.0, "round"), (59.5, "feature\nframes"), (65.5, "hand\nmasks")):
        ax.text(x, top, t, ha="center", va="bottom", fontsize=7.4, color=INK2, linespacing=0.95)
    n_hi = sum(c.startswith("high") for c in cells)
    for lab, lo, hi, col in (("HER2 high", 0, n_hi - 1, HIGH), ("HER2 low", n_hi + 1, len(cells), LOW)):
        ax.annotate("", xy=(-7.6, lo - 0.3), xytext=(-7.6, hi + 0.3),
                    arrowprops=dict(arrowstyle="-", color=col, lw=2.2), annotation_clip=False)
        ax.text(-8.6, (lo + hi) / 2, lab, rotation=90, ha="center", va="center", fontsize=8.4, color=col,
                fontweight="bold")
    ax.set_xlim(-1, 67.5)
    ax.set_ylim(len(cells) + 0.6, -1.6)
    ax.set_yticks([])
    steps = int(feat.laser_step.max())
    ax.set_xticks(range(0, steps + 1, 10))
    ax.set_xlabel("laser-power step  (one step $\\approx$ 200 frames $\\approx$ 9 s;  power rises left to right)",
                  fontsize=8.2, color=INK2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_bounds(0, steps)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_title(f"(a) {len(feat)} feature frames from {len(cells)} cells, and the hand masks that check them",
                 fontsize=9.2, color=INK, loc="left", pad=6)
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=FEAT, ms=6, label="feature frame (measured)"),
                       Line2D([], [], marker="o", ls="", mfc="none", mec=MASK, ms=6, mew=1.1,
                              label="hand-masked frame (checks the model, never measured)"),
                       # a ring over a square: both kinds of frame fall in the same laser step
                       Line2D([], [], marker="o", ls="", mfc=FEAT, mec=MASK, ms=6, mew=1.1,
                              label="both, same laser step")],
              loc="upper center", bbox_to_anchor=(0.47, -0.13), ncol=3, fontsize=7.6, frameon=False,
              columnspacing=1.4, handletextpad=0.3)

    bx = fig.add_subplot(gs[1], sharey=ax)
    lo_y, hi_y = min(trap_y.min(), y_lo.min()), max(trap_y.max(), y_hi.max())
    bx.axvspan(lo_y - 8, 480, color="#e9e8e2", zorder=0)
    bx.axvspan(515, hi_y + 8, color="#e9e8e2", zorder=0)
    for c in cells:
        col = HIGH if c.startswith("high") else LOW
        bx.plot([lo_y - 8, y_lo[c]], [y[c]] * 2, color=GRID, lw=0.8, zorder=1)
        bx.plot([y_lo[c], y_hi[c]], [y[c]] * 2, color=col, lw=3.2, alpha=0.35, solid_capstyle="round", zorder=2)
        bx.scatter(trap_y[c], y[c], s=30, color=col, zorder=3)
    bx.set_xlim(lo_y - 8, hi_y + 8)
    odd = [c for c in cells if c.startswith("low") and trap_y[c] > 500]
    for c in odd:   # a low cell recorded at the high cells' trap position
        bx.annotate(f"{short(c)} traps\nwith the high cells", xy=(trap_y[c] - 3, y[c]), xytext=(487, y[c] - 1.7),
                    fontsize=7, color=INK2, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    bx.set_title("(b) cell height (px)", fontsize=9.2, color=INK, loc="left", pad=6)
    bx.set_xlabel("px from the top of the frame", fontsize=7.6, color=INK2)
    bx.legend(handles=[Line2D([], [], marker="o", ls="", color=HIGH, ms=5.5, label="HER2 high cell"),
                       Line2D([], [], marker="o", ls="", color=LOW, ms=5.5, label="HER2 low cell"),
                       Line2D([], [], color=INK2, lw=3.2, alpha=0.35, label="median (dot) and\nrange over the video"),
                       Patch(facecolor="#e9e8e2", edgecolor="none", label="two groups\n(likely sessions)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=7.2, frameon=False,
              handletextpad=0.4, borderaxespad=0)
    bx.tick_params(colors=INK2, labelsize=7.6, length=0, left=False, labelleft=False)
    bx.spines[["top", "right", "left"]].set_visible(False)
    bx.spines["bottom"].set_color(GRID)
    for axis in (ax, bx):
        axis.set_facecolor("white")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {a.output}")
    if a.table_output:
        write_models_table(a.model, cells, res, a.table_output)
    if a.rounds_output:
        plot_rounds(a.model, cells, feat.cell.value_counts().to_dict(), a.rounds_output)


if __name__ == "__main__":
    main()
