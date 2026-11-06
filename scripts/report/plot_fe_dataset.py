#!/usr/bin/env python3
"""
Dataset Organization figure for the feature-extraction reports (22 October report onward).

Segmentation is closed; what matters now is which frames the features come from and what checks them.
One row per cell (HER2 high above low), one column per laser-power step:

  * feature frames -- up to 6 per laser step (the step frame and 5 more within the step; one square per
                      step); segmented, gated, measured
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

    python scripts/report/plot_fe_dataset.py --model nnunet1000_loocv --output ../masters_thesis/images/fe_dataset.png \
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
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from celldform.features.segmenters import MODELS, plan, round_of  # noqa: E402

HIGH, LOW = "#2a78d6", "#eb6834"
FEAT, MASK = "#9fc9a8", "#1b5e2b"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
FRAMES_PER_STEP = 200
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


ROUND_COLS = ["#6a51a3", "#1b8a6b", "#b8327a", "#8a7a12", "#4d6a8a"]   # 5-fold rounds R1..R5
POWER_LO, POWER_HI = "#fbeec4", "#d9861c"                            # laser power, low -> high


def plot_overview(model: str, feat: pd.DataFrame, chk: pd.DataFrame, output: Path, example: str = "high_cell2",
                  zoom_step: int = 7) -> None:
    """The dataset on one figure.
      (a) one recording: the 50 laser steps (power rising), the hand-masked frames along it, and one step
          enlarged -- its 6 feature frames at their offsets, and the hand-masked frame that falls in it;
      (b) the 15 cells, one row each: the round whose checkpoint segments it (the one that never saw it),
          its feature frames and hand masks, and the trapped cell's height in the frame (two groups =
          two recording positions)."""
    from matplotlib.colors import LinearSegmentedColormap
    cells = sorted(feat.cell.unique(), key=lambda c: (not c.startswith("high"), int(c.split("cell")[1])))
    rnd = round_of(model)
    recs = {r["round"]: r for r in plan(model)}
    n_rounds = len(recs)
    rcol = (lambda k: ROUND_COLS[(k - 1) % len(ROUND_COLS)]) if n_rounds <= len(ROUND_COLS) else (lambda k: VAL)
    power = LinearSegmentedColormap.from_list("power", [POWER_LO, POWER_HI])
    n_steps = int(feat.laser_step.max()) + 1

    fig = plt.figure(figsize=(7.4, 6.9), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.3], hspace=0.16)

    # ── (a) one recording ───────────────────────────────────────────────────
    ax = fig.add_subplot(outer[0])
    ax.set_xlim(-0.5, n_steps + 0.5)
    ax.set_ylim(-2.75, 1.7)
    ax.axis("off")
    for s in range(n_steps):
        ax.add_patch(plt.Rectangle((s, 0), 0.92, 0.42, color=power(s / (n_steps - 1)), lw=0))
    hm = chk[chk.cell == example]
    ax.scatter(hm.frame_index / FRAMES_PER_STEP, [0.68] * len(hm), s=16, facecolor="white", edgecolor=MASK, lw=1.0,
               zorder=3)
    ax.text(-0.6, 0.21, "50 laser\nsteps", ha="right", va="center", fontsize=6.8, color=INK2, linespacing=0.9)
    ax.text(-0.6, 0.72, "hand masks", ha="right", va="center", fontsize=6.8, color=MASK)
    ax.annotate("", xy=(n_steps, 1.08), xytext=(0, 1.08),
                arrowprops=dict(arrowstyle="-|>", color=POWER_HI, lw=1.1, mutation_scale=8))
    # the enlarged step
    z0, z1, zy = 4.0, n_steps - 4.0, -1.55
    ax.add_patch(plt.Polygon([[zoom_step, 0], [zoom_step + 0.92, 0], [z1, zy + 0.5], [z0, zy + 0.5]],
                             color=power(zoom_step / (n_steps - 1)), alpha=0.28, lw=0))
    ax.add_patch(plt.Rectangle((zoom_step - 0.08, -0.06), 1.08, 0.54, fill=False, ec=INK, lw=0.9))
    ax.plot([z0, z1], [zy, zy], color=INK2, lw=1.0)
    span = z1 - z0
    for t in range(0, 201, 50):
        x = z0 + span * t / 200
        ax.plot([x, x], [zy - 0.08, zy + 0.08], color=INK2, lw=0.8)
        ax.text(x, zy - 0.2, f"{t}", ha="center", va="top", fontsize=6.4, color=INK2)
    ax.text(z1 + 0.6, zy, "frames into\nthe step", ha="left", va="center", fontsize=6.4, color=INK2, linespacing=0.9)
    offs = sorted(feat[feat.cell == example].offset.unique())
    for o in offs:
        x = z0 + span * o / 200
        ax.scatter([x], [zy + 0.3], marker="s", s=34, color=FEAT, edgecolor="#3f7a4d", lw=0.6, zorder=3)
    k = hm[hm.laser_step == zoom_step]
    for _, r in k.iterrows():
        x = z0 + span * (r.frame_index % FRAMES_PER_STEP) / 200
        ax.scatter([x], [zy + 0.3], s=58, facecolor="none", edgecolor=MASK, lw=1.2, zorder=4)
        ax.annotate("hand-masked frame", xy=(x, zy + 0.42), xytext=(x + 7.5, zy + 0.95), fontsize=6.6, color=MASK,
                    ha="left", va="center", arrowprops=dict(arrowstyle="-", color=MASK, lw=0.7))
    ax.text(z0, zy + 0.95, f"step {zoom_step} enlarged: {len(offs)} feature frames (squares)",
            ha="left", va="center", fontsize=6.9, color="#3f7a4d", fontweight="bold")
    ax.text(n_steps / 2, -2.6, "5 cells (high-1, -9, -10, low-5, -6) start their step frames at offset 100, so "
            "they have 5 per step; a frame already hand-masked is not taken twice",
            ha="center", va="center", fontsize=6.5, color=INK2, style="italic")
    ax.text(-6.5, 1.5, f"(a) One recording ({short(example)}): 22 fps, about 10 000 frames; the laser power rises "
            f"one step every 200 frames ($\\approx$9 s)", ha="left", va="center", fontsize=8.8, color=INK)

    # ── (b) the 15 cells ────────────────────────────────────────────────────
    inner = outer[1].subgridspec(1, 4, width_ratios=[1.05, 1.5, 0.45, 1.6], wspace=0.08)
    y = {c: i + (0.6 if c.startswith("low") else 0) for i, c in enumerate(cells)}
    ymax = max(y.values()) + 0.7
    axs = [fig.add_subplot(inner[j]) for j in range(4)]
    for a_ in axs:
        a_.set_ylim(ymax, -1.3)
        a_.set_yticks([])
        a_.spines[["top", "right", "left"]].set_visible(False)
        a_.spines["bottom"].set_color(GRID)
        a_.tick_params(colors=INK2, labelsize=6.6, length=0)
    # round chips
    a0 = axs[0]
    a0.set_xlim(0, 1)
    a0.spines["bottom"].set_visible(False)
    a0.set_xticks([])
    for c in cells:
        col = HIGH if c.startswith("high") else LOW
        a0.text(0.0, y[c], short(c), ha="left", va="center", fontsize=7.6, color=col, fontweight="bold")
        k_ = rnd[c]
        a0.add_patch(FancyBboxPatch((0.5, y[c] - 0.34), 0.42, 0.68, boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=rcol(k_), edgecolor="none"))
        a0.text(0.71, y[c], f"R{k_}", ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
    a0.text(0.71, -0.95, "round\ncheckpoint", ha="center", va="bottom", fontsize=6.6, color=INK2, linespacing=0.9)
    # feature frames
    a1 = axs[1]
    nf = feat.cell.value_counts()
    a1.barh([y[c] for c in cells], [nf[c] for c in cells], height=0.62, color=FEAT, lw=0)
    for c in cells:
        a1.text(nf[c] + 4, y[c], f"{nf[c]}", ha="left", va="center", fontsize=6.6, color=INK2)
    a1.set_xlim(0, 340)
    a1.set_xticks([0, 100, 200, 300])
    a1.set_title("feature frames", fontsize=7.2, color="#3f7a4d", pad=1, y=0.985)
    # hand masks
    a2 = axs[2]
    nh = chk.cell.value_counts()
    a2.set_xlim(0, 1)
    a2.set_xticks([])
    a2.spines["bottom"].set_visible(False)
    for c in cells:
        a2.scatter([0.28], [y[c]], s=26, facecolor="white", edgecolor=MASK, lw=1.0)
        a2.text(0.5, y[c], f"{nh.get(c, 0)}", ha="left", va="center", fontsize=6.6, color=MASK)
    a2.set_title("hand\nmasks", fontsize=7.2, color=MASK, pad=1, y=0.97, linespacing=0.9)
    # trap height
    a3 = axs[3]
    ty = feat.groupby("cell").trap_y.first()
    lo_, hi_ = feat.groupby("cell").local_trap_y.min(), feat.groupby("cell").local_trap_y.max()
    x_lo, x_hi = min(ty.min(), lo_.min()) - 8, max(ty.max(), hi_.max()) + 8
    gap = np.sort(ty.values)
    cut = gap[np.argmax(np.diff(gap))] + np.max(np.diff(gap)) / 2
    rows = [min(y.values()) - 0.6, max(y.values()) + 0.6]   # shade the cell rows only: the header stays on white
    a3.fill_betweenx(rows, x_lo, cut - 18, color="#efeee9", zorder=0, lw=0)
    a3.fill_betweenx(rows, cut + 18, x_hi, color="#efeee9", zorder=0, lw=0)
    for c in cells:
        col = HIGH if c.startswith("high") else LOW
        a3.plot([lo_[c], hi_[c]], [y[c]] * 2, color=col, lw=3.0, alpha=0.3, solid_capstyle="round")
        a3.scatter([ty[c]], [y[c]], s=20, color=col, zorder=3)
    a3.set_xlim(x_lo, x_hi)
    odd = [c for c in cells if not c.startswith("high") and ty[c] > cut]
    for c in odd:
        a3.annotate(f"{short(c)}: recorded\nat the high\ncells' position", xy=(ty[c] - 3, y[c]),
                    xytext=(cut, y[c] + 1.7), fontsize=6.3, color=INK2, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    a3.set_title("trapped cell's height y (px)", fontsize=7.2, color=INK, pad=1, y=0.985)
    a3.text(0.5, -0.075, "dot: median over the video; bar: range over the steps", transform=a3.transAxes,
            ha="center", va="top", fontsize=6.2, color=INK2)
    n_hi = sum(c.startswith("high") for c in cells)
    for lab, ys, col in (("HER2 high", [y[c] for c in cells[:n_hi]], HIGH), ("HER2 low", [y[c] for c in cells[n_hi:]], LOW)):
        a0.annotate("", xy=(-0.08, min(ys) - 0.4), xytext=(-0.08, max(ys) + 0.4), xycoords="data",
                    arrowprops=dict(arrowstyle="-", color=col, lw=2.0), annotation_clip=False)
        a0.text(-0.17, (min(ys) + max(ys)) / 2, lab, rotation=90, ha="center", va="center", fontsize=7,
                color=col, fontweight="bold", clip_on=False)
    r0 = recs[min(recs)]
    a0.text(-0.42, -2.3, f"(b) The {len(cells)} cells, each segmented by the one round model that never saw it "
            f"({len(r0['train'])} + {len(r0['val'])} cells trained it)", transform=a0.transData, ha="left",
            va="center", fontsize=8.8, color=INK, clip_on=False)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {output}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=list(MODELS), default="nnunet1000_loocv")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds-output", type=Path)
    ap.add_argument("--table-output", type=Path)
    a = ap.parse_args()
    res = REPO / "results" / "features" / a.model
    feat = pd.read_csv(res / "features_frames.csv")
    chk = pd.read_csv(res / "checks_test.csv")
    cells = sorted(feat.cell.unique(), key=lambda c: (not c.startswith("high"), int(c.split("cell")[1])))
    if sorted(round_of(a.model)) != sorted(cells):
        raise SystemExit("the run's rounds do not match the extracted cells")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    plot_overview(a.model, feat, chk, a.output)
    if a.table_output:
        write_models_table(a.model, cells, res, a.table_output)
    if a.rounds_output:
        plot_rounds(a.model, cells, feat.cell.value_counts().to_dict(), a.rounds_output)


if __name__ == "__main__":
    main()
