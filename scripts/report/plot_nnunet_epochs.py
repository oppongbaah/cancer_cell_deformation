#!/usr/bin/env python3
"""
nnU-Net at 100 against 1000 epochs on the same randswap rounds (8 October report).

Reads both runs' native eval logs (results/<exp>/eval_log_native.jsonl, the latest entry per
round and cell) and nnU-Net's own training logs (checkpoints/<exp>/Dataset*/*/fold_*/
training_log_*.txt). Nothing is typed in here.

    python scripts/report/plot_nnunet_epochs.py \
        --out ../masters_thesis/images --tables ../masters_thesis/main_work/tables

Writes nnunet_epochs.png -- (a) every cell's Test DSC at 100 and 1000 epochs, (b) each round's
internal validation Dice over the 1000 epochs -- and nnunet_epochs.tex.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent.parent
SHORT_EXP, SHORT_TRAINER = "randswap/experiment_1_nnunet", "nnUNetTrainer_100epochs"
LONG_EXP, LONG_TRAINER = "randswap/experiment_1b_nnunet_1000ep", "nnUNetTrainer"
# GPUs the two runs trained on (1 October: the local workstation; 8 October: the NSU cluster).
GPU = {SHORT_EXP: "RTX 4060, 8\\,GB", LONG_EXP: "Tesla V100, 32\\,GB"}
TIE = 0.01            # a cell counts as better/worse only beyond this margin (as in plot_cv.py)

# nnU-Net's report colour (deep magenta, C5 in plot_experiments.py); same model at both lengths,
# so one hue -- told apart by marker fill, always with a legend.
NNU = "#b0306a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def short(cell: str) -> str:
    return cell.replace("_cell", "-")


def scores(exp: str) -> dict[tuple[str, str], dict]:
    """(round, cell or 'ALL') -> metrics of class 1, the latest log entry winning."""
    last = {}
    for line in (REPO / "results" / exp / "eval_log_native.jsonl").read_text().splitlines():
        r = json.loads(line)
        last[(r["fold"], r["filter"] or "ALL")] = r["metrics"]["1"]
    return last


def train_logs(exp: str, trainer: str) -> list[Path]:
    folds = sorted((REPO / "checkpoints" / exp).glob(f"Dataset*/{trainer}__nnUNetPlans__2d/fold_*"))
    return [sorted(f.glob("training_log_*.txt"))[-1] for f in folds if (f / "checkpoint_final.pth").exists()]


def epoch_seconds(logs: list[Path]) -> list[float]:
    return [float(t) for p in logs for t in re.findall(r"Epoch time: ([\d.]+) s", p.read_text())]


def cell_val_dice(log: Path) -> list[float]:
    """Per-epoch pseudo Dice of the trapped-cell class on the round's validation cell."""
    return [float(re.findall(r"[\d.]+(?=\))", x)[0]) for x in re.findall(r"Pseudo dice \[([^\]]*)\]", log.read_text())]


def ema(xs: list[float], a: float = 0.9) -> list[float]:
    out, m = [], None
    for x in xs:                 # nnU-Net's own smoothing: ema = 0.9 * ema + 0.1 * current
        m = x if m is None else a * m + (1 - a) * x
        out.append(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="folder for nnunet_epochs.png")
    ap.add_argument("--tables", type=Path, required=True, help="folder for nnunet_epochs.tex")
    a = ap.parse_args()

    S, L = scores(SHORT_EXP), scores(LONG_EXP)
    key = lambda k: (not k[1].startswith("high"), int(k[1].split("cell")[1]))
    cells = sorted((k for k in S if k[1] != "ALL"), key=key)
    if sorted(k for k in L if k[1] != "ALL") != sorted(cells):
        raise SystemExit("the two runs did not score the same rounds and cells")
    rounds = sorted({k[0] for k in cells}, key=int)
    d = {k: L[k]["dsc"] - S[k]["dsc"] for k in cells}

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10.4, 6.6), facecolor="white")
    gs = fig.add_gridspec(2, 5, height_ratios=[1.5, 1], hspace=0.75, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    bxs = [fig.add_subplot(gs[1, i]) for i in range(5)]
    for axis in (ax, *bxs):
        axis.set_facecolor(SURF)
        for sp in ("top", "right"):
            axis.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            axis.spines[sp].set_color(GRID)
        axis.tick_params(colors=INK2, labelsize=8.5, length=0)
        axis.grid(axis="y", color=GRID, lw=0.8)
        axis.set_axisbelow(True)

    lo = min(min(S[k]["dsc"], L[k]["dsc"]) for k in cells)
    for x, k in enumerate(cells):
        s, l = S[k]["dsc"], L[k]["dsc"]
        ax.plot([x - 0.12, x + 0.12], [s, l], color=GRID, lw=2.5, zorder=2)
        ax.scatter([x - 0.12], [s], s=62, marker="o", facecolor="white", edgecolor=NNU, linewidth=1.8, zorder=4)
        ax.scatter([x + 0.12], [l], s=62, marker="o", color=NNU, edgecolor="white", linewidth=1.5, zorder=4)
        ax.text(x, max(s, l) + 0.008, "0.000" if abs(d[k]) < 5e-4 else f"{d[k]:+.3f}".replace("-", "−"),
                ha="center", va="bottom", fontsize=7.2, color=INK if abs(d[k]) > TIE else INK2)
        ax.text(x, lo - 0.02, f"R{k[0]}", ha="center", va="top", fontsize=7.2, color=INK2)
    n_high = sum(k[1].startswith("high") for k in cells)
    ax.axvline(n_high - 0.5, color=GRID, lw=1)
    ax.set_xticks(range(len(cells)), [short(k[1]) for k in cells], fontsize=8.6, color=INK)
    ax.set_xlim(-0.7, len(cells) - 0.3)
    ax.set_ylim(lo - 0.05, 1.0)
    ax.set_ylabel("Test DSC (native)", color=INK2, fontsize=9)
    ms, ml = st.mean(S[k]["dsc"] for k in cells), st.mean(L[k]["dsc"] for k in cells)
    ax.set_title(f"(a) Same 15 cells, same rounds — per-cell mean: 100 epochs {ms:.3f}, 1000 epochs {ml:.3f}"
                 f"  ·  number = 1000 minus 100", loc="left", fontsize=9.5, color=INK)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=NNU,
                              markeredgewidth=1.8, label="nnU-Net, 100 epochs (1 Oct)"),
                       Line2D([], [], marker="o", ls="", color=NNU, label="nnU-Net, 1000 epochs (8 Oct)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, frameon=False, ncol=2)

    logs = train_logs(LONG_EXP, LONG_TRAINER)
    curves = {r: ema(cell_val_dice(p)) for r, p in zip(rounds, logs)}
    val_cell = {r: json.loads((REPO / "results" / LONG_EXP / f"nnunet_fold_{r}_split.json").read_text())
                ["cells"]["val"][0] for r in rounds}
    floor = 0.85
    for i, (bx, (r, c)) in enumerate(zip(bxs, curves.items())):
        bx.plot(range(len(c)), c, color=NNU, lw=1.3, zorder=3)
        bx.axvline(100, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
        bx.set_xlim(0, 1000)
        bx.set_ylim(floor, 1.0)
        bx.set_xticks([0, 500, 1000])
        ticks = bx.get_xticklabels()
        ticks[0].set_horizontalalignment("left")      # keep neighbouring panels' "1000" and "0" apart
        ticks[-1].set_horizontalalignment("right")
        if i:
            bx.tick_params(labelleft=False)
        bx.set_title(f"R{r} · val. {short(val_cell[r])}", fontsize=8.6, color=INK, loc="left")
        bx.text(995, floor + 0.006, f"end {c[-1]:.3f}", ha="right", va="bottom", fontsize=7.4, color=INK2)
    bxs[0].set_ylabel("Val. cell DSC (EMA)", color=INK2, fontsize=9)
    bxs[2].set_xlabel("Epoch  (dashed: epoch 100)", color=INK2, fontsize=9)
    fig.text(0.125, 0.39, "(b) The 1000-epoch run: trapped-cell Dice on each round's validation cell "
             "(nnU-Net's smoothed pseudo Dice; not a test score)", fontsize=9.5, color=INK)
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "nnunet_epochs.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── table ─────────────────────────────────────────────────────────────────
    def pooled(m: dict, ks) -> dict:
        tp, fp, fn = (sum(m[k][x] for k in ks) for x in ("tp", "fp", "fn"))
        return {"pooled": 2 * tp / (2 * tp + fp + fn), "prec": tp / (tp + fp), "rec": tp / (tp + fn)}

    def b(x, y):                 # bold the higher of two scores
        f = lambda v, on: rf"\textbf{{{v:.3f}}}" if on else f"{v:.3f}"
        return f(x, x > y), f(y, y > x)

    def worst(m: dict) -> str:
        k = min(cells, key=lambda c: m[c]["dsc"])
        return f"{short(k[1])} ({m[k]['dsc']:.3f})"

    pr = lambda p: f"{p['prec']:.3f} / {p['rec']:.3f}"
    rows = [(f"Pooled DSC, round {r}", *b(pooled(S, [k for k in cells if k[0] == r])["pooled"],
                                           pooled(L, [k for k in cells if k[0] == r])["pooled"])) for r in rounds]
    PS, PL = pooled(S, cells), pooled(L, cells)
    rows += [(r"\textbf{Pooled DSC, all 15 cells}", *b(PS["pooled"], PL["pooled"])),
             ("Per-cell mean DSC", *b(ms, ml)),
             ("Precision / recall, 15 cells", pr(PS), pr(PL)),
             None]
    hi = lambda m: st.mean(m[k]["dsc"] for k in cells if k[1].startswith("high"))
    lw = lambda m: st.mean(m[k]["dsc"] for k in cells if k[1].startswith("low"))
    better, worse = sum(v > TIE for v in d.values()), sum(v < -TIE for v in d.values())
    up = sum(v > 0 for v in d.values())
    r_up = sum(pooled(L, [k for k in cells if k[0] == r])["pooled"] > pooled(S, [k for k in cells if k[0] == r])["pooled"]
               for r in rounds)
    rows += [(rf"\hiHER{{High HER2}} mean ({n_high} cells)", *b(hi(S), hi(L))),
             (rf"\loHER{{Low HER2}} mean ({len(cells) - n_high} cells)", *b(lw(S), lw(L))),
             ("Worst cell", worst(S), worst(L)),
             (rf"Cells higher at 1000 / by $>{TIE}$", "---", f"{up} of {len(cells)} / {better} ({worse} lower by $>{TIE}$)"),
             ("Rounds higher at 1000 (Pooled DSC)", "---", f"{r_up} of {len(rounds)}"),
             None]
    es, el = epoch_seconds(train_logs(SHORT_EXP, SHORT_TRAINER)), epoch_seconds(logs)
    n_s, n_l = len(train_logs(SHORT_EXP, SHORT_TRAINER)), len(logs)
    rows += [("Epochs (last epoch kept)", "100", "1000"),
             ("GPU", GPU[SHORT_EXP], GPU[LONG_EXP]),
             ("Time per epoch", f"{st.median(es):.0f} s", f"{st.median(el):.0f} s"),
             ("Training time per round", f"{sum(es) / n_s / 3600:.1f} h", f"{sum(el) / n_l / 3600:.1f} h"),
             ("Time per frame (to features)", r"\multicolumn{2}{c}{unchanged --- same network and plan}")]
    lines = [r"% GENERATED by scripts/report/plot_nnunet_epochs.py -- do not edit by hand.",
             r"\begin{tabular}{lcc}", r"\toprule",
             r" \footnotesize\emph{nnU-Net, same 15 cells, same rounds} & \textbf{100 epochs} & \textbf{1000 epochs} \\",
             r"\midrule"]
    for r in rows:
        if r is None:
            lines.append(r"\midrule")
        elif len(r) == 2:
            lines.append(f"{r[0]} & {r[1]} \\\\")
        else:
            lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    a.tables.mkdir(parents=True, exist_ok=True)
    (a.tables / "nnunet_epochs.tex").write_text("\n".join(lines))
    print(f"-> {a.out / 'nnunet_epochs.png'}, {a.tables / 'nnunet_epochs.tex'}")


if __name__ == "__main__":
    main()
