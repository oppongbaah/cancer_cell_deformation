#!/usr/bin/env python3
"""
Leave-one-cell-out against the 5-fold cross-validation, for the U-Net and nnU-Net (15 October report).

Reads the four runs' native eval logs (results/<exp>/eval_log_native.jsonl, the latest entry per
cell). Nothing is typed in here. A run that has not scored every cell yet (nnU-Net LOOCV while its
rounds are still coming in) is drawn with the cells it has, and the table compares it only on
those cells -- the "same cells" rows say how many.

    python scripts/report/plot_loocv.py \
        --out ../masters_thesis/main_work/images --tables ../masters_thesis/main_work/tables

Writes loocv.png -- every cell's Test DSC under both splits, (a) U-Net, (b) nnU-Net -- and loocv.tex.
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
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parent.parent.parent
# (label, colour, 5-fold exp, LOOCV exp) -- the colours of plot_cv.py / plot_model_comparison.py
MODELS = [("U-Net", "#4a3aa7", "randswap/experiment_0_cv_baseline", "randswap/experiment_4_loocv/unet"),
          ("nnU-Net", "#b0306a", "randswap/experiment_1b_nnunet_1000ep", "randswap/experiment_4_loocv/nnunet")]
TRAIN_CELLS = {"5-fold": 11, "LOOCV": 13}
TIE = 0.01            # a cell counts as better/worse only beyond this margin (as in plot_cv.py)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def short(cell: str) -> str:
    return cell.replace("_cell", "-")


def scores(exp: str) -> dict[str, dict]:
    """cell -> class-1 metrics plus its round, the latest log entry winning."""
    last = {}
    for line in (REPO / "results" / exp / "eval_log_native.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not r["filter"]:
            continue
        rnd = r.get("fold") or re.search(r"fold_(\d+)", r["checkpoint"]).group(1)
        last[r["filter"]] = {**r["metrics"]["1"], "round": int(rnd)}
    return last


def pooled(m: dict, cells) -> float:
    tp, fp, fn = (sum(m[c][x] for c in cells) for x in ("tp", "fp", "fn"))
    return 2 * tp / (2 * tp + fp + fn)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="folder for loocv.png")
    ap.add_argument("--tables", type=Path, required=True, help="folder for loocv.tex")
    a = ap.parse_args()

    runs = {name: (scores(k5), scores(kl)) for name, _, k5, kl in MODELS}
    key = lambda c: (not c.startswith("high"), int(c.split("cell")[1]))
    cells = sorted(runs["U-Net"][0], key=key)
    for name, (K, L) in runs.items():
        if sorted(K) != sorted(cells) or not set(L) <= set(cells):
            raise SystemExit(f"{name}: the runs did not score the same cells")
    n_high = sum(c.startswith("high") for c in cells)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 6.8), facecolor="white", gridspec_kw={"hspace": 0.62})
    for ax, (name, col, _, _) in zip(axes, MODELS):
        K, L = runs[name]
        ax.set_facecolor(SURF)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        lo = min([K[c]["dsc"] for c in cells] + [L[c]["dsc"] for c in L])
        for x, c in enumerate(cells):
            k = K[c]["dsc"]
            ax.scatter([x - 0.12], [k], s=62, marker="o", facecolor="white", edgecolor=col, linewidth=1.8, zorder=4)
            if c not in L:
                ax.text(x - 0.12, k - 0.012, "not yet", ha="center", va="top", fontsize=6.8, color=INK2, style="italic")
                continue
            l, d = L[c]["dsc"], L[c]["dsc"] - k
            ax.plot([x - 0.12, x + 0.12], [k, l], color=GRID, lw=2.5, zorder=2)
            ax.scatter([x + 0.12], [l], s=62, marker="o", color=col, edgecolor="white", linewidth=1.5, zorder=4)
            ax.text(x, max(k, l) + 0.012, "0.000" if abs(d) < 5e-4 else f"{d:+.3f}".replace("-", "−"),
                    ha="center", va="bottom", fontsize=7.2, color=INK if abs(d) > TIE else INK2,
                    fontweight="bold" if abs(d) > TIE else "normal")
        ax.axvline(n_high - 0.5, color=GRID, lw=1)
        ax.set_xticks(range(len(cells)), [short(c) for c in cells], fontsize=8.6, color=INK)
        ax.set_xlim(-0.7, len(cells) - 0.3)
        ax.set_ylim(max(0.0, lo - 0.04), 1.03)
        ax.set_ylabel("Test DSC (native)", color=INK2, fontsize=9)
        both = [c for c in cells if c in L]
        ax.set_title(f"({'ab'[MODELS.index(next(m for m in MODELS if m[0] == name))]}) {name} — per-cell mean"
                     f" on {len(both)} cells: 5-fold {st.mean(K[c]['dsc'] for c in both):.3f}, "
                     f"leave-one-out {st.mean(L[c]['dsc'] for c in both):.3f}  ·  number = LOOCV minus 5-fold",
                     loc="left", fontsize=9.5, color=INK)
        ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=col,
                                  markeredgewidth=1.8, label=f"{name}, 5-fold CV (11 training cells)"),
                           Line2D([], [], marker="o", ls="", color=col,
                                  label=f"{name}, leave-one-cell-out (13 training cells)")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False, ncol=2)
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "loocv.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── table ─────────────────────────────────────────────────────────────────
    def f(v, on=False):
        return rf"\textbf{{{v:.3f}}}" if on else f"{v:.3f}"

    def pair(fn, name, cs):          # (5-fold, LOOCV) cells, the higher one bold
        K, L = runs[name]
        x, y = fn(K, cs), fn(L, cs)
        return f(x, x > y + 5e-4), f(y, y > x + 5e-4)

    mean = lambda m, cs: st.mean(m[c]["dsc"] for c in cs)
    sd = lambda m, cs: st.stdev(m[c]["dsc"] for c in cs)
    cols, notes, partial = [], [], []
    for name, *_ in MODELS:
        K, L = runs[name]
        cs = [c for c in cells if c in L]
        if len(cs) < len(cells):     # the 5-fold column is re-averaged over these cells only -- say so
            allk = [K[c]["dsc"] for c in cells]
            partial.append((name, len(cs), [short(c) for c in cells if c not in L],
                            st.mean(allk), st.stdev(allk), pooled(K, cells)))
        d = [L[c]["dsc"] - K[c]["dsc"] for c in cs]
        hi, lw = [c for c in cs if c.startswith("high")], [c for c in cs if c.startswith("low")]
        worst = lambda m: (lambda c: f"{short(c)} ({m[c]['dsc']:.3f})")(min(cs, key=lambda c: m[c]["dsc"]))
        cols.append({
            "cells": (f"{len(cs)}", f"{len(cs)} of {len(cells)}" if len(cs) < len(cells) else f"{len(cs)}"),
            "rounds": (f"{len({K[c]['round'] for c in cells})}", f"{len({L[c]['round'] for c in cs})}"
                       + ("" if len(cs) == len(cells) else f" of {len(cells)}")),
            "pooled": pair(pooled, name, cs),
            "mean": tuple(rf"{v} $\pm$ {sd(m, cs):.3f}" for v, m in zip(pair(mean, name, cs), (K, L))),
            "hi": pair(mean, name, hi), "lo": pair(mean, name, lw),
            "worst": (worst(K), worst(L)),
            "moved": ("---", f"{sum(v > TIE for v in d)} up / {sum(v < -TIE for v in d)} down"),
            "p": ("---", f"{wilcoxon(d).pvalue:.2f}"),
        })
    rows = [("Training cells per round", "cells", None),
            ("Rounds trained", "rounds", None),
            ("Cells compared", "cells", None),
            None,
            (r"\textbf{Per-cell mean DSC}", "mean", None),     # the headline (2026-10-15)
            ("Pooled DSC", "pooled", None),
            (r"\hiHER{High HER2} mean", "hi", None),
            (r"\loHER{Low HER2} mean", "lo", None),
            ("Worst cell", "worst", None),
            None,
            (rf"Cells moved by $>{TIE}$", "moved", None),
            ("Wilcoxon $p$, per cell", "p", None)]
    dag = lambda n: r"$^\dagger$" if any(p[0] == n for p in partial) else ""
    lines = [r"% GENERATED by scripts/report/plot_loocv.py -- do not edit by hand.",
             r"\begin{tabular}{lcccc}", r"\toprule",
             rf" & \multicolumn{{2}}{{c}}{{\textcolor{{ecE}}{{\textbf{{U-Net}}}}{dag('U-Net')}}} & "
             rf"\multicolumn{{2}}{{c}}{{\textcolor{{ecF}}{{\textbf{{nnU-Net}}}}, 1000 ep.{dag('nnU-Net')}}} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r" & 5-fold & LOOCV & 5-fold & LOOCV \\",
             r"\midrule"]
    for r in rows:
        if r is None:
            lines.append(r"\midrule")
            continue
        label, k, _ = r
        if label.startswith("Training cells"):
            vals = [str(TRAIN_CELLS["5-fold"]), str(TRAIN_CELLS["LOOCV"])] * 2
        else:
            vals = [v for c in cols for v in c[k]]
        lines.append(" & ".join([label, *vals]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    for n, k, miss, m, sdv, pl in partial:
        lines.append(rf"\par\vspace{{3pt}}{{\scriptsize $^\dagger$\,{n}: both columns cover only the {k} cells "
                     rf"LOOCV has scored ({', '.join(miss)} missing), so its 5-fold numbers here differ from "
                     rf"the 15-cell ones: 5-fold per-cell mean on all 15 cells {m:.3f} $\pm$ {sdv:.3f} "
                     rf"(Pooled {pl:.3f}).}}")
    lines.append("")
    a.tables.mkdir(parents=True, exist_ok=True)
    (a.tables / "loocv.tex").write_text("\n".join(lines))
    print(f"-> {a.out / 'loocv.png'}, {a.tables / 'loocv.tex'}")


if __name__ == "__main__":
    main()
