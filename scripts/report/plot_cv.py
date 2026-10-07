#!/usr/bin/env python3
"""
Figure and tables for the randswap cross-validation (1 October report).

Reads what scripts/training/summarize_cv.py writes for one CV run:
results/<exp>/<run>_cv_cells.csv, <run>_cv_rounds.csv and <run>_cv_summary.json.
Nothing is typed in here.

    python scripts/report/plot_cv.py --exp randswap/experiment_0_cv_baseline --run cv_baseline \
        --out ../masters_thesis/images --tables ../masters_thesis/main_work/tables

Writes cv_cells.png (every cell's Test DSC, grouped by HER2 class, labelled with the round
that tested it), cv_rounds.tex and cv_cells.tex.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Categorical slots 1-2 of the reference palette (validated, light): blue, orange.
HIGH, LOW = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
# Benchmark: ours = slot 7 violet (Experiment 4's colour), nnU-Net = deep magenta (C5 in
# plot_experiments.py) -- validated all-pairs with the report's other series.
OURS, NNU = "#4a3aa7", "#b0306a"


def short(cell: str) -> str:
    return cell.replace("_cell", "-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--compare-exp", default=None, help="a second CV run to benchmark against (nnU-Net)")
    ap.add_argument("--compare-run", default=None)
    ap.add_argument("--distill-exp", default=None, help="student CV to set against ours and nnU-Net")
    ap.add_argument("--distill-run", default=None)
    a = ap.parse_args()
    if a.distill_exp:      # Experiments 6 and 7: the ensemble and MegaNet, each against ours
        trio(a, "randswap/experiment_2_ensemble", "ensemble", "Ensemble", "U-Net + nnU-Net", ENSEMBLE, "D",
             "ensemble", "ensemble_tta")
        trio(a, a.distill_exp, a.distill_run, "MegaNet", "student", MEGANET, "^", "meganet", "meganet")
        return
    if a.compare_exp:
        benchmark(a)
        return
    R = Path("results") / a.exp
    cells = list(csv.DictReader((R / f"{a.run}_cv_cells.csv").open()))
    rounds = list(csv.DictReader((R / f"{a.run}_cv_rounds.csv").open()))
    summ = json.loads((R / f"{a.run}_cv_summary.json").read_text())
    for c in cells:
        for k in ("dsc", "precision", "recall"):
            c[k] = float(c[k])
    key = lambda c: (c["her2"] != "high", int(c["cell"].split("cell")[1]))
    cells.sort(key=key)

    fig, ax = plt.subplots(figsize=(10.4, 3.9), facecolor="white")
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    xs = list(range(len(cells)))
    n_high = sum(c["her2"] == "high" for c in cells)
    for x, c in zip(xs, cells):
        col = HIGH if c["her2"] == "high" else LOW
        ax.scatter([x], [c["dsc"]], s=70, color=col, edgecolor="white", linewidth=1.6, zorder=4)
        ax.text(x + 0.16, c["dsc"], f"{c['dsc']:.2f}", ha="left", va="center", fontsize=7.6, color=INK)
        ax.text(x, 0.505, f"R{c['fold']}", ha="center", va="bottom", fontsize=7.2, color=INK2)
    for lo, hi, mean, col in ((0, n_high, summ["high_mean_dsc"], HIGH),
                              (n_high, len(cells), summ["low_mean_dsc"], LOW)):
        ax.hlines(mean, lo - 0.4, hi - 0.6, color=col, lw=1.6, ls=(0, (4, 3)), zorder=3)
        ax.text(hi - 0.55, mean, f" {mean:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(n_high - 0.5, color=GRID, lw=1)
    ax.set_xticks(xs, [short(c["cell"]) for c in cells], fontsize=8.6, color=INK)
    ax.set_xlim(-0.7, len(cells) - 0.1)
    ax.set_ylim(0.5, 1.04)
    ax.set_ylabel("Test DSC (native)", color=INK2, fontsize=9)
    ax.set_title(f"Every cell tested once, by a model that never saw it — mean over 15 cells "
                 f"{summ['per_cell_mean_dsc']:.3f} ± {summ['per_cell_sd']:.3f}; R = the round that "
                 f"tested it", loc="left", fontsize=9.5, color=INK)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=HIGH, label="high HER2"),
                       Line2D([], [], marker="o", ls="", color=LOW, label="low HER2"),
                       Line2D([], [], color=INK2, lw=1.6, ls=(0, (4, 3)), label="class mean")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False, ncol=3)
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "cv_cells.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    a.tables.mkdir(parents=True, exist_ok=True)
    by_fold = {}
    for c in cells:
        by_fold.setdefault(c["fold"], []).append(c)
    lines = [r"% GENERATED by scripts/report/plot_cv.py -- do not edit by hand.",
             r"\begin{tabular}{clcccc}", r"\toprule",
             r"\textbf{Round} & \textbf{Test cells (Test DSC)} & \textbf{Val. cell} & "
             r"\textbf{Best epoch} & \textbf{Internal val.} \\", r"\midrule"]
    for r in rounds:
        tc = ", ".join(f"{short(c['cell'])} ({c['dsc']:.2f})" for c in sorted(by_fold[r["fold"]], key=key))
        lines.append(f"{r['fold']} & {tc} & {short(r['val_cell'])} & {r['best_epoch']} & "
                     f"{float(r['internal_val_dsc']):.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (a.tables / "cv_rounds.tex").write_text("\n".join(lines))

    lines = [r"% GENERATED by scripts/report/plot_cv.py -- do not edit by hand.",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r" & \textbf{Cells} & \textbf{Per-cell mean DSC} & \textbf{Pooled DSC} & \textbf{Worst cell} \\",
             r"\midrule"]
    def row(name, cs, pooled=None):
        m = sum(c["dsc"] for c in cs) / len(cs)
        w = min(cs, key=lambda c: c["dsc"])
        return (f"{name} & {len(cs)} & {m:.3f} & {pooled if pooled else '---'} & "
                f"{short(w['cell'])} ({w['dsc']:.3f}) \\\\")
    lines.append(row(r"\textbf{All cells}", cells, f"\\textbf{{{summ['pooled_dsc']:.3f}}}"))
    lines.append(row("high HER2", [c for c in cells if c["her2"] == "high"]))
    lines.append(row("low HER2", [c for c in cells if c["her2"] == "low"]))
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (a.tables / "cv_cells.tex").write_text("\n".join(lines))
    print(f"-> {a.out / 'cv_cells.png'}, {a.tables}/cv_rounds.tex, cv_cells.tex")


def _cells(exp: str, run: str) -> dict[str, dict]:
    rows = csv.DictReader((Path("results") / exp / f"{run}_cv_cells.csv").open())
    return {r["cell"]: {k: (float(v) if k in ("dsc", "precision", "recall") else
                            int(v) if k in ("tp", "fp", "fn") else v) for k, v in r.items()} for r in rows}


def _train_hours(paths: list[Path], pattern: str) -> float:
    """Summed epoch times over every round's log, in hours."""
    tot = 0.0
    for p in paths:
        tot += sum(float(t) for t in re.findall(pattern, p.read_text()))
    return tot / 3600


def benchmark(a) -> None:
    """Ours against nnU-Net on the SAME randswap rounds: per-cell dumbbell + one LaTeX table."""
    ours, nn = _cells(a.exp, a.run), _cells(a.compare_exp, a.compare_run)
    if not set(nn) <= set(ours):
        raise SystemExit(f"cells not in our CV: {sorted(set(nn) - set(ours))}")
    # While nnU-Net is still running, compare on the cells it has scored so far -- ours
    # restricted to the same cells, so every row is a like-for-like pair.
    key = lambda c: (ours[c]["her2"] != "high", int(c.split("cell")[1]))
    cells = sorted(nn, key=key)
    rounds = sorted({nn[c]["fold"] for c in cells}, key=int)
    n_rounds = len({ours[c]["fold"] for c in ours})
    partial = len(cells) < len(ours)
    span = rounds[0] if len(rounds) == 1 else (f"{rounds[0]}–{rounds[-1]}" if
            [int(r) for r in rounds] == list(range(int(rounds[0]), int(rounds[-1]) + 1)) else ", ".join(rounds))
    scope = (f"round{'s' if len(rounds) > 1 else ''} {span} of {n_rounds}, "
             f"{len(cells)} of {len(ours)} cells" if partial else f"same {len(cells)} cells, same rounds")

    fig, ax = plt.subplots(figsize=(max(6.4, 0.62 * len(cells) + 1.2), 3.9), facecolor="white")
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    lo = min(min(ours[c]["dsc"], nn[c]["dsc"]) for c in cells)
    for x, c in enumerate(cells):
        o, n = ours[c]["dsc"], nn[c]["dsc"]
        ax.plot([x - 0.12, x + 0.12], [o, n], color=GRID, lw=2.5, zorder=2)
        ax.scatter([x - 0.12], [o], s=62, marker="s", color=OURS, edgecolor="white", linewidth=1.5, zorder=4)
        ax.scatter([x + 0.12], [n], s=62, marker="o", color=NNU, edgecolor="white", linewidth=1.5, zorder=4)
        d = n - o
        ax.text(x, max(o, n) + 0.012, "0.000" if abs(d) < 5e-4 else f"{d:+.3f}".replace("-", "−"),
                ha="center", va="bottom", fontsize=7.2,
                color=INK if abs(d) >= 0.02 else INK2)
        ax.text(x, lo - 0.035, f"R{ours[c]['fold']}", ha="center", va="top", fontsize=7.2, color=INK2)
    n_high = sum(ours[c]["her2"] == "high" for c in cells)
    ax.axvline(n_high - 0.5, color=GRID, lw=1)
    ax.set_xticks(range(len(cells)), [short(c) for c in cells], fontsize=8.6, color=INK)
    ax.set_xlim(-0.7, len(cells) - 0.3)
    ax.set_ylim(lo - 0.08, 1.04)
    ax.set_ylabel("Test DSC (native)", color=INK2, fontsize=9)
    mo, mn = st.mean(ours[c]["dsc"] for c in cells), st.mean(nn[c]["dsc"] for c in cells)
    ax.set_title(f"{scope[0].upper() + scope[1:]}\nper-cell mean: U-Net {mo:.3f}, nnU-Net {mn:.3f}"
                 f"  ·  number = nnU-Net minus U-Net", loc="left", fontsize=9.5, color=INK)
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=OURS, label="U-Net (crop 384, Lovász)"),
                       Line2D([], [], marker="o", ls="", color=NNU, label="nnU-Net (raw frames, 100 epochs)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False, ncol=2)
    fig.savefig(a.out / "cv_benchmark.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    def agg(cs: dict) -> dict:
        d = [cs[c]["dsc"] for c in cells]
        tp, fp, fn = (sum(cs[c][k] for c in cells) for k in ("tp", "fp", "fn"))
        w = min(cells, key=lambda c: cs[c]["dsc"])
        return {"mean": st.mean(d), "sd": st.stdev(d), "pooled": 2 * tp / (2 * tp + fp + fn),
                "prec": tp / (tp + fp), "rec": tp / (tp + fn),
                "high": st.mean(cs[c]["dsc"] for c in cells if cs[c]["her2"] == "high"),
                "low": st.mean(cs[c]["dsc"] for c in cells if cs[c]["her2"] == "low"),
                "worst": f"{short(w)} ({cs[w]['dsc']:.3f})"}
    n_hi = sum(ours[c]["her2"] == "high" for c in cells)
    A, B = agg(ours), agg(nn)
    # A cell counts as won only by a margin above TIE (0.01): differences of 0.000-0.002 are
    # inside run-to-run noise and would otherwise read as wins.
    TIE = 0.01
    better = sum(nn[c]["dsc"] - ours[c]["dsc"] > TIE for c in cells)
    worse = sum(ours[c]["dsc"] - nn[c]["dsc"] > TIE for c in cells)
    ties = len(cells) - better - worse
    lat_p = Path("results") / a.compare_exp / "latency_ms.json"
    lat = json.loads(lat_p.read_text()) if lat_p.exists() else None
    # Training time PER ROUND (mean over the rounds that finished), so a partial run compares fairly.
    our_logs = sorted((Path("logs") / a.exp).glob(f"{a.run}_fold_[0-9]*.log"))
    h_ours = _train_hours(our_logs, r"Epoch \d+/\d+ .*?\(([\d.]+)s\)") / len(our_logs)
    nn_logs = [sorted(f.glob("training_log_*.txt"))[-1]
               for f in sorted((Path("checkpoints") / a.compare_exp).glob("Dataset*/*/fold_*"))
               if (f / "checkpoint_final.pth").exists()]
    h_nn = _train_hours(nn_logs, r"Epoch time: ([\d.]+) s") / max(len(nn_logs), 1)
    fps = lambda ms: f"{ms:.0f} ms ({1000 / ms:.0f} fps)"

    def b(x, y, hi=True):           # bold the better of two numbers
        bx, by = (x > y, y > x) if hi else (x < y, y < x)
        f = lambda v, on: rf"\textbf{{{v:.3f}}}" if on else f"{v:.3f}"
        return f(x, bx), f(y, by)
    def per_round(cs: dict, r: str) -> dict:
        tp, fp, fn = (sum(cs[c][k] for c in cells if cs[c]["fold"] == r) for k in ("tp", "fp", "fn"))
        return {"pooled": 2 * tp / (2 * tp + fp + fn), "prec": tp / (tp + fp), "rec": tp / (tp + fn)}
    PR = {r: (per_round(ours, r), per_round(nn, r)) for r in rounds}
    pr = lambda m: f"{m['prec']:.3f} / {m['rec']:.3f}"
    # Four sections (user, 2026-10-01): Pooled DSC per round | precision / recall per round |
    # per-class and per-cell | method and cost. The per-cell mean was dropped from this table.
    rows = [(f"Pooled DSC, round {r}", *b(PR[r][0]["pooled"], PR[r][1]["pooled"])) for r in rounds]
    rows += [(rf"\textbf{{Pooled DSC, {'all ' if not partial else ''}{len(cells)} cells}}", *b(A["pooled"], B["pooled"]))]
    rows += [None]
    rows += [(f"Precision / recall, round {r}", pr(PR[r][0]), pr(PR[r][1])) for r in rounds]
    rows += [(f"Precision / recall, {len(cells)} cells", pr(A), pr(B))]
    rows += [None,
        (rf"\hiHER{{High HER2}} mean ({n_hi} cell{'s' * (n_hi != 1)})", *b(A["high"], B["high"])),
        (rf"\loHER{{Low HER2}} mean ({len(cells) - n_hi} cell{'s' * (len(cells) - n_hi != 1)})",
         *b(A["low"], B["low"])),
        ("Worst cell", A["worst"], B["worst"]),
        (rf"Cells better by $>{TIE}$ ({ties} tied)", f"{worse} of {len(cells)}", f"{better} of {len(cells)}"),
        None,
        ("Input", r"denoise + CLAHE, $384$ crop", r"raw frame, $1024\times1280$ patch"),
        ("Epoch chosen", "best on the val.\\ cell", "last (100)"),
        ("Training time per round", f"{60 * h_ours:.0f} min", f"{h_nn:.1f} h"),
        # ours: same-session measurement with nnU-Net once it exists; until then the 384-crop
        # full-pipeline figure of Experiment 3 (results/postswap/latency_ms.json, same span)
        ("Time per frame (to features)", fps(lat["ours_crop384"] if lat else json.loads(
            Path("results/postswap/latency_ms.json").read_text())["single_384_crop"]),
         fps(lat["nnunet_tta"]) if lat else r"\emph{measured after round 5}"),
    ]
    if lat and "nnunet_forward_only_fp16" in lat:
        rows.append((r"\quad without TTA / network alone, fp16", "---",
                     f"{lat['nnunet_no_tta']:.0f} ms / {lat['nnunet_forward_only_fp16']:.0f} ms"))
    lines = [r"% GENERATED by scripts/report/plot_cv.py --compare-exp -- do not edit by hand.",
             r"\begin{tabular}{lcc}", r"\toprule",
             rf" \footnotesize\emph{{{scope}}} & \textbf{{U-Net}} & \textbf{{nnU-Net}} \\", r"\midrule"]
    lines += [r"\midrule" if r is None else " & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (a.tables / "cv_benchmark.tex").write_text("\n".join(lines))
    print(f"-> {a.out / 'cv_benchmark.png'}, {a.tables / 'cv_benchmark.tex'}")


MEGANET = "#7a4a00"    # dark brown: separates from every report series all-pairs (validate_palette.js)
ENSEMBLE = "#3d3d3a"   # dark neutral + diamond: a combination of two models, not a new series (8 hues
                       # cannot all be told apart); always direct-labelled
NN_EXP = "randswap/experiment_1_nnunet"
LATENCY = Path("results") / NN_EXP / "latency_ms.json"


def _ms(key: str) -> str:
    v = json.loads(LATENCY.read_text())[key]
    return f"{v:.0f} ms ({1000 / v:.0f} fps)"


def trio(a, exp: str, run: str, name: str, sub: str, colour: str, marker: str, stem: str, lat_key: str) -> None:
    """A model against ours, with nnU-Net drawn as the reference line: per-cell figure <stem>_cells.png
    and table <stem>_cv.tex (Pooled DSC per round | per-cell, HER2 class, worst | wins and speed)."""
    ours = _cells(a.exp, a.run)
    new = _cells(exp, run)
    nn = _cells(NN_EXP, "nnunet")
    key = lambda c: (ours[c]["her2"] != "high", int(c.split("cell")[1]))
    cells = sorted(ours, key=key)

    fig, ax = plt.subplots(figsize=(10.4, 3.9), facecolor="white")
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    lo = min(min(ours[c]["dsc"], new[c]["dsc"]) for c in cells)
    for x, c in enumerate(cells):
        o, s_, n = ours[c]["dsc"], new[c]["dsc"], nn[c]["dsc"]
        ax.plot([x - 0.12, x + 0.12], [o, s_], color=GRID, lw=2.5, zorder=2)
        ax.scatter([x - 0.12], [o], s=62, marker="s", color=OURS, edgecolor="white", linewidth=1.5, zorder=4)
        ax.scatter([x + 0.12], [s_], s=74, marker=marker, color=colour, edgecolor="white", linewidth=1.2, zorder=4)
        ax.hlines(n, x - 0.3, x + 0.3, color=NNU, lw=1.6, zorder=3)
        d = s_ - o
        ax.text(x, max(o, s_, n) + 0.012, "0.000" if abs(d) < 5e-4 else f"{d:+.3f}".replace("-", "−"),
                ha="center", va="bottom", fontsize=7.2, color=INK if abs(d) >= 0.02 else INK2)
        ax.text(x, lo - 0.035, f"R{ours[c]['fold']}", ha="center", va="top", fontsize=7.2, color=INK2)
    n_high = sum(ours[c]["her2"] == "high" for c in cells)
    ax.axvline(n_high - 0.5, color=GRID, lw=1)
    ax.set_xticks(range(len(cells)), [short(c) for c in cells], fontsize=8.6, color=INK)
    ax.set_xlim(-0.7, len(cells) - 0.3)
    ax.set_ylim(lo - 0.08, 1.04)
    ax.set_ylabel("Test DSC (native)", color=INK2, fontsize=9)
    mo, ms = st.mean(ours[c]["dsc"] for c in cells), st.mean(new[c]["dsc"] for c in cells)
    ax.set_title(f"Same 15 cells, same rounds — per-cell mean: U-Net {mo:.3f}, {name} {ms:.3f}"
                 f"  ·  number = {name} minus U-Net", loc="left", fontsize=9.5, color=INK)
    ax.legend(handles=[Line2D([], [], marker="s", ls="", color=OURS, label="U-Net (CV model)"),
                       Line2D([], [], marker=marker, ls="", color=colour, label=f"{name} ({sub})"),
                       Line2D([], [], color=NNU, lw=1.6, label="nnU-Net alone, for reference")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False, ncol=3)
    fig.savefig(a.out / f"{stem}_cells.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    def pooled(cs, cl):
        tp, fp, fn = (sum(cs[c][k] for c in cl) for k in ("tp", "fp", "fn"))
        return 2 * tp / (2 * tp + fp + fn)
    rounds = sorted({ours[c]["fold"] for c in cells}, key=int)
    TIE = 0.01
    better = sum(new[c]["dsc"] - ours[c]["dsc"] > TIE for c in cells)
    worse = sum(ours[c]["dsc"] - new[c]["dsc"] > TIE for c in cells)
    hi = [c for c in cells if ours[c]["her2"] == "high"]
    lo_ = [c for c in cells if ours[c]["her2"] == "low"]
    m = lambda cs, cl: st.mean(cs[c]["dsc"] for c in cl)
    worst = lambda cs: min(cells, key=lambda c: cs[c]["dsc"])

    def row(label, f):          # bold = the better of ours and the new model (nnU-Net is the reference)
        o, s_ = f(ours), f(new)
        bo, bs = (rf"\textbf{{{o:.3f}}}", f"{s_:.3f}") if o > s_ else (f"{o:.3f}", rf"\textbf{{{s_:.3f}}}")
        return f"{label} & {bo} & {bs} & {f(nn):.3f} \\\\"
    lines = [rf"% GENERATED by scripts/report/plot_cv.py ({stem}) -- do not edit by hand.",
             r"\begin{tabular}{lccc}", r"\toprule",
             rf" & \textbf{{U-Net}} & \textbf{{{name}}} & \textbf{{nnU-Net}} \\",
             rf" & \footnotesize(CV model) & \footnotesize({sub}) & \footnotesize(alone) \\", r"\midrule"]
    for r in rounds:
        cl = [c for c in cells if ours[c]["fold"] == r]
        lines.append(row(f"Pooled DSC, round {r}", lambda cs: pooled(cs, cl)))
    lines.append(row(r"\textbf{Pooled DSC, all 15 cells}", lambda cs: pooled(cs, cells)))
    lines += [r"\midrule", row("Per-cell mean DSC", lambda cs: m(cs, cells)),
              row(rf"\hiHER{{High HER2}} mean ({len(hi)} cells)", lambda cs: m(cs, hi)),
              row(rf"\loHER{{Low HER2}} mean ({len(lo_)} cells)", lambda cs: m(cs, lo_)),
              f"Worst cell & {short(worst(ours))} {ours[worst(ours)]['dsc']:.3f} & "
              f"{short(worst(new))} {new[worst(new)]['dsc']:.3f} & {short(worst(nn))} {nn[worst(nn)]['dsc']:.3f} \\\\",
              r"\midrule",
              f"Cells won / lost vs U-Net ($>{TIE}$) & & {better} / {worse} of {len(cells)} & \\\\",
              f"Time per frame & {_ms('ours_crop384')} & {_ms(lat_key)} & {_ms('nnunet_tta')} \\\\",
              r"\bottomrule", r"\end{tabular}", ""]
    (a.tables / f"{stem}_cv.tex").write_text("\n".join(lines))
    print(f"-> {a.out / (stem + '_cells.png')}, {a.tables / (stem + '_cv.tex')}")


if __name__ == "__main__":
    main()
