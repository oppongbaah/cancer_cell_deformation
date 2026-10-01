#!/usr/bin/env python3
"""
Figures for the experiments chapter of the 1 October report.

Every number is read from the experiments' NATIVE-resolution eval logs
(results/postswap/<experiment>/eval_log_native.jsonl, scored by eval_native.py at
1228x922 so runs trained at 256, 512 or on a crop sit on one grid) and from
results/postswap/latency_ms.json. Nothing is typed in here.

    python scripts/report/plot_experiments.py --out ../masters_thesis/images

Writes:
  exp_summary.png   (a) pooled Dice of every run, experiment by experiment, with each run's
                        worst-to-best cell range and the baseline's seed band;
                    (b) accuracy against speed, with the camera's real-time budget.
  exp_1a.png, exp_1b.png, exp_2.png, exp_3.png   per-cell testing results per experiment.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path("results/postswap")
CELLS = ["high_cell1", "high_cell9", "low_cell5", "low_cell6"]
CELL_LABEL = {c: c.replace("_cell", "-") for c in CELLS}

# Categorical slots 1-4 of the reference palette, validated (scripts/validate_palette.js,
# light): CVD worst adjacent dE 9.1, normal-vision 22.9. Two slots sit below 3:1 on the
# surface, so every mark is direct-labelled and the report carries the tables.
GREY, C1A, C1B, C2, C3 = "#8a8985", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C4 = "#4a3aa7"   # exp 4 (cross-validation): slot 7, validated adjacent to the others
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1", "#fcfcfb"


def load(exp: str) -> dict[str, dict]:
    """run -> {cell: metrics, 'ALL': metrics}; the latest entry per (run, scope) wins."""
    out: dict[str, dict] = {}
    for line in (R / exp / "eval_log_native.jsonl").read_text().splitlines():
        r = json.loads(line)
        run = Path(r["checkpoint"]).parent.name
        out.setdefault(run, {})[r["filter"] or "ALL"] = r["metrics"]["1"]
    return out


def style(ax, ylab=None):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if ylab:
        ax.set_ylabel(ylab, color=INK2, fontsize=9)


def pooled(m):
    return m["ALL"]["dsc"]


def cell_range(m):
    if "_range" in m:                      # cross-validation: worst..best over all 15 cells
        return m["_range"]
    v = [m[c]["dsc"] for c in CELLS]
    return min(v), max(v)


def load_cv(exp: str = "randswap/experiment_0_cv_baseline", run: str = "cv_baseline") -> dict:
    """The CV result in the same shape as an eval-log run: pooled DSC + 15-cell range."""
    import csv
    base = Path("results") / exp
    summ = json.loads((base / f"{run}_cv_summary.json").read_text())
    d = [float(r["dsc"]) for r in csv.DictReader((base / f"{run}_cv_cells.csv").open())]
    return {"ALL": {"dsc": summ["pooled_dsc"]}, "_range": (min(d), max(d)), "_n": len(d)}


def summary(out: Path) -> None:
    base, e1a, e1b = load("experiment_0a_seed_baseline"), load("experiment_1a_optimizer"), load("experiment_1b_loss")
    e2, e3 = load("experiment_2_resolution"), load("experiment_3_cropping")
    lat = json.loads((R / "latency_ms.json").read_text())

    seeds = [f"postswap_ftl_cw142_s{s}" for s in (42, 43, 44, 45)]
    lov = [f"e1b_lovasz_s{s}" for s in (42, 43, 44, 45)]
    cv = load_cv()
    # (group label, colour, [(run label, metrics, kind)])   kind: ens | single | cv
    # Seeded experiments show their ENSEMBLE only (user, 2026-10-01); single runs stay.
    groups = [
        ("Baseline\nensemble of 4, 256", GREY, [("ensemble", base["ENSEMBLE_baseline_x4"], "ens")]),
        ("1a  SGD\n256", C1A, [("SGD s43", e1a["e1a_sgd_nesterov"], "single")]),
        ("1b  + Lovász\nensemble of 4, 256", C1B, [("ensemble", e1b["ENSEMBLE_lovasz_x4"], "ens")]),
        ("2  resolution\nLovász s42, batch 4×4", C2, [("resize 256", e2["e2a_res256_b4"], "single"),
                                                     ("resize 512", e2["e2b_res512_b4"], "single")]),
        ("3  cropping\nLovász s42, batch 4×4", C3, [("crop 512", e3["e3a_crop512_b4"], "single"),
                                                   ("crop 384", e3["e3b_crop384_b4"], "single")]),
        ("4  cross-validation\ncrop 384, 5 rounds", C4, [("crop 384,\n15 random cells", cv, "cv")]),
    ]

    fig = plt.figure(figsize=(12.4, 8.6), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1], hspace=0.42)
    ax = fig.add_subplot(gs[0])
    style(ax, "Pooled DSC, trapped cell (native 1228×922)")

    # Reference = the baseline ENSEMBLE (2026-10-01, user: the ensemble is the better
    # comparison than a seed mean, which no single model achieves).
    ref = pooled(base["ENSEMBLE_baseline_x4"])
    ax.axhline(ref, color=GREY, lw=1, ls=(0, (4, 3)))

    x, xt, xl, best = 0.0, [], [], (None, -1)
    for gi, (glabel, col, runs) in enumerate(groups):
        x0 = x
        seedx, seedv = [], []
        for label, m, kind in runs:
            lo, hi = cell_range(m)
            v = pooled(m)
            ax.vlines(x, lo, hi, color=col, lw=2, alpha=0.35, zorder=2)        # worst..best cell
            mk = {"seed": "o", "single": "o", "ens": "D", "cv": "s"}[kind]
            ax.scatter([x], [v], s=70 if kind != "seed" else 46, marker=mk, color=col,
                       edgecolor="white", linewidth=1.6, zorder=4)
            is_best = label == "crop 384"
            ax.text(x + 0.13, v + (0.016 if kind == "ens" else 0), f"{v:.3f}", ha="left",
                    va="center", fontsize=7.8, color=INK,
                    fontweight="bold" if is_best else "normal")
            ax.text(x, min(lo, v) - 0.014, label + ("\n(best of 1–3)" if is_best else ""), ha="center",
                    va="top", fontsize=7.4, color=INK if is_best else INK2,
                    fontweight="bold" if is_best else "normal")
            if kind == "seed":
                seedx.append(x); seedv.append(v)
            if v > best[1]:
                best = (x, v)
            x += 1.15 if kind != "seed" else 0.95
        xt.append((x0 + x - 1) / 2); xl.append(glabel)
        if col == C4:
            # The CV is scored on 15 RANDOMLY drawn cells, every other run on the 4
            # hand-picked ones: shade it so the two are not read as one comparison.
            ax.axvspan(x0 - 0.55, x - 0.6, color=C4, alpha=0.06, lw=0, zorder=0)
            ax.text((x0 + x - 1.15) / 2, 0.985, "different test set", ha="center", va="top",
                    fontsize=7.6, color=C4, style="italic")
        x += 0.9
        if gi < len(groups) - 1:
            ax.axvline(x - 0.95, color=GRID, lw=1)
    ax.set_xticks(xt, xl, fontsize=8.6, color=INK)
    ax.set_xlim(-1.2, x - 0.5)
    ax.set_ylim(0.44, 1.0)
    ax.set_title("(a)  Every run on one grid — dot = Pooled DSC, bar = worst to best testing cell",
                 loc="left", fontsize=10, color=INK, pad=10)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, label="single model / one seed"),
                       Line2D([], [], marker="D", ls="", color=MUTED, label="ensemble of four"),
                       Line2D([], [], marker="s", ls="", color=C4, label="cross-validation, 15 random cells"),
                       Line2D([], [], color=MUTED, lw=2, alpha=0.5, label="range across the testing cells (4; CV: 15)"),
                       Line2D([], [], color=GREY, lw=1, ls=(0, (4, 3)),
                              label=f"baseline ensemble ({ref:.3f}), the reference")],
              loc="lower right", fontsize=7.4, frameon=False, ncol=3)

    # (b) accuracy against speed
    bx = fig.add_subplot(gs[1])
    style(bx, "Pooled DSC (native)")
    budget = 1000 / 22
    bx.axvspan(budget, 70, color="#e34948", alpha=0.06, lw=0)
    bx.axvline(budget, color="#e34948", lw=1, ls=(0, (4, 3)))
    bx.text(budget + 0.6, 0.905, "slower than the\n22 fps camera", fontsize=7.8, color=INK2, va="top")
    pts = [("baseline ensemble", lat["ensemble4_256"], ref, GREY),
           ("SGD s43", lat["single_256"], pooled(e1a["e1a_sgd_nesterov"]), C1A),
           ("resize 256 (Lovász s42)", lat["single_256"], pooled(e2["e2a_res256_b4"]), C2),
           ("Lovász ensemble", lat["ensemble4_256"], pooled(e1b["ENSEMBLE_lovasz_x4"]), C1B),
           ("resize 512", lat["single_512_resize"], pooled(e2["e2b_res512_b4"]), C2),
           ("crop 512", lat["single_512_crop"], pooled(e3["e3a_crop512_b4"]), C3),
           ("crop 384", lat["single_384_crop"], pooled(e3["e3b_crop384_b4"]), C3),
           ("CV, 15 cells", lat["single_384_crop"], pooled(cv), C4)]
    offs = {"SGD s43": (6, 5), "resize 256 (Lovász s42)": (6, -6),
            "baseline ensemble": (-9, -6), "Lovász ensemble": (-9, 2), "resize 512": (6, 4),
            "crop 512": (-9, 7), "crop 384": (8, 0), "CV, 15 cells": (8, 0)}
    for label, ms, v, col in pts:
        bx.scatter([ms], [v], s=64, color=col, edgecolor="white", linewidth=1.6, zorder=4,
                   marker="D" if "ensemble" in label else "s" if label.startswith("CV") else "o")
        dx, dy = offs[label]
        bx.annotate(f"{label}  {v:.3f}", (ms, v), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.8, color=INK, ha="left" if dx > 0 else "right", va="center")
    bx.set_xlim(18, 70)
    bx.set_ylim(min(p[2] for p in pts) - 0.03, 0.92)
    bx.set_xlabel("full pipeline, ms per frame (fp32, RTX 4060)", color=INK2, fontsize=9)
    bx.set_title("(b)  Accuracy against speed — only the left of the dashed line keeps up with the camera",
                 loc="left", fontsize=10, color=INK, pad=8)
    fig.savefig(out / "exp_summary.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def per_cell(out: Path, name: str, title: str, runs: list[tuple[str, dict, str, str]]) -> None:
    """Per-cell testing Dice: one row of dots per run across the 4 cells + pooled."""
    fig, ax = plt.subplots(figsize=(8.6, 2.9 + 0.08 * len(runs)), facecolor="white")
    style(ax, "Dice (native)")
    scopes = CELLS + ["ALL"]
    n = len(runs)
    width = 0.72
    for i, (label, m, col, ls) in enumerate(runs):
        dx = -width / 2 + width * (i + 0.5) / n
        xs = [k + dx for k in range(len(scopes))]
        ys = [m[s]["dsc"] for s in scopes]
        # Dots only: the y-axis starts above 0, and a stem from a cut axis would overstate
        # differences the way a truncated bar does.
        ax.scatter(xs, ys, s=58, color=col, edgecolor="white", linewidth=1.2, zorder=4,
                   alpha=1 if ls == "-" else 0.6)
        for xx, yy in zip(xs, ys):
            ax.text(xx, yy + 0.012, f"{yy:.2f}", ha="center", va="bottom", fontsize=6.6, color=INK,
                    rotation=90 if n > 4 else 0)
    ax.set_xticks(range(len(scopes)), [CELL_LABEL.get(s, "Pooled") for s in scopes], fontsize=9, color=INK)
    ax.axvline(len(CELLS) - 0.5, color=GRID, lw=1)
    ax.set_ylim(0.4 if all(m[s]["dsc"] > 0.45 for _, m, _, _ in runs for s in scopes) else 0, 1.06)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, alpha=1 if ls == "-" else 0.6, label=l)
                       for l, _, c, ls in runs],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(n, 5), fontsize=7.8, frameon=False)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def table(path: Path, rows: list[tuple[str, dict]], bold: str | None = None,
          midrule_after: list[int] | None = None) -> None:
    """LaTeX table of testing results (native): per cell (each cell's Test DSC), Pooled DSC (over every test
    frame's pixels), prec/rec. The per-cell mean was dropped 2026-10-01: with ~50 frames per
    testing cell it barely differs from the pooled score, and the cells are shown anyway."""
    lines = [r"% GENERATED by scripts/report/plot_experiments.py -- do not edit by hand.",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"\textbf{Model} & \textbf{high-1} & \textbf{high-9} & \textbf{low-5} & \textbf{low-6} & "
             r"\textbf{Pooled DSC} & \textbf{Prec.\,/\,Rec.} \\", r"\midrule"]
    for i, (label, m) in enumerate(rows, 1):
        cells = [m[c]["dsc"] for c in CELLS]
        vals = [f"{v:.3f}" for v in cells] + [f"{m['ALL']['dsc']:.3f}"]
        pr = (f"{m['ALL']['precision']:.3f} / {m['ALL']['recall']:.3f}" if "precision" in m["ALL"] else "---")
        if label == bold:
            vals = [rf"\textbf{{{v}}}" for v in vals]
            label = rf"\textbf{{{label}}}"
        lines.append(" & ".join([label, *vals, pr]) + r" \\")
        if midrule_after and i in midrule_after:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def start_table(path: Path) -> None:
    """Starting Point table: the 24 September baseline on the 256 grid it was reported on
    (eval_log.jsonl) beside the same models re-scored at native 1228x922 (eval_log_native.jsonl)."""
    def pooled(name: str) -> dict[str, float]:
        out = {}
        for line in (R / "experiment_0a_seed_baseline" / name).read_text().splitlines():
            r = json.loads(line)
            if r["filter"] is None:
                out[Path(r["checkpoint"]).parent.name] = r["metrics"]["1"]["dsc"]
        return out
    seeds = [f"postswap_ftl_cw142_s{k}" for k in (42, 43, 44, 45)]
    g, n = pooled("eval_log.jsonl"), pooled("eval_log_native.jsonl")
    def ms(d):
        v = [d[s] for s in seeds]
        return f"{st.mean(v):.3f} $\\pm$ {st.stdev(v):.3f}"
    ens = "ENSEMBLE_baseline_x4"
    lines = [r"% GENERATED by scripts/report/plot_experiments.py -- do not edit by hand.",
             r"\begin{tabular}{lcc}", r"\toprule",
             r" & \textbf{256 grid} & \textbf{Native} \\",
             r" & (24 Sept.) & ($1228\times922$) \\", r"\midrule",
             f"Single model (4 seeds) & {ms(g)} & {ms(n)} \\\\",
             f"\\textbf{{Ensemble of four}} & \\textbf{{{g[ens]:.3f}}} & \\textbf{{{n[ens]:.3f}}} \\\\",
             r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def seed_mean(runs: dict, names: list[str]) -> dict:
    return {s: {"dsc": st.mean(runs[n][s]["dsc"] for n in names)} for s in CELLS + ["ALL"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tables", type=Path, default=None, help="where the LaTeX tables go")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summary(a.out)
    base, e1a, e1b = load("experiment_0a_seed_baseline"), load("experiment_1a_optimizer"), load("experiment_1b_loss")
    e2, e3 = load("experiment_2_resolution"), load("experiment_3_cropping")
    per_cell(a.out, "exp_1a", "Experiment 1a — SGD against the AdamW baseline, same seed (43)",
             [("baseline s43 (AdamW)", base["postswap_ftl_cw142_s43"], GREY, "-"),
              ("SGD, Nesterov 0.99", e1a["e1a_sgd_nesterov"], C1A, "-")])
    per_cell(a.out, "exp_1b", "Experiment 1b — Lovász against the baseline: ensemble of four seeds each",
             [("baseline ensemble", base["ENSEMBLE_baseline_x4"], GREY, "-"),
              ("Lovász ensemble", e1b["ENSEMBLE_lovasz_x4"], C1B, "-")])
    per_cell(a.out, "exp_2", "Experiment 2 — whole frame resized to 256 vs 512 (Lovász, seed 42, batch 4×4)",
             [("resize 256", e2["e2a_res256_b4"], GREY, "-"), ("resize 512", e2["e2b_res512_b4"], C2, "-")])
    per_cell(a.out, "exp_3", "Experiment 3 — native crops against the resizes (Lovász, seed 42, batch 4×4)",
             [("resize 256", e2["e2a_res256_b4"], GREY, ":"), ("resize 512", e2["e2b_res512_b4"], C2, ":"),
              ("crop 512", e3["e3a_crop512_b4"], C3, ":"), ("crop 384", e3["e3b_crop384_b4"], C3, "-")])
    if a.tables:
        a.tables.mkdir(parents=True, exist_ok=True)
        seeds = [f"postswap_ftl_cw142_s{k}" for k in (42, 43, 44, 45)]
        lov = [f"e1b_lovasz_s{k}" for k in (42, 43, 44, 45)]
        start_table(a.tables / "start.tex")
        table(a.tables / "exp_1a.tex", [("baseline s43 (AdamW)", base["postswap_ftl_cw142_s43"]),
                                         ("SGD, Nesterov 0.99", e1a["e1a_sgd_nesterov"])])
        table(a.tables / "exp_1b.tex",
              [(f"baseline s{n[-2:]}", base[n]) for n in seeds] + [("baseline ensemble", base["ENSEMBLE_baseline_x4"])]
              + [(f"Lov\\'asz s{n[-2:]}", e1b[n]) for n in lov] + [("Lov\\'asz ensemble", e1b["ENSEMBLE_lovasz_x4"])],
              midrule_after=[5])
        table(a.tables / "exp_2.tex", [("resize 256", e2["e2a_res256_b4"]), ("resize 512", e2["e2b_res512_b4"])],
              bold="resize 512")
        table(a.tables / "exp_3.tex", [("resize 256", e2["e2a_res256_b4"]), ("resize 512", e2["e2b_res512_b4"]),
                                        ("crop 512", e3["e3a_crop512_b4"]), ("crop 384", e3["e3b_crop384_b4"])],
              bold="crop 384")
    print(f"-> {a.out}/exp_summary.png, exp_1a.png, exp_1b.png, exp_2.png, exp_3.png")


if __name__ == "__main__":
    main()
