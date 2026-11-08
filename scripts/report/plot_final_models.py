#!/usr/bin/env python3
"""
"Final Models at a Glance" -- the closing page of the 8 October report (written 2026-10-21).

The two models the project keeps, each estimated by BOTH cross-validations:

  U-Net    (crop 384, our recipe)  -> live tracking, every camera frame
  nnU-Net  (1000 epochs)           -> the classifier's masks, one frame per laser step

Reads the four runs' native eval logs, the latency benchmark, and the two deployment models'
training logs. Nothing is typed in here except what each model is FOR.

    python scripts/report/plot_final_models.py --out ../masters_thesis/main_work/images --tables ../masters_thesis/main_work/tables

Writes final_models.png -- (a) accuracy against time per frame, every cell's score behind each
estimate, the camera's real-time budget shaded; (b) every cell's score under both CVs, both models --
and final_models.tex, the two model cards.
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
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent.parent
RS = REPO / "results" / "randswap"
MODELS = {   # name -> colour, 5-fold exp, LOOCV exp, latency key
    "U-Net": ("#4a3aa7", "experiment_0_cv_baseline", "experiment_4_loocv/unet", "ours_crop384"),
    "nnU-Net": ("#b0306a", "experiment_1b_nnunet_1000ep", "experiment_4_loocv/nnunet", "nnunet_tta"),
}
CAMERA_FPS = 22.0
HIGH, LOW = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def short(c: str) -> str:
    return c.replace("_cell", "-")


def scores(exp: str) -> dict[str, dict]:
    out = {}
    for line in (RS / exp / "eval_log_native.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["filter"]:
            out[r["filter"]] = r["metrics"]["1"]
    return out


def unet_deploy() -> tuple[int, float, int]:
    """(epochs, training hours, frames) of the U-Net deployment model, from its log and split record."""
    log = (REPO / "logs/randswap/experiment_0_cv_baseline/cv_baseline_fold_deploy.log").read_text()
    secs = [float(s) for s in re.findall(r"Epoch \d+/\d+ .*\(([\d.]+)s\)", log)]
    split = json.loads((RS / "experiment_0_cv_baseline/cv_baseline_fold_deploy_split.json").read_text())
    return len(secs), sum(secs) / 3600, sum(split["annotated_frames_per_cell"].values()) + split["n_legacy_frames"]


def nnunet_deploy() -> tuple[int, float, int]:
    log = sorted((REPO / "logs/hpc").glob("loocv_nnunet_deploy_*.out"))[-1].read_text()
    secs = [float(s) for s in re.findall(r"Epoch time: ([\d.]+) s", log)]
    rec = json.loads((RS / "experiment_4_loocv/nnunet/nnunet_deploy.jsonl").read_text().splitlines()[-1])
    return len(secs), sum(secs) / 3600, rec["frames"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    a = ap.parse_args()
    lat = json.loads((RS / "experiment_1_nnunet/latency_ms.json").read_text())
    S = {n: {"5-fold": scores(k5), "LOOCV": scores(kl)} for n, (_, k5, kl, _) in MODELS.items()}
    cells = sorted(S["U-Net"]["5-fold"], key=lambda c: (not c.startswith("high"), int(c.split("cell")[1])))
    for n in S:
        for p in S[n]:
            if sorted(S[n][p]) != sorted(cells):
                raise SystemExit(f"{n} {p}: not all 15 cells scored")

    fig = plt.figure(figsize=(10.6, 4.35), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.28)

    # ── (a) accuracy vs time per frame ───────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    budget = 1000 / CAMERA_FPS
    ax.axvspan(10, budget, color="#e8f3ea", zorder=0)
    ax.axvline(budget, color="#1b7a3a", lw=1, ls=(0, (4, 3)))
    ax.text(budget * 1.04, 0.61, f"real-time budget, {CAMERA_FPS:.0f} fps camera ({budget:.1f} ms)", rotation=90,
            ha="left", va="bottom", fontsize=7.4, color="#1b7a3a")
    rng = np.random.default_rng(0)
    for n, (col, _, _, key) in MODELS.items():
        t = lat[key]
        for p, off, filled in (("5-fold", 0.91, False), ("LOOCV", 1.10, True)):   # same model, same speed: offset to separate
            x = t * off
            v = [S[n][p][c]["dsc"] for c in cells]
            ax.scatter(x * np.exp(rng.uniform(-0.05, 0.05, len(v))), v, s=9, color=col, alpha=0.28, lw=0, zorder=2)
            m = st.mean(v)
            ax.scatter([x], [m], s=70, marker="o", facecolor=col if filled else "white", edgecolor=col, lw=1.8, zorder=4)
            ax.text(x * (0.87 if p == "5-fold" else 1.15), m, f"{m:.3f}", ha="right" if p == "5-fold" else "left",
                    va="center", fontsize=7.8, color=INK)
        ax.text(t, 1.005, n, ha="center", va="bottom", fontsize=9.5, color=col, weight="bold")
    ax.set_xscale("log")
    ax.set_xlim(18, 1100)
    ax.set_ylim(0.6, 1.03)
    ax.set_xticks([20, 45, 100, 200, 400, 1000], ["20", "45", "100", "200", "400", "1000"])
    ax.set_xlabel("Time per frame (ms, log scale)", fontsize=9, color=INK2)
    ax.set_ylabel("Test DSC (native)", fontsize=9, color=INK2)
    ax.set_title("(a) Accuracy against speed", loc="left", fontsize=9.8, color=INK, pad=14)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2,
                              markeredgewidth=1.6, label="5-fold CV, per-cell mean"),
                       Line2D([], [], marker="o", ls="", color=INK2, label="leave-one-cell-out, per-cell mean"),
                       Line2D([], [], marker="o", ls="", color=INK2, alpha=0.3, markersize=4, label="one cell")],
              loc="lower right", fontsize=7.4, frameon=False)

    # ── (b) every cell under both CVs ────────────────────────────────────────
    bx = fig.add_subplot(gs[1])
    ys = {c: i + (0.6 if not c.startswith("high") else 0) for i, c in enumerate(cells)}
    for n, (col, *_), dy in ((k, v, d) for (k, v), d in zip(MODELS.items(), (-0.17, 0.17))):
        for c in cells:
            y = ys[c] + dy
            v5, vl = S[n]["5-fold"][c]["dsc"], S[n]["LOOCV"][c]["dsc"]
            bx.plot([v5, vl], [y, y], color=col, lw=2.2, alpha=0.45, solid_capstyle="round", zorder=2)
            bx.scatter([v5], [y], s=24, facecolor="white", edgecolor=col, lw=1.4, zorder=3)
            bx.scatter([vl], [y], s=24, color=col, lw=0, zorder=4)
    bx.set_yticks([ys[c] for c in cells], [short(c) for c in cells], fontsize=8.2)
    for lab, c in zip(bx.get_yticklabels(), cells):
        lab.set_color(HIGH if c.startswith("high") else LOW)
    bx.invert_yaxis()
    bx.set_xlim(0.6, 1.0)
    bx.set_xlabel("Test DSC (native)", fontsize=9, color=INK2)
    bx.set_title("(b) Every cell, both cross-validations", loc="left", fontsize=9.8, color=INK, pad=14)
    lo_hi = [(c, abs(S["nnU-Net"]["5-fold"][c]["dsc"] - S["nnU-Net"]["LOOCV"][c]["dsc"])) for c in cells]
    c_max = max(lo_hi, key=lambda x: x[1])[0]
    bx.annotate("nnU-Net's only cell where\nits two CVs disagree", xy=(S["nnU-Net"]["LOOCV"][c_max]["dsc"], ys[c_max] + 0.17),
                xytext=(0.63, ys[c_max] + 1.6), fontsize=7.4, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    bx.legend(handles=[Line2D([], [], color=MODELS["U-Net"][0], lw=2.2, label="U-Net"),
                       Line2D([], [], color=MODELS["nnU-Net"][0], lw=2.2, label="nnU-Net"),
                       Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2, label="5-fold"),
                       Line2D([], [], marker="o", ls="", color=INK2, label="LOOCV")],
              loc="upper center", bbox_to_anchor=(0.5, -0.11), fontsize=7.6, frameon=False, ncol=4)
    for axis in (ax, bx):
        axis.set_facecolor(SURF)
        for sp in ("top", "right"):
            axis.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            axis.spines[sp].set_color(GRID)
        axis.tick_params(colors=INK2, labelsize=8.2, length=0)
        axis.grid(color=GRID, lw=0.8)
        axis.set_axisbelow(True)
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "final_models.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── model cards ──────────────────────────────────────────────────────────
    ue, uh, uf = unet_deploy()
    ne, nh, nf = nnunet_deploy()
    cell_mean = lambda m: st.mean(v["dsc"] for v in m.values())
    cell_sd = lambda m: st.stdev(v["dsc"] for v in m.values())
    cell_avg = lambda m, k: st.mean(v[k] for v in m.values())
    cls = lambda m, h: st.mean(v["dsc"] for c, v in m.items() if c.startswith(h))
    worst = lambda m: (lambda c: f"{short(c)} {m[c]['dsc']:.2f}")(min(m, key=lambda c: m[c]["dsc"]))

    def both(fn, n):
        # LOOCV value in the EXP 8 colour (ecH), so the two estimates read apart at a glance
        return f"{fn(S[n]['5-fold'])} / \\textcolor{{ecH}}{{{fn(S[n]['LOOCV'])}}}"

    rows = [
        ("Job", r"live tracking, every frame", r"classifier masks, 1 frame per step"),
        ("Final model", f"all 15 cells + legacy ({uf} frames)", f"all 15 cells + legacy ({nf} frames)"),
        ("Training", f"{ue} epochs (CV median), {60 * uh:.0f} min", f"{ne} epochs, {nh:.1f} h, one V100"),
        ("Time per frame", f"\\textbf{{{lat['ours_crop384']:.0f}\\,ms}} --- real time",
         f"{lat['nnunet_tta']:.0f}\\,ms ({lat['nnunet_no_tta']:.0f}\\,ms, no TTA)"),
        None,
        (r"\textbf{Per-cell mean DSC}$^\dagger$", both(lambda m: f"{cell_mean(m):.3f}", "U-Net"),
         both(lambda m: f"\\textbf{{{cell_mean(m):.3f}}}", "nnU-Net")),
        (r"\quad $\pm$ SD across cells", both(lambda m: f"{cell_sd(m):.3f}", "U-Net"),
         both(lambda m: f"{cell_sd(m):.3f}", "nnU-Net")),
        # per-cell means, like the headline DSC; precision < recall = the mask runs too large
        ("Precision", both(lambda m: f"{cell_avg(m, 'precision'):.3f}", "U-Net"),
         both(lambda m: f"{cell_avg(m, 'precision'):.3f}", "nnU-Net")),
        ("Recall", both(lambda m: f"{cell_avg(m, 'recall'):.3f}", "U-Net"),
         both(lambda m: f"{cell_avg(m, 'recall'):.3f}", "nnU-Net")),
        (r"\hiHER{High HER2} cells", both(lambda m: f"{cls(m, 'high'):.3f}", "U-Net"),
         both(lambda m: f"{cls(m, 'high'):.3f}", "nnU-Net")),
        (r"\loHER{Low HER2} cells", both(lambda m: f"{cls(m, 'low'):.3f}", "U-Net"),
         both(lambda m: f"{cls(m, 'low'):.3f}", "nnU-Net")),
        ("Worst cell", both(worst, "U-Net"), both(worst, "nnU-Net")),
    ]
    lines = [r"% GENERATED by scripts/report/plot_final_models.py -- do not edit by hand.",
             r"\begin{tabular}{p{0.24\linewidth}p{0.33\linewidth}p{0.33\linewidth}}", r"\toprule",
             r" & \textcolor{ecE}{\textbf{U-Net}} & \textcolor{ecF}{\textbf{nnU-Net}} \\",
             r"\midrule"]
    for r in rows:
        lines.append(r"\midrule" if r is None else " & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    a.tables.mkdir(parents=True, exist_ok=True)
    (a.tables / "final_models.tex").write_text("\n".join(lines))
    print(f"-> {a.out / 'final_models.png'}, {a.tables / 'final_models.tex'}")


if __name__ == "__main__":
    main()
