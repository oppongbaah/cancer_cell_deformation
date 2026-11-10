"""
Thesis figure and numbers for Section 4.3 (Developing the Segmentation Models), 2026-11-09.

    python scripts/report/plot_development_thesis.py --img ../masters_thesis/main_report/images/png

Reads only the evaluation logs on disk (trapped-cell Dice per development cell, scored at the native 1228 x 922
resolution): results/postswap/<experiment>/eval_log_native.jsonl, and the preswap ledger
results/preswap/preswap_eval_log.jsonl (scored on the 256 x 256 grid, two or four development cells).
Prints per-cell mean Dice of every run and writes fig_ch4_development.png:
  (a) four seeds of the baseline loss against four seeds with the Lovasz term, per development cell
  (b) the four inputs (resize 256, resize 512, crop 512, crop 384), per development cell
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent.parent
R = ROOT / "results"
CELLS = ["high_cell1", "high_cell9", "low_cell5", "low_cell6"]
LABEL = {"high_cell1": "high-1", "high_cell9": "high-9", "low_cell5": "low-5", "low_cell6": "low-6"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BASE, LOV = "#8a8985", "#1b7a3a"
INPUTS = {"e2a_res256_b4": ("Resize 256", "#c9c8c3", "o"), "e2b_res512_b4": ("Resize 512", "#8a8985", "s"),
          "e3a_crop512_b4": ("Crop 512", "#4a8fb8", "D"), "e3b_crop384_b4": ("Crop 384", "#1b4f7a", "^")}

for f in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyretermes-*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})


def native(exp):
    """{run: {cell: dice}} from one postswap experiment's native log (single models only, last entry wins)."""
    out = {}
    for line in open(R / "postswap" / exp / "eval_log_native.jsonl"):
        r = json.loads(line)
        if r.get("members"):
            run = "ENSEMBLE:" + Path(r["checkpoint"]).parent.name if r.get("checkpoint") else "ENSEMBLE"
        else:
            run = Path(r["checkpoint"]).parent.name
        if r.get("filter") in CELLS:
            out.setdefault(run, {})[r["filter"]] = r["metrics"]["1"]["dsc"]
    return out


def show(name, d):
    v = [d[c] for c in CELLS]
    print(f"  {name:34s} per-cell mean {st.mean(v):.3f}  lowest {min(v):.3f}  "
          + "  ".join(f"{LABEL[c]} {d[c]:.3f}" for c in CELLS))
    return st.mean(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, required=True)
    a = ap.parse_args()

    print("== Postswap, four development cells, native resolution")
    base = native("experiment_0a_seed_baseline")
    lov = native("experiment_1b_loss")
    sgd = native("experiment_1a_optimizer")
    res = native("experiment_2_resolution")
    crop = native("experiment_3_cropping")
    bs = [f"postswap_ftl_cw142_s{s}" for s in (42, 43, 44, 45)]
    ls = [f"e1b_lovasz_s{s}" for s in (42, 43, 44, 45)]
    mb = [show(r, base[r]) for r in bs]
    ml = [show(r, lov[r]) for r in ls]
    for k, v in list(base.items()) + list(lov.items()):
        if k.startswith("ENSEMBLE"):
            show(k, v)
    print(f"  baseline 4 seeds {st.mean(mb):.3f} +- {st.stdev(mb):.3f}; Lovasz {st.mean(ml):.3f} +- {st.stdev(ml):.3f}; "
          f"paired differences {[round(y - x, 3) for x, y in zip(mb, ml)]}")
    print(f"  lowest cell baseline {min(min(base[r].values()) for r in bs):.3f}, "
          f"Lovasz {min(min(lov[r].values()) for r in ls):.3f}")
    show("e1a_sgd_nesterov (vs s43)", sgd["e1a_sgd_nesterov"])
    for r in INPUTS:
        show(r, res.get(r) or crop.get(r))

    print("\n== Preswap ledger (256 x 256 grid; per-cell mean where per-cell rows exist)")
    pre = {}
    for line in open(R / "preswap" / "preswap_eval_log.jsonl"):
        r = json.loads(line)
        pre.setdefault(Path(r["checkpoint"]).parent.name, {})[r.get("filter") or "ALL"] = r["metrics"]["1"]["dsc"]
    for k, v in pre.items():
        cells = {c: x for c, x in v.items() if c != "ALL"}
        pcm = f"per-cell mean {st.mean(cells.values()):.3f} over {len(cells)}" if cells else ""
        print(f"  {k:48s} pooled {v.get('ALL', float('nan')):.3f}  {pcm}")

    # ---------------- figure ----------------
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 2.6), gridspec_kw={"wspace": 0.28})
    x = np.arange(len(CELLS))
    for runs, d, col, dx in ((bs, base, BASE, -0.12), (ls, lov, LOV, 0.12)):
        for r in runs:
            ax.scatter(x + dx, [d[r][c] for c in CELLS], s=16, color=col, lw=0, alpha=0.9, zorder=3)
        ax.plot([], [])
    ax.set_xticks(x, [LABEL[c] for c in CELLS], fontsize=8)
    ax.set_ylabel("Dice, trapped cell", fontsize=8)
    ax.set_ylim(0.45, 1.0)
    ax.set_title("(a) Four seeds, $256 \\times 256$ input", fontsize=8.5, color=INK, loc="left")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BASE, ms=4.5, label="Focal Tversky + CE"),
                       Line2D([], [], marker="o", ls="", color=LOV, ms=4.5, label="+ Lovász")],
              loc="lower right", fontsize=7.5, frameon=False)
    for k, (r, (lab, col, mk)) in enumerate(INPUTS.items()):
        d = res.get(r) or crop.get(r)
        bx.plot(x + (k - 1.5) * 0.13, [d[c] for c in CELLS], ls="", color=col, marker=mk, ms=4.5, label=lab,
                zorder=3)
    bx.set_xticks(x, [LABEL[c] for c in CELLS], fontsize=8)
    bx.set_ylim(0.45, 1.0)
    bx.set_title("(b) Input of the network, one seed", fontsize=8.5, color=INK, loc="left")
    bx.legend(loc="lower right", fontsize=7.5, frameon=False, ncol=2)
    for axx in (ax, bx):
        axx.tick_params(labelsize=7.5)
        axx.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        axx.set_axisbelow(True)
    a.img.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.img / "fig_ch4_development.png", dpi=400, bbox_inches="tight", facecolor="white")
    print(f"\n-> {a.img / 'fig_ch4_development.png'}")


if __name__ == "__main__":
    main()
