"""
Rate of deformation (RoD) per cell: measured over the full ramp, in real time, and predicted early.

    python scripts/biomechanics/rate_of_deformation.py --model nnunet1000_loocv [--img <dir>] [--tables <dir>]

RoD = slope of the strain (%) of the cell's length against the laser step (celldform.biomechanics.rod), for the
length along the beam (main) and across it, with a Newey-West 95 % interval. Usable cells, passing frames only.

    per cell     the full-ramp RoD of every cell, with its interval
    real time    the RoD from the first k steps only, k = 5, 10, ..., 50: how wide is the interval after k steps,
                 and how far is the estimate from the full-ramp RoD?
    early        leave-one-cell-out: the left-out cell's RoD after k steps, estimated four ways, against its full-ramp
                 RoD (the target):
                   own           its own k steps only
                   shrunk        own, combined with the prior from the other 13 cells (pooled)
                   shrunk+class  the same with a class-specific prior (the cell's HER2 class -- what a classifier
                                 would supply; the true class, so an upper bound on what classifying adds)
                   prior only    the other cells' mean RoD, no own data

Outputs in results/biomechanics/<model>/: rod_cells.csv, rod_realtime.csv, rod_early.csv, rod_cells.tex,
rod_realtime.tex, rod_cells.png, rod_realtime.png.

Final model: the settings and the priors from ALL usable cells (pooled, and per HER2 class), for estimating a new
cell's RoD (celldform.biomechanics.rod.estimate) -> checkpoints/biomechanics/<model>/rod_model.json
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from celldform.analysis_config import dump as dump_config, load as load_config  # noqa: E402
from celldform.biomechanics.rod import FRAMES_PER_STEP, Rod, prior, rate, shrink, step_strain  # noqa: E402
from celldform.utils.runlog import start_run_log  # noqa: E402
from celldform import final_models  # noqa: E402

LABELS = {"len_par": "along the beam", "len_perp": "across the beam", "volume": "of the volume"}


def configure(cfg) -> None:
    """Settings from configs/analysis/regression.yaml (rod_measures, base_steps, hac_lags, realtime_steps, fps).
    The tables and figures expect len_par (the main measure) and len_perp."""
    global CFG, MEASURES, KS, STEP_SECONDS
    CFG = cfg
    MEASURES = {m: LABELS.get(m, m) for m in cfg.rod_measures}
    KS = list(cfg.realtime_steps)
    STEP_SECONDS = FRAMES_PER_STEP / cfg.fps


def _strain(frames: pd.DataFrame, measure: str) -> pd.Series:
    return step_strain(frames, measure, base_steps=CFG.base_steps)


def _rate(strain: pd.Series) -> Rod:
    return rate(strain, lags=CFG.hac_lags)


configure(load_config("regression"))
HIGH, LOW = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
METHODS = {"own": ("own steps only", "#52514e", "-"), "shrunk": ("shrunk, pooled prior", "#1b7a3a", "-"),
           "shrunk_class": ("shrunk, class prior", "#7a4fd6", "--"), "prior": ("prior only (no own data)", "#b0aea8", ":")}


def short(c: str) -> str:
    return c.replace("_cell", "-")


def load(model: str) -> tuple[pd.DataFrame, dict]:
    res = ROOT / "results" / "features" / model
    cells = pd.read_csv(res / "features_cells.csv")
    cells = cells[cells.usable]
    f = pd.read_csv(res / "features_frames.csv")
    f = f[f.cell.isin(cells.cell)]
    strains = {(c, m): _strain(g, m) for c, g in f.groupby("cell") for m in MEASURES}
    her2 = cells.set_index("cell").her2.to_dict()
    return pd.DataFrame({"cell": list(her2), "her2": list(her2.values())}), strains


def per_cell(cells: pd.DataFrame, strains: dict) -> pd.DataFrame:
    rows = []
    for c, h in zip(cells.cell, cells.her2):
        row = {"cell": c, "her2": h}
        for m in MEASURES:
            r = _rate(strains[(c, m)])
            lo, hi = r.ci
            row |= {f"{m}_rod": r.slope, f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_se": r.se,
                    f"{m}_per_s": r.slope / STEP_SECONDS, f"{m}_steps": r.n_steps}
        rows.append(row)
    t = pd.DataFrame(rows)
    t["_n"] = t.cell.str.extract(r"(\d+)$")[0].astype(int)
    return t.sort_values(["her2", "_n"]).drop(columns="_n").reset_index(drop=True)


def realtime(cells: pd.DataFrame, strains: dict, full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in cells.cell:
        for m in MEASURES:
            st = strains[(c, m)]
            target = full.set_index("cell").loc[c, f"{m}_rod"]
            for k in KS:
                r = _rate(st[st.index < k])
                rows.append({"cell": c, "measure": m, "k": k, "rod": r.slope, "half_width": 1.96 * r.se,
                             "abs_error": abs(r.slope - target)})
    return pd.DataFrame(rows)


def early(cells: pd.DataFrame, strains: dict, full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fullrods = {(c, m): _rate(strains[(c, m)]) for c in cells.cell for m in MEASURES}
    her2 = dict(zip(cells.cell, cells.her2))
    for c in cells.cell:
        others = [o for o in cells.cell if o != c]
        same = [o for o in others if her2[o] == her2[c]]
        for m in MEASURES:
            target = fullrods[(c, m)].slope
            pm, ps = prior([fullrods[(o, m)] for o in others])
            cm, cs = prior([fullrods[(o, m)] for o in same])
            st = strains[(c, m)]
            for k in KS:
                own = _rate(st[st.index < k])
                est = {"own": own.slope, "shrunk": shrink(own, pm, ps).slope,
                       "shrunk_class": shrink(own, cm, cs).slope, "prior": pm}
                rows += [{"cell": c, "measure": m, "k": k, "method": meth, "abs_error": abs(v - target)}
                         for meth, v in est.items()]
    return pd.DataFrame(rows)


def save_model(model: str, cells: pd.DataFrame, full: pd.DataFrame) -> None:
    """The RoD model: the method's settings and, per measure, the prior (mean and between-cell SD of the full-ramp
    RoD) from all usable cells -- pooled and per HER2 class. With early.csv's leave-one-cell-out errors."""
    priors = {}
    for m in MEASURES:
        rods = {c: Rod(r[f"{m}_rod"], r[f"{m}_se"], int(r[f"{m}_steps"])) for c, r in full.set_index("cell").iterrows()}
        her2 = dict(zip(full.cell, full.her2))
        priors[m] = {}
        for key, group in (("pooled", list(rods)), ("high", [c for c in rods if her2[c] == "high"]),
                           ("low", [c for c in rods if her2[c] == "low"])):
            mean, sd = prior([rods[c] for c in group])
            priors[m][key] = {"mean": mean, "sd": sd, "n_cells": len(group)}
    obj = {"info": final_models.info(model, cells.cell, cells.her2), "measures": list(MEASURES),
           "base_steps": CFG.base_steps, "hac_lags": CFG.hac_lags, "fps": CFG.fps, "frames_per_step": FRAMES_PER_STEP,
           "step_seconds": STEP_SECONDS, "units": "% strain per laser step", "priors": priors}
    folder = final_models.ckpt_dir("biomechanics", model)
    final_models.to_json(obj, folder / "rod_model.json")
    final_models.write_manifest(folder, [{"file": "rod_model.json", "measures": " ".join(MEASURES),
                                          "n_cells": len(cells), "created": obj["info"]["created"]}])
    print(f"RoD model (priors from {len(cells)} cells) -> {folder / 'rod_model.json'}")


def num(v: float, k: int = 3) -> str:
    return f"${v:+.{k}f}$"


def cells_latex(t: pd.DataFrame, out: Path) -> None:
    def cell_cols(r, m):
        sig = r[f"{m}_lo"] > 0 or r[f"{m}_hi"] < 0
        val = num(r[f"{m}_rod"], 2)
        val = f"\\textbf{{{val}}}" if sig else val
        return f"{val} & [{num(r[f'{m}_lo'], 2)}, {num(r[f'{m}_hi'], 2)}]"
    rows = []
    for _, r in t.iterrows():
        name = ("\\hiHER{" if r.her2 == "high" else "\\loHER{") + short(r.cell) + "}"
        rows.append(f"{name} & {cell_cols(r, 'len_par')} & {r['len_par_per_s'] * 1000:+.1f} & {cell_cols(r, 'len_perp')} \\\\")
    out.write_text("\\begin{tabular}{l c c c c c}\n\\toprule\n"
                   " & \\multicolumn{3}{c}{RoD along the beam} & \\multicolumn{2}{c}{RoD across the beam} \\\\\n"
                   "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
                   "Cell & \\%/step & 95\\,\\% CI & $10^{-3}$\\,\\%/s & \\%/step & 95\\,\\% CI \\\\\n\\midrule\n"
                   + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def realtime_latex(rt: pd.DataFrame, ea: pd.DataFrame, out: Path) -> None:
    r = rt[rt.measure == "len_par"].groupby("k").agg(hw=("half_width", "median"), err=("abs_error", "median"))
    e = ea[ea.measure == "len_par"].groupby(["k", "method"]).abs_error.median().unstack()
    rows = [f"{k} & {k * STEP_SECONDS / 60:.1f} & $\\pm${r.loc[k, 'hw']:.3f} & {e.loc[k, 'own']:.3f} & "
            f"{e.loc[k, 'shrunk']:.3f} & {e.loc[k, 'shrunk_class']:.3f} & {e.loc[k, 'prior']:.3f} \\\\" for k in KS]
    out.write_text("\\begin{tabular}{c c c c c c c}\n\\toprule\n"
                   "\\multicolumn{2}{c}{Watched} & 95\\,\\% CI & \\multicolumn{4}{c}{Error against the full-ramp RoD "
                   "(median, \\%/step)} \\\\\n\\cmidrule(lr){1-2}\\cmidrule(lr){4-7}\n"
                   "steps & min & half-width & own steps & shrunk, pooled & shrunk, class & prior only \\\\\n"
                   "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def plot_cells(t: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    y = np.arange(len(t))[::-1]
    for ax, (m, label) in zip(axes, MEASURES.items()):
        for yi, (_, r) in zip(y, t.iterrows()):
            col = HIGH if r.her2 == "high" else LOW
            ax.plot([r[f"{m}_lo"], r[f"{m}_hi"]], [yi, yi], color=col, lw=1.6, solid_capstyle="round", zorder=2)
            ax.scatter(r[f"{m}_rod"], yi, s=34, color=col, edgecolor="white", lw=1.2, zorder=3)
        for h, col in (("high", HIGH), ("low", LOW)):
            ax.axvline(t[t.her2 == h][f"{m}_rod"].mean(), color=col, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        ax.set_title(f"RoD {label} (% per laser step)", fontsize=9, loc="left", color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5, colors=INK2)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [short(c) for c in t.cell], fontsize=8)
    for lab, h in zip(axes[0].get_yticklabels(), t.her2):
        lab.set_color(HIGH if h == "high" else LOW)
    handles = [plt.Line2D([], [], color=HIGH, marker="o", lw=1.6), plt.Line2D([], [], color=LOW, marker="o", lw=1.6),
               plt.Line2D([], [], color=INK2, ls=(0, (4, 3)), lw=1)]
    fig.legend(handles, ["high HER2 cell, 95 % CI", "low HER2 cell, 95 % CI", "class mean"], loc="lower center",
               ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_realtime(rt: pd.DataFrame, ea: pd.DataFrame, out: Path) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.6))
    r = rt[rt.measure == "len_par"]
    for c, g in r.groupby("cell"):
        ax.plot(g.k, g.half_width, color=INK2, lw=0.6, alpha=0.35)
    med = r.groupby("k").half_width.median()
    ax.plot(med.index, med.values, color=INK, lw=2.2, marker="o", ms=4)
    ax.set_yscale("log")
    ax.set_title("(a) Precision of the RoD after k steps", fontsize=9, loc="left", color=INK)
    ax.set_ylabel("95 % CI half-width (% per step)", fontsize=8, color=INK2)
    e = ea[ea.measure == "len_par"].groupby(["k", "method"]).abs_error.median().unstack()
    for meth, (label, col, ls) in METHODS.items():
        bx.plot(e.index, e[meth], color=col, ls=ls, lw=2, marker="o", ms=3.5, label=label)
    bx.set_title("(b) Early estimate against the full-ramp RoD", fontsize=9, loc="left", color=INK)
    bx.set_ylabel("median error (% per step)", fontsize=8, color=INK2)
    bx.legend(fontsize=7.2, frameon=False)
    for a in (ax, bx):
        a.set_xlabel("laser steps watched", fontsize=8, color=INK2)
        a.grid(color=GRID, lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=7.5, colors=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nnunet1000_loocv", help="feature run in results/features/")
    ap.add_argument("--img", type=Path, default=None, help="also write the figures here")
    ap.add_argument("--tables", type=Path, default=None, help="also write the LaTeX tables here")
    ap.add_argument("--config", type=Path, default=None,
                    help="settings file (default configs/analysis/regression.yaml)")
    a = ap.parse_args()
    start_run_log(f"biomechanics/{a.model}", "rate_of_deformation")
    configure(load_config("regression", a.config))
    out = ROOT / "results" / "biomechanics" / a.model
    out.mkdir(parents=True, exist_ok=True)
    dump_config(CFG, out / "config_rod.yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cells, strains = load(a.model)
        full = per_cell(cells, strains)
        rt = realtime(cells, strains, full)
        ea = early(cells, strains, full)
    full.to_csv(out / "rod_cells.csv", index=False)
    save_model(a.model, cells, full)
    rt.to_csv(out / "rod_realtime.csv", index=False)
    ea.to_csv(out / "rod_early.csv", index=False)
    print(full.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nreal time (length along the beam, medians over cells):")
    print(rt[rt.measure == "len_par"].groupby("k")[["half_width", "abs_error"]].median().to_string())
    print("\nearly estimate, median abs error (length along the beam):")
    print(ea[ea.measure == "len_par"].groupby(["k", "method"]).abs_error.median().unstack().to_string())
    for d_img, d_tab in [(out, out)] + ([(a.img, a.tables)] if a.img and a.tables else []):
        cells_latex(full, d_tab / "rod_cells.tex")
        realtime_latex(rt, ea, d_tab / "rod_realtime.tex")
        plot_cells(full, d_img / "rod_cells.png")
        plot_realtime(rt, ea, d_img / "rod_realtime.png")
    print(f"wrote {out}" + (f", {a.img}, {a.tables}" if a.img else ""))


if __name__ == "__main__":
    main()
