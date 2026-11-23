"""
Deformation regression: does the strain grow with the laser power, and differently for high and low HER2 cells?

    python scripts/biomechanics/deformation_regression.py --model nnunet1000_loocv [--img <dir>] [--tables <dir>]

One value per cell and laser step: the median of the step's passing feature frames (usable cells only), as a strain
in % against the cell's own baseline (median over steps 0-9), for three measures -- length along the beam, length
across it, and the spheroid volume. Each is fitted with a linear mixed-effects model:

    strain = b0 + b1 step + b2 high + b3 step x high  +  (u0 + u1 step | cell)  +  e

    b1   the typical low cell's strain per step (its compliance)
    b3   how much more (or less) a high cell strains per step -- the HER2 effect on deformation
    u1   each cell's own slope: the random slope makes the CELL the unit, not the ~700 step values

Fitted by REML (statsmodels MixedLM). A second fit adds the cell's baseline length (standardised) and its
interaction with the step, to ask whether size accounts for any HER2 effect.

How well it fits -- R^2 for mixed models (Nakagawa & Schielzeth 2013; random slopes as in Johnson 2014):
    marginal     variance of the fixed-effect prediction (step, HER2) / total variance
    conditional  (fixed + random-effect variance) / total variance -- the share explained once every cell has
                 its own line

Does HER2 help predict a new cell? -- leave-one-cell-out: each cell in turn is left out, both models are fitted
on the other 13, and the left-out cell's whole strain curve is predicted from the fixed effects alone (a new cell
has no fitted line of its own):
    with HER2     strain ~ step * high
    without HER2  strain ~ step
    zero          strain = 0 at every step (no deformation)
scored by the RMSE (percentage points of strain) over the cell's steps; the paired comparison is a Wilcoxon
signed-rank test over the 14 cells. HER2 and the recording session are the
same split in this dataset, so b3 cannot be told apart from a session effect.

Outputs in results/biomechanics/<model>/: deformation_regression.csv / .tex / .png (coefficients, fitted lines),
deformation_fit.csv / .tex (R^2 and leave-one-cell-out errors), deformation_loco.csv / .png (every cell's errors).
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
from celldform.utils.runlog import start_run_log  # noqa: E402

LABELS = {"len_par": "Length along the beam", "len_perp": "Length across the beam", "volume": "Volume"}


def configure(cfg) -> None:
    """Settings from configs/analysis/regression.yaml (--config for another file)."""
    global CFG, MEASURES, BASE_STEPS, RE_FORMULA
    CFG = cfg
    MEASURES = {m: LABELS.get(m, m) for m in cfg.measures}
    BASE_STEPS = cfg.base_steps
    RE_FORMULA = "~laser_step" if cfg.random_slope else "1"


configure(load_config("regression"))
HIGH, LOW = "#2a78d6", "#eb6834"   # the report's HER2 colours
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def strains(model: str) -> pd.DataFrame:
    """One row per cell and laser step: strain (%) of each measure against the cell's baseline."""
    res = ROOT / "results" / "features" / model
    cells = pd.read_csv(res / "features_cells.csv")
    cells = cells[cells.usable]
    f = pd.read_csv(res / "features_frames.csv")
    f = f[f.passed & f.cell.isin(cells.cell)]
    s = f.groupby(["cell", "her2", "laser_step"])[list(MEASURES)].median().reset_index()
    for m in MEASURES:
        base = s[s.laser_step < BASE_STEPS].groupby("cell")[m].median()
        s[f"{m}_strain"] = (s[m] / s.cell.map(base) - 1) * 100
    s["high"] = (s.her2 == "high").astype(int)
    size = cells.set_index("cell").len_par_base
    s["size_z"] = s.cell.map((size - size.mean()) / size.std())
    return s


def fit(s: pd.DataFrame, y: str, formula_rhs: str):
    import statsmodels.formula.api as smf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("lbfgs", "powell", "nm"):
            m = smf.mixedlm(f"{y} ~ {formula_rhs}", s, groups="cell", re_formula=RE_FORMULA).fit(
                reml=True, method=method)
            if m.converged:
                return m
    return m


def r_squared(m) -> tuple[float, float]:
    """Marginal and conditional R^2 of a random-intercept-and-slope MixedLM fit."""
    fe = m.fe_params
    X = m.model.exog
    var_f = float(np.var(X @ fe.to_numpy()))
    Z = m.model.exog_re                                    # columns: intercept, laser_step
    S = m.cov_re.to_numpy()
    var_r = float(np.mean(np.einsum("ij,jk,ik->i", Z, S, Z)))
    var_e = float(m.scale)
    tot = var_f + var_r + var_e
    return var_f / tot, (var_f + var_r) / tot


def loco(s: pd.DataFrame, y: str) -> pd.DataFrame:
    """Leave-one-cell-out: each cell's strain curve predicted from the other 13 (fixed effects only)."""
    rows = []
    for cell in sorted(s.cell.unique()):
        train, test = s[s.cell != cell], s[(s.cell == cell) & s[y].notna()]
        with_h, without = fit(train, y, "laser_step * high"), fit(train, y, "laser_step")
        rmse = lambda pred: float(np.sqrt(np.mean((test[y].to_numpy() - pred) ** 2)))   # noqa: E731
        rows.append({"cell": cell, "high": int(test.high.iloc[0]),
                     "rmse_with_her2": rmse(np.asarray(with_h.predict(test))),
                     "rmse_without_her2": rmse(np.asarray(without.predict(test))),
                     "rmse_zero": rmse(np.zeros(len(test)))})
    return pd.DataFrame(rows)


def summarise(s: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, fits = [], {}
    for meas, label in MEASURES.items():
        y = f"{meas}_strain"
        m = fit(s, y, "laser_step * high")
        a = fit(s, y, "laser_step * high + laser_step * size_z") if CFG.size_adjusted else m
        fits[meas] = m
        ci, ci_a = m.conf_int(), a.conf_int()
        slope_sd = (float(np.sqrt(max(m.cov_re.loc["laser_step", "laser_step"], 0)))
                    if "laser_step" in m.cov_re.index else np.nan)
        rows.append({"measure": meas, "label": label,
                     "low_slope": m.fe_params["laser_step"], "low_slope_lo": ci.loc["laser_step", 0],
                     "low_slope_hi": ci.loc["laser_step", 1], "low_slope_p": m.pvalues["laser_step"],
                     "her2_slope": m.fe_params["laser_step:high"], "her2_slope_lo": ci.loc["laser_step:high", 0],
                     "her2_slope_hi": ci.loc["laser_step:high", 1], "her2_slope_p": m.pvalues["laser_step:high"],
                     "her2_slope_size_adj": a.fe_params["laser_step:high"],
                     "her2_slope_size_adj_lo": ci_a.loc["laser_step:high", 0],
                     "her2_slope_size_adj_hi": ci_a.loc["laser_step:high", 1],
                     "her2_slope_size_adj_p": a.pvalues["laser_step:high"],
                     "cell_slope_sd": slope_sd, "converged": bool(m.converged and a.converged),
                     "n_cells": s.cell.nunique(), "n_steps": int(s[y].notna().sum())})
    return pd.DataFrame(rows), fits


def fit_quality(s: pd.DataFrame, fits: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import wilcoxon
    rows, per_cell = [], []
    for meas, label in MEASURES.items():
        y = f"{meas}_strain"
        r2m, r2c = r_squared(fits[meas])
        lc = loco(s, y).assign(measure=meas)
        per_cell.append(lc)
        diff = lc.rmse_with_her2 - lc.rmse_without_her2
        rows.append({"measure": meas, "label": label, "r2_marginal": r2m, "r2_conditional": r2c,
                     "rmse_with_her2": lc.rmse_with_her2.mean(), "rmse_without_her2": lc.rmse_without_her2.mean(),
                     "rmse_zero": lc.rmse_zero.mean(), "cells_her2_better": int((diff < 0).sum()),
                     "n_cells": len(lc), "wilcoxon_p": float(wilcoxon(lc.rmse_with_her2, lc.rmse_without_her2).pvalue)})
    return pd.DataFrame(rows), pd.concat(per_cell, ignore_index=True)


def fit_latex(q: pd.DataFrame, out: Path) -> None:
    rows = [f"{r['label']} & {r['r2_marginal']:.3f} & {r['r2_conditional']:.2f} & {r['rmse_zero']:.2f} & "
            f"{r['rmse_without_her2']:.2f} & {r['rmse_with_her2']:.2f} & {r['cells_her2_better']}/{r['n_cells']} & "
            f"{r['wilcoxon_p']:.2f} \\\\" for _, r in q.iterrows()]
    out.write_text("\\begin{tabular}{l c c c c c c c}\n\\toprule\n"
                   " & \\multicolumn{2}{c}{$R^2$} & \\multicolumn{5}{c}{Leave-one-cell-out: error on the left-out cell "
                   "(RMSE, \\% strain)} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-8}\n"
                   "Strain of the & marginal & conditional & no deformation & without HER2 & with HER2 & "
                   "HER2 helps & $p$ \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def plot_loco(lc: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(11, 3.3))
    for ax, (meas, label) in zip(axes, MEASURES.items()):
        d = lc[lc.measure == meas]
        for _, r in d.iterrows():
            col = HIGH if r.high else LOW
            ax.plot([0, 1], [r.rmse_without_her2, r.rmse_with_her2], color=col, lw=1.0, alpha=0.7, zorder=2)
            ax.scatter([0, 1], [r.rmse_without_her2, r.rmse_with_her2], s=16, color=col, edgecolor="white",
                       lw=0.8, zorder=3)
        ax.set_xticks([0, 1], ["without HER2", "with HER2"], fontsize=8, color=INK2)
        ax.set_xlim(-0.35, 1.35)
        ax.set_title(f"{label}", fontsize=9, loc="left", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=7.5, colors=INK2)
    axes[0].set_ylabel("error on the left-out cell\n(RMSE, % strain)", fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], color=HIGH, marker="o", lw=1), plt.Line2D([], [], color=LOW, marker="o", lw=1)]
    fig.legend(handles, ["high HER2 cell", "low HER2 cell"], loc="lower center", ncol=2, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def latex(t: pd.DataFrame, out: Path) -> None:
    num = lambda v: f"${v:+.3f}$"   # noqa: E731  (in math mode, so the minus is a minus)
    ci = lambda r, k: f"{num(r[k])} [{num(r[k + '_lo'])}, {num(r[k + '_hi'])}]"   # noqa: E731
    p = lambda v: "$<$0.001" if v < 0.001 else f"{v:.2f}"                       # noqa: E731
    rows = [f"{r['label']} & {ci(r, 'low_slope')} & {ci(r, 'her2_slope')} & {p(r['her2_slope_p'])} & "
            f"{r['cell_slope_sd']:.3f} & {ci(r, 'her2_slope_size_adj')} \\\\" for _, r in t.iterrows()]
    out.write_text("\\begin{tabular}{l c c c c c}\n\\toprule\n"
                   " & \\multicolumn{1}{c}{Low cells} & \\multicolumn{2}{c}{High $-$ low} & Between cells & "
                   "High $-$ low, \\\\\n"
                   "Strain of the & slope $\\beta_1$ & slope $\\beta_3$ & $p$ & slope SD & size-adjusted \\\\\n"
                   "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def plot(s: pd.DataFrame, fits: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(11, 3.6), sharex=True)
    steps = np.arange(int(s.laser_step.max()) + 1)
    for ax, (meas, label) in zip(axes, MEASURES.items()):
        y = f"{meas}_strain"
        for cell, g in s.groupby("cell"):
            ax.plot(g.laser_step, g[y], color=HIGH if g.high.iloc[0] else LOW, lw=0.7, alpha=0.3, zorder=1)
        m = fits[meas]
        fe, cov = m.fe_params, m.cov_params().loc[m.fe_params.index, m.fe_params.index]
        for h, col, name in ((0, LOW, "low"), (1, HIGH, "high")):
            X = pd.DataFrame({"Intercept": 1.0, "laser_step": steps, "high": h, "laser_step:high": steps * h})[fe.index]
            mu = X.to_numpy() @ fe.to_numpy()
            se = np.sqrt(np.einsum("ij,jk,ik->i", X.to_numpy(), cov.to_numpy(), X.to_numpy()))
            ax.fill_between(steps, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=0.18, lw=0, zorder=2)
            ax.plot(steps, mu, color=col, lw=2.2, zorder=3)
            ax.text(steps[-1] + 1, mu[-1], name, color=INK2, fontsize=7.5, va="center")
        ax.axhline(0, color=INK2, lw=0.6, zorder=0)
        ax.set_title(f"{label} (%)", fontsize=9, loc="left", color=INK)
        ax.set_xlabel("laser step (power increases)", fontsize=8, color=INK2)
        ax.set_xlim(0, steps[-1] + 6)
        lo, hi = np.nanpercentile(s[y], [1, 99])
        ax.set_ylim(min(lo, -5), max(hi, 5))
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7.5, colors=INK2)
    axes[0].set_ylabel("strain against the baseline (%)", fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], color=HIGH, lw=2.2), plt.Line2D([], [], color=LOW, lw=2.2),
               plt.Line2D([], [], color=INK2, lw=0.7, alpha=0.5)]
    fig.legend(handles, ["high HER2: fitted mean, 95 % CI", "low HER2: fitted mean, 95 % CI", "one cell"],
               loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nnunet1000_loocv", help="feature run in results/features/")
    ap.add_argument("--img", type=Path, default=None, help="also write the figure here")
    ap.add_argument("--tables", type=Path, default=None, help="also write the LaTeX table here")
    ap.add_argument("--config", type=Path, default=None,
                    help="settings file (default configs/analysis/regression.yaml)")
    a = ap.parse_args()
    start_run_log(f"biomechanics/{a.model}", "deformation_regression")
    configure(load_config("regression", a.config))
    out = ROOT / "results" / "biomechanics" / a.model
    out.mkdir(parents=True, exist_ok=True)
    dump_config(CFG, out / "config_regression.yaml")
    s = strains(a.model)
    t, fits = summarise(s)
    t.to_csv(out / "deformation_regression.csv", index=False)
    print(f"{s.cell.nunique()} cells, {len(s)} cell-steps")
    print(t.drop(columns=["label"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    for meas, m in fits.items():
        print(f"\n{MEASURES[meas]}:\n{m.summary().tables[1]}")
    q, lc = fit_quality(s, fits)
    q.to_csv(out / "deformation_fit.csv", index=False)
    lc.to_csv(out / "deformation_loco.csv", index=False)
    print("\nFIT AND LEAVE-ONE-CELL-OUT PREDICTION:")
    print(q.drop(columns=["label"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    latex(t, out / "deformation_regression.tex")
    fit_latex(q, out / "deformation_fit.tex")
    plot(s, fits, out / "deformation_regression.png")
    plot_loco(lc, out / "deformation_loco.png")
    if a.tables:
        latex(t, a.tables / "deformation_regression.tex")
        fit_latex(q, a.tables / "deformation_fit.tex")
    if a.img:
        plot(s, fits, a.img / "deformation_regression.png")
        plot_loco(lc, a.img / "deformation_loco.png")
    print(f"wrote {out}" + (f", {a.img}" if a.img else "") + (f", {a.tables}" if a.tables else ""))


if __name__ == "__main__":
    main()
