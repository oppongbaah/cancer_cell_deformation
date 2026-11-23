"""
How many cells would a conclusive experiment need, and how small a deformation can the system resolve?

    python scripts/classification/power_resolution.py --model nnunet1000_loocv [--tables <report tables dir>]

Cells needed (from the usable cells of features_cells.csv and the selected features of benchmark_cells.py)
    per feature   Cohen's d between the classes (pooled SD), and the cells per class a two-sided Mann--Whitney
                  test needs to find a difference that size at alpha 0.05 with 80 % power:
                      n = 2 (z_0.975 + z_0.8)^2 / d^2 / 0.955     (0.955: its efficiency against the t-test)
                  d from 14 cells is itself uncertain (its 95 % interval is roughly +/-1.1), so n is an order
                  of magnitude, not a design figure.
    classifier    the cells (balanced classes) a one-sided binomial test needs to show that a classifier of a
                  given true accuracy beats 50 % at alpha 0.05 with 80 % power.

Resolution (from the passing feature frames: length along the beam, L_par)
    frame noise   median over cells and steps of SD/mean of L_par within a laser step (~6 frames)
    step noise    (a) frame noise / sqrt(frames per step) x 1.25 (the median's efficiency)
                  (b) from the step-to-step differences of each cell's step medians: SD(diff)/sqrt(2) -- an
                      upper bound, it also contains the cell's real step-to-step change
    detectable    a change between two steps: 2.77 x step noise (95 %); the strain over the ramp (last ten steps
                  against the baseline ten): 1.96 x sqrt(2/10) x step noise x 1.25; a compliance: 1.96 x the
                  standard error of the slope over 50 steps (x 1.1, the Theil--Sen efficiency)
    compared with the observed per-cell compliances and the difference between the class means.
    Random noise only: a segmentation error that grows with the laser power (high-1) is not in it.

Outputs in results/classification/<model>/: power.csv, resolution.csv, classify_power.tex, classify_resolution.tex.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts/classification/benchmark_cells.py")
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)
BENCH_NAMES = _bench.NAMES

ALPHA, POWER, MW_EFFICIENCY = 0.05, 0.80, 0.955
STEPS, BASE = 50, 10
NAMES = BENCH_NAMES                  # every per-cell feature's report name (benchmark_cells.NAMES)


def cells_per_class(d: float) -> float:
    return 2 * (norm.ppf(1 - ALPHA / 2) + norm.ppf(POWER)) ** 2 / d ** 2 / MW_EFFICIENCY


def cells_for_accuracy(acc: float) -> int:
    for n in range(4, 2000):
        k = int(binom.isf(ALPHA, n, 0.5)) + 1          # smallest k with P(X >= k | chance) <= alpha
        while binom.sf(k - 1, n, 0.5) > ALPHA:
            k += 1
        if binom.sf(k - 1, n, acc) >= POWER:
            return n
    return -1


def power(model: str, cells: pd.DataFrame) -> pd.DataFrame:
    sel_path = ROOT / "results" / "classification" / model / "feature_selection.csv"
    sel = pd.read_csv(sel_path)
    chosen = sel[sel.selected].sort_values("rank").feature.tolist()
    high = cells.her2 == "high"
    rows = []
    for f in chosen:
        a, b = cells.loc[high, f], cells.loc[~high, f]
        sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
        d = (a.mean() - b.mean()) / sp
        rows.append({"feature": f, "high_mean": a.mean(), "low_mean": b.mean(), "cohens_d": d,
                     "cells_per_class": int(np.ceil(cells_per_class(d)))})
    return pd.DataFrame(rows)


def resolution(model: str, cells: pd.DataFrame) -> dict:
    f = pd.read_csv(ROOT / "results" / "features" / model / "features_frames.csv")
    f = f[f.passed & f.cell.isin(cells.cell)]
    step = f.groupby(["cell", "laser_step"]).len_par
    per_step = pd.DataFrame({"rel_sd": step.std() / step.mean(), "n": step.size()})
    frame_noise, frames_per_step = per_step.rel_sd.median(), per_step.n.median()
    med = step.median().rename("L").reset_index()
    med["strain"] = med.groupby("cell").L.transform(lambda x: x / x.iloc[:BASE].median())
    diff_noise = med.groupby("cell").strain.apply(lambda x: x.diff().std() / np.sqrt(2)).median()
    sxx = ((np.arange(STEPS) - (STEPS - 1) / 2) ** 2).sum()
    out = {"length_px": f.len_par.median(), "frame_noise": frame_noise, "frames_per_step": frames_per_step,
           "observed_compliance_abs_median": cells.compliance_par.abs().median(),
           "observed_compliance_between_cell_sd": cells.compliance_par.std(),
           "class_difference_compliance": abs(cells[cells.her2 == "high"].compliance_par.mean()
                                              - cells[cells.her2 == "low"].compliance_par.mean())}
    for tag, s in (("model", 1.2533 * frame_noise / np.sqrt(frames_per_step)), ("diff", diff_noise)):
        out |= {f"step_noise_{tag}": s, f"step_change_{tag}": 2.77 * s,
                f"ramp_strain_{tag}": 1.96 * np.sqrt(2 / BASE) * s * 1.25,
                f"compliance_{tag}": 1.96 * s / np.sqrt(sxx) * 1.1}
    return out


def power_latex(p: pd.DataFrame, acc: dict, out: Path) -> None:
    rows = [f"{NAMES[r.feature]} & {r.cohens_d:+.2f} & {r.cells_per_class} \\\\" for r in p.itertuples()]
    rows += ["\\midrule", "\\multicolumn{3}{l}{\\emph{A classifier beating chance (cells in all, half of each class)}} \\\\"]
    rows += [f"true accuracy {int(a * 100)}\\,\\% & & {n} \\\\" for a, n in acc.items()]
    out.write_text("\\begin{tabular}{l c c}\n\\toprule\n"
                   "Feature & Cohen's $d$ (high $-$ low) & Cells per class needed \\\\\n\\midrule\n"
                   + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def resolution_latex(r: dict, out: Path) -> None:
    pct = lambda v, k=2: f"{v * 100:.{k}f}\\,\\%"   # noqa: E731
    rows = [f"One frame, length along the beam & \\multicolumn{{2}}{{c}}{{{pct(r['frame_noise'])} "
            f"(= {r['frame_noise'] * r['length_px']:.2f}\\,px of {r['length_px']:.0f}\\,px)}} \\\\",
            f"One laser step ($\\approx${r['frames_per_step']:.0f} frames) & {pct(r['step_noise_model'])} & "
            f"{pct(r['step_noise_diff'])} \\\\",
            "\\midrule",
            f"Smallest change between two steps & {pct(r['step_change_model'])} & {pct(r['step_change_diff'])} \\\\",
            f"Smallest strain over the ramp & {pct(r['ramp_strain_model'])} & {pct(r['ramp_strain_diff'])} \\\\",
            f"Smallest compliance (per step) & {pct(r['compliance_model'], 3)} & {pct(r['compliance_diff'], 3)} \\\\",
            "\\midrule",
            f"\\emph{{Observed}}: compliance, median $|k_\\parallel|$ & \\multicolumn{{2}}{{c}}"
            f"{{{pct(r['observed_compliance_abs_median'], 3)} per step}} \\\\",
            f"\\emph{{Observed}}: spread between cells (SD) & \\multicolumn{{2}}{{c}}"
            f"{{{pct(r['observed_compliance_between_cell_sd'], 3)} per step}} \\\\",
            f"\\emph{{Observed}}: high $-$ low class means & \\multicolumn{{2}}{{c}}"
            f"{{{pct(r['class_difference_compliance'], 3)} per step}} \\\\"]
    out.write_text("\\begin{tabular}{l c c}\n\\toprule\n"
                   "Noise and the smallest detectable deformation (95\\,\\%) & from frame noise & "
                   "from step differences \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nnunet1000_loocv", help="feature run in results/features/")
    ap.add_argument("--tables", type=Path, default=None, help="also write the LaTeX tables here")
    a = ap.parse_args()
    out = ROOT / "results" / "classification" / a.model
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "results" / "features" / a.model / "features_cells.csv")
    cells = cells[cells.usable].copy()
    cells["circularity"] = cells["circularity_median"]

    p = power(a.model, cells)
    acc = {x: cells_for_accuracy(x) for x in (0.7, 0.75, 0.8, 0.9)}
    r = resolution(a.model, cells)
    p.to_csv(out / "power.csv", index=False)
    pd.Series(r).to_csv(out / "resolution.csv", header=["value"])
    print(p.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("cells needed for a classifier of true accuracy:", acc)
    print(pd.Series(r).to_string(float_format=lambda v: f"{v:.5g}"))
    for d in [out] + ([a.tables] if a.tables else []):
        power_latex(p, acc, d / "classify_power.tex")
        resolution_latex(r, d / "classify_resolution.tex")
    print(f"wrote {out}" + (f", {a.tables}" if a.tables else ""))


if __name__ == "__main__":
    main()
