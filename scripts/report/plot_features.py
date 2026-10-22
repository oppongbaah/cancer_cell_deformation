#!/usr/bin/env python3
"""
Feature-extraction figures and tables (22 October report).

Reads what scripts/features/extract_features.py wrote under results/features/<model>/ --
checks_test.csv, calibration.json, cell_validation.csv, features_frames.csv, features_cells.csv,
image_quality_session.csv -- plus the predicted masks and raw frames for the example tiles.
Nothing is typed in here.

    python scripts/report/plot_features.py --model nnunet100_5fold \
        --out ../masters_thesis/images --tables ../masters_thesis/main_work/tables

Writes
  features_validation.png   (a) predicted vs hand-drawn area on the annotated frames, passed / rejected;
                            (b) one example tile per kind of rejection, on the raw frame
  features_trajectories.png area and aspect ratio of every usable cell across the 50 laser steps
  features_validation.tex   what the checks catch, and the feature error of frames that pass
  features_cells.tex        per cell: annotated-frame Dice, size bias, validated, feature frames passed
  image_quality.tex         focus / illumination / contrast / noise, high vs low HER2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

HIGH, LOW = "#2a78d6", "#eb6834"                 # plot_cv.py's HER2 colours
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SHAPE = ["area", "perimeter", "major_axis", "minor_axis", "aspect_ratio", "eccentricity", "circularity", "solidity"]
LABEL = {"area": "Area", "perimeter": "Perimeter", "major_axis": "Major axis", "minor_axis": "Minor axis",
         "aspect_ratio": "Aspect ratio", "eccentricity": "Eccentricity$^\\ast$", "circularity": "Circularity",
         "solidity": "Solidity"}
IQ = {"focus_frame": "Focus, whole frame", "focus_cell": "Focus, around the cell", "illum_mean": "Illumination, mean",
      "illum_gradient": "Illumination, unevenness", "contrast_rms": "Contrast, whole frame",
      "contrast_cell": "Contrast, cell vs surround", "noise_sigma": "Noise"}


def short(c: str) -> str:
    return c.replace("_cell", "-")


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def write_sample(frames: pd.DataFrame, cells: pd.DataFrame, order: list, out: Path, n: int = 5) -> None:
    """Five consecutive rows of features_frames.csv -- one usable cell, centred on a frame the checks dropped,
    so the empty shape features of a dropped frame show -- split into (a) the frame and its checks, (b) the
    15 shape features, (c) the 7 image measures."""
    bad = frames[~frames.passed & frames.cell.map(cells["usable"]).astype(bool) & frames.laser_step.between(2, 47)]
    centre = bad.assign(o=bad.cell.map(order.index)).sort_values(["o", "laser_step"]).iloc[0]
    rows = frames[frames.cell == centre.cell].sort_values("laser_step")
    k = int(np.flatnonzero(rows.stem.values == centre.stem)[0])
    rows = rows.iloc[max(k - n // 2, 0):max(k - n // 2, 0) + n]

    def num(v, d=3):
        if isinstance(v, (bool, np.bool_)):
            return r"\checkmark" if v else r"\texttimes"
        if pd.isna(v):
            return "--"
        return f"{v:.{d}g}" if isinstance(v, float) else str(v)

    def block(title: str, cols: list[tuple[str, str]]) -> list[str]:
        out_ = [rf"\multicolumn{{{len(cols)}}}{{l}}{{\emph{{{title}}}}} \\", r"\midrule",
                " & ".join(rf"\texttt{{{c.replace('_', chr(92) + '_')}}}" for c, _ in cols) + r" \\", r"\midrule"]
        for _, r in rows.iterrows():
            cells_ = []
            for c, kind in cols:
                v = r[c]
                if kind == "text":
                    cells_.append(str(v).replace("_", r"\_") if not pd.isna(v) else "--")
                else:
                    cells_.append(num(v))
            out_.append(" & ".join(cells_) + r" \\")
        return out_

    a_cols = [(c, "text" if c in ("cell", "her2", "reasons") else "num") for c in
              ("cell", "her2", "frame_index", "laser_step", "n_regions", "trap_dist", "area_ratio", "boundary_conf",
               "uncertain_frac", "touches_border", "passed", "reasons")]
    b_cols = [(c, "num") for c in ("laser_step", "area", "perimeter", "major_axis", "minor_axis", "aspect_ratio", "eccentricity",
                                   "circularity", "solidity", "hu_0", "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6")]
    c_cols = [(c, "num") for c in ("laser_step", "focus_frame", "focus_cell", "illum_mean", "illum_gradient", "contrast_rms",
                                   "contrast_cell", "noise_sigma")]
    lines = [r"% GENERATED by scripts/report/plot_features.py -- do not edit by hand."]
    for title, cols in (("(a) the frame and its checks", a_cols), ("(b) the 15 shape features (empty where the frame "
                        "failed a check)", b_cols), ("(c) the 7 image measures (every frame)", c_cols)):
        spec = "l" * (len(cols) - 1) + (r"p{4.2cm}" if cols[-1][0] == "reasons" else "l")   # reasons wrap
        lines += [r"\begin{adjustbox}{max width=\linewidth}", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
                  *block(title, cols), r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\par\vspace{6pt}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="nnunet100_5fold")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    a = ap.parse_args()
    R = REPO / "results" / "features" / a.model
    test = pd.read_csv(R / "checks_test.csv")
    frames = pd.read_csv(R / "features_frames.csv")
    cells = pd.read_csv(R / "features_cells.csv", index_col="cell")
    cv = pd.read_csv(R / "cell_validation.csv", index_col="cell")
    cal = json.loads((R / "calibration.json").read_text())
    sess = pd.read_csv(R / "image_quality_session.csv")
    a.out.mkdir(parents=True, exist_ok=True)
    a.tables.mkdir(parents=True, exist_ok=True)
    order = sorted(cells.index, key=lambda c: (cells.loc[c, "her2"] != "high", int(c.split("cell")[1])))

    # ── figure 1: validation ────────────────────────────────────────────────
    # predicted vs ground-truth area on every hand-masked frame (one panel; the dropped-frame examples are in
    # plot_region_selection.py's figures, drawn on the masks)
    fig, ax = plt.subplots(figsize=(6.2, 4.6), facecolor="white")
    style(ax)
    lim = [test[["area", "gt_area"]].min().min() * 0.9, test[["area", "gt_area"]].max().max() * 1.05]
    ax.plot(lim, lim, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    for h, col in (("high", HIGH), ("low", LOW)):
        t = test[test.her2 == h]
        ax.scatter(t.gt_area[t.passed], t.area[t.passed], s=14, color=col, alpha=0.75, lw=0, zorder=3)
        ax.scatter(t.gt_area[~t.passed], t.area[~t.passed], s=26, marker="x", color=col, lw=1.3, zorder=4)
    h1 = test[test.cell == "high_cell1"]
    ax.annotate("high-1: drawn ~1.4$\\times$ too large\non every frame $\\rightarrow$ cell rejected",
                xy=(h1.gt_area.median(), h1.area.median()), xytext=(lim[0] + 40, lim[1] * 0.9),
                fontsize=7.6, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Ground-truth area (px)", fontsize=9, color=INK2)
    ax.set_ylabel("Predicted area (px)", fontsize=9, color=INK2)
    ax.set_title("Hand-masked frames, segmented by the model that never saw the cell", loc="left", fontsize=9.5,
                 color=INK)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=HIGH, label="high HER2"),
                       Line2D([], [], marker="o", ls="", color=LOW, label="low HER2"),
                       Line2D([], [], marker="o", ls="", color=INK2, label="frame passed"),
                       Line2D([], [], marker="x", ls="", color=INK2, label="frame rejected")],
              loc="lower right", fontsize=7.4, frameon=False)
    fig.savefig(a.out / "features_validation.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── figure 2: trajectories over the laser steps ─────────────────────────
    use = [c for c in order if cells.loc[c, "usable"]]
    f2 = frames[frames.cell.isin(use) & frames.passed].copy()
    base = f2[f2.laser_step <= 9].groupby("cell")["area"].median()
    f2["area_rel"] = f2.area / f2.cell.map(base)
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.6), facecolor="white", gridspec_kw={"wspace": 0.22})
    for ax, col, lab in ((axs[0], "area_rel", "Area / its median over steps 0-9"),
                         (axs[1], "aspect_ratio", "Aspect ratio (major / minor axis)")):
        style(ax)
        for c in use:
            t = f2[f2.cell == c].sort_values("laser_step")
            ax.plot(t.laser_step, t[col], color=HIGH if cells.loc[c, "her2"] == "high" else LOW, lw=0.9, alpha=0.55)
        for h, colr in (("high", HIGH), ("low", LOW)):
            g = f2[f2.her2 == h].groupby("laser_step")
            n_cells = f2.loc[f2.her2 == h, "cell"].nunique()
            m = g[col].median()[g["cell"].nunique() >= n_cells / 2]     # median only where half the cells have a frame
            ax.plot(m.index, m.values, color=colr, lw=2.4)
        ax.set_xlabel("Laser step (one frame per step; power increases)", fontsize=9, color=INK2)
        ax.set_title(lab, loc="left", fontsize=9.5, color=INK)
    axs[1].legend(handles=[Line2D([], [], color=HIGH, lw=2.4, label=f"high HER2 ({sum(cells.loc[use, 'her2'] == 'high')} cells)"),
                           Line2D([], [], color=LOW, lw=2.4, label=f"low HER2 ({sum(cells.loc[use, 'her2'] == 'low')} cells)"),
                           Line2D([], [], color=INK2, lw=0.9, alpha=0.6, label="one cell (thin); class median (thick)")],
                  loc="upper left", fontsize=7.6, frameon=False)
    fig.savefig(a.out / "features_trajectories.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── tables ──────────────────────────────────────────────────────────────
    gen = r"% GENERATED by scripts/report/plot_features.py -- do not edit by hand."
    g = cal["gates"]
    # the feature error of the frames that pass, against the same feature on the ground-truth mask, for all
    # 15 shape features. Per frame: |predicted - ground truth| / |ground truth| for the size and shape
    # measures; for eccentricity and the Hu moments, whose typical value is near 0 or of either sign,
    # |predicted - ground truth| / the median |ground truth| over the frames (marked *). STABLE: 90 % of
    # frames within STABLE_PCT of the ground truth and no more than STABLE_FLIPS of them flipping sign.
    STABLE_PCT, STABLE_FLIPS = 15.0, 0.05
    ok = test[test.passed]
    feats = SHAPE + [f"hu_{k}" for k in range(7)]
    lines = [gen, r"\begin{tabular}{l c c c c}", r"\toprule",
             r"\textbf{Feature} & \textbf{Median frame} $\tilde{e}$ & \textbf{90\,\% of frames} $e_{90}$ & \textbf{Sign flips} $f$ & "
             r"\textbf{Stable} \\", r"\midrule"]
    for f in feats:
        both = ok[[f, f"gt_{f}"]].dropna()          # a frame without the measure (e.g. a degenerate fit) is skipped
        pr, gt = both[f], both[f"gt_{f}"]
        diff = (pr - gt).abs()
        scaled = f == "eccentricity" or f.startswith("hu_")
        err = 100 * (diff / gt.abs().median() if scaled else diff / gt.abs())
        signed = f.startswith("hu_")                  # only the Hu moments can change sign
        flips = float((np.sign(pr) != np.sign(gt)).mean()) if signed else 0.0
        med, p90 = err.median(), err.quantile(0.9)
        stable = p90 <= STABLE_PCT and flips <= STABLE_FLIPS
        name = (LABEL.get(f, f.replace("hu_", "Hu moment ")) + ("$^\\ast$" if scaled else "")).replace("Eccentricity$^\\ast$$^\\ast$", "Eccentricity$^\\ast$")
        if f == "hu_0":
            lines.append(r"\midrule")
        lines.append(f"{name} & {med:.1f}\\,\\% & {p90:.1f}\\,\\% & "
                     + (f"{100 * flips:.0f}\\,\\%" if signed else "--") + " & "
                     + (r"\textcolor{tpcol}{\textbf{yes}}" if stable else r"\textcolor{warnred}{\textbf{no}}") + r" \\")
    lines.append(rf"\midrule \multicolumn{{5}}{{l}}{{\footnotesize {len(ok)} hand-masked frames that pass the checks. "
                 rf"Stable: 90\,\% of frames within {STABLE_PCT:.0f}\,\%,}} \\"
                 rf" \multicolumn{{5}}{{l}}{{\footnotesize and at most {100 * STABLE_FLIPS:.0f}\,\% of them flipping sign.}} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (a.tables / "features_validation.tex").write_text("\n".join(lines))

    # five consecutive rows of features_frames.csv, around a dropped frame of a usable cell, in three parts
    write_sample(frames, cells, order, a.tables / "features_sample.tex")

    lines = [gen, r"\begin{tabular}{lccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{\emph{Annotated frames}} & \multicolumn{2}{c}{\emph{Feature frames}} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
             r"Cell & Dice & Size & Validated & Passed & Usable \\", r"\midrule"]
    for c in order:
        ok = lambda v: r"\ding{51}" if v else r"\textcolor{warnred}{\ding{55}}"
        tag = r"\hiHER" if cells.loc[c, "her2"] == "high" else r"\loHER"
        lines.append(rf"{tag}{{{short(c)}}} & {cv.loc[c, 'dsc']:.3f} & {cv.loc[c, 'size_bias']:.2f}$\times$ & "
                     rf"{ok(cv.loc[c, 'validated'])} & {int(cells.loc[c, 'passed'])}/{int(cells.loc[c, 'frames'])} & "
                     rf"{ok(cells.loc[c, 'usable'])} \\")
    lines += [r"\midrule", rf"\textbf{{Total}} & & & {int(cv.validated.sum())}/{len(cv)} & "
              rf"{int(cells.passed.sum())}/{int(cells.frames.sum())} & {int(cells.usable.sum())}/{len(cells)} \\",
              r"\bottomrule", r"\end{tabular}", ""]
    (a.tables / "features_cells.tex").write_text("\n".join(lines))

    lines = [gen, r"\begin{tabular}{lccccc}", r"\toprule",
             r"\emph{Per-cell means} & \hiHER{High} & \loHER{Low} & Cohen's $d$ & AUC & $p$ \\", r"\midrule"]
    for _, r in sess.iterrows():
        strong = abs(r.auc_high_vs_low - 0.5) >= 0.3
        name = rf"\textbf{{{IQ[r.measure]}}}" if strong else IQ[r.measure]
        p = f"{r.p_mannwhitney:.3f}" if r.p_mannwhitney >= 0.001 else "$<$0.001"
        lines.append(f"{name} & {r.high_mean:.3g} & {r.low_mean:.3g} & {r.cohens_d:+.2f} & {r.auc_high_vs_low:.2f} & "
                     f"{p} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (a.tables / "image_quality.tex").write_text("\n".join(lines))

    # each image measure per cell, grouped by RECORDING POSITION (the trap's height in the frame), not by HER2:
    # the 9 high cells + low-1 trap at y ~535 px, the other 5 low cells at y ~460 px. Colour stays HER2. If a
    # measure follows the session, the two groups separate completely, with low-1 inside its session's group.
    from scipy.stats import mannwhitneyu
    per = frames.groupby("cell").agg(her2=("her2", "first"), ty=("trap_y", "first"), **{m: (m, "mean") for m in IQ})
    gap = np.sort(per.ty.values)                         # split at the largest gap between trap heights
    cut = gap[np.argmax(np.diff(gap))] + np.max(np.diff(gap)) / 2
    per["upper"] = per.ty > cut
    ms = list(IQ)
    fig, axs = plt.subplots(1, len(ms), figsize=(10.4, 3.2), facecolor="white", gridspec_kw={"wspace": 0.55})
    rng = np.random.default_rng(0)
    for ax, m in zip(axs, ms):
        style(ax)
        for x, grp in enumerate((per[per.upper], per[~per.upper])):
            xs = x + rng.uniform(-0.13, 0.13, len(grp))
            cols = [HIGH if h == "high" else LOW for h in grp.her2]
            ax.scatter(xs, grp[m], s=22, c=cols, alpha=0.85, lw=0, zorder=3)
            ax.plot([x - 0.27, x + 0.27], [grp[m].median()] * 2, color=INK2, lw=2, zorder=2)
            if "low_cell1" in grp.index:
                k = list(grp.index).index("low_cell1")
                ax.scatter(xs[k], grp.loc["low_cell1", m], s=60, facecolor="none", edgecolor=INK, lw=1, zorder=4)
        u = mannwhitneyu(per[per.upper][m], per[~per.upper][m])
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"y≈{per[per.upper].ty.median():.0f}", f"y≈{per[~per.upper].ty.median():.0f}"],
                           fontsize=7.4)
        ax.set_xlim(-0.6, 1.6)
        ax.tick_params(axis="y", labelsize=7)
        sig = u.pvalue < 0.05
        ax.set_title(IQ[m].replace(", ", ",\n"), fontsize=7.8, color=INK if sig else INK2,
                     fontweight="bold" if sig else "normal", pad=4)
        ax.text(0.5, -0.22, f"p = {u.pvalue:.3f}" if u.pvalue >= 0.001 else "p < 0.001", transform=ax.transAxes,
                ha="center", fontsize=7.2, color=INK if sig else INK2)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=HIGH, label="HER2 high cell"),
                        Line2D([], [], marker="o", ls="", color=LOW, label="HER2 low cell"),
                        Line2D([], [], color=INK2, lw=2, label="median of the group"),
                        Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=7, label="low-1")],
               loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=7.4, frameon=False,
               title="grouped by the trap's height y (px): 9 high cells + low-1  |  5 low cells", title_fontsize=7.6)
    fig.savefig(a.out / "features_session_cells.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {a.out}/features_{{validation,trajectories}}.png, {a.tables}/{{features_validation,features_cells,image_quality}}.tex")


if __name__ == "__main__":
    main()
