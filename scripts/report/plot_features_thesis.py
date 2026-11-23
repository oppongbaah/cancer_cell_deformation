"""
Thesis figure and numbers for Section 4.5 (Feature Extraction Quality), 2026-11-09.

    python scripts/report/plot_features_thesis.py --img ../masters_thesis/main_report/images/png

Reads only results/features/nnunet1000_loocv/ (each cell segmented by the leave-one-cell-out nnU-Net model that
did not see it): checks_test.csv (the annotated test frames, predicted and ground-truth features, frame checks),
features_frames.csv (every feature frame and whether it passed), cell_validation.csv, cell_feature_check.csv,
calibration.json, multi_region_feature.csv. Same error definitions as scripts/report/plot_features.py.
Prints the frame-check, cell-validation and feature-reliability numbers and writes fig_ch4_validation.png
(thesis Fig. "Validation of the 15 cells"; redrawn from the 10/29 report's features_validation.png):
  (a) Dice of every annotated frame per cell, with the cell's mean over the passing frames (limit 0.85)
  (b) predicted / ground-truth area of the same frames, with the cell's median (limits 0.80-1.25);
      the right margin gives the annotated frames and the feature frames that pass the frame checks
With --reliability it also writes the older fig_ch4_features.png (no longer in the thesis since 2026-11-09;
the reliability results are stated in the text and the selection figure of Chapter 3 shows them per candidate):
  (a) per-frame features: error of 90 % of the passing annotated frames against the ground truth (15 % limit)
  (b) per-cell deformation features: Spearman rho of predicted against ground truth across the 15 cells (0.8)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
R = ROOT / "results" / "features" / "nnunet1000_loocv"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
HIGH, LOW = "#2a78d6", "#eb6834"                   # as in plot_dataset_thesis.py
GOOD_DSC, SIZE_BAND = 0.85, (0.80, 1.25)            # cell validation limits (thesis Eq. cell_validation)
FRAME = {"area": "Area", "perimeter": "Perimeter", "major_axis": "Major axis", "minor_axis": "Minor axis",
         "aspect_ratio": "Aspect ratio", "circularity": "Circularity", "solidity": "Solidity",
         "eccentricity": "Eccentricity", **{f"hu_{k}": f"Hu moment {k}" for k in range(7)},
         "len_par": "Length along the beam", "len_perp": "Length across the beam", "volume": "Volume",
         "ellip_f2": "Elongation $f_2$", "roughness": "Roughness"}
SCALED = {"eccentricity", "ellip_f2", "roughness", *[f"hu_{k}" for k in range(7)]}
ANGLES = {"orientation": "Orientation", "ellip_phase": "Elongation direction"}
CELL = {"len_par_base": "Baseline length along", "len_perp_base": "Baseline length across",
        "volume_change": "Volume change", "orient_spread": "Orientation spread",
        "compliance_perp": "Compliance across", "nonlinearity": "Nonlinearity", "compliance_par": "Compliance along",
        "roughness_median": "Roughness", "ellip_change": "Elongation change", "poisson": "Transverse ratio",
        "onset_step": "Onset"}

for f in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyretermes-*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, required=True)
    ap.add_argument("--reliability", action="store_true", help="also write the older fig_ch4_features.png")
    a = ap.parse_args()

    fr = pd.read_csv(R / "features_frames.csv")
    test = pd.read_csv(R / "checks_test.csv")
    cv = pd.read_csv(R / "cell_validation.csv", index_col="cell")
    chk = pd.read_csv(R / "cell_feature_check.csv", index_col="cell")
    cal = json.loads((R / "calibration.json").read_text())
    multi = pd.read_csv(R / "multi_region_feature.csv")

    # ---- frame checks on the feature frames
    print(f"== Feature frames: {int(fr.passed.sum())} of {len(fr)} pass")
    for g in ("high", "low"):
        x = fr[fr.her2 == g]
        print(f"   {g}: {int((~x.passed).sum())} of {len(x)} dropped")
    per = fr.groupby("cell").passed.agg(["sum", "size"])
    print("   per cell passed:", {c: f"{int(r['sum'])}/{int(r['size'])}" for c, r in per.iterrows()})
    mf = multi.drop_duplicates("stem")
    print(f"   frames with more than one trapped-cell region: {len(mf)} "
          f"({mf.cell.value_counts().to_dict()})")
    print(f"== Annotated test frames: {cal['frames']}, pass {cal['passed']}; bad masks (Dice < {cal['good_dsc']}) "
          f"{cal['bad_masks']}, caught {cal['bad_masks_caught']}; good masks dropped {cal['good_masks_rejected']}; "
          f"Dice passed {cal['dsc_passed_mean']:.3f}, dropped {cal['dsc_failed_mean']:.3f}; by HER2 "
          f"{ {g: (v['passed'], round(v['dsc_passed_mean'], 3)) for g, v in cal['by_her2'].items()} }")
    if "cross_check" in cal:
        print("   cross-check:", cal["cross_check"])

    # ---- cell validation
    print("\n== Cell validation (cell: Dice, size ratio, validated)")
    for c, r in cv.iterrows():
        print(f"   {c:12s} {r.dsc:.3f} {r.size_bias:.2f} {r.validated}")
    print(f"   validated {int(cv.validated.sum())}/15; lowest Dice {cv.dsc.min():.3f} ({cv.dsc.idxmin()}); size "
          f"{cv.size_bias.min():.2f}-{cv.size_bias.max():.2f}")

    # ---- per-frame feature error on the passing annotated frames
    ok = test[test.passed]
    rows = []
    for f, lab in FRAME.items():
        both = ok[[f, f"gt_{f}"]].dropna()
        pr, gt = both[f], both[f"gt_{f}"]
        err = 100 * (pr - gt).abs() / (gt.abs().median() if f in SCALED else gt.abs())
        flips = float((np.sign(pr) != np.sign(gt)).mean()) if f.startswith("hu_") else 0.0
        rows.append((lab, err.median(), err.quantile(0.9), flips, err.quantile(0.9) <= 15 and flips <= 0.05))
    E = pd.DataFrame(rows, columns=["feature", "med", "p90", "flips", "stable"])
    print(f"\n== Per-frame feature error, {len(ok)} passing annotated frames")
    print(E.round(3).to_string(index=False))
    for f, lab in ANGLES.items():
        both = ok[[f, f"gt_{f}"]].dropna()
        d = ((both[f] - both[f"gt_{f}"] + 90) % 180 - 90).abs()
        print(f"   {lab}: median {d.median():.1f} deg, 90 % {d.quantile(0.9):.1f} deg")

    # ---- per-cell deformation features, predicted against ground truth
    v = chk[chk.validated]
    rr = []
    for k, lab in CELL.items():
        both = v[[f"pred_{k}", f"gt_{k}"]].dropna()
        rr.append((lab, spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic, len(both)))
    C = pd.DataFrame(rr, columns=["feature", "rho", "n"])
    print("\n== Per-cell deformation features, Spearman rho across cells")
    print(C.round(2).to_string(index=False))

    # ---- figure: validation of the 15 cells on their annotated frames
    a.img.mkdir(parents=True, exist_ok=True)
    order = sorted(cv.index, key=lambda c: (not c.startswith("high"), int(c.split("cell")[1])))
    names = [c.replace("_cell", "-") for c in order]
    rng = np.random.default_rng(0)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.6, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    ax.axvline(GOOD_DSC, color=INK2, lw=0.8, ls="--", zorder=1)
    bx.axvspan(*SIZE_BAND, color=GRID, alpha=0.7, lw=0, zorder=0)
    bx.axvline(1, color=INK2, lw=0.8, ls="--", zorder=1)
    for i, c in enumerate(order):
        t = test[test.cell == c]
        col = HIGH if c.startswith("high") else LOW
        y = i + rng.uniform(-0.2, 0.2, len(t))
        ps = t.passed.to_numpy()
        for axx, v in ((ax, t.dsc.to_numpy()), (bx, (t.area / t.gt_area).to_numpy())):
            axx.scatter(v[ps], y[ps], s=7, color=col, alpha=0.75, lw=0, zorder=3)
            axx.scatter(v[~ps], y[~ps], s=13, marker="x", color=col, lw=0.8, zorder=4)
        ax.plot([cv.dsc[c]] * 2, [i - 0.33, i + 0.33], color=INK, lw=1.4, zorder=5)
        bx.plot([cv.size_bias[c]] * 2, [i - 0.33, i + 0.33], color=INK, lw=1.4, zorder=5)
        bx.text(1.14, i, f"{int(ps.sum())} of {len(t)}", transform=bx.get_yaxis_transform(), va="center",
                ha="center", fontsize=7.5, color=INK2)
        bx.text(1.42, i, f"{int(per.loc[c, 'sum'])} of {int(per.loc[c, 'size'])}", transform=bx.get_yaxis_transform(),
                va="center", ha="center", fontsize=7.5, color=INK2)
    # right margin: frames that pass the frame checks, annotated (the points drawn) and feature frames
    for x, head in ((1.14, "Annotated\nframes\npassed"), (1.42, "Feature\nframes\npassed")):
        bx.text(x, -1.25, head, transform=bx.get_yaxis_transform(), va="center", ha="center", fontsize=7.5, color=INK)
    n_high = sum(c.startswith("high") for c in order)
    for axx in (ax, bx):
        axx.axhline(n_high - 0.5, color=INK2, lw=0.5)
        axx.tick_params(labelsize=7.5)
        axx.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        axx.set_axisbelow(True)
    ax.set_yticks(range(len(order)), names)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(min(0.0, test.dsc.min() - 0.03), 1.01)
    ax.set_xlabel("Dice coefficient", fontsize=8)
    ax.set_title("(a) Overlap with the ground truth", fontsize=8.5, color=INK, loc="left")
    bx.set_xlabel("Predicted area / ground-truth area", fontsize=8)
    bx.set_title("(b) Size ratio", fontsize=8.5, color=INK, loc="left")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=HIGH, ms=4, label="High HER2 cell, frame passed"),
                        Line2D([], [], marker="o", ls="", color=LOW, ms=4, label="Low HER2 cell, frame passed"),
                        Line2D([], [], marker="x", ls="", color=INK2, ms=4, label="Frame dropped by the checks"),
                        Line2D([], [], color=INK, lw=1.4, label="Value of the cell"),
                        Line2D([], [], color=INK2, lw=0.8, ls="--", label="Limit 0.85 (a), ratio 1 (b)"),
                        plt.Rectangle((0, 0), 1, 1, color=GRID, label="Allowed size ratio, 0.80 to 1.25")],
               loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7.5, frameon=False)
    fig.savefig(a.img / "fig_ch4_validation.png", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n-> {a.img / 'fig_ch4_validation.png'}")
    print("   frames outside the plotted ranges:", int((test.dsc < ax.get_xlim()[0]).sum()))
    ap_ = test.groupby("cell").passed.agg(["sum", "size"])
    print("   annotated frames passed:", {c: f"{int(r['sum'])}/{int(r['size'])}" for c, r in ap_.iterrows()},
          f"total {int(test.passed.sum())}/{len(test)}")

    # ---- older figure: reliability of the features (not in the thesis; --reliability)
    if a.reliability:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 3.6), gridspec_kw={"wspace": 0.95, "width_ratios": [1.1, 1]})
        E = E.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(E))
        for i, r in E.iterrows():
            ax.scatter(r.p90, i, s=24, facecolor=INK if r.stable else "white", edgecolor=INK, lw=1, zorder=3)
        ax.axvline(15, color=INK2, lw=0.8, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(0.05, 400)
        ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
        ax.set_yticks(y, E.feature, fontsize=7.5)
        ax.set_xlabel("Error of 90% of frames (%)", fontsize=8)
        ax.set_title("(a) Per-frame features", fontsize=8.5, color=INK, loc="left")
        C = C.sort_values("rho").reset_index(drop=True)
        y2 = np.arange(len(C))
        for i, r in C.iterrows():
            bx.scatter(r.rho, i, s=24, facecolor=INK if r.rho >= 0.8 else "white", edgecolor=INK, lw=1, zorder=3)
        bx.axvline(0.8, color=INK2, lw=0.8, ls="--", zorder=1)
        bx.set_xlim(0, 1.02)
        bx.set_yticks(y2, C.feature, fontsize=7.5)
        bx.set_xlabel(r"Spearman $\rho$, predicted against ground truth", fontsize=8)
        bx.set_title("(b) Per-cell deformation features", fontsize=8.5, color=INK, loc="left")
        for axx in (ax, bx):
            axx.tick_params(axis="x", labelsize=7.5)
            axx.grid(axis="x", color=GRID, lw=0.6, zorder=0)
            axx.set_axisbelow(True)
        fig.legend(handles=[Line2D([], [], marker="o", ls="", color=INK, ms=5, label="Meets the limit"),
                            Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=5, label="Fails the limit"),
                            Line2D([], [], color=INK2, lw=0.8, ls="--", label="Limit")],
                   loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=7.5, frameon=False)
        a.img.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.img / "fig_ch4_features.png", dpi=400, bbox_inches="tight", facecolor="white")
        print(f"\n-> {a.img / 'fig_ch4_features.png'}")


if __name__ == "__main__":
    main()
