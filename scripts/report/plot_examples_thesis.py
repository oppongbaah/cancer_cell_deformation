"""
Thesis figure for Section 4.4.4 (Final Models): frames, ground truth and nnU-Net masks, 2026-11-09.

    python scripts/report/plot_examples_thesis.py --img ../masters_thesis/main_report/images/png

Each cell is segmented by the leave-one-cell-out nnU-Net model that never saw it (round r tests one cell):
  data/nnunet/predictions/experiment_4_loocv_nnunet/round<r>/<stem>.png       prediction, 0/1/2, 1228 x 922
  data/nnunet/raw/Dataset502_CellTrapLOOCV/labelsTr/<stem>.png                 ground truth
  data/nnunet/raw/Dataset502_CellTrapLOOCV/imagesTs_round<r>/<stem>_0000.png   frame
  results/features/nnunet1000_loocv/checks_test.csv                            laser step of each test frame
Cells chosen by rule from the leave-one-cell-out Dice: best (low-5), median (high-9), weakest apart from high-10
(high-1), and high-10. Frames chosen by rule: the test frame nearest sampling interval 10 (low power) and nearest
interval 40 (high power). Prints the pooled Dice of each cell over its 20 test frames as a check against the log.
Colours (agreed for all mask figures): ground truth green (trapped cell) / red (other object); prediction white /
grey; difference yellow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
P = ROOT / "data" / "nnunet" / "predictions" / "experiment_4_loocv_nnunet"
D = ROOT / "data" / "nnunet" / "raw" / "Dataset502_CellTrapLOOCV"
CELLS = [("low_cell5", 2, "low-5"), ("high_cell9", 1, "high-9"), ("high_cell1", 13, "high-1"),
         ("high_cell10", 14, "high-10")]
STEPS = (10, 40)
HALF = 75                                     # crop: 150 x 150 px around the trapped cell of the ground truth
INK, INK2 = "#0b0b0b", "#52514e"
GREEN, RED, WHITE, GREY, YELLOW = (0.20, 0.75, 0.30), (0.85, 0.20, 0.20), (1, 1, 1), (0.55, 0.55, 0.55), (1.0, 0.85, 0.0)

for f in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyretermes-*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman"],
                     "font.size": 9})


def colour(lab, c1, c2):
    out = np.zeros(lab.shape + (3,))
    out[lab == 1] = c1
    out[lab == 2] = c2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, required=True)
    a = ap.parse_args()
    test = pd.read_csv(ROOT / "results" / "features" / "nnunet1000_loocv" / "checks_test.csv")

    for cell, rnd, lab in CELLS:
        files = sorted((P / f"round{rnd}").glob("*.png"))
        tp = fp = fn = 0
        for f in files:
            p, g = np.array(Image.open(f)) == 1, np.array(Image.open(D / "labelsTr" / f.name)) == 1
            tp, fp, fn = tp + (p & g).sum(), fp + (p & ~g).sum(), fn + (~p & g).sum()
        cell_dice = 2 * tp / (2 * tp + fp + fn)
        print(f"{lab}: round {rnd}, {len(files)} test frames, Dice {cell_dice:.3f}")
        t = test[test.cell == cell].set_index("stem")
        stems = [(t.laser_step - s).abs().idxmin() for s in STEPS]

        fig = plt.figure(figsize=(6.5, 1.36))
        gs = fig.add_gridspec(1, 9, width_ratios=[1, 1, 1, 1, 0.18, 1, 1, 1, 1], wspace=0.05,
                              left=0.0, right=1.0, bottom=0.0, top=0.64)
        for k, stem in enumerate(stems):
            step = int(t.laser_step[stem])
            frame = np.array(Image.open(D / f"imagesTs_round{rnd}" / f"{stem}_0000.png").convert("L"))
            gt = np.array(Image.open(D / "labelsTr" / f"{stem}.png"))
            pr = np.array(Image.open(P / f"round{rnd}" / f"{stem}.png"))
            yy, xx = np.nonzero(gt == 1)
            cy, cx = int(yy.mean()), int(xx.mean())
            sl = (slice(max(cy - HALF, 0), cy + HALF), slice(max(cx - HALF, 0), cx + HALF))
            g, p = gt[sl] == 1, pr[sl] == 1
            dice = 2 * (p & g).sum() / (p.sum() + g.sum())
            diff = np.zeros(g.shape + (3,))
            diff[p & g] = GREEN                      # agreement: the trapped cell, green as in the ground truth
            diff[p ^ g] = YELLOW
            panels = [(frame[sl], "gray"), (colour(gt[sl], GREEN, RED), None), (colour(pr[sl], WHITE, GREY), None),
                      (diff, None)]
            axs = [fig.add_subplot(gs[0, 5 * k + j]) for j in range(4)]
            for ax, (im, cmap), ttl in zip(axs, panels, ("Frame", "Ground truth", "Predicted", "Difference")):
                ax.imshow(im, cmap=cmap, interpolation="nearest")
                ax.set_xticks([]), ax.set_yticks([])
                ax.set_title(ttl, fontsize=7, color=INK, pad=2)
                for sp in ax.spines.values():
                    sp.set_color(INK2), sp.set_linewidth(0.5)
            axs[3].text(0.96, 0.04, f"Dice {dice:.2f}", transform=axs[3].transAxes, ha="right", va="bottom",
                        fontsize=6, color="white")
            power = "Low" if k == 0 else "High"
            fig.text(0.245 if k == 0 else 0.755, 0.755, f"{power} laser power, interval {step + 1}", ha="center",
                     fontsize=7.5, color=INK)
            print(f"  {lab} {stem} interval {step + 1}: frame Dice {dice:.3f}")
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(fc=GREEN, label="Ground truth, trapped cell"), Patch(fc=RED, label="other objects"),
                            Patch(fc=WHITE, ec=INK2, lw=0.5, label="Predicted, trapped cell"),
                            Patch(fc=GREY, label="other objects"), Patch(fc=YELLOW, label="Difference")],
                   loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=6.8, frameon=False,
                   handlelength=1.1, columnspacing=1.3, handletextpad=0.4)
        a.img.mkdir(parents=True, exist_ok=True)
        out = a.img / f"fig_ch4_examples_{lab.replace('-', '')}.png"
        fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
