"""
Thesis figure and table rows for Section 4.4.4 (Final Models), 2026-11-09.

    python scripts/report/plot_final_models_thesis.py --img ../masters_thesis/main_report/images/png

The final models are the 15 leave-one-cell-out models of each network (thesis 3.4.6). Reads:
  logs/randswap/experiment_4_loocv/unet/loocv_baseline_fold_<k>.log          U-Net, one log per round
  checkpoints/randswap/experiment_4_loocv/nnunet/Dataset502_CellTrapLOOCV/
      nnUNetTrainer__nnUNetPlans__2d/fold_<k-1>/training_log_*.txt           nnU-Net, fold k-1 = round k
  results/randswap/experiment_4_loocv/{unet/loocv_baseline,nnunet/nnunet}_cv_cells.csv   test scores per cell
Writes fig_ch4_training.png: loss and Dice of the validation cell per epoch, median and interquartile range
over the 15 rounds, for each network; prints the per-cell test table (Dice, precision, recall).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent.parent
ULOG = ROOT / "logs" / "randswap" / "experiment_4_loocv" / "unet"
NDIR = (ROOT / "checkpoints" / "randswap" / "experiment_4_loocv" / "nnunet" / "Dataset502_CellTrapLOOCV"
        / "nnUNetTrainer__nnUNetPlans__2d")
RES = ROOT / "results" / "randswap" / "experiment_4_loocv"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
UNET, NNUNET, TRAIN = "#5546c4", "#b8336a", "#8a8985"

for f in Path("/usr/share/texmf/fonts/opentype/public/tex-gyre").glob("texgyretermes-*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "Times New Roman"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})


def unet_rounds():
    out, best = [], []
    pat = re.compile(r"^Epoch (\d+)/\d+\s+train_loss=([\d.]+)\s+train_DSC=([\d.]+)\s+val_loss=([\d.]+)\s+val_DSC=([\d.]+)")
    for k in range(1, 16):
        rows, b = [], None
        for line in open(ULOG / f"loocv_baseline_fold_{k}.log"):
            m = pat.match(line)
            if m:
                rows.append([float(x) for x in m.groups()])
            m = re.search(r"Reloaded best checkpoint for final evaluation: epoch (\d+)", line)
            if m:
                b = int(m.group(1))
        out.append(pd.DataFrame(rows, columns=["epoch", "train_loss", "train_dsc", "val_loss", "val_dsc"]))
        best.append(b)
    return out, best


def nnunet_rounds():
    out = []
    for k in range(15):
        log = sorted((NDIR / f"fold_{k}").glob("training_log_*.txt"))[-1]
        rows, cur = {}, None
        for line in open(log):
            if m := re.search(r": Epoch (\d+)\s*$", line):
                cur = int(m.group(1))
                rows[cur] = {}
            elif cur is not None and (m := re.search(r": (train_loss|val_loss) (-?[\d.]+)", line)):
                rows[cur][m.group(1)] = float(m.group(2))
            elif cur is not None and (m := re.search(r"Pseudo dice \[np\.float32\(([\d.]+)\)", line)):
                rows[cur]["val_dsc"] = float(m.group(1))       # first foreground class = trapped cell
        out.append(pd.DataFrame.from_dict(rows, orient="index").rename_axis("epoch").reset_index())
    return out


def band(ax, runs, col, color, label, min_runs=8):
    n = max(len(r) for r in runs)
    m = np.full((len(runs), n), np.nan)
    for i, r in enumerate(runs):
        m[i, :len(r)] = r[col].values
    keep = np.sum(~np.isnan(m), axis=0) >= min_runs
    x = np.arange(1, n + 1)[keep]
    q1, med, q3 = (np.nanpercentile(m[:, keep], q, axis=0) for q in (25, 50, 75))
    ax.fill_between(x, q1, q3, color=color, alpha=0.18, lw=0)
    ax.plot(x, med, color=color, lw=1.4, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, required=True)
    a = ap.parse_args()

    ur, ubest = unet_rounds()
    nr = nnunet_rounds()
    print(f"U-Net: epochs run {min(len(r) for r in ur)}-{max(len(r) for r in ur)}, best epochs {ubest}, "
          f"median {int(np.median(ubest))}; final validation Dice at best epoch: "
          f"median {np.median([r.set_index('epoch').val_dsc[b] for r, b in zip(ur, ubest)]):.3f}")
    print(f"nnU-Net: epochs {[len(r) for r in nr][:3]}...; last-epoch pseudo Dice median "
          f"{np.median([r.val_dsc.iloc[-1] for r in nr]):.3f}")

    fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.6), gridspec_kw={"hspace": 0.55, "wspace": 0.28})
    for row, (runs, name, col) in enumerate(((ur, "U-Net", UNET), (nr, "nnU-Net", NNUNET))):
        ax, bx = axs[row]
        band(ax, runs, "train_loss", TRAIN, "Training")
        band(ax, runs, "val_loss", col, "Validation cell")
        band(bx, runs, "val_dsc", col, "Validation cell")
        if name == "U-Net":
            for axx in (ax, bx):
                axx.axvline(np.median(ubest), color=INK2, lw=0.7, ls="--", zorder=1)
            bx.text(np.median(ubest) + 1, 0.42, f"median best\nepoch {int(np.median(ubest))}", fontsize=7, color=INK2)
        letter = "ab" if row == 0 else "cd"
        ax.set_title(f"({letter[0]}) {name}, loss", fontsize=8.5, color=INK, loc="left")
        bx.set_title(f"({letter[1]}) {name}, Dice of the trapped cell", fontsize=8.5, color=INK, loc="left")
        bx.set_ylim(0.4 if name == "U-Net" else 0.8, 1.0)
        if name == "nnU-Net":
            for axx in (ax, bx):
                axx.set_xscale("log")
                axx.set_xlim(1, 1000)
        for axx in (ax, bx):
            axx.set_xlabel("Epoch" + (" (log scale)" if name == "nnU-Net" else ""), fontsize=8)
            axx.tick_params(labelsize=7.5)
            axx.grid(axis="y", color=GRID, lw=0.6, zorder=0)
            axx.set_axisbelow(True)
        ax.set_ylabel("Loss", fontsize=8)
        bx.set_ylabel("Dice", fontsize=8)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    a.img.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.img / "fig_ch4_training.png", dpi=400, bbox_inches="tight", facecolor="white")
    print(f"-> {a.img / 'fig_ch4_training.png'}")

    u = pd.read_csv(RES / "unet" / "loocv_baseline_cv_cells.csv").set_index("cell")
    n = pd.read_csv(RES / "nnunet" / "nnunet_cv_cells.csv").set_index("cell")
    cells = sorted(u.index, key=lambda c: (not c.startswith("high"), int(c.split("cell")[1])))
    print("\n== Table rows: cell & U-Net Dice & P & R & nnU-Net Dice & P & R")
    for c in cells:
        lab = c.replace("_cell", "-")
        print(f"        {lab:8s} & {u.dsc[c]:.3f} & {u.precision[c]:.3f} & {u.recall[c]:.3f} & "
              f"{n.dsc[c]:.3f} & {n.precision[c]:.3f} & {n.recall[c]:.3f} \\\\")
    print(f"        Mean & ${u.dsc.mean():.3f} \\pm {u.dsc.std():.3f}$ & {u.precision.mean():.3f} & {u.recall.mean():.3f} & "
          f"${n.dsc.mean():.3f} \\pm {n.dsc.std():.3f}$ & {n.precision.mean():.3f} & {n.recall.mean():.3f} \\\\")


if __name__ == "__main__":
    main()
