#!/usr/bin/env python3
"""
How the extraction picks the trapped cell, and why it drops a frame -- drawn on the predicted masks
(22 October report, feature extraction).

  Examples prefer frames whose laser step also has a hand-masked frame, so the ground-truth row is from
  the same step as the frame above it (feature frames themselves are never hand-masked).

  (a) Frames where the model marks more than one trapped-cell region: the region selected (green; nearest
      the video's trap) and the others (white, as in the predicted mask), the trap and its 45 px distance limit (yellow), and the
      size and distance of every region. Chosen from the run's own frames: one per cell, preferring
      frames where the LARGEST region is not the one kept.
  (b) Frames the frame checks dropped, one per kind of failure (distance, merged outline, size jump,
      unsure outline): the selected region (green), any other region (white), and the failed check with
      its value and limit. Preferring frames with a single failed check, from cells that are used.
      (Solidity = area / convex hull: a dent, or a neighbour fused on, lowers it.)

Everything is read from the extraction's outputs: results/features/<model>/features_frames.csv (the
checks and reasons; every feature frame, up to 6 per laser step) and data/features/<model>/{feature,dense}/labels/
(the masks: the step frames and the other frames of a step were segmented from two folders). The masks are shown as they
are, label 1 only, on black -- no frame underneath.

    python scripts/report/plot_region_selection.py --model nnunet1000_loocv --output ../masters_thesis/images/region_selection.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

REPO = Path(__file__).resolve().parent.parent.parent
FACTORY = REPO / "data" / "frames" / "03_mask_factory"     # the step frames
DENSE = REPO / "data" / "frames" / "05_dense_frames"       # the other feature frames of each step
sys.path.insert(0, str(REPO))
from celldform.features.quality import Gates, regions, select_trapped_region  # noqa: E402
from celldform.features.segmenters import five_fold_records  # noqa: E402

KEPT, OTHER, TRAP = (0.18, 0.80, 0.30), (1.0, 1.0, 1.0), "#ffd21f"   # selected green; another region ALSO
                                         # marked as trapped cell stays white, as in the predicted mask
DECOY = (0.88, 0.22, 0.20)               # ground truth: other-object (decoy) class, red as in the annotation tool
PRED_DECOY = (0.42, 0.42, 0.42)          # predicted: other-object class, grey
PRED, HIGH = "#ffffff", "#2a78d6"        # predicted mask (white) and row-label colours, as in plot_high1_exclusion.py
INK, INK2, BAD = "#0b0b0b", "#52514e", "#c62828"
G = Gates()
NEAR_PX = 160            # the view holds, in full, every object whose centre is within this of the trap
# kind of failure -> (pattern in the reasons, how it reads in the figure)
KINDS = {
    "distance": (r"px from the trap", "not in the trap"),
    "solidity": (r"solidity", "dented or merged outline"),
    "area": (r"area x", "sudden size jump"),
    "confidence": (r"boundary confidence", "model unsure of the outline"),
}


GT = {}   # cell -> {stem: hand mask} -- each cell's hand-masked (5-fold test) frames
for _r in five_fold_records().values():
    for _s, _m in _r["test_masks_native"].items():
        GT.setdefault(re.search(r"(high|low)_cell\d+", _s).group(0), {})[_s] = REPO / _m


def index_of(stem: str) -> int:
    return int(re.search(r"_(\d+)$", stem).group(1))


def short(c: str) -> str:
    return c.replace("_cell", "-")


def view(lab: np.ndarray, row: pd.Series, half: int = 70) -> dict:
    """One example's tiles: frame, ground truth, predicted mask and selection, all in one square view that
    holds every object near the trap (predicted or hand-drawn) and the trap's limit circle."""
    rs = regions(lab == 1)
    kept, _ = select_trapped_region(lab == 1, (row.trap_x, row.trap_y))
    tx, ty = row.local_trap_x, row.local_trap_y
    # feature frames have no hand mask (by design): the nearest hand-masked frame of the same cell
    gstem, gpath = min(GT[row.cell].items(), key=lambda kv: abs(index_of(kv[0]) - row.frame_index))
    g_full = cv2.imread(str(gpath), cv2.IMREAD_UNCHANGED)
    # zoom out to every object near the trap -- predicted or hand-drawn, cell or decoy -- in full
    n_cc, cc, _, cents = cv2.connectedComponentsWithStats(((lab > 0) | (g_full > 0)).astype(np.uint8), connectivity=8)
    near = [k for k in range(1, n_cc) if np.hypot(cents[k][0] - tx, cents[k][1] - ty) <= NEAR_PX]
    show = np.isin(cc, near) | (lab == 1)       # ... and every trapped-cell region, however far (it may be the one selected)
    ys, xs = np.nonzero(show) if show.any() else (np.array([ty]), np.array([tx]))
    r_lim = G.max_trap_dist
    x0, x1 = min(xs.min(), tx - r_lim) - 10, max(xs.max(), tx + r_lim) + 10
    y0, y1 = min(ys.min(), ty - r_lim) - 10, max(ys.max(), ty + r_lim) + 10
    side = int(max(2 * half, x1 - x0, y1 - y0))
    cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
    ox, oy = cx - side // 2, cy - side // 2
    rgb = np.zeros((side, side, 3))
    info = []
    solidity = np.nan

    def crop(img: np.ndarray) -> np.ndarray:          # the same square window, zero-padded at the frame edge
        pad = cv2.copyMakeBorder(img, side, side, side, side, cv2.BORDER_CONSTANT)
        return pad[oy + side:oy + 2 * side, ox + side:ox + 2 * side]
    model = np.zeros((side, side, 3))
    lab_c = crop(lab)
    model[lab_c == 1] = matplotlib.colors.to_rgb(PRED)   # the prediction as it is, as in the high-1
    model[lab_c == 2] = PRED_DECOY                       # figure: trapped-cell class white, other objects grey
    for r in sorted(rs, key=lambda r: r["mask"].sum(), reverse=True):
        is_kept = kept is not None and r["label"] == kept["label"]
        if is_kept:
            cnts, _ = cv2.findContours(r["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            hull = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea)))   # as in quality.py
            solidity = float(r["mask"].sum()) / hull if hull > 0 else np.nan
        m = np.zeros((side, side), bool)
        yy, xx = np.nonzero(r["mask"])
        ok = (yy - oy >= 0) & (yy - oy < side) & (xx - ox >= 0) & (xx - ox < side)
        m[yy[ok] - oy, xx[ok] - ox] = True
        rgb[m] = KEPT if is_kept else OTHER
        d = float(np.hypot(r["centroid"][0] - tx, r["centroid"][1] - ty))
        info.append({"kept": is_kept, "area": r["area"], "dist": d})
    info.sort(key=lambda i: not i["kept"])
    raw_p = FACTORY / f"{row.stem}.jpg"
    raw = cv2.imread(str(raw_p if raw_p.exists() else DENSE / f"{row.stem}.jpg"), cv2.IMREAD_GRAYSCALE)
    g = crop(g_full)
    gt = np.zeros((side, side, 3))
    gt[g == 1] = KEPT
    gt[g == 2] = DECOY
    return {"rgb": rgb, "trap": (tx - ox, ty - oy), "side": side, "info": info, "raw": crop(raw), "model": model,
            "gt": gt, "gt_step": index_of(gstem) // 200, "solidity": solidity}


def plain(ax, img: np.ndarray, title: str | None = None, gray: bool = False) -> None:
    ax.imshow(img, cmap="gray" if gray else None, vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c9c8c3" if gray else "#000000")
    if title:
        ax.set_title(title, fontsize=7.6, color=INK, pad=3, fontweight="bold")


def draw(ax, v: dict, title: str | None, lines: list[tuple[str, str]], bad: bool) -> None:
    ax.imshow(v["rgb"], interpolation="nearest")
    ax.add_patch(Circle(v["trap"], G.max_trap_dist, fill=False, ec=TRAP, lw=1.0, ls=(0, (3, 2))))
    ax.plot(*v["trap"], marker="+", color=TRAP, ms=8, mew=1.4)
    ax.set_xlim(-0.5, v["side"] - 0.5)
    ax.set_ylim(v["side"] - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(BAD if bad else "#2e7d32")
        sp.set_linewidth(1.6)
    if title:
        ax.set_title(title, fontsize=7.6, color=INK, pad=3, fontweight="bold")
    ax.text(0.03, 0.03, f"{v['side']} px", transform=ax.transAxes, fontsize=6, color="#bdbdbd", va="bottom")
    for k, (text, col) in enumerate(lines):
        ax.text(0.5, -0.04 - 0.075 * k, text, transform=ax.transAxes, ha="center", va="top", fontsize=6.9,
                color=col)


def gt_same_step(r: pd.Series) -> bool:
    """A hand-masked frame of the cell in the same laser step (so the ground-truth row lines up)."""
    return any(index_of(s) // 200 == r.laser_step for s in GT[r.cell])


def pick_multi(df: pd.DataFrame, labels: dict[str, Path], n: int) -> list[tuple[pd.Series, dict]]:
    """n/2 frames WITH a hand mask in the same step, then n/2 WITHOUT, different cells within each half,
    preferring frames where the largest region is not the one selected."""
    cand = []
    for _, r in df[(df.n_regions > 1) & df.passed].iterrows():
        v = view(cv2.imread(str(labels[r.stem]), cv2.IMREAD_UNCHANGED), r)
        largest_not_kept = max(v["info"], key=lambda i: i["area"])["kept"] is False
        cand.append((not largest_not_kept, -max(i["area"] for i in v["info"] if not i["kept"]),
                     gt_same_step(r), r, v))
    cand.sort(key=lambda c: c[:2])
    out = []
    for aligned in (True, False):
        half, seen = [], set()
        for *_, al, r, v in cand:
            if al == aligned and r.cell not in seen and len(half) < n // 2:
                half.append((r, v))
                seen.add(r.cell)
        for *_, al, r, v in cand:        # fill from the same half if there are too few cells
            if al == aligned and len(half) < n // 2 and all(r.stem != h[0].stem for h in half):
                half.append((r, v))
        out += half
    return out


def pick_dropped(df: pd.DataFrame, labels: dict[str, Path], unusable: set[str], n: int) -> list[tuple[str, pd.Series, dict]]:
    """One frame per kind of failure, n/2 of them WITH a hand mask in the same step and n/2 WITHOUT. The
    distance check is shown by its clear case (no cell in the trap, >= 2x the limit), which has no hand
    mask in its step; the other kinds fill the two halves in KINDS order."""
    bad = df[~df.passed].copy()
    bad["n_reasons"] = bad.reasons.str.count(";") + 1
    bad["used"] = ~bad.cell.isin(unusable)
    bad["aligned"] = [gt_same_step(x) for _, x in bad.iterrows()]
    best = {}                             # kind -> {aligned: best row}
    for kind, (pat, _) in KINDS.items():
        c = bad[bad.reasons.str.split(";").str[0].str.contains(pat)]
        if kind == "distance":
            c = c[c.trap_dist >= 2 * G.max_trap_dist]
        for al in (True, False):
            cc = c[c.aligned == al].sort_values(["used", "n_reasons"], ascending=[False, True])
            if not cc.empty:
                best.setdefault(kind, {})[al] = cc.iloc[0]
    want = {True: n // 2, False: n - n // 2}
    plan = {True: [], False: []}
    for kind in sorted(best, key=lambda k: (len(best[k]) > 1, list(KINDS).index(k))):   # constrained kinds first
        for al in sorted(best[kind], key=lambda a: len(plan[a]) - want[a]):
            if len(plan[al]) < want[al]:
                plan[al].append((kind, best[kind][al]))
                break
    order = list(KINDS)
    out = []
    for al in (True, False):
        for kind, r in sorted(plan[al], key=lambda kr: order.index(kr[0])):
            out.append((kind, r, view(cv2.imread(str(labels[r.stem]), cv2.IMREAD_UNCHANGED), r)))
    return out


def reason_lines(r: pd.Series, v: dict) -> list[tuple[str, str]]:
    """The failed checks with enough digits to see the value is past its limit (45.3 vs 45, not 45 vs 45)."""
    lines = []
    for w in r.reasons.split("; "):
        if "from the trap" in w:
            lines.append(f"{r.trap_dist:.1f} px off (limit {G.max_trap_dist:.0f})")
        elif w.startswith("solidity"):
            lines.append(f"solidity {v['solidity']:.3f} (min {G.min_solidity})")
        elif w.startswith("area x"):
            lines.append(f"size {r.area_ratio:.2f}× nearby ({G.area_ratio_range[0]}–{G.area_ratio_range[1]})")
        elif w.startswith("boundary confidence"):
            lines.append(f"confidence {r.boundary_conf:.3f} (min {G.min_boundary_conf})")
        else:
            lines.append(w)
    return [(t, BAD) for t in lines]


def write_checks(test: pd.DataFrame, out: Path, good_dsc: float = 0.85) -> None:
    """The five frame checks: how each value is computed, its limit, and where the GOOD masks of the
    hand-masked frames (Dice >= good_dsc with the ground truth) fall on it -- the evidence for the limit."""
    g = test[test.dsc >= good_dsc]
    lo, hi = G.area_ratio_range
    rows = [   # check, the report's equation for it, limit, where the good masks fall
        ("Distance from the trap", r"\eqref{eq:dist}", rf"$\leq$ {G.max_trap_dist:.0f}\,px",
         rf"99th pct {g.trap_dist.quantile(0.99):.1f}\,px"),
        ("Solidity", r"\eqref{eq:solidity}", rf"$\geq$ {G.min_solidity}", rf"1st pct {g.solidity.quantile(0.01):.3f}"),
        ("Size vs nearby steps", r"\eqref{eq:size}", rf"{lo}--{hi}$\times$",
         rf"1st--99th pct {g.area_ratio.quantile(0.01):.2f}--{g.area_ratio.quantile(0.99):.2f}"),
        ("Outline confidence", r"\eqref{eq:conf}", rf"$\geq$ {G.min_boundary_conf}",
         rf"1st pct {g.boundary_conf.quantile(0.01):.3f}"),
        ("Frame edge", "--", "no", f"{int(g.touches_border.sum())} of {len(g)} touch"),
    ]
    lines = [r"% GENERATED by scripts/report/plot_region_selection.py -- do not edit by hand.",
             r"\begin{tabular}{l c l l}", r"\toprule",
             rf"\textbf{{Check}} & \textbf{{Eq.}} & \textbf{{Limit}} & \textbf{{Good masks ({len(g)})}} \\",
             r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"-> {out}")


def write_cross_check(calib: dict, out: Path) -> None:
    """The frame checks on frames that did not set their limits (leave-one-cell-out), from calibration.json."""
    cc = calib["cross_check"]
    rows = [("Limits used (rounded outward, set on all frames)", cc["used"]),
            ("Rule, set on all frames", cc["rule_on_all_frames"]),
            (r"Rule, set \emph{without} the cell it is applied to", cc["rule_without_the_cell"])]
    lines = [r"% GENERATED by scripts/report/plot_region_selection.py -- do not edit by hand.",
             r"\begin{tabular}{l c c}", r"\toprule",
             r"\textbf{Limits} & \textbf{Good masks dropped} & \textbf{Bad masks caught} \\", r"\midrule"]
    for name, v in rows:
        lines.append(f"{name} & {v['good_dropped']} of {v['good']} ({100 * v['good_dropped'] / v['good']:.1f}\\,\\%) & "
                     f"{v['bad_caught']} of {v['bad']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"-> {out}")


def write_rejections(df: pd.DataFrame, out: Path) -> None:
    """Every rejected feature frame, grouped by the first check it failed, by cell: how many, and in which laser
    steps (all the checks a frame failed are in features_frames.csv)."""
    kinds = {**{k: v for k, v in KINDS.items()}, "edge": (r"cut by the frame edge", "cut by the frame edge"),
             "none": (r"no cell near the trap", "no cell found")}
    bad = df[~df.passed].copy()
    bad["kind"] = [next((k for k, (pat, _) in kinds.items() if re.search(pat, r.split(";")[0])), "none")
                   for r in bad.reasons]

    def steps(st: pd.Series) -> str:
        u = sorted(set(st))
        return ", ".join(map(str, u)) if len(u) <= 4 else f"{u[0]}--{u[-1]}"
    lines = [r"% GENERATED by scripts/report/plot_region_selection.py -- do not edit by hand.",
             r"\begin{tabular}{l r p{0.58\linewidth}}", r"\toprule",
             r"\textbf{First check failed} & \textbf{Frames} & \textbf{Cell: frames (laser steps)} \\", r"\midrule"]
    for kind, (_, label) in kinds.items():
        g = bad[bad.kind == kind]
        if g.empty:
            continue
        where = "; ".join(f"\\code{{{short(c)}}}: {len(gg)} ({steps(gg.laser_step)})"
                          for c, gg in sorted(g.groupby("cell"), key=lambda cg: -len(cg[1])))
        lines.append(f"{label[0].upper() + label[1:]} & {len(g)} & {where}" + r" \\")
    lines += [r"\midrule", f"Total & {len(bad)} of {len(df):,} & ".replace(",", r"\,") + r"\\", r"\bottomrule", r"\end{tabular}", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="nnunet1000_loocv")
    ap.add_argument("--output", type=Path, required=True, help="figure (a): frames with more than one region")
    ap.add_argument("--dropped-output", type=Path, required=True, help="figure (b): dropped frames")
    ap.add_argument("--table-output", type=Path, help="table: every rejected frame, by the check it failed")
    ap.add_argument("--checks-output", type=Path, help="table: how each frame check is computed, its limit, good masks")
    ap.add_argument("--cross-output", type=Path, help="table: the checks on frames that did not set their limits")
    ap.add_argument("--n", type=int, default=4, help="tiles per row")
    a = ap.parse_args()
    res = REPO / "results/features" / a.model
    df = pd.read_csv(res / "features_frames.csv")
    df["reasons"] = df.reasons.fillna("")
    cells = pd.read_csv(res / "features_cells.csv", index_col="cell")
    labels = {}                 # stem -> predicted mask; a frame in both folders is taken from the step frames
    for sub in ("feature", "dense"):
        for q in (REPO / "data/features" / a.model / sub / "labels").glob("*.png"):
            labels.setdefault(q.stem, q)
    if a.table_output:
        write_rejections(df, a.table_output)
    if a.checks_output:
        write_checks(pd.read_csv(res / "checks_test.csv"), a.checks_output)
    if a.cross_output:
        write_cross_check(json.loads((res / "calibration.json").read_text()), a.cross_output)
    multi = pick_multi(df, labels, a.n)
    dropped = pick_dropped(df, labels, set(cells.index[~cells.usable]), a.n)

    rowlab = ["frame", "ground\ntruth", "predicted\nmask", "selection"]

    def figure(examples: list, title: str, out: Path) -> None:
        """One page-wide figure: per example (column) the frame, ground truth, predicted mask and selection."""
        n = len(examples)
        fig = plt.figure(figsize=(7.4, 8.9), facecolor="white")
        gs = fig.add_gridspec(5, n, height_ratios=[1, 1, 1, 1, 0.5], hspace=0.05, wspace=0.05,
                              top=0.875, bottom=0.0, left=0.075, right=0.995)
        for c, (r, v, lines, bad) in enumerate(examples):
            ax = fig.add_subplot(gs[0, c])
            plain(ax, v["raw"], short(r.cell), gray=True)
            ax.text(0.04, 0.95, f"step {r.laser_step}", transform=ax.transAxes, fontsize=7.4, color=INK, va="top",
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            gx = fig.add_subplot(gs[1, c])
            if v["gt_step"] == r.laser_step:     # a hand mask of the same laser step: comparable position
                plain(gx, v["gt"])
                gx.text(0.04, 0.95, f"step {v['gt_step']}", transform=gx.transAxes, fontsize=7.4, color=INK,
                        va="top", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            else:                                # a mask from another step would show the cell elsewhere
                plain(gx, np.zeros_like(v["gt"]))
                gx.text(0.5, 0.5, f"no hand mask\nin step {r.laser_step}\n\n(nearest: step {v['gt_step']},\n"
                        f"a different moment)", transform=gx.transAxes, ha="center", va="center", fontsize=7.4,
                        color="#bdbdbd", linespacing=1.15)
            plain(fig.add_subplot(gs[2, c]), v["model"])
            draw(fig.add_subplot(gs[3, c]), v, None, lines, bad)
            if c == 0:
                for k, axx in enumerate(fig.axes[-4:]):
                    axx.set_ylabel(rowlab[k], fontsize=8, color=HIGH, fontweight="bold", linespacing=0.95)
                    axx.yaxis.set_label_coords(-0.06, 0.5)       # one column for every row label
        fig.text(0.075, 0.995, title, fontsize=9, color=INK, va="top")
        cols = [ex[0] for ex in examples]
        k = sum(gt_same_step(r) for r in cols)           # aligned examples come first
        for lo, hi, text in ((0, k, "ground truth in the same step"), (k, len(cols), "no ground truth in that step")):
            if hi > lo:
                x0 = fig.axes[4 * lo].get_position().x0
                x1 = fig.axes[4 * (hi - 1)].get_position().x1
                y = fig.axes[0].get_position().y1 + 0.03
                fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color=INK2, lw=1))
                fig.text((x0 + x1) / 2, y + 0.006, text, ha="center", va="bottom", fontsize=8, color=INK2,
                         fontstyle="italic")
        fig.legend(handles=[Patch(fc=PRED, ec="#555", lw=0.6, label="predicted: trapped cell"),
                            Patch(color=PRED_DECOY, label="predicted: other object"),
                            Patch(color=KEPT, label="ground truth: trapped cell / selected region"),
                            Patch(color=DECOY, label="ground truth: decoy"),
                            Patch(fc=OTHER, ec="#555", lw=0.6, label="also marked trapped, not selected"),
                            Line2D([], [], color=TRAP, ls=(0, (3, 2)), marker="+", ms=7, label="trap, 45 px limit")],
                   loc="upper left", bbox_to_anchor=(0.068, 0.975), ncol=3, fontsize=7, frameon=False,
                   handlelength=1.2, columnspacing=0.9, handletextpad=0.4)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"-> {out}")

    passed = []
    for r, v in multi:
        lines = [(f"{'selected' if inf['kept'] else 'other'}: {inf['area']} px, {inf['dist']:.0f} px off",
                  "#1b7a3a" if inf["kept"] else INK2) for inf in v["info"]][:3]
        passed.append((r, v, lines + [("→ passed every check", "#1b7a3a")], False))
    figure(passed, "More than one region marked as the trapped cell: the one nearest the trap is selected",
           a.output)
    figure([(r, v, [(KINDS[kind][1], INK)] + reason_lines(r, v) + [("→ dropped", BAD)], True)
            for kind, r, v in dropped], "Frames dropped by the frame checks, one per kind of failure", a.dropped_output)


if __name__ == "__main__":
    main()
