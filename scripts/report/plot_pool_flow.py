#!/usr/bin/env python3
"""
Provenance diagram of the frame pools: which cells go where, what draws from
what, and where the evaluation firewall sits.

This replaces the pool-summary *table* in the progress reports. A table can
carry the counts but not the relationships — that the annotation pool and the
mask factory are sampled *in parallel* from the same raw videos, on different
frame grids, sharing only the frames the two grids happen to have in common;
that legacy frames enter from a different source entirely; and that the holdout
pools sit behind a hard boundary.

Corrected 2026-09-30: earlier versions drew an arrow from the mask factory into
the annotation pool ("draws N frames at different indices"). That was wrong. The
domain frames were extracted from the raw videos on their own grid (every ~500th
frame, vs. the factory's every 200th); the overlap is counted here byte-for-byte. Those needed a paragraph of prose under the table; here they
are the picture.

Every count is read off disk, never hardcoded, so the figure cannot drift from
the data the way a hand-maintained table does. Run it again after any pool
change and the numbers follow.

Usage
-----
python scripts/report/plot_pool_flow.py --output results/pool_flow.png
python scripts/report/plot_pool_flow.py --data-root data --output results/pool_flow.png
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Report palette (matches the LaTeX preamble's \definecolor values).
NSU_GREEN = "#226644"
NSU_GOLD = "#D6B02C"
WARN_RED = "#B02A2A"
INK = "#2B2B2B"
MUTED = "#6B7075"
BOX_FILL = "#F4F5F6"
BOX_EDGE = "#C9CDD2"
ARROW = "#9AA0A6"

_EXTS = (".jpg", ".jpeg", ".png")


def _frames(d: Path) -> list[Path]:
    """Image files directly in `d` — non-recursive, matching every pipeline script."""
    if not d.exists():
        return []
    return [p for p in d.iterdir() if p.suffix.lower() in _EXTS and p.is_file()]


def _cells(paths: list[Path], strip_prefix: str = "") -> set[str]:
    out = set()
    for p in paths:
        stem = p.stem[len(strip_prefix):] if strip_prefix and p.stem.startswith(strip_prefix) else p.stem
        parts = stem.split("_")
        if len(parts) >= 2 and parts[1].startswith("cell"):
            out.add(f"{parts[0]}_{parts[1]}")
    return out


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _box(ax, x, y, title, lines, accent, w=None, reserved=False):
    """A rounded panel with a colored identity bar down its left edge.

    Width is derived from the longest line so text can never overflow the frame
    — the failure the first draft of this diagram had. The bar carries identity
    so body text stays in ink tokens; colored body text reads as emphasis,
    which is not what a category marker means.
    """
    pad, lh = 0.22, 0.30
    longest = max([len(title) * 0.098] + [len(l) * 0.077 for l in lines])
    w = w or longest + 2 * pad
    h = 0.42 + lh * len(lines) + 0.24

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                linewidth=1.0, edgecolor=BOX_EDGE, facecolor=BOX_FILL, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y), 0.06, h, boxstyle="round,pad=0,rounding_size=0.02",
                                linewidth=0, facecolor=accent, zorder=4))
    ax.text(x + pad, y + h - 0.30, title, fontsize=12, fontweight="bold",
            color=INK, va="center", ha="left", zorder=5)
    if reserved:
        ax.text(x + w - pad, y + h - 0.30, "reserved", fontsize=9.5, color=WARN_RED,
                fontweight="bold", va="center", ha="right", zorder=5)
    for i, ln in enumerate(lines):
        ax.text(x + pad, y + h - 0.66 - lh * i, ln, fontsize=10.5, color=MUTED,
                va="center", ha="left", zorder=5)
    return x + w, y + h


def _arrow(ax, xy_from, xy_to, label=None, label_xy=None, head=True):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>" if head else "-",
                                 mutation_scale=13, linewidth=1.4, color=ARROW,
                                 shrinkA=0, shrinkB=0, zorder=2))
    if label:
        ax.text(*(label_xy or ((xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2)),
                label, fontsize=10, color=MUTED, ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="none"))


def main() -> None:
    p = argparse.ArgumentParser(description="Frame-pool provenance diagram")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("results/pool_flow.png"))
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    fr = args.data_root / "frames"
    factory = _frames(fr / "03_mask_factory")
    pool = _frames(fr / "01_annotate_pool")
    holdout = _frames(fr / "02_unet_holdout")
    blind = _frames(fr / "02_unet_holdout" / "final_test")
    masks = args.data_root / "masks" / "01_annotate_pool_multiclass"
    hold_masks = args.data_root / "masks" / "02_unet_holdout_multiclass"
    if not factory or not pool:
        sys.exit(f"No frames found under {fr} — run from the repo root.")

    domain = [q for q in pool if q.stem.startswith("domain_")]
    # Frames present in BOTH pools, byte-identical -- the only coupling between them.
    fact_md5 = {_md5(q) for q in factory}
    shared = [q for q in domain if _md5(q) in fact_md5]
    n_dom_cells = len(_cells(domain, strip_prefix="domain_"))
    # Classifier features come only from factory frames the U-Net never trained on
    # (2026-09-30). Frames off the step-200 grid are top-ups extracted to replace
    # shared ones -- high-10's 50 at indices 100, 300, ..., 9900.
    n_features = len(factory) - len(shared)
    off_grid = [q for q in factory if int(q.stem.rsplit("_", 1)[1]) % 200]
    off_cells = sorted({c.replace("_cell", "-") for c in _cells(off_grid)})
    legacy = [q for q in pool if not q.stem.startswith("domain_")]
    ann = sum(1 for q in pool if (masks / f"{q.stem}.png").exists())
    hold_ann = sum(1 for q in holdout if (hold_masks / f"{q.stem}.png").exists())
    blind_ann = sum(1 for q in blind if (hold_masks / f"{q.stem}.png").exists())
    n_fact, n_hold, n_blind = len(_cells(factory)), len(_cells(holdout)), len(_cells(blind))
    total_cells = n_fact + n_hold + n_blind

    fig, ax = plt.subplots(figsize=(14.6, 5.9))
    ax.set_xlim(0, 14.6); ax.set_ylim(-0.05, 5.75); ax.axis("off")

    # Display names are ROLES, not folder paths (2026-09-30, user decision): the
    # folders on disk stay 03_mask_factory / 01_annotate_pool / 02_unet_holdout.

    # ── sources (left column) ───────────────────────────────────────────────
    src_r, _ = _box(ax, 0.95, 4.35, f"{total_cells} cells", ["one raw video each"], MUTED)
    leg_r, _ = _box(ax, 1.40, 2.55, "2021 legacy archive",
                    ["older microscope, no HER2 label"], MUTED)

    # ── frame pools (middle column) ─────────────────────────────────────────
    fac_r, fac_t = _box(ax, 5.55, 4.15, "Feature factory",
                        [f"{n_fact} cells \u00b7 {len(factory)} frames",
                         "none trained on \u00b7 HER2 label known per cell"]
                        + ([f"{len(shared)} shared frames skipped"] if shared else []),
                        NSU_GREEN)
    ann_r, ann_t = _box(ax, 5.55, 2.20, "Annotation pool",
                        [f"{len(pool)} frames \u00b7 {ann}/{len(pool)} annotated",
                         f"{len(domain)} domain-adapt + {len(legacy)} legacy",
                         "hand-drawn masks"], NSU_GREEN)

    grid_label = f"all {n_fact} cells \u00b7 every 200th frame"
    if off_grid:
        shift = min(int(q.stem.rsplit("_", 1)[1]) % 200 for q in off_grid)
        grid_label += f"\n({', '.join(off_cells)}: shifted {shift} frames off it)"
    _arrow(ax, (src_r, 4.78), (5.55, 4.78), grid_label, (4.12, 5.16 if off_grid else 5.00))
    # Second, parallel draw from the same videos into the training pool: an elbow
    # down the gutter between the columns.
    gx = 5.05
    _arrow(ax, (src_r, 4.55), (gx, 4.55), head=False)
    _arrow(ax, (gx, 4.55), (gx, 3.40), head=False)
    _arrow(ax, (gx, 3.40), (5.55, 3.40))
    ax.text(gx - 0.12, 3.98, f"{len(domain)} frames \u00b7 {n_dom_cells} cells\nown grid (~every 500th)",
            fontsize=10, color=MUTED, ha="right", va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="none"))
    # The overlap: a plain connector, not an arrow -- neither pool feeds the other. Drawn
    # only while one exists; the 57 shared frames were removed from the factory 2026-09-30.
    if shared:
        ax.plot([6.55, 6.55], [4.15, ann_t], color=ARROW, linewidth=1.2,
                linestyle=(0, (3, 3)), zorder=2)
        ax.text(6.70, (4.15 + ann_t) / 2, f"{len(shared)} frames in both pools (identical)",
                fontsize=10, color=MUTED, ha="left", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="none"))
    _arrow(ax, (leg_r, 2.92), (5.55, 2.92), f"{len(legacy)} frames", ((leg_r + 5.55) / 2, 3.14))

    # ── the two learning stages (right column) ──────────────────────────────
    sx = 10.35
    cls_r, cls_t = _box(ax, sx, 4.15, "Classification",
                        ["SVM \u00b7 RF \u00b7 DT \u2192 HER2 high / low",
                         "learns from morphology features"], INK, w=3.05)
    seg_r, seg_t = _box(ax, sx, 2.35, "Segmentation",
                        ["U-Net, ensemble of four seeds",
                         "learns from the hand-drawn masks"], INK, w=3.05)
    cls_mid, seg_mid = (4.15 + cls_t) / 2, (2.35 + seg_t) / 2
    _arrow(ax, (fac_r, cls_mid), (sx, cls_mid), "features", ((fac_r + sx) / 2, cls_mid + 0.20))
    _arrow(ax, (ann_r, seg_mid), (sx, seg_mid), "trains", ((ann_r + sx) / 2, seg_mid + 0.20))
    # The trained U-Net is what turns factory frames into masks, and so into features.
    lx = sx + 0.55
    ax.add_patch(FancyArrowPatch((lx, seg_t), (lx, 3.95), arrowstyle="-", linewidth=1.3,
                                 color=ARROW, linestyle=(0, (3, 2.5)), zorder=2))
    ax.add_patch(FancyArrowPatch((lx, 3.95), (8.55, 3.95), arrowstyle="-", linewidth=1.3,
                                 color=ARROW, linestyle=(0, (3, 2.5)), zorder=2))
    ax.add_patch(FancyArrowPatch((8.55, 3.95), (8.55, 4.15), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.3, color=ARROW, shrinkA=0, shrinkB=0, zorder=2))
    ax.text((lx + 8.55) / 2, 3.86, "trained U-Net segments these frames",
            fontsize=9.5, color=MUTED, style="italic", ha="center", va="top", zorder=5)

    # ── the firewall ────────────────────────────────────────────────────────
    ax.plot([0.15, 14.45], [1.88, 1.88], color=WARN_RED, linewidth=1.3,
            linestyle=(0, (7, 4)), zorder=1)
    ax.text(1.45, 1.96, "evaluation firewall \u2014 no training run has ever seen a cell below this line",
            fontsize=10.5, color=WARN_RED, style="italic", va="bottom", ha="left", zorder=5)

    # ── the testing set: one pool, scoring both stages ──────────────────────
    # The separate blind test was retired 2026-09-10; the four held-out cells are one
    # set. It is called the testing set in the reports (2026-09-30, user decision), but
    # it also guided the segmentation recipe, and the box says so rather than hide it.
    if blind:
        print(f"   WARNING: {len(blind)} frames in final_test/ are not drawn -- the figure "
              "assumes the blind test is retired")
    pend = len(holdout) - hold_ann
    tx, tw = 6.20, 8.05
    _, test_t = _box(ax, tx, -0.02, f"Testing set \u2014 {n_hold} cells no model has trained on",
                     [f"{n_hold} cells \u00b7 {len(holdout)} frames \u00b7 {hold_ann}/{len(holdout)} hand-annotated"
                      + (f" \u00b7 {pend} pending" if pend else "") + " \u00b7 high and low HER2",
                      "hand-drawn masks score the segmentation; known HER2 labels score the classifier",
                      "caveat: also used to choose the segmentation recipe"], NSU_GOLD, w=tw)
    # allocation trunk: source down, along the gutter above the box, into it
    _arrow(ax, (1.28, 4.35), (1.28, 1.70), head=False)
    _arrow(ax, (1.28, 1.70), (7.0, 1.70), head=False)
    _arrow(ax, (7.0, 1.70), (7.0, test_t), f"{n_hold} cells", (6.35, 1.70))
    # scoring arrows, in the testing set's gold: up across the firewall into each stage
    ux = sx + 1.20                                         # riser into segmentation
    ax.add_patch(FancyArrowPatch((ux, test_t), (ux, 2.35), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.6, color=NSU_GOLD, shrinkA=0, shrinkB=0, zorder=2))
    ax.text(ux, 2.10, "scores masks", fontsize=9.5, color=INK, ha="center", va="center", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"))
    rx = tx + tw - 0.25                                    # right-hand riser to classification
    ax.add_patch(FancyArrowPatch((rx, test_t), (rx, cls_mid), arrowstyle="-", linewidth=1.6,
                                 color=NSU_GOLD, shrinkA=0, shrinkB=0, zorder=2))
    ax.add_patch(FancyArrowPatch((rx, cls_mid), (cls_r, cls_mid), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.6, color=NSU_GOLD, shrinkA=0, shrinkB=0, zorder=2))
    ax.text(rx, (seg_t + 4.15) / 2, "scores\nHER2 calls", fontsize=9.5, color=INK, ha="center",
            va="center", zorder=5, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"))

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"-> {args.output}")
    print(f"   {total_cells} cells = {n_fact} factory + {n_hold} testing + {n_blind} blind")
    print(f"   01_annotate_pool {len(pool)} ({len(domain)} domain + {len(legacy)} legacy), {ann} annotated")
    print(f"   03_mask_factory  {len(factory)} frames, {len(shared)} also in 01_annotate_pool, "
          f"{n_features} feed features, {len(off_grid)} off the step-200 grid")
    print(f"   02_unet_holdout  {len(holdout)} frames, {hold_ann} annotated")
    print(f"   final_test       {len(blind)} frames, {blind_ann} annotated")


if __name__ == "__main__":
    main()
