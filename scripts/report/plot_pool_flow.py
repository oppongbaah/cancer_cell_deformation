#!/usr/bin/env python3
"""
Provenance diagram of the frame pools: which cells go where, what draws from
what, and where the evaluation firewall sits.

This replaces the pool-summary *table* in the progress reports. A table can
carry the counts but not the relationships — that the annotation pool is
*drawn from* the mask factory rather than parallel to it, that legacy frames
enter from a different source entirely, and that the holdout pools sit behind
a hard boundary. Those needed a paragraph of prose under the table; here they
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
    legacy = [q for q in pool if not q.stem.startswith("domain_")]
    ann = sum(1 for q in pool if (masks / f"{q.stem}.png").exists())
    hold_ann = sum(1 for q in holdout if (hold_masks / f"{q.stem}.png").exists())
    blind_ann = sum(1 for q in blind if (hold_masks / f"{q.stem}.png").exists())
    n_fact, n_hold, n_blind = len(_cells(factory)), len(_cells(holdout)), len(_cells(blind))
    total_cells = n_fact + n_hold + n_blind

    fig, ax = plt.subplots(figsize=(11.5, 5.35))
    ax.set_xlim(0, 11.5); ax.set_ylim(-0.05, 5.55); ax.axis("off")

    # ── sources (left column) ───────────────────────────────────────────────
    src_r, _ = _box(ax, 0.95, 4.35, f"{total_cells} cells", ["one raw video each"], MUTED)
    leg_r, _ = _box(ax, 1.40, 2.55, "2021 legacy archive",
                    ["older microscope, no HER2 label"], MUTED)

    # ── training / feature side (right column) ──────────────────────────────
    fac_r, fac_t = _box(ax, 5.55, 4.15, "03_mask_factory",
                        [f"{n_fact} cells \u00b7 {len(factory)} frames",
                         "frozen U-Net \u2192 classifier features"], NSU_GREEN)
    ann_r, ann_t = _box(ax, 5.55, 2.20, "01_annotate_pool",
                        [f"{len(pool)} frames \u00b7 {ann}/{len(pool)} annotated",
                         f"{len(domain)} domain-adapt + {len(legacy)} legacy",
                         "trains the U-Net"], NSU_GREEN)

    _arrow(ax, (src_r, 4.78), (5.55, 4.78), f"all {n_fact} cells", (4.35, 5.00))
    _arrow(ax, (6.55, 4.15), (6.55, ann_t), f"draws {len(domain)} frames\nat different indices",
           (8.05, 3.92))
    _arrow(ax, (leg_r, 2.92), (5.55, 2.92), f"{len(legacy)} frames", ((leg_r + 5.55) / 2, 3.14))

    # ── allocation trunk: source down to the evaluation roles ───────────────
    _arrow(ax, (1.28, 4.35), (1.28, 1.30), head=False)
    if blind:
        _arrow(ax, (1.28, 1.30), (9.30, 1.30), head=False)
        _arrow(ax, (3.15, 1.30), (3.15, 1.02), f"{n_hold} cells", (3.15, 1.52))
        _arrow(ax, (9.30, 1.30), (9.30, 1.02), f"{n_blind} cell", (9.30, 1.52))
    else:
        _arrow(ax, (1.28, 1.30), (5.20, 1.30), head=False)
        _arrow(ax, (5.20, 1.30), (5.20, 1.02), f"{n_hold} cells", (5.20, 1.54))

    # ── the firewall ────────────────────────────────────────────────────────
    ax.plot([0.15, 11.35], [1.88, 1.88], color=WARN_RED, linewidth=1.3,
            linestyle=(0, (7, 4)), zorder=1)
    ax.text(11.35, 1.96, "evaluation firewall \u2014 no training run has ever seen a cell below this line",
            fontsize=10.5, color=WARN_RED, style="italic", va="bottom", ha="right", zorder=5)

    # ── evaluation side ─────────────────────────────────────────────────────
    # The blind test was retired at the 2026-09-10 supervisor meeting, so all
    # evaluation cells now sit in one development set. Keep drawing the separate
    # blind-test box only while a final_test/ pool actually exists — otherwise the
    # diagram would assert a reserved cell the project no longer holds.
    if blind:
        _box(ax, 0.95, -0.02, "02_unet_holdout \u2014 development set",
             [f"{n_hold} cells \u00b7 {len(holdout)} frames \u00b7 {hold_ann}/{len(holdout)} annotated",
              "evaluated repeatedly to guide tuning"], NSU_GOLD)
        _box(ax, 6.30, -0.02, ".../final_test \u2014 blind test",
             [f"{n_blind} cell \u00b7 {len(blind)} frames \u00b7 "
              + ("deliberately unannotated" if blind_ann == 0 else f"{blind_ann} annotated (!)"),
              "evaluated once, at the very end"], WARN_RED, reserved=True)
    else:
        pend = len(holdout) - hold_ann
        _box(ax, 2.60, -0.02, "02_unet_holdout \u2014 development set",
             [f"{n_hold} cells \u00b7 {len(holdout)} frames \u00b7 {hold_ann}/{len(holdout)} annotated"
              + (f" \u00b7 {pend} pending" if pend else ""),
              "all four cells guide tuning \u2014 no blind test is held back"], NSU_GOLD, w=6.3)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"-> {args.output}")
    print(f"   {total_cells} cells = {n_fact} factory + {n_hold} development + {n_blind} blind")
    print(f"   01_annotate_pool {len(pool)} ({len(domain)} domain + {len(legacy)} legacy), {ann} annotated")
    print(f"   03_mask_factory  {len(factory)} frames")
    print(f"   02_unet_holdout  {len(holdout)} frames, {hold_ann} annotated")
    print(f"   final_test       {len(blind)} frames, {blind_ann} annotated")


if __name__ == "__main__":
    main()
