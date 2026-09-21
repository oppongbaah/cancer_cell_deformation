#!/usr/bin/env python3
"""
Horizontal bar chart ranking every checkpoint in results/preswap/preswap_eval_log.jsonl
by its pooled trapped-cell DSC on the development set.

Companion to scripts/report/plot_percell_metrics.py: that one compares a handful of
runs in detail, this one places all of them on a single scale so a new result
can be read against the whole history at once.

Numbers come out of the eval log, never typed in, so the ranking always
reflects the runs that actually happened. Checkpoints with no pooled entry
(evaluated per-cell only) are skipped rather than guessed at.

Usage
-----
python scripts/report/plot_sweep_dsc.py --output results/sweep_dsc.png \\
    --highlight multiclass_experiment_256_bf64_scale_cos \\
    --warn multiclass_experiment_256_bf64_reg \\
    --label multiclass_experiment_256_bf64_scale_cos="Scale augmentation"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NSU_GREEN = "#226644"
WARN_RED = "#B02A2A"
NEUTRAL = "#B9BDC2"
GRID = "#D9D9D9"
INK = "#2B2B2B"


def _pair(spec: str) -> tuple[str, str]:
    key, _, label = spec.partition("=")
    return key.strip(), (label.strip() or key.strip())


def main() -> None:
    p = argparse.ArgumentParser(description="Rank all checkpoints by pooled holdout DSC")
    p.add_argument("--log", type=Path, default=Path("results/preswap/preswap_eval_log.jsonl"))
    p.add_argument("--class-id", default="1")
    p.add_argument("--metric", default="dsc")
    p.add_argument("--highlight", action="append", default=[],
                   help="Checkpoint dir name to draw in the accent color, repeatable")
    p.add_argument("--warn", action="append", default=[],
                   help="Checkpoint dir name to draw in the warning color, repeatable")
    p.add_argument("--label", action="append", default=[], metavar="KEY=LABEL",
                   help="Override the display name for one checkpoint, repeatable")
    p.add_argument("--exclude", action="append", default=[],
                   help="Checkpoint dir name to leave out entirely, repeatable")
    p.add_argument("--band", metavar="N=LABEL",
                   help="Shade the top N ranked bars as one group and annotate it. Use when the "
                        "ranking spans two training setups that are not comparable to each other, "
                        "so a reader can see which bars belong to which.")
    p.add_argument("--band-rest-label", default=None,
                   help="Annotation for the bars below the band.")
    p.add_argument("--row-height", type=float, default=0.34,
                   help="Vertical inches per bar. Lower it when the chart has to share a "
                        "printed page with body text rather than stranding one.")
    p.add_argument("--xlabel", default="Pooled trapped-cell DSC (101-frame development set)")
    p.add_argument("--output", type=Path, default=Path("results/sweep_dsc.png"))
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    if not args.log.exists():
        sys.exit(f"Eval log not found: {args.log}")
    labels = dict(_pair(x) for x in args.label)

    # Newest pooled entry per checkpoint directory.
    best: dict[str, dict] = {}
    for line in args.log.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("filter") is not None or args.class_id not in r.get("metrics", {}):
            continue
        name = Path(r["checkpoint"]).parent.name
        if name in args.exclude:
            continue
        if name not in best or r.get("timestamp", "") > best[name].get("timestamp", ""):
            best[name] = r

    if not best:
        sys.exit("No pooled evaluations found in the log.")

    rows = sorted(((n, r["metrics"][args.class_id][args.metric]) for n, r in best.items()),
                  key=lambda t: t[1])
    names = [labels.get(n, n.replace("multiclass_experiment_256", "").lstrip("_") or "original")
             for n, _ in rows]
    # A label appearing twice reads as a duplicate bar rather than a controlled
    # comparison, which defeats the point of showing both. Refuse it outright so the
    # caller has to disambiguate rather than shipping an ambiguous chart.
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        sys.exit("Duplicate bar labels — disambiguate them with --label: "
                 + "; ".join(sorted(dupes)))

    vals = [v for _, v in rows]
    colors = [NSU_GREEN if n in args.highlight else WARN_RED if n in args.warn else NEUTRAL
              for n, _ in rows]

    fig, ax = plt.subplots(figsize=(9.0, args.row_height * len(rows) + 1.1))
    ax.barh(range(len(rows)), vals, color=colors, height=0.72, zorder=3)
    for i, v in enumerate(vals):
        emphasised = colors[i] != NEUTRAL
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", ha="left", fontsize=11.5,
                color=colors[i] if emphasised else INK,
                fontweight="bold" if emphasised else "normal", zorder=4)

    if args.band:
        n_band, _, band_label = args.band.partition("=")
        n_band = int(n_band)
        top = len(rows) - 0.5                      # rows are plotted bottom-up
        split = len(rows) - n_band - 0.5
        ax.axhspan(split, top, color=NSU_GREEN, alpha=0.06, zorder=1)
        ax.axhline(split, color=GRID, linewidth=0.9, zorder=2)
        xr = max(vals) * 1.36          # must match the set_xlim below
        ax.text(xr * 0.98, (split + top) / 2, band_label, rotation=270, va="center",
                ha="center", fontsize=10, color=NSU_GREEN, linespacing=1.2)
        if args.band_rest_label:
            ax.text(xr * 0.98, split / 2, args.band_rest_label, rotation=270, va="center",
                    ha="center", fontsize=10, color="#7A7F85", linespacing=1.2)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=11.5, color=INK)
    ax.set_xlim(0, max(vals) * (1.36 if args.band else 1.16))
    ax.set_xlabel(args.xlabel, fontsize=12, color=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=INK, labelsize=11)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"-> {args.output}  ({len(rows)} checkpoints)")
    for (n, v) in reversed(rows):
        print(f"   {v:.3f}  {n}")


if __name__ == "__main__":
    main()
