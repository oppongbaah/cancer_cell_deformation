#!/usr/bin/env python3
"""
Grouped bar chart of per-cell holdout metrics, read straight out of
results/preswap/preswap_eval_log.jsonl — one panel per evaluation scope (each
development-set cell, plus the pooled figure), grouped bars per metric,
one bar per checkpoint.

This exists because the per-cell comparison is the single most-repeated
figure in the progress reports, and pooling the two cells into one number
hides the finding that matters: they fail in opposite directions, and a
change that helps one can silently degrade the other. Any bar that comes
in below its baseline is labelled in red for exactly that reason.

Numbers are never typed in — they are looked up in the eval log by
(checkpoint, filter), so a figure in a report always matches the run that
produced it. The first --checkpoint given is treated as the baseline that
later ones are compared against.

Usage
-----
python scripts/report/plot_percell_metrics.py \\
    --checkpoint multiclass_experiment_256=Original \\
    --checkpoint multiclass_experiment_256_bf64_scale="Scale only" \\
    --checkpoint multiclass_experiment_256_bf64_allaug=Combined \\
    --scope high_cell9="high-9 (over-segments)" \\
    --scope low_cell6="low-6 (under-detects)" \\
    --scope pooled=Pooled \\
    --output results/percell_metrics.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Matches the progress-report LaTeX preamble's \definecolor values, so a figure
# dropped into a report doesn't clash with the document around it.
NSU_GREEN = "#226644"
NSU_GOLD = "#D6B02C"
WARN_RED = "#B02A2A"
NEUTRAL = "#9AA0A6"
GRID = "#D9D9D9"
INK = "#2B2B2B"

# Bar colors in --checkpoint order: baseline first, then the variants.
SERIES_COLORS = [NEUTRAL, NSU_GOLD, NSU_GREEN, "#4A7FB5", "#8E6FB0"]


def _pair(spec: str) -> tuple[str, str]:
    """`key=Label` -> (key, Label); a bare `key` labels itself."""
    key, _, label = spec.partition("=")
    return key.strip(), (label.strip() or key.strip())


def _load(log_path: Path) -> list[dict]:
    if not log_path.exists():
        sys.exit(f"Eval log not found: {log_path}")
    rows = []
    for n, line in enumerate(log_path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  warning: skipping unparseable line {n}", file=sys.stderr)
    return rows


def _lookup(rows: list[dict], ckpt_key: str, scope_key: str, cls: str) -> dict:
    """Newest entry for one checkpoint directory and one evaluation scope.

    Matched on the checkpoint's *directory name*, exactly — never a substring.
    Substring matching silently resolves `multiclass_experiment_256` to
    `multiclass_experiment_256_bf64_allaug`, which would quietly plot one
    checkpoint's numbers under another's label.
    """
    want_filter = None if scope_key.lower() == "pooled" else scope_key
    hits = [
        r for r in rows
        if Path(r.get("checkpoint", "")).parent.name == ckpt_key
        and r.get("filter") == want_filter
        and cls in r.get("metrics", {})
    ]
    if not hits:
        sys.exit(f"No eval-log entry for checkpoint '{ckpt_key}' with filter "
                 f"{want_filter!r}. Run scripts/training/evaluate_unet.py for it first.")
    return max(hits, key=lambda r: r.get("timestamp", ""))["metrics"][cls]


def main() -> None:
    p = argparse.ArgumentParser(description="Per-cell holdout metric bar chart")
    p.add_argument("--log", type=Path, default=Path("results/preswap/preswap_eval_log.jsonl"))
    p.add_argument("--checkpoint", action="append", required=True, metavar="KEY[=LABEL]",
                   help="Checkpoint name substring, repeatable. The first is the baseline.")
    p.add_argument("--scope", action="append", required=True, metavar="KEY[=LABEL]",
                   help="`--filter` value from the eval log, or `pooled`. One panel each.")
    p.add_argument("--metric", action="append", metavar="KEY[=LABEL]",
                   help="Metric key(s) from the log (default: dsc, precision)")
    p.add_argument("--class-id", default="1",
                   help="Class key in the log's metrics dict (default: 1 = trapped cell)")
    p.add_argument("--highlight", default=None, metavar="KEY",
                   help="Checkpoint key to draw in the accent color; every other series goes "
                        "muted. Without it, colors follow --checkpoint order. Use this so the "
                        "accent lands on the result being argued for, not on whichever run "
                        "happens to be listed last.")
    p.add_argument("--title", default=None)
    p.add_argument("--output", type=Path, default=Path("results/percell_metrics.png"))
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    checkpoints = [_pair(c) for c in args.checkpoint]
    scopes = [_pair(s) for s in args.scope]
    metrics = [_pair(m) for m in (args.metric or ["dsc=DSC", "precision=Precision"])]
    rows = _load(args.log)

    # values[scope][metric][checkpoint]
    values = {
        sk: {mk: [_lookup(rows, ck, sk, args.class_id)[mk] for ck, _ in checkpoints]
             for mk, _ in metrics}
        for sk, _ in scopes
    }

    fig, axes = plt.subplots(1, len(scopes), figsize=(3.6 * len(scopes), 3.5), sharey=True)
    if len(scopes) == 1:
        axes = [axes]

    n_series = len(checkpoints)
    bar_w = 0.8 / n_series
    if args.highlight is not None:
        keys = [ck for ck, _ in checkpoints]
        if args.highlight not in keys:
            sys.exit(f"--highlight '{args.highlight}' is not one of the --checkpoint keys: {keys}")
        hi = keys.index(args.highlight)
        # Muted colors are handed out in order of appearance so that two
        # non-highlighted series never collide into the same swatch.
        muted = [NEUTRAL, NSU_GOLD, "#8E6FB0", "#4A7FB5"]
        series_colors, k = [], 0
        for i in range(n_series):
            if i == hi:
                series_colors.append(NSU_GREEN)
            else:
                series_colors.append(muted[k % len(muted)])
                k += 1
    else:
        series_colors = [SERIES_COLORS[i % len(SERIES_COLORS)] for i in range(n_series)]

    for ax, (scope_key, scope_label) in zip(axes, scopes):
        for i, (_ck, ck_label) in enumerate(checkpoints):
            xs, heights = [], []
            for j, (mk, _ml) in enumerate(metrics):
                xs.append(j - 0.4 + bar_w * (i + 0.5))
                heights.append(values[scope_key][mk][i])
            bars = ax.bar(xs, heights, bar_w * 0.92,
                          color=series_colors[i],
                          label=ck_label if ax is axes[0] else None,
                          edgecolor="white", linewidth=0.6, zorder=3)
            for (mk, _ml), bar, h in zip(metrics, bars, heights):
                baseline = values[scope_key][mk][0]
                # Red marks a variant that came in under the baseline — the
                # regression a pooled average would have hidden.
                worse = i > 0 and h < baseline
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=9.5,
                        color=WARN_RED if worse else INK,
                        fontweight="bold" if worse else "normal", zorder=4)

        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([ml for _mk, ml in metrics], fontsize=11.5)
        ax.set_title(scope_label, fontsize=12.5, color=INK, pad=9)
        ax.set_ylim(0, 1.0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="both", length=0, colors=INK, labelsize=10.5)

    axes[0].set_ylabel("Trapped-cell score", fontsize=11.5, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_series, frameon=False,
               fontsize=11.5, bbox_to_anchor=(0.5, -0.02))
    if args.title:
        fig.suptitle(args.title, fontsize=12, color=INK, y=1.0)

    fig.tight_layout(rect=(0, 0.06, 1, 0.99))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"-> {args.output}")
    for sk, sl in scopes:
        for mk, ml in metrics:
            cells = "  ".join(f"{cl}={values[sk][mk][i]:.3f}"
                              for i, (_c, cl) in enumerate(checkpoints))
            print(f"   {sl:<26} {ml:<10} {cells}")


if __name__ == "__main__":
    main()
