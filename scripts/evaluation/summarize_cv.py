#!/usr/bin/env python3
"""
Summarise a randswap cross-validation: every round, every cell, one headline.

    python scripts/evaluation/summarize_cv.py --exp randswap/experiment_0_cv_baseline --run cv_baseline

Reads, per fold k: results/<exp>/<run>_fold_<k>_split.json (who was where),
logs/<exp>/<run>_fold_<k>.log (best epoch, internal validation Dice) and
results/<exp>/eval_log_native.jsonl (test-cell scores at native resolution, with raw pixel
counts). Writes results/<exp>/<run>_cv_rounds.csv, <run>_cv_cells.csv, <run>_cv_summary.json.

Headline = PER-CELL MEAN Dice over every cell (each tested once, by a model that never saw
it), so a cell with 50 annotated frames counts no more than one with 20. Pooled Dice (all
test frames' pixels together) is reported beside it. The median best epoch sets the
deployment model's fixed training length.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path


def best_epoch(log: Path) -> tuple[int | None, float | None]:
    text = log.read_text() if log.exists() else ""
    m = re.search(r"best=([\d.]+) at epoch (\d+)", text)
    if m:
        return int(m.group(2)), float(m.group(1))
    vals = [(int(e), float(v)) for e, v in re.findall(r"Epoch (\d+)/\d+ .*?val_DSC=([\d.]+)", text)]
    if not vals:
        return None, None
    e, v = max(vals, key=lambda t: t[1])
    return e, v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    R, L = Path("results") / args.exp, Path("logs") / args.exp

    scores: dict[tuple[str, str], dict] = {}
    for line in (R / "eval_log_native.jsonl").read_text().splitlines():
        r = json.loads(line)
        # Our runs: the round is in the checkpoint path (fold_<k>/unet_best.pt). nnU-Net runs log
        # it explicitly as "fold" -- nnU-Net's own fold folders are 0-based.
        m = re.search(r"/fold_(\w+)/unet_best\.pt$", r["checkpoint"])
        fold = r.get("fold") or (m.group(1) if m else None)
        if fold and r["filter"]:
            scores[(fold, r["filter"])] = r["metrics"]["1"]            # latest entry wins

    rounds, cells = [], []
    # CV rounds only, in numeric order ("deploy" is the all-cell model, not a round; and a string
    # sort would put round 10 before round 2)
    splits = [p for p in R.glob(f"{args.run}_fold_*_split.json") if p.stem.split("_fold_")[1][:-6].isdigit()]
    for split in sorted(splits, key=lambda p: int(p.stem.split("_fold_")[1][:-6])):
        rec = json.loads(split.read_text())
        k = rec["fold"]
        if k == "deploy":           # the all-cell deployment model: no test cells, nothing to summarise
            continue
        ep, val = best_epoch(L / f"{args.run}_fold_{k}.log")
        row = {"fold": k, "test_cells": " ".join(rec["cells"]["test"]),
               "val_cell": " ".join(rec["cells"]["val"]), "crop_center": rec.get("crop_center"),
               "best_epoch": ep, "internal_val_dsc": val}
        for c in rec["cells"]["test"]:
            m = scores.get((k, c))
            if m is None:
                continue
            row[c] = round(m["dsc"], 4)
            cells.append({"cell": c, "her2": rec["her2"][c], "fold": k, "dsc": m["dsc"],
                          "precision": m["precision"], "recall": m["recall"],
                          "tp": m.get("tp"), "fp": m.get("fp"), "fn": m.get("fn")})
        rounds.append(row)

    if not cells:
        raise SystemExit("no scored test cells yet")
    dsc = [c["dsc"] for c in cells]
    summary = {"rounds_done": len({c["fold"] for c in cells}),   # rounds with scored cells
                "cells_scored": len(cells),
               "per_cell_mean_dsc": st.mean(dsc), "per_cell_sd": st.stdev(dsc) if len(dsc) > 1 else None,
               "worst_cell": min(cells, key=lambda c: c["dsc"])["cell"], "worst_dsc": min(dsc)}
    for cls in ("high", "low"):
        v = [c["dsc"] for c in cells if c["her2"] == cls]
        if v:
            summary[f"{cls}_mean_dsc"] = st.mean(v)
    if all(c["tp"] is not None for c in cells):
        tp, fp, fn = (sum(c[k] for c in cells) for k in ("tp", "fp", "fn"))
        summary["pooled_dsc"] = 2 * tp / (2 * tp + fp + fn)
    eps = [r["best_epoch"] for r in rounds if r["best_epoch"]]
    if eps:
        summary["median_best_epoch"] = st.median(eps)

    with (R / f"{args.run}_cv_rounds.csv").open("w", newline="") as fh:
        keys = ["fold", "test_cells", "val_cell", "crop_center", "best_epoch", "internal_val_dsc"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rounds)
    with (R / f"{args.run}_cv_cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0])); w.writeheader()
        w.writerows(sorted(cells, key=lambda c: (c["her2"], c["cell"])))
    (R / f"{args.run}_cv_summary.json").write_text(json.dumps(summary, indent=1))

    print(f"{'fold':>4}  {'test cells':<34} {'val':<12} {'best ep':>7} {'int. val':>8}")
    for r in rounds:
        print(f"{r['fold']:>4}  {r['test_cells']:<34} {r['val_cell']:<12} {str(r['best_epoch']):>7} "
              f"{'-' if r['internal_val_dsc'] is None else round(r['internal_val_dsc'], 4):>8}")
    print()
    for c in sorted(cells, key=lambda c: (c["her2"], c["cell"])):
        print(f"  {c['cell'].replace('_cell', '-'):<9} fold {c['fold']}  DSC {c['dsc']:.3f}  "
              f"prec {c['precision']:.3f}  rec {c['recall']:.3f}")
    print()
    for k, v in summary.items():
        print(f"  {k:<18} {round(v, 4) if isinstance(v, float) else v}")


if __name__ == "__main__":
    main()
