#!/usr/bin/env python3
"""
Generate the progress report's LaTeX tables from the data.

The charts in this repo already read from disk, but the report's tables were
hand-typed, which meant a pool change could silently leave them stale. These
emit `.tex` fragments the report `\\input{}`s, so every number in the document
comes from one of these scripts.

Two tables:

  percell   per-cell frame counts and annotation status. Counts frames in
            01_annotate_pool (domain_ prefix), 03_mask_factory and
            02_unet_holdout, and checks each frame for a mask.
  headline  development-set results for the runs named on the command line,
            looked up in results/holdout_eval_log.jsonl by (checkpoint, filter).
  baseline  one checkpoint, broken out per development cell plus pooled.
  decoy     where each cell's false positives land — on annotated decoy
            objects (class 2) or on background. Loads the model, so it is
            slower than the others; it is the only table that cannot be
            answered from the eval log alone.

Usage
-----
python scripts/make_report_tables.py percell --output ../masters_thesis/main_work/tables/percell.tex
python scripts/make_report_tables.py headline --output ../masters_thesis/main_work/tables/headline.tex \\
    --run multiclass_experiment_256_bf64_base_cos="No augmentation (noise only)" \\
    --run multiclass_experiment_256_bf64_scale_cos="Scale augmentation" \\
    --run multiclass_experiment_256_bf64_allaug="Combined augmentation" \\
    --baseline multiclass_experiment_256_bf64_base_cos \\
    --emph multiclass_experiment_256_bf64_scale_cos
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_EXTS = (".jpg", ".jpeg", ".png")


def _tex(s: str) -> str:
    return s.replace("_", r"\_")


def _frames(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return [p for p in d.iterdir() if p.suffix.lower() in _EXTS and p.is_file()]


def _cell(stem: str) -> str | None:
    s = stem[len("domain_"):] if stem.startswith("domain_") else stem
    parts = s.split("_")
    if len(parts) >= 2 and parts[1].startswith("cell"):
        return f"{parts[0]}-{parts[1].replace('cell', '')}"
    return None


def _key(cell: str) -> tuple[int, int]:
    her2, n = cell.split("-")
    return (0 if her2 == "high" else 1, int(n))


# ── per-cell table ──────────────────────────────────────────────────────────
def percell(root: Path) -> str:
    fr, mk = root / "frames", root / "masks"
    train, factory, hold = defaultdict(int), defaultdict(int), defaultdict(int)
    ann = defaultdict(int)

    pool_masks = mk / "01_annotate_pool_multiclass"
    for p in _frames(fr / "01_annotate_pool"):
        c = _cell(p.stem)
        if c is None:                     # legacy frames carry no cell id
            continue
        train[c] += 1
        ann[c] += (pool_masks / f"{p.stem}.png").exists()
    for p in _frames(fr / "03_mask_factory"):
        if (c := _cell(p.stem)):
            factory[c] += 1

    hold_masks = mk / "02_unet_holdout_multiclass"
    hold_ann = defaultdict(int)
    for p in _frames(fr / "02_unet_holdout"):
        if (c := _cell(p.stem)):
            hold[c] += 1
            hold_ann[c] += (hold_masks / f"{p.stem}.png").exists()
    blind, blind_ann = defaultdict(int), defaultdict(int)
    for p in _frames(fr / "02_unet_holdout" / "final_test"):
        if (c := _cell(p.stem)):
            blind[c] += 1
            blind_ann[c] += (hold_masks / f"{p.stem}.png").exists()

    n_legacy = sum(1 for p in _frames(fr / "01_annotate_pool") if _cell(p.stem) is None)
    n_legacy_ann = sum(1 for p in _frames(fr / "01_annotate_pool")
                       if _cell(p.stem) is None and (pool_masks / f"{p.stem}.png").exists())
    n_legacy_hold = len(_frames(fr / "02_unet_holdout" / "legacy_holdout"))

    def row(c: str) -> str:
        if c in hold:
            mark = (rf"\checkmark~\textbf{{{hold_ann[c]}/{hold[c]}}}"
                    if hold_ann[c] else r"\textcolor{warnred}{\textbf{unannotated}}")
            return (rf"\textbf{{{c}}} & \multicolumn{{2}}{{c}}{{\textit{{dev.\ set, {hold[c]}fr}}}} & {mark}")
        if c in blind:
            mark = (r"\textcolor{warnred}{\textbf{reserved}}" if not blind_ann[c]
                    else rf"\textcolor{{warnred}}{{\textbf{{{blind_ann[c]} annotated (!)}}}}")
            return (rf"\textbf{{{c}}} & \multicolumn{{2}}{{c}}{{\textit{{blind test, {blind[c]}fr}}}} & {mark}")
        mark = r"\checkmark" if ann[c] == train[c] else rf"\textcolor{{warnred}}{{{ann[c]}/{train[c]}}}"
        return f"{c} & {train[c]} & {factory.get(c, 0)} & {mark}"

    cells = sorted(set(train) | set(factory) | set(hold) | set(blind), key=_key)
    highs = [c for c in cells if c.startswith("high")]
    lows = [c for c in cells if c.startswith("low")]
    body = []
    for i in range(max(len(highs), len(lows))):
        left = row(highs[i]) if i < len(highs) else " & & & "
        right = row(lows[i]) if i < len(lows) else " & & & "
        body.append(f"{left} & {right} \\\\")

    return "\n".join([
        r"% GENERATED by scripts/make_report_tables.py percell — do not edit by hand.",
        r"\captionof{table}{Per-cell frame counts and annotation status.}\label{tab:percell}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccc|lccc}", r"\toprule",
        r"\textbf{Cell} & \textbf{Train} & \textbf{Factory} & \textbf{Ann.} & "
        r"\textbf{Cell} & \textbf{Train} & \textbf{Factory} & \textbf{Ann.} \\",
        r"\midrule", *body, r"\midrule",
        rf"\multicolumn{{4}}{{l}}{{Legacy 2021 (no cell ID): \textbf{{{n_legacy}}} frames, "
        rf"{'all annotated' if n_legacy_ann == n_legacy else f'{n_legacy_ann}/{n_legacy} annotated'}}} & "
        rf"\multicolumn{{4}}{{l}}{{\code{{legacy\_holdout}} (diag.-only): {n_legacy_hold} frames, reserved}} \\",
        r"\bottomrule", r"\end{tabular}", ""])


# ── headline results table ──────────────────────────────────────────────────
def headline(log: Path, runs, baseline: str | None, emph: set[str], extra) -> str:
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]

    def pooled(ck: str) -> dict:
        hits = [r for r in rows if Path(r["checkpoint"]).parent.name == ck
                and r.get("filter") is None and "1" in r.get("metrics", {})]
        if not hits:
            sys.exit(f"No pooled eval-log entry for '{ck}'. Run scripts/evaluate_unet.py for it first.")
        return max(hits, key=lambda r: r.get("timestamp", ""))["metrics"]["1"]

    base = pooled(baseline)["dsc"] if baseline else None
    out = []
    for ck, label in runs:
        m = pooled(ck)
        d = "---" if base is None or ck == baseline else f"{m['dsc'] - base:+.3f}"
        if d != "---":
            d = (rf"\textbf{{${d[0]}${d[1:]}}}" if not d.startswith("-")
                 else rf"\textcolor{{warnred}}{{${d[0]}${d[1:]}}}")
        cells = [label, f"{m['dsc']:.3f}", f"{m['precision']:.3f}", f"{m['recall']:.3f}", d]
        if ck in emph:
            cells = [rf"\textbf{{{c}}}" if i < 4 else c for i, c in enumerate(cells)]
        out.append(" & ".join(cells) + r" \\")
    for ck, label in extra:
        m = pooled(ck)
        out.append(" & ".join([rf"\textit{{{label}}}"] +
                              [rf"\textit{{{m[k]:.3f}}}" for k in ("dsc", "precision", "recall")] +
                              [r"\textit{---}"]) + r" \\")

    return "\n".join([
        r"% GENERATED by scripts/make_report_tables.py headline — do not edit by hand.",
        r"\captionof{table}{Development-set results. All runs share an identical config apart from",
        r"augmentation; any italic row additionally used the older schedule and split, so it is not",
        r"directly comparable to the rows above it.}\label{tab:headline}",
        r"\begin{tabularx}{\linewidth}{Xcccc}", r"\toprule",
        r"\textbf{Run} & \textbf{Pooled DSC} & \textbf{Precision} & \textbf{Recall} & "
        r"\textbf{vs.\ no aug.} \\", r"\midrule",
        *out[:len(runs)],
        *([r"\midrule"] + out[len(runs):] if extra else []),
        r"\bottomrule", r"\end{tabularx}", ""])


def baseline(log: Path, ck: str, cells) -> str:
    """One checkpoint, per development cell plus pooled — read from the eval log."""
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]

    def get(scope):
        want = None if scope == "pooled" else scope
        hits = [r for r in rows if Path(r["checkpoint"]).parent.name == ck
                and r.get("filter") == want and "1" in r.get("metrics", {})]
        return max(hits, key=lambda r: r.get("timestamp", "")) if hits else None

    out = []
    for key, label in cells:
        r = get(key)
        if r is None:
            out.append(f"{label} & \\multicolumn{{4}}{{c}}{{\\textit{{not yet evaluated}}}} \\\\")
            continue
        m = r["metrics"]["1"]
        out.append(f"{label} & {r['n_images']} & {m['dsc']:.3f} & "
                   f"{m['precision']:.3f} & {m['recall']:.3f} \\\\")
    p = get("pooled")
    if p:
        m = p["metrics"]["1"]
        out += [r"\midrule",
                f"\\textit{{Pooled}} & \\textit{{{p['n_images']}}} & \\textbf{{{m['dsc']:.3f}}} & "
                f"\\textit{{{m['precision']:.3f}}} & \\textit{{{m['recall']:.3f}}} \\\\"]
    return "\n".join([
        r"% GENERATED by scripts/make_report_tables.py baseline — do not edit by hand.",
        r"\captionof{table}{Baseline: the settled scale-augmentation checkpoint on the",
        r"restructured four-cell development set. Every experiment this week is measured",
        r"against these numbers.}\label{tab:baseline}",
        r"\begin{tabularx}{\linewidth}{Xcccc}", r"\toprule",
        r"\textbf{Cell} & \textbf{Frames} & \textbf{DSC} & \textbf{Precision} & \textbf{Recall} \\",
        r"\midrule", *out, r"\bottomrule", r"\end{tabularx}", ""])


def decoy(root: Path, ck: str, cells, base_features: int) -> str:
    """Where each cell's false positives land: annotated decoy, or background."""
    import cv2, numpy as np
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from celldform.segmentation.unet import UNet

    model = UNet.from_checkpoint(Path(f"checkpoints/{ck}/unet_best.pt"),
                                 in_channels=1, base_features=base_features, out_channels=3)
    mk = root / "masks" / "02_unet_holdout_multiclass"
    pre = root / "preprocessed" / "evaluate" / "02_unet_holdout" / "inputs"
    out = []
    for key, label in cells:
        tot = on2 = 0
        for q in sorted(mk.glob(f"{key}_*.png")):
            pin = pre / f"{q.stem}.png"
            if not pin.exists():
                continue
            gt = cv2.imread(str(q), cv2.IMREAD_GRAYSCALE)
            small = cv2.imread(str(pin), cv2.IMREAD_GRAYSCALE)
            pred = cv2.resize(model.predict(small.astype(np.float32) / 255.0, binary_output=True),
                              (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            fp = (pred > 0) & (gt != 1)
            tot += int(fp.sum()); on2 += int((fp & (gt == 2)).sum())
        if not tot:
            continue
        frac = on2 / tot
        verdict = (r"\textcolor{warnred}{\textbf{decoy confusion}}" if frac > 0.4
                   else "background over-spill")
        pct = lambda v: f"{v*100:.0f}\\%"          # escape: bare % comments out the row
        cell_txt = f"\\textbf{{{pct(frac)}}}" if frac > 0.4 else pct(frac)
        out.append(f"{label} & {tot:,} & {cell_txt} & {pct(1-frac)} & {verdict} \\\\")
    return "\n".join([
        r"% GENERATED by scripts/make_report_tables.py decoy — do not edit by hand.",
        r"\captionof{table}{Where each cell's false positives land. ``On decoy'' means the",
        r"model predicted \emph{trapped cell} on pixels the annotator labelled as an untrapped",
        r"object.}\label{tab:decoy}",
        r"\begin{tabularx}{\linewidth}{Xcccl}", r"\toprule",
        r"\textbf{Cell} & \textbf{FP pixels} & \textbf{On decoy} & \textbf{On background} & \textbf{Verdict} \\",
        r"\midrule", *out, r"\bottomrule", r"\end{tabularx}", ""])


def _pair(spec: str):
    k, _, v = spec.partition("=")
    return k.strip(), (v.strip() or k.strip())


def main() -> None:
    p = argparse.ArgumentParser(description="Emit the report's LaTeX tables from the data")
    p.add_argument("table", choices=["percell", "headline", "baseline", "decoy"])
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--log", type=Path, default=Path("results/holdout_eval_log.jsonl"))
    p.add_argument("--run", action="append", default=[], metavar="KEY[=LABEL]")
    p.add_argument("--extra-run", action="append", default=[], metavar="KEY[=LABEL]",
                   help="Row rendered in italics, below a rule — for runs that are not comparable")
    p.add_argument("--baseline", default=None, help="Run the delta column is measured against")
    p.add_argument("--emph", action="append", default=[], help="Run to bold, repeatable")
    p.add_argument("--checkpoint", default=None, help="Checkpoint dir name (baseline/decoy)")
    p.add_argument("--cell", action="append", default=[], metavar="KEY[=LABEL]",
                   help="Development cell, repeatable (baseline/decoy)")
    p.add_argument("--base-features", type=int, default=64)
    p.add_argument("--output", type=Path, default=None, help="Default: stdout")
    args = p.parse_args()

    if args.table == "percell":
        text = percell(args.data_root)
    elif args.table == "baseline":
        if not (args.checkpoint and args.cell):
            sys.exit("baseline needs --checkpoint and at least one --cell")
        text = baseline(args.log, args.checkpoint, [_pair(c) for c in args.cell])
    elif args.table == "decoy":
        if not (args.checkpoint and args.cell):
            sys.exit("decoy needs --checkpoint and at least one --cell")
        text = decoy(args.data_root, args.checkpoint, [_pair(c) for c in args.cell],
                     args.base_features)
    else:
        if not args.run:
            sys.exit("headline needs at least one --run")
        text = headline(args.log, [_pair(r) for r in args.run], args.baseline,
                        set(args.emph), [_pair(r) for r in args.extra_run])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"-> {args.output}  ({len(text.splitlines())} lines)")
    else:
        print(text)


if __name__ == "__main__":
    main()
