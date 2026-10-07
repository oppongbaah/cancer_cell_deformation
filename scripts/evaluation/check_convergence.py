#!/usr/bin/env python3
"""
Flag training runs that converged poorly, before their numbers enter a mean.

Internal validation carries **no signal for comparing recipes** — the 2026-09-10
controlled re-run separated two checkpoints by 0.001 internally while the
development set separated the same two by 0.066. But it does detect a run that
simply trained badly, and that is a different question.

The case this exists for: `postswap_ftl_cw142_s44` reached internal val 0.905
against its siblings' 0.933 and 0.940, and went on to score 0.507 on `high-1`
against their 0.818 and 0.833 — dragging the 3-seed mean down and roughly
tripling the evaluated error bar. The bad run announced itself on a number that was
already being printed; nothing was looking at it.

Usage
-----
python scripts/evaluation/check_convergence.py                       # all postswap seeds
python scripts/evaluation/check_convergence.py --runs A B C
python scripts/evaluation/check_convergence.py --tolerance 0.015
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import torch

CKPT_ROOT = Path("checkpoints/postswap/experiment_0a_seed_baseline")
TOLERANCE = 0.02   # how far below the best sibling a run may sit before it is suspect


def read_run(d: Path) -> dict | None:
    ck = d / "unet_best.pt"
    if not ck.exists():
        return None
    m = torch.load(ck, map_location="cpu", weights_only=False)
    return {"run": d.name, "epoch": m.get("epoch"), "val_dsc": float(m.get("val_dsc", float("nan")))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-root", type=Path, default=CKPT_ROOT)
    p.add_argument("--runs", nargs="*", default=None,
                   help="Run names to compare (default: every seed of the baseline recipe)")
    p.add_argument("--tolerance", type=float, default=TOLERANCE,
                   help=f"Flag a run this far below the best sibling (default {TOLERANCE})")
    args = p.parse_args()

    names = args.runs or sorted(d.name for d in args.ckpt_root.iterdir()
                                if d.is_dir() and (d / "unet_best.pt").exists())
    runs = [r for r in (read_run(args.ckpt_root / n) for n in names) if r]
    if len(runs) < 2:
        raise SystemExit(f"need at least 2 runs to compare, found {len(runs)} under {args.ckpt_root}")

    best = max(r["val_dsc"] for r in runs)
    for r in runs:
        r["delta"] = r["val_dsc"] - best
        r["suspect"] = r["delta"] < -args.tolerance

    print(f"Internal validation across {len(runs)} runs  (tolerance {args.tolerance})\n")
    print(f"  {'run':<44}{'epoch':>7}{'val_DSC':>10}{'vs best':>10}")
    print("  " + "-" * 70)
    for r in sorted(runs, key=lambda x: -x["val_dsc"]):
        flag = "  <-- SUSPECT" if r["suspect"] else ""
        print(f"  {r['run']:<44}{r['epoch']:>7}{r['val_dsc']:>10.4f}{r['delta']:>+10.4f}{flag}")

    vals = [r["val_dsc"] for r in runs]
    print(f"\n  mean {st.mean(vals):.4f}   sd {st.stdev(vals):.4f}   "
          f"range {max(vals) - min(vals):.4f}")

    bad = [r for r in runs if r["suspect"]]
    if not bad:
        print("\n  All runs converged comparably. Their evaluated numbers can be compared.")
        return

    print(f"\n  {len(bad)} run(s) converged materially worse than the rest:")
    for r in bad:
        print(f"      {r['run']}  ({r['delta']:+.4f})")
    print("""
  This is NOT a recipe comparison — internal validation cannot rank recipes here.
  It says these runs trained badly, and their development-set scores will drag a
  multi-seed mean down for a reason that has nothing to do with the change under test.

  Re-run them with a different seed before pooling:
      python scripts/training/train_experiment.py postswap/experiment_0a_seed_baseline <run-name>

  Then re-check. Keep the tolerance fixed rather than tuning it to admit a run.""")
    sys.exit(1)


if __name__ == "__main__":
    main()
