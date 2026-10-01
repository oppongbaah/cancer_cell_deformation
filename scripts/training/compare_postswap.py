#!/usr/bin/env python3
"""
Compare postswap runs against the series baseline, on the development set.

Exists to enforce the rule in configs/postswap/README.md that a single-seed
delta smaller than the noise floor is not a result. Measured spread on this
pipeline is sd ~0.047 pooled (3 seeds, split pinned), so a difference between
two single runs has SE ~ sd*sqrt(2) ~ 0.066 and needs roughly 2x that to be
believable. Deltas under that threshold are printed as INCONCLUSIVE, however
tempting they look -- this project has twice recorded a sub-threshold delta as
a finding and had to retract it.

Usage
-----
python scripts/training/compare_postswap.py                       # every run in the log
python scripts/training/compare_postswap.py --runs A B            # just these, vs. baseline
python scripts/training/compare_postswap.py --seeds 42 43 44      # pool seeds of one recipe
"""
from __future__ import annotations
import argparse, json, statistics as st
from collections import defaultdict
from pathlib import Path

LOG = Path("results/postswap/experiment_0a_seed_baseline/eval_log.jsonl")
BASELINE = "postswap_ftl_cw142_s42"
CELLS = ["high_cell1", "high_cell9", "low_cell5", "low_cell6"]
SD = 0.049                      # evaluated sd over FOUR seeds (s42-s45), split pinned, complete
                                # 203-frame testing set (2026-09-24). Was 0.057 over three seeds
                                # (and 0.047 on 172 frames); updated 2026-09-30.
THRESHOLD = 2 * SD * (2 ** 0.5) # ~0.139 -- smallest believable single-run difference


def load(log: Path) -> dict[str, dict[str, dict]]:
    if not log.exists():
        raise SystemExit(f"No eval log at {log}. Run scripts/training/run_postswap.sh first.")
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        run = Path(r["checkpoint"]).parent.name
        out[run][r.get("filter") or "ALL CELLS"] = r["metrics"]["1"]   # class 1 = trapped cell
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path, default=LOG)
    p.add_argument("--runs", nargs="*", default=None, help="Run names to show")
    p.add_argument("--baseline", default=BASELINE)
    p.add_argument("--seeds", nargs="*", default=None,
                   help="Run names to aggregate as seeds of one recipe (mean +/- sd)")
    args = p.parse_args()

    data = load(args.log)
    if args.seeds:
        names = [n for n in args.seeds if n in data]
        if not names:
            raise SystemExit("none of those runs are in the log")
        print(f"Aggregate over {len(names)} seeds: {', '.join(names)}\n")
        print(f"{'cell':<10}" + "".join(f"{n.split('_')[-1]:>9}" for n in names)
              + f"{'mean':>9}{'sd':>8}")
        for c in CELLS + ["ALL CELLS"]:
            v = [data[n][c]["dsc"] for n in names if c in data[n]]
            if len(v) < 2:
                continue
            lbl = c.replace("_cell", "-") if c != "ALL CELLS" else "ALL CELLS"
            print(f"{lbl:<10}" + "".join(f"{x:>9.3f}" for x in v)
                  + f"{st.mean(v):>9.3f}{st.stdev(v):>8.3f}")
        return

    base = data.get(args.baseline)
    if base is None:
        raise SystemExit(f"baseline {args.baseline} not in the log")
    runs = args.runs or [r for r in data if r != args.baseline]
    runs = [r for r in runs if r != args.baseline and r in data]
    if not runs:
        print(f"Baseline only ({args.baseline}):")
        for c in CELLS + ["ALL CELLS"]:
            if c in base:
                lbl = c.replace("_cell", "-") if c != "ALL CELLS" else "ALL CELLS"
                print(f"  {lbl:<10}{base[c]['dsc']:>7.3f}")
        return

    print(f"Baseline: {args.baseline}")
    print(f"Noise floor: single-run differences below {THRESHOLD:.3f} are not "
          f"interpretable (evaluated sd {SD}).\n")
    for run in runs:
        m = data[run]
        print(f"--- {run} ---")
        print(f"{'cell':<10}{'baseline':>10}{'this run':>10}{'delta':>9}")
        for c in CELLS + ["ALL CELLS"]:
            if c not in m or c not in base:
                continue
            b, x = base[c]["dsc"], m[c]["dsc"]
            lbl = c.replace("_cell", "-") if c != "ALL CELLS" else "ALL CELLS"
            print(f"{lbl:<10}{b:>10.3f}{x:>10.3f}{x - b:>+9.3f}")
        if "ALL CELLS" in m and "ALL CELLS" in base:
            d = m["ALL CELLS"]["dsc"] - base["ALL CELLS"]["dsc"]
            if abs(d) < THRESHOLD:
                print(f"\n  => INCONCLUSIVE. Evaluated delta {d:+.3f} is inside the "
                      f"+/-{THRESHOLD:.3f} noise floor.\n     Log as screened, not as a "
                      f"result. Three seeds on BOTH arms to decide it.")
            else:
                print(f"\n  => Evaluated delta {d:+.3f} exceeds the {THRESHOLD:.3f} noise "
                      f"floor -- worth confirming\n     with three seeds on both arms.")
        print()


if __name__ == "__main__":
    main()
