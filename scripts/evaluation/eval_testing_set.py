#!/usr/bin/env python3
"""
Score one checkpoint on the testing set (the four held-out cells, on disk
02_unet_holdout / preprocessed postswap_devset): each cell, then all cells pooled.

    python scripts/evaluation/eval_testing_set.py <checkpoint.pt> <config.yaml> <eval_log.jsonl> [pool]

pool defaults to postswap_devset (256x256); a 512 model needs postswap_devset_512. Each of the
five scores is one evaluate_unet.py run appending one line to the eval log. Runs with whichever
python launched it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CELLS = ["high_cell1", "high_cell9", "low_cell5", "low_cell6"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("log", type=Path, help="eval log (JSONL) the scores are appended to")
    ap.add_argument("pool", nargs="?", default="postswap_devset")
    args = ap.parse_args()
    os.chdir(ROOT)

    if not args.checkpoint.is_file():
        sys.exit(f"no checkpoint: {args.checkpoint}")
    for cell in CELLS + [None]:                      # None: all cells pooled
        cmd = [sys.executable, "scripts/evaluation/evaluate_unet.py", "--checkpoint", str(args.checkpoint),
               "--config", str(args.config), "--pool", args.pool, "--results-log", str(args.log)]
        if cell:
            cmd += ["--filter", cell]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    print(f"  scored {args.checkpoint.parent.name} on the testing set -> {args.log}")


if __name__ == "__main__":
    main()
