#!/usr/bin/env python3
"""
Build one randswap fold: draw the cell split from the config, then preprocess its data.

    python scripts/data/build_randswap.py --config configs/randswap/<exp>/<exp>.yaml --fold 1

Reads the config's ``split:`` section (whole-cell counts, seed, stratification, sources)
and writes, under ``<split.data_root>/fold_<k>/``:

    train/inputs, train/labels   every frame of the train + val cells, plus legacy frames
    test/inputs,  test/labels    split.test_frames_per_cell evenly spaced frames per test cell
    split.json                   the record: cells per role, frame counts, test frames,
                                 crop centres, percentages, and the config's git commit

The crop centre (when ``preprocessing.crop_size`` is set) is the centre of the
trapped-cell region of THIS FOLD's training cells only -- the test cells' masks are never
read to place it. Legacy frames are cropped around their own trap.

The split is a draw: it refuses to run unless the config file is committed to git and
unmodified, so the history shows the seed was fixed before any split was seen. Accept
the result as it comes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from celldform.config import load as load_config
from celldform.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from celldform.splits import assign, collect_frames, describe, pick_test_frames, roster_of


def committed(path: Path) -> str:
    """Commit hash of the last commit touching ``path``; exits if untracked or modified."""
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True)
    if git("ls-files", "--error-unmatch", str(path)).returncode:
        sys.exit(f"{path} is not committed. Commit the config BEFORE the first draw.")
    if git("diff", "--quiet", "HEAD", "--", str(path)).returncode:
        sys.exit(f"{path} has uncommitted changes. Commit them before drawing.")
    return git("log", "-1", "--format=%H", "--", str(path)).stdout.strip()


def trap_centre(masks: list[Path]) -> tuple[int, int]:
    """Centre (x, y) of the union bounding box of trapped-cell (label 1) pixels."""
    x0 = y0 = 10 ** 9
    x1 = y1 = -1
    for m in masks:
        ys, xs = np.nonzero(cv2.imread(str(m), cv2.IMREAD_GRAYSCALE) == 1)
        if len(xs):
            x0, x1, y0, y1 = min(x0, xs.min()), max(x1, xs.max()), min(y0, ys.min()), max(y1, ys.max())
    if x1 < 0:
        raise ValueError("no trapped-cell pixels to place the crop")
    return int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fold", required=True, help="1..k, or 'deploy' (every cell trains: the deployment model)")
    args = ap.parse_args()

    commit = committed(args.config)
    conf = load_config(str(args.config))
    sp, pp = conf.split, conf.preprocessing
    if sp is None or not sp.data_root:
        sys.exit("config has no split section with a data_root")

    frames = collect_frames(sp.sources)
    roster = roster_of(frames)
    a = assign(roster, sp, args.fold)
    print(describe(a, frames))

    train_frames = [f for f in frames if a.role(f.cell) in ("train", "val")]
    test = [t for c in a.test for t in pick_test_frames(frames, c, sp.test_frames_per_cell)]

    crop, legacy_crop = None, None
    if pp.crop_size:
        crop = trap_centre([f.mask for f in frames if f.cell in a.train])
        legacy = [f.mask for f in frames if f.cell is None]
        legacy_crop = trap_centre(legacy) if legacy else crop
        print(f"  crop {pp.crop_size}: centre {crop} from the {len(a.train)} training cells; "
              f"legacy {legacy_crop}")

    out = Path(sp.data_root) / f"fold_{a.fold}"
    if out.exists():
        shutil.rmtree(out)
    cfg = PreprocessingConfig(output_size=tuple(pp.target_size), median_kernel=pp.median_kernel,
                              clahe_clip_limit=pp.clahe_clip_limit,
                              clahe_tile_grid=tuple(pp.clahe_tile_grid),
                              morph_kernel_size=pp.morph_kernel_size, crop_size=pp.crop_size)
    size = (cfg.output_size[1], cfg.output_size[0])

    def write(group, items):
        (out / group / "inputs").mkdir(parents=True)
        (out / group / "labels").mkdir(parents=True)
        for f in items:
            cfg.crop_center = legacy_crop if f.cell is None else crop
            pipe = PreprocessingPipeline(cfg)
            img = cv2.imread(str(f.image), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(out / group / "inputs" / f"{f.stem}.png"),
                        (pipe(img) * 255).astype(np.uint8))
            mask = cv2.imread(str(f.mask), cv2.IMREAD_GRAYSCALE)
            if cfg.crop_size:
                x0, y0, x1, y1 = pipe.crop_box(*mask.shape[:2])
                mask = mask[y0:y1, x0:x1]
            if mask.shape[::-1] != size:
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out / group / "labels" / f"{f.stem}.png"), mask)

    write("train", train_frames)
    if test:
        write("test", test)

    per_cell = {c: sum(1 for f in frames if f.cell == c) for c in roster}
    record = {
        "fold": a.fold, "config": str(args.config), "config_commit": commit,
        "cell_split_seed": sp.cell_split_seed, "stratify_by": sp.stratify_by,
        "counts": {"train": len(a.train), "val": len(a.val), "test": len(a.test)},
        "percent_cells": {r: round(100 * len(v) / len(roster), 1)
                          for r, v in (("train", a.train), ("val", a.val), ("test", a.test))},
        "cells": {"train": a.train, "val": a.val, "test": a.test},
        "her2": roster, "annotated_frames_per_cell": per_cell,
        "n_legacy_frames": sum(1 for f in frames if f.cell is None),
        "test_frames": {c: [t.stem for t in test if t.cell == c] for c in a.test},
        "test_masks_native": {t.stem: str(t.mask) for t in test},
        "crop_size": pp.crop_size, "crop_center": crop, "legacy_crop_center": legacy_crop,
    }
    (out / "split.json").write_text(json.dumps(record, indent=1))
    print(f"  -> {out}  ({len(train_frames)} train/val frames, {len(test)} test frames)")


if __name__ == "__main__":
    main()
