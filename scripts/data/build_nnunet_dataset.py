#!/usr/bin/env python3
"""
Build the nnU-Net benchmark dataset from the randswap cross-validation -- same frames, same folds.

    python scripts/data/build_nnunet_dataset.py \
        --config configs/randswap/experiment_0_cv_baseline/cv_baseline.yaml \
        --splits results/randswap/experiment_0_cv_baseline --run cv_baseline

What nnU-Net gets is what our U-Net got, at the resolution it was annotated at:
  * every annotated frame (all 15 cells + the 200 legacy frames), RAW -- 1228x922 grayscale,
    no median/CLAHE/crop: nnU-Net plans its own normalisation and patch size (user decision,
    2026-10-01: benchmark nnU-Net as it is normally used);
  * the 3-class masks (0 background, 1 trapped cell, 2 decoy), unchanged;
  * splits_final.json with one entry per CV round, copied from the round's split record:
    train = that round's 11 training cells + legacy, val = its validation cell. The round's
    3 test cells are in neither -- they are predicted afterwards, never trained on.

nnU-Net folds are 0-based: nnU-Net fold k-1 is randswap round k.

One dataset holds every frame, so the dataset fingerprint (image shape, spacing) sees the test
cells too. For 2D PNGs with per-image z-score normalisation that fingerprint carries no label or
intensity information into training -- every frame has the same shape and spacing.

Writes <nnUNet_raw>/Dataset<ID>_<name>/{imagesTr,labelsTr,imagesTs_round<k>,dataset.json,
test_frames.json} and <nnUNet_preprocessed>/Dataset<ID>_<name>/splits_final.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from celldform.config import load as load_config
from celldform.splits import collect_frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True, help="results folder holding <run>_fold_<k>_split.json")
    ap.add_argument("--run", required=True)
    ap.add_argument("--raw", type=Path, default=Path("data/nnunet/raw"))
    ap.add_argument("--preprocessed", type=Path, default=Path("data/nnunet/preprocessed"))
    ap.add_argument("--dataset-id", type=int, default=501)
    ap.add_argument("--name", default="CellTrapRandswap")
    a = ap.parse_args()

    conf = load_config(str(a.config))
    frames = collect_frames(conf.split.sources)
    recs = sorted((json.loads(p.read_text()) for p in a.splits.glob(f"{a.run}_fold_*_split.json")),
                  key=lambda r: r["fold"])
    recs = [r for r in recs if r["fold"] != "final"]
    if not recs:
        raise SystemExit(f"no split records in {a.splits}")

    ds = f"Dataset{a.dataset_id:03d}_{a.name}"
    out = a.raw / ds
    if out.exists():
        shutil.rmtree(out)
    (out / "imagesTr").mkdir(parents=True)
    (out / "labelsTr").mkdir()

    shape = None
    for f in frames:
        img = cv2.imread(str(f.image), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(f.mask), cv2.IMREAD_GRAYSCALE)
        if img.shape != mask.shape:
            raise SystemExit(f"{f.stem}: image {img.shape} vs mask {mask.shape}")
        if mask.max() > 2:
            raise SystemExit(f"{f.stem}: label {mask.max()} outside 0/1/2")
        shape = shape or img.shape
        cv2.imwrite(str(out / "imagesTr" / f"{f.stem}_0000.png"), img)
        cv2.imwrite(str(out / "labelsTr" / f"{f.stem}.png"), mask.astype(np.uint8))

    # Each round's test frames, as a separate input folder for nnUNetv2_predict.
    tests = {}
    for r in recs:
        d = out / f"imagesTs_round{r['fold']}"
        d.mkdir()
        stems = [s for c in r["cells"]["test"] for s in r["test_frames"][c]]
        for s in stems:
            shutil.copy(out / "imagesTr" / f"{s}_0000.png", d / f"{s}_0000.png")
        tests[r["fold"]] = {"stems": stems, "masks": r["test_masks_native"], "cells": r["cells"],
                            "her2": r["her2"]}
    (out / "test_frames.json").write_text(json.dumps(tests, indent=1))

    (out / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "brightfield"},
        "labels": {"background": 0, "trapped_cell": 1, "decoy": 2},
        "numTraining": len(frames), "file_ending": ".png",
        "description": "randswap CV frames, raw 1228x922; folds = randswap rounds (nnU-Net fold k-1 = round k)",
    }, indent=1))

    by_cell = {}
    for f in frames:
        by_cell.setdefault(f.cell, []).append(f.stem)        # None = legacy
    splits = []
    for r in recs:
        train = sorted(by_cell[None] + [s for c in r["cells"]["train"] for s in by_cell[c]])
        val = sorted(s for c in r["cells"]["val"] for s in by_cell[c])
        test_cells = set(r["cells"]["test"])
        assert not any(s in test_cells for s in (r["cells"]["train"] + r["cells"]["val"]))
        splits.append({"train": train, "val": val})
    pp = a.preprocessed / ds
    pp.mkdir(parents=True, exist_ok=True)
    (pp / "splits_final.json").write_text(json.dumps(splits, indent=1))

    print(f"-> {out}: {len(frames)} frames ({len(by_cell[None])} legacy), {shape[1]}x{shape[0]}")
    for r, s in zip(recs, splits):
        print(f"  round {r['fold']} (nnU-Net fold {int(r['fold']) - 1}): train {len(s['train'])}  "
              f"val {len(s['val'])} ({', '.join(r['cells']['val'])})  "
              f"test {len(tests[r['fold']]['stems'])} ({', '.join(r['cells']['test'])})")
    print(f"-> {pp / 'splits_final.json'}")


if __name__ == "__main__":
    main()
