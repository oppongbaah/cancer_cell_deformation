#!/usr/bin/env python3
"""
Look at a teacher target (.npy) from the distillation data: the frame, its label, and the teacher's
per-class probabilities as heat maps.

    python scripts/report/view_soft_target.py --round 4                       # a random training frame
    python scripts/report/view_soft_target.py --round 4 --stem high_cell7_002525
    python scripts/report/view_soft_target.py --round 4 --n 6 --output soft.png   # a grid of 6

Each .npy in data/preprocessed/randswap/experiment_3_distill/fold_<k>/train/soft/ is a float16 array of
shape (3, 384, 384): the probability nnU-Net (the teacher) gives each pixel for background, trapped
cell and decoy, over the same 384 crop the student sees. Panels: input | label (hand-drawn for annotated
frames, the teacher's argmax for teacher-labelled ones; green = cell, red = decoy) | p(cell) | p(decoy).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("data/preprocessed/randswap/experiment_3_distill")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--round", required=True)
    ap.add_argument("--stem", nargs="*")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--output", type=Path, default=Path("results/randswap/experiment_3_distill/pilot/soft_targets.png"))
    a = ap.parse_args()
    d = DATA / f"fold_{a.round}"
    rec = json.loads((d / "split.json").read_text())
    extra = set(rec["extra_stems"])
    stems = a.stem or random.Random(0).sample(sorted(p.stem for p in (d / "train" / "soft").glob("*.npy")), a.n)

    fig, axes = plt.subplots(len(stems), 4, figsize=(12, 3.1 * len(stems)), squeeze=False)
    for row, s in zip(axes, stems):
        soft = np.load(d / "train" / "soft" / f"{s}.npy").astype(np.float32)
        img = cv2.imread(str(d / "train" / "inputs" / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(str(d / "train" / "labels" / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        ov = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        ov[lab == 1] = (0.45 * ov[lab == 1] + [0, 140, 0]).astype(np.uint8)
        ov[lab == 2] = (0.45 * ov[lab == 2] + [140, 0, 0]).astype(np.uint8)
        src = "teacher label" if s in extra else "hand-drawn label"
        for ax, im, title, kw in ((row[0], img, s, dict(cmap="gray")), (row[1], ov, src, {}),
                                  (row[2], soft[1], "teacher p(cell)", dict(cmap="viridis", vmin=0, vmax=1)),
                                  (row[3], soft[2], "teacher p(decoy)", dict(cmap="viridis", vmin=0, vmax=1))):
            h = ax.imshow(im, **kw)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.colorbar(h, ax=row[3], fraction=0.046)
        print(f"{s}: shape {soft.shape}, dtype float16, p sums to {soft.sum(0).mean():.3f}, "
              f"cell pixels p>0.5: {(soft[1] > 0.5).sum()}, max p(cell) {soft[1].max():.3f}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.output, dpi=110)
    print(f"-> {a.output}")


if __name__ == "__main__":
    main()
