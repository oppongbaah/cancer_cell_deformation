#!/usr/bin/env python3
"""
Colorized masks — render saved masks as real color PNGs, viewable
directly in any image viewer (no napari needed).

Multiclass masks (label ids 0/1/2) look almost pure black in a plain image
viewer, since a pixel value of 1 or 2 out of 0-255 is visually indistinguishable
from background — see scripts/annotation/annotate.py's docstring for why. This renders
them as a visible colorized mask instead: background (label 0) = black,
label 1 (trapped cell) = green, label 2 (other/decoy object) = red. Legacy
binary masks (0/255) also work — the foreground is rendered green.

The raw frame is never loaded or blended in — this is the mask alone,
color-coded, not an overlay on top of the source image.

Usage
-----
python scripts/annotation/preview_masks.py                                    # 01_annotate_pool_multiclass
python scripts/annotation/preview_masks.py --pool 01_annotate_pool            # binary masks
python scripts/annotation/preview_masks.py --mask-dir data/masks/01_annotate_pool_multiclass \\
                                 --output-dir data/masks/01_annotate_pool_multiclass_preview
python scripts/annotation/preview_masks.py --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# BGR colors (OpenCV convention)
_LABEL_COLORS = {
    1: (0, 200, 0),    # trapped cell -> green
    2: (0, 0, 220),    # other/decoy object -> red
}
_BINARY_COLOR = (0, 200, 0)  # legacy 0/255 masks -> green foreground


def _load_gray(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _colorize(mask: np.ndarray) -> np.ndarray:
    """Color-code a label mask onto a black background; returns a BGR image."""
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)

    if set(np.unique(mask).tolist()).issubset({0, 255}):
        colored[mask > 127] = _BINARY_COLOR
    else:
        for label, color in _LABEL_COLORS.items():
            colored[mask == label] = color

    return colored


def main() -> None:
    parser = argparse.ArgumentParser(description="Render colorized mask previews (black background)")
    parser.add_argument("--pool", default="01_annotate_pool_multiclass",
                        help="Pool folder name under data/masks/")
    parser.add_argument("--mask-dir", type=Path, help="Override mask directory")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--limit", type=int, default=None, help="Only preview the first N masks")
    args = parser.parse_args()

    mask_dir: Path = args.mask_dir or (Path("data/masks") / args.pool)
    output_dir: Path = args.output_dir or (Path("data/masks") / f"{args.pool}_preview")

    if not mask_dir.exists():
        sys.exit(f"Mask directory not found: {mask_dir}")

    masks = sorted(p for p in mask_dir.iterdir() if p.suffix.lower() == ".png")
    if args.limit:
        masks = masks[:args.limit]
    if not masks:
        sys.exit(f"No masks found in {mask_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Masks:  {mask_dir}  ({len(masks)})")
    print(f"Output: {output_dir}\n")

    n_written = 0
    for mask_path in masks:
        mask = _load_gray(mask_path)
        preview = _colorize(mask)
        out_path = output_dir / mask_path.name
        cv2.imwrite(str(out_path), preview)
        n_written += 1
        print(f"  {mask_path.name} -> {out_path}")

    print(f"\nDone. {n_written} preview(s) written to {output_dir}")
    print("Legend: black = background, green = trapped cell (label 1), "
          "red = other/decoy object (label 2)")


if __name__ == "__main__":
    main()
