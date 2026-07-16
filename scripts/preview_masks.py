#!/usr/bin/env python3
"""
Colorized mask preview — render saved masks overlaid on their raw frames as
real color PNGs, viewable directly in any image viewer (no napari needed).

Multiclass masks (label ids 0/1/2) look almost pure black in a plain image
viewer, since a pixel value of 1 or 2 out of 0-255 is visually indistinguishable
from background — see scripts/annotate.py's docstring for why. This renders
them as a visible overlay instead: label 1 (trapped cell) = green, label 2
(other/decoy object) = red. Legacy binary masks (0/255) also work — the
foreground is rendered green.

Usage
-----
python scripts/preview_masks.py                                    # 01_annotate_pool_multiclass
python scripts/preview_masks.py --pool 01_annotate_pool            # binary masks
python scripts/preview_masks.py --mask-dir data/masks/01_annotate_pool_multiclass \\
                                 --image-dir data/frames/01_annotate_pool \\
                                 --output-dir data/masks/01_annotate_pool_multiclass_preview
python scripts/preview_masks.py --alpha 0.5 --limit 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# BGR colors (OpenCV convention)
_LABEL_COLORS = {
    1: (0, 200, 0),    # trapped cell -> green
    2: (0, 0, 220),    # other/decoy object -> red
}
_BINARY_COLOR = (0, 200, 0)  # legacy 0/255 masks -> green foreground


def _load_gray(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _find_frame(image_dir: Path, stem: str) -> Path | None:
    for ext in _EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _overlay(frame: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend a color-coded mask onto a grayscale frame; returns a BGR image."""
    base = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    color_layer = np.zeros_like(base)

    if set(np.unique(mask).tolist()).issubset({0, 255}):
        color_layer[mask > 127] = _BINARY_COLOR
    else:
        for label, color in _LABEL_COLORS.items():
            color_layer[mask == label] = color

    painted = np.any(color_layer != 0, axis=2)
    blended_full = cv2.addWeighted(base, 1 - alpha, color_layer, alpha, 0)
    blended = base.copy()
    blended[painted] = blended_full[painted]
    return blended


def main() -> None:
    parser = argparse.ArgumentParser(description="Render colorized mask-on-frame previews")
    parser.add_argument("--pool", default="01_annotate_pool_multiclass",
                        help="Pool folder name under data/masks/ "
                             "(raw frame pool is inferred by stripping '_multiclass')")
    parser.add_argument("--image-dir", type=Path, help="Override raw frame directory")
    parser.add_argument("--mask-dir", type=Path, help="Override mask directory")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay blend strength (0-1)")
    parser.add_argument("--limit", type=int, default=None, help="Only preview the first N masks")
    args = parser.parse_args()

    mask_dir: Path = args.mask_dir or (Path("data/masks") / args.pool)
    image_pool = args.pool.removesuffix("_multiclass")
    image_dir: Path = args.image_dir or (Path("data/frames") / image_pool)
    output_dir: Path = args.output_dir or (Path("data/masks") / f"{args.pool}_preview")

    if not mask_dir.exists():
        sys.exit(f"Mask directory not found: {mask_dir}")
    if not image_dir.exists():
        sys.exit(f"Image directory not found: {image_dir}")

    masks = sorted(p for p in mask_dir.iterdir() if p.suffix.lower() == ".png")
    if args.limit:
        masks = masks[:args.limit]
    if not masks:
        sys.exit(f"No masks found in {mask_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Masks:  {mask_dir}  ({len(masks)})")
    print(f"Frames: {image_dir}")
    print(f"Output: {output_dir}\n")

    n_written, n_missing_frame = 0, 0
    for mask_path in masks:
        frame_path = _find_frame(image_dir, mask_path.stem)
        frame = _load_gray(frame_path) if frame_path else None
        mask = _load_gray(mask_path)

        if frame is None:
            n_missing_frame += 1
            print(f"  [skip] no matching frame for {mask_path.name}")
            continue

        preview = _overlay(frame, mask, args.alpha)
        out_path = output_dir / mask_path.name
        cv2.imwrite(str(out_path), preview)
        n_written += 1
        print(f"  {mask_path.name} -> {out_path}")

    print(f"\nDone. {n_written} preview(s) written"
          + (f", {n_missing_frame} skipped (no matching frame)" if n_missing_frame else "")
          + f" to {output_dir}")
    print("Legend: green = trapped cell (label 1), red = other/decoy object (label 2)")


if __name__ == "__main__":
    main()
