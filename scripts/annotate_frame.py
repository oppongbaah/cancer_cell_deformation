#!/usr/bin/env python3
"""
Annotate a single frame in napari.

Opens one image with a raw greyscale layer, a CLAHE-enhanced preview layer
(toggle visibility in the layer list), and an empty Labels layer to paint on.

Keybindings
-----------
S   Save mask and close
Q   Quit without saving

The mask is written as a uint8 PNG with pixel values 0 (background) / 255 (cell).
If the output path already contains a mask, it is pre-loaded so you can correct it.

Usage
-----
python scripts/annotate_frame.py data/frames/01_annotate_pool/frame_0000.jpg
python scripts/annotate_frame.py data/frames/01_annotate_pool/frame_0000.jpg \\
                                 --mask-out data/masks/01_annotate_pool/frame_0000.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import napari
import numpy as np


def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read {path}")
    return img


def _clahe_preview(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def _default_mask_path(image_path: Path) -> Path:
    # mirrors the pool convention: data/frames/<pool>/x.jpg → data/masks/<pool>/x.png
    parts = image_path.parts
    try:
        frames_idx = parts.index("frames")
        pool = parts[frames_idx + 1]
        return Path("data/masks") / pool / image_path.with_suffix(".png").name
    except (ValueError, IndexError):
        return image_path.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate a single frame in napari")
    parser.add_argument("image", type=Path, help="Path to the image file to annotate")
    parser.add_argument(
        "--mask-out",
        type=Path,
        default=None,
        help="Where to save the mask PNG (default: mirrors data/masks/<pool>/ structure)",
    )
    args = parser.parse_args()

    image_path: Path = args.image
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    mask_out: Path = args.mask_out or _default_mask_path(image_path)

    img = _load_gray(image_path)
    enhanced = _clahe_preview(img)

    # Pre-load existing mask so corrections are easy
    if mask_out.exists():
        existing = cv2.imread(str(mask_out), cv2.IMREAD_GRAYSCALE)
        labels = (existing > 0).astype(np.int32) if existing is not None else np.zeros(img.shape, dtype=np.int32)
        print(f"Pre-loaded existing mask from {mask_out}")
    else:
        labels = np.zeros(img.shape, dtype=np.int32)

    print(f"Image:    {image_path}")
    print(f"Mask out: {mask_out}")
    print("\nKeybindings:  S = save & close   Q = quit without saving\n")

    saved = {"done": False}

    viewer = napari.Viewer(title=f"celldform annotator — {image_path.name}")
    viewer.add_image(img, name="raw", colormap="gray")
    viewer.add_image(enhanced, name="enhanced", colormap="gray", visible=False)
    viewer.add_labels(labels, name="mask", opacity=0.5)

    @viewer.bind_key("s")
    def _save_and_close(v: napari.Viewer) -> None:
        binary = (v.layers["mask"].data > 0).astype(np.uint8) * 255
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_out), binary)
        saved["done"] = True
        print(f"Saved → {mask_out}")
        v.close()

    @viewer.bind_key("q")
    def _quit(v: napari.Viewer) -> None:
        print("Quit without saving.")
        v.close()

    napari.run()

    if not saved["done"]:
        print("No mask saved.")


if __name__ == "__main__":
    main()
