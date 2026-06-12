#!/usr/bin/env python3
"""
Interactive cell annotation tool — napari-based.

Opens images from an annotation pool one at a time.  Two image layers are
available: the raw greyscale frame and a CLAHE-enhanced preview (toggle
visibility in the layer list) to help identify cell boundaries.  Paint the
cell body on the 'mask' Labels layer, then press a key to save and continue.

Keybindings
-----------
S   Save current mask as a binary PNG and advance to the next image
N   Skip (no mask saved) and advance
P   Go back to the previous image

Masks are written as uint8 PNGs with pixel values 0 (background) / 255 (cell).
Already-annotated images are skipped automatically; use --redo to revisit them.

Usage
-----
python scripts/annotate.py
python scripts/annotate.py --image-dir data/frames/01_annotate_pool \\
                           --mask-dir  data/masks/01_annotate_pool
python scripts/annotate.py --redo
python scripts/annotate.py --pool 02_unet_holdout   # annotate holdout set
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import napari
import numpy as np

_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read {path}")
    return img


def _clahe_preview(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def _save_mask(labels: np.ndarray, path: Path) -> None:
    binary = (labels > 0).astype(np.uint8) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), binary)


def _update_viewer(viewer: napari.Viewer, path: Path, idx: int, total: int) -> None:
    img = _load_gray(path)
    enhanced = _clahe_preview(img)
    blank = np.zeros(img.shape, dtype=np.int32)

    if "raw" in viewer.layers:
        viewer.layers["raw"].data = img
        viewer.layers["enhanced"].data = enhanced
        viewer.layers["mask"].data = blank
    else:
        viewer.add_image(img, name="raw", colormap="gray")
        viewer.add_image(enhanced, name="enhanced", colormap="gray", visible=False)
        viewer.add_labels(blank, name="mask", opacity=0.5)

    viewer.title = f"celldform annotator  [{idx + 1}/{total}]  {path.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="napari cell annotation tool")
    parser.add_argument("--pool", default="01_annotate_pool",
                        help="Pool folder name under data/frames/ and data/masks/")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--redo", action="store_true",
                        help="Re-annotate images that already have masks")
    args = parser.parse_args()

    if args.image_dir and args.mask_dir:
        image_dir: Path = args.image_dir
        mask_dir: Path = args.mask_dir
    else:
        image_dir = Path("data/frames") / args.pool
        mask_dir = Path("data/masks") / args.pool

    if not image_dir.exists():
        sys.exit(f"Image directory not found: {image_dir}")

    all_images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _EXTS)
    if not all_images:
        sys.exit(f"No images found in {image_dir}")

    if args.redo:
        todo = all_images
    else:
        todo = [p for p in all_images
                if not (mask_dir / p.with_suffix(".png").name).exists()]

    if not todo:
        print("All images already annotated.  Use --redo to re-annotate.")
        return

    already_done = len(all_images) - len(todo)
    print(f"Pool:       {image_dir}")
    print(f"Masks:      {mask_dir}")
    print(f"To annotate: {len(todo)} / {len(all_images)}  "
          f"({already_done} already done)")
    print("\nKeybindings:  S = save & next   N = skip   P = previous\n")

    state = {"idx": 0, "saved": 0, "skipped": 0}

    viewer = napari.Viewer(title="celldform annotator")
    _update_viewer(viewer, todo[0], 0, len(todo))

    def _advance(delta: int) -> None:
        nxt = state["idx"] + delta
        if nxt < 0:
            print("Already at the first image.")
            return
        if nxt >= len(todo):
            print(f"\nAll images processed.  "
                  f"Saved: {state['saved']}, Skipped: {state['skipped']}")
            viewer.close()
            return
        state["idx"] = nxt
        _update_viewer(viewer, todo[nxt], nxt, len(todo))

    @viewer.bind_key("s")
    def _save_and_next(viewer: napari.Viewer) -> None:
        idx = state["idx"]
        out = mask_dir / todo[idx].with_suffix(".png").name
        _save_mask(viewer.layers["mask"].data, out)
        state["saved"] += 1
        print(f"  [{state['idx'] + 1}/{len(todo)}] Saved  → {out}")
        _advance(+1)

    @viewer.bind_key("n")
    def _skip(viewer: napari.Viewer) -> None:
        print(f"  [{state['idx'] + 1}/{len(todo)}] Skipped  {todo[state['idx']].name}")
        state["skipped"] += 1
        _advance(+1)

    @viewer.bind_key("p")
    def _prev(viewer: napari.Viewer) -> None:
        _advance(-1)

    napari.run()
    print(f"\nSession ended.  Saved: {state['saved']}, Skipped: {state['skipped']}")


if __name__ == "__main__":
    main()
