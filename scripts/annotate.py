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
B   Go back to the previous image (reclaimed from napari's default "toggle
    preserve labels" shortcut on the mask layer)
C   Return to the current frame — the last one reached via S/N — after
    browsing away with N/B (a no-op if you're already there)
P   Toggle paint mode on the mask layer (napari's default binding — left
    alone; use the mask layer's mode buttons for erase/fill/pick)

Masks are written as uint8 PNGs with pixel values 0 (background) / 255 (cell).
Every image in the pool is always loaded, in order — including ones you've
already annotated — so N/B move freely through the whole set for review or
correction. The viewer opens on the first not-yet-annotated image (e.g.
6/240 if the first 5 are already done), not the start of the pool. Revisiting
an already-annotated frame loads its saved mask into the Labels layer (not
blank), and the window title shows "[saved]" for frames that already have a
mask on disk.

Multi-class mode (--multiclass): label 1 = trapped cell, label 2 = other/decoy
object (untrapped cell, debris, lookalike blob), 0 = background. Masks are
written with raw label values (0/1/2), not binarized. Each frame is seeded
 from its existing binary mask in data/masks/<pool>/ (if present) as label 1,
so you only need to paint decoy objects — not redraw the trapped cell.
Switch the active paint label via the "label" spinbox in napari's Labels
layer controls panel, or with "-" / "=" to step the current label down/up,
before painting a different class.

Usage
-----
python scripts/annotate.py
python scripts/annotate.py --image-dir data/frames/01_annotate_pool \\
                           --mask-dir  data/masks/01_annotate_pool
python scripts/annotate.py --pool 02_unet_holdout   # annotate holdout set
python scripts/annotate.py --multiclass             # 3-class experiment (same 40 images)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if sys.platform == "linux" and not Path("/dev/dri").exists():
    # WSL2 without GPU passthrough has no DRM render nodes, so EGL's
    # hardware path (dri2/zink) fails. GLX still works via Mesa's
    # llvmpipe software renderer, so force Qt onto that path.
    os.environ.setdefault("QT_XCB_GL_INTEGRATION", "xcb_glx")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import cv2
import napari
import numpy as np
from napari.utils.colormaps import DirectLabelColormap

_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Fixed label->color mapping for --multiclass mode. Without this, napari's default
# Labels colormap is auto-generated and can reassign colors when the layer's .data is
# swapped out for a new array on every frame change (as _update_viewer does, rather
# than creating a fresh Labels layer each time) — observed as label 1 (trapped cell)
# and label 2 (other/decoy) sometimes rendering as the same color after navigating
# with B/N. A DirectLabelColormap is a plain fixed dict, immune to that.
_MULTICLASS_COLORMAP = DirectLabelColormap(
    color_dict={0: "transparent", 1: "green", 2: "red", None: "transparent"},
)


def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read {path}")
    return img


def _clahe_preview(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def _save_mask(labels: np.ndarray, path: Path, multiclass: bool = False) -> None:
    if multiclass:
        out = labels.astype(np.uint8)
    else:
        out = (labels > 0).astype(np.uint8) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)


def _load_existing_or_blank(
    img_shape: tuple,
    mask_path: Path,
    seed_path: Path | None = None,
    multiclass: bool = False,
) -> np.ndarray:
    """Load a previously saved mask for editing, seed from another mask, or start blank.

    Priority: existing mask at ``mask_path`` > seed mask at ``seed_path`` > blank.
    Without this, navigating back to an already-annotated image would silently
    discard whatever was painted before.
    """
    if mask_path.exists():
        raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            return raw.astype(np.int32) if multiclass else (raw > 127).astype(np.int32)

    if seed_path is not None and seed_path.exists():
        raw = cv2.imread(str(seed_path), cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            # Seed source is always a binary mask — becomes label 1 (trapped cell).
            return (raw > 127).astype(np.int32)

    return np.zeros(img_shape, dtype=np.int32)


def _update_viewer(
    viewer: napari.Viewer,
    path: Path,
    idx: int,
    total: int,
    mask_dir: Path,
    seed_dir: Path | None = None,
    multiclass: bool = False,
) -> None:
    img = _load_gray(path)
    enhanced = _clahe_preview(img)
    mask_path = mask_dir / path.with_suffix(".png").name
    seed_path = (seed_dir / path.with_suffix(".png").name) if seed_dir is not None else None
    mask = _load_existing_or_blank(img.shape, mask_path, seed_path, multiclass)

    if "raw" in viewer.layers:
        viewer.layers["raw"].data = img
        viewer.layers["enhanced"].data = enhanced
        viewer.layers["mask"].data = mask
    else:
        viewer.add_image(img, name="raw", colormap="gray")
        viewer.add_image(enhanced, name="enhanced", colormap="gray", visible=False)
        label_colormap = _MULTICLASS_COLORMAP if multiclass else None
        viewer.add_labels(mask, name="mask", opacity=0.5, colormap=label_colormap)

    status = "  [saved]" if mask_path.exists() else ""
    viewer.title = f"celldform annotator  [{idx + 1}/{total}]  {path.name}{status}"


def main() -> None:
    parser = argparse.ArgumentParser(description="napari cell annotation tool")
    parser.add_argument("--pool", default="01_annotate_pool",
                        help="Pool folder name under data/frames/ and data/masks/")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--multiclass", action="store_true",
                        help="Annotate 3 classes (0=background, 1=trapped cell, "
                             "2=other/decoy object) instead of binary. Masks default "
                             "to data/masks/<pool>_multiclass/ and are seeded from the "
                             "existing binary mask (label 1) when available.")
    args = parser.parse_args()

    if args.image_dir and args.mask_dir:
        image_dir: Path = args.image_dir
        mask_dir: Path = args.mask_dir
        seed_dir: Path | None = None
    else:
        image_dir = Path("data/frames") / args.pool
        binary_mask_dir = Path("data/masks") / args.pool
        if args.multiclass:
            mask_dir = Path("data/masks") / f"{args.pool}_multiclass"
            seed_dir = binary_mask_dir
        else:
            mask_dir = binary_mask_dir
            seed_dir = None

    if not image_dir.exists():
        sys.exit(f"Image directory not found: {image_dir}")

    all_images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _EXTS)
    if not all_images:
        sys.exit(f"No images found in {image_dir}")

    # Always load the full pool, in order, so N/B can freely browse and
    # correct earlier annotations rather than only stepping through
    # not-yet-annotated frames.
    todo = all_images

    def _has_mask(p: Path) -> bool:
        return (mask_dir / p.with_suffix(".png").name).exists()

    already_done = sum(1 for p in all_images if _has_mask(p))
    start_idx = next((i for i, p in enumerate(all_images) if not _has_mask(p)), 0)

    print(f"Pool:       {image_dir}")
    print(f"Masks:      {mask_dir}")
    if seed_dir is not None:
        print(f"Seed from:  {seed_dir}  (label 1, when no multiclass mask exists yet)")
    print(f"Images:     {len(all_images)}  ({already_done} already annotated — shown as [saved])")
    if already_done < len(all_images):
        print(f"Starting at the first unannotated image: {start_idx + 1}/{len(all_images)}")
    else:
        print("All images already annotated — starting from the beginning.")
    print("\nKeybindings:  S = save & next   N = skip   B = previous   "
          "C = return to current   P = toggle paint\n")
    if args.multiclass:
        print("Multi-class mode: label 1 = trapped cell, label 2 = other/decoy object.")
        print('Switch labels with the "label" spinbox in the Labels layer controls, or "-"/"=".\n')

    state = {"idx": start_idx, "anchor": start_idx, "saved": 0, "skipped": 0}

    viewer = napari.Viewer(title="celldform annotator")
    _update_viewer(viewer, todo[start_idx], start_idx, len(todo), mask_dir, seed_dir, args.multiclass)

    def _advance(delta: int, update_anchor: bool = True) -> None:
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
        if update_anchor:
            state["anchor"] = nxt
        _update_viewer(viewer, todo[nxt], nxt, len(todo), mask_dir, seed_dir, args.multiclass)

    @viewer.bind_key("s")
    def _save_and_next(viewer: napari.Viewer) -> None:
        idx = state["idx"]
        out = mask_dir / todo[idx].with_suffix(".png").name
        _save_mask(viewer.layers["mask"].data, out, args.multiclass)
        state["saved"] += 1
        print(f"  [{state['idx'] + 1}/{len(todo)}] Saved  → {out}")
        _advance(+1)

    @viewer.bind_key("n")
    def _skip(viewer: napari.Viewer) -> None:
        print(f"  [{state['idx'] + 1}/{len(todo)}] Skipped  {todo[state['idx']].name}")
        state["skipped"] += 1
        _advance(+1)

    @viewer.bind_key("b")
    def _prev(viewer: napari.Viewer) -> None:
        _advance(-1, update_anchor=False)

    # napari's Labels layer binds "B" to "toggle preserve labels" by default
    # (napari.utils.shortcuts), which shadows the viewer-level "b" above
    # whenever the mask layer — the one you're painting on — is active. Reclaim
    # "B" on that layer specifically for "previous". "P" is left as napari's
    # default paint-mode toggle.
    @viewer.layers["mask"].bind_key("b", overwrite=True)
    def _prev_from_mask_layer(layer) -> None:
        _advance(-1, update_anchor=False)

    @viewer.bind_key("c")
    def _return_to_current(viewer: napari.Viewer) -> None:
        anchor = state["anchor"]
        if state["idx"] == anchor:
            print("Already at the current frame.")
            return
        state["idx"] = anchor
        _update_viewer(viewer, todo[anchor], anchor, len(todo), mask_dir, seed_dir, args.multiclass)

    napari.run()
    print(f"\nSession ended.  Saved: {state['saved']}, Skipped: {state['skipped']}")


if __name__ == "__main__":
    main()
