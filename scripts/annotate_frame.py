#!/usr/bin/env python3
"""
Annotate or correct specific frames in napari, one after the other.

Unlike scripts/annotate.py (which walks a whole pool), this opens exactly the
frames you name — full paths or bare names resolved against the pool — making
it the tool for fixing masks flagged by scripts/validate_masks.py. Each frame
shows a raw greyscale layer, a CLAHE-enhanced preview layer (toggle visibility
in the layer list), and a Labels layer to paint on.

Keybindings
-----------
S   Save mask and advance to the next frame (closes after the last)
N   Skip (no save) and advance
B   Go back to the previous frame
Q   Quit the session

Binary mode (default): masks are written as uint8 PNG, 0 (background) /
255 (cell); an existing mask is pre-loaded binarised.

Multi-class mode (--multiclass): label 1 = trapped cell (green), label 2 =
other/decoy object (red), 0 = background. Masks default to
data/masks/<pool>_multiclass/ and are written with raw label values (0/1/2),
not binarised. Existing masks are pre-loaded with their raw values, and any
stray labels outside 0/1/2 are rendered magenta so they're easy to spot and
erase; saving while strays remain prints a warning.

Usage
-----
python scripts/annotate_frame.py data/frames/01_annotate_pool/frame_0000.jpg
python scripts/annotate_frame.py --multiclass \\
    domain_high_cell8_000000 domain_low_cell3_000000 domain_low_cell4_000000
python scripts/annotate_frame.py frame_0000.jpg --mask-out custom/path.png
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

_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_VALID_MULTICLASS = {0, 1, 2}

# Same fixed mapping as scripts/annotate.py, except unknown labels render
# magenta instead of transparent — this script's job is correcting flagged
# masks, and invisible stray labels (e.g. an accidental 3 or 253) are exactly
# what needs to be found and erased.
_MULTICLASS_COLORMAP = DirectLabelColormap(
    color_dict={0: "transparent", 1: "green", 2: "red", None: "magenta"},
)


def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read {path}")
    return img


def _clahe_preview(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def _resolve_image(arg: str, pool_dir: Path) -> Path:
    """Resolve a CLI argument to an image path: as given, or against the pool."""
    p = Path(arg)
    if p.exists():
        return p
    candidates = [pool_dir / arg] if p.suffix else [
        (pool_dir / arg).with_suffix(ext) for ext in _EXTS
    ]
    for c in candidates:
        if c.exists():
            return c
    sys.exit(f"Image not found: {arg} (also tried under {pool_dir})")


def _default_mask_path(image_path: Path, multiclass: bool) -> Path:
    # mirrors the pool convention: data/frames/<pool>/x.jpg → data/masks/<pool>/x.png
    parts = image_path.parts
    try:
        frames_idx = parts.index("frames")
        pool = parts[frames_idx + 1] + ("_multiclass" if multiclass else "")
        return Path("data/masks") / pool / image_path.with_suffix(".png").name
    except (ValueError, IndexError):
        return image_path.with_suffix(".png")


def _load_existing_or_blank(img_shape: tuple, mask_path: Path, multiclass: bool) -> np.ndarray:
    if mask_path.exists():
        raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            print(f"  Pre-loaded existing mask from {mask_path}")
            if multiclass:
                stray = sorted(int(v) for v in set(np.unique(raw)) - _VALID_MULTICLASS)
                if stray:
                    print(f"  ⚠ Stray label values {stray} present — shown in magenta; erase them.")
                return raw.astype(np.int32)
            return (raw > 127).astype(np.int32)
    return np.zeros(img_shape, dtype=np.int32)


def _save_mask(labels: np.ndarray, path: Path, multiclass: bool) -> None:
    if multiclass:
        out = labels.astype(np.uint8)
        stray = sorted(int(v) for v in set(np.unique(out)) - _VALID_MULTICLASS)
        if stray:
            print(f"  ⚠ Saved mask still contains stray label values {stray} — "
                  f"validate_masks.py will flag it.")
    else:
        out = (labels > 0).astype(np.uint8) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate/correct specific frames in napari, one after the other"
    )
    parser.add_argument("images", nargs="+",
                        help="Image files to annotate — full paths, or bare names "
                             "resolved against data/frames/<pool>/")
    parser.add_argument("--pool", default="01_annotate_pool",
                        help="Pool folder used to resolve bare names and default mask paths")
    parser.add_argument("--multiclass", action="store_true",
                        help="Edit 3-class masks (0=background, 1=trapped cell, 2=other/decoy); "
                             "saves raw label values to data/masks/<pool>_multiclass/")
    parser.add_argument("--mask-out", type=Path, default=None,
                        help="Explicit output mask path (single image only; default mirrors "
                             "data/masks/<pool>[_multiclass]/ structure)")
    args = parser.parse_args()

    if args.mask_out is not None and len(args.images) > 1:
        sys.exit("--mask-out only makes sense with a single image; "
                 "multiple images use the data/masks/<pool>/ convention.")

    pool_dir = Path("data/frames") / args.pool
    images = [_resolve_image(a, pool_dir) for a in args.images]
    mask_paths = [
        args.mask_out or _default_mask_path(p, args.multiclass) for p in images
    ]

    print(f"Frames to edit: {len(images)}")
    print("Keybindings:  S = save & next   N = skip   B = previous   Q = quit\n")
    if args.multiclass:
        print("Multi-class mode: label 1 = trapped cell (green), label 2 = other/decoy (red).")
        print("Stray labels outside 0/1/2 render magenta — erase them before saving.\n")

    state = {"idx": 0, "saved": 0, "skipped": 0}
    viewer = napari.Viewer(title="celldform frame annotator")

    def _show(idx: int) -> None:
        path = images[idx]
        img = _load_gray(path)
        enhanced = _clahe_preview(img)
        print(f"[{idx + 1}/{len(images)}] {path.name}")
        mask = _load_existing_or_blank(img.shape, mask_paths[idx], args.multiclass)
        if "raw" in viewer.layers:
            viewer.layers["raw"].data = img
            viewer.layers["enhanced"].data = enhanced
            viewer.layers["mask"].data = mask
        else:
            viewer.add_image(img, name="raw", colormap="gray")
            viewer.add_image(enhanced, name="enhanced", colormap="gray", visible=False)
            viewer.add_labels(
                mask, name="mask", opacity=0.5,
                colormap=_MULTICLASS_COLORMAP if args.multiclass else None,
            )
        status = "  [saved]" if mask_paths[idx].exists() else ""
        viewer.title = f"celldform frame annotator  [{idx + 1}/{len(images)}]  {path.name}{status}"

    def _advance(delta: int) -> None:
        nxt = state["idx"] + delta
        if nxt < 0:
            print("Already at the first frame.")
            return
        if nxt >= len(images):
            print(f"\nAll frames processed.  Saved: {state['saved']}, Skipped: {state['skipped']}")
            viewer.close()
            return
        state["idx"] = nxt
        _show(nxt)

    _show(0)

    @viewer.bind_key("s")
    def _save_and_next(v: napari.Viewer) -> None:
        idx = state["idx"]
        _save_mask(v.layers["mask"].data, mask_paths[idx], args.multiclass)
        state["saved"] += 1
        print(f"  Saved → {mask_paths[idx]}")
        _advance(+1)

    @viewer.bind_key("n")
    def _skip(v: napari.Viewer) -> None:
        print(f"  Skipped {images[state['idx']].name}")
        state["skipped"] += 1
        _advance(+1)

    @viewer.bind_key("b")
    def _prev(v: napari.Viewer) -> None:
        _advance(-1)

    # napari's Labels layer binds "B" to "toggle preserve labels" by default,
    # shadowing the viewer-level binding while the mask layer is active —
    # reclaim it (same workaround as scripts/annotate.py).
    @viewer.layers["mask"].bind_key("b", overwrite=True)
    def _prev_from_mask_layer(layer) -> None:
        _advance(-1)

    @viewer.bind_key("q")
    def _quit(v: napari.Viewer) -> None:
        print("Quit.")
        v.close()

    napari.run()
    print(f"\nSession ended.  Saved: {state['saved']}, Skipped: {state['skipped']}")


if __name__ == "__main__":
    main()
