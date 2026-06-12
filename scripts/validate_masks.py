#!/usr/bin/env python3
"""
Validate that every image in an annotation pool has a corresponding binary mask.

Checks performed per mask
--------------------------
1. Exists     — a PNG file is present alongside the source image
2. Readable   — the file is a valid image OpenCV can decode
3. Binary     — pixel values are strictly 0 or 255
4. Non-empty  — at least one cell pixel (255) is present

Exit code 0 if all masks pass; non-zero otherwise (suitable for CI / pre-train
gate scripts).

Usage
-----
python scripts/validate_masks.py                          # default: 01_annotate_pool
python scripts/validate_masks.py --pool 02_unet_holdout
python scripts/validate_masks.py --image-dir data/frames/01_annotate_pool \\
                                  --mask-dir  data/masks/01_annotate_pool
"""
             
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def validate(image_dir: Path, mask_dir: Path) -> dict:
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _EXTS)

    results: dict = {
        "total": len(images),
        "ok": 0,
        "missing": [],
        "unreadable": [],
        "not_binary": [],
        "empty": [],
    }

    for img_path in images:
        mask_path = mask_dir / img_path.with_suffix(".png").name

        if not mask_path.exists():
            results["missing"].append(img_path.name)
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            results["unreadable"].append(img_path.name)
            continue

        if not set(np.unique(mask)).issubset({0, 255}):
            results["not_binary"].append(img_path.name)
            continue

        if not np.any(mask == 255):
            results["empty"].append(img_path.name)
            continue

        results["ok"] += 1

    return results


def _print_list(label: str, names: list[str]) -> None:
    if not names:
        return
    print(f"\n{label} ({len(names)}):")
    for name in names:
        print(f"  - {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate annotation masks")
    parser.add_argument("--pool", default="01_annotate_pool",
                        help="Pool folder name under data/frames/ and data/masks/")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    args = parser.parse_args()

    if args.image_dir and args.mask_dir:
        image_dir: Path = args.image_dir
        mask_dir: Path = args.mask_dir
    else:
        image_dir = Path("data/frames") / args.pool
        mask_dir = Path("data/masks") / args.pool

    if not image_dir.exists():
        sys.exit(f"Image directory not found: {image_dir}")

    print(f"Images: {image_dir}")
    print(f"Masks:  {mask_dir}\n")

    r = validate(image_dir, mask_dir)

    pct = r["ok"] / r["total"] * 100 if r["total"] else 0.0
    print(f"Progress: {r['ok']}/{r['total']} valid  ({pct:.1f}%)")

    _print_list("Missing", r["missing"])
    _print_list("Unreadable", r["unreadable"])
    _print_list("Not binary (values other than 0/255)", r["not_binary"])
    _print_list("Empty (no cell pixels)", r["empty"])

    n_issues = sum(len(r[k]) for k in ("missing", "unreadable", "not_binary", "empty"))
    if n_issues == 0:
        print("\nAll masks valid.")
        sys.exit(0)
    else:
        print(f"\n{n_issues} issue(s) found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
