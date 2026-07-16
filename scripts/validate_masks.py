#!/usr/bin/env python3
"""
Validate the masks that are present in the annotation pool.

Checks performed per mask
--------------------------
1. Readable   — the file is a valid image OpenCV can decode
2. Binary     — pixel values are strictly 0 or 255 (--multiclass: 0, 1, or 2)
3. Non-empty  — at least one cell pixel (255) is present
                (--multiclass: at least one trapped-cell pixel, label 1)

Exit code 0 if all present masks pass; non-zero otherwise (suitable for CI /
pre-train gate scripts).

Usage
-----
python scripts/validate_masks.py                          # default: 01_annotate_pool
python scripts/validate_masks.py --pool 02_unet_holdout
python scripts/validate_masks.py --image-dir data/frames/01_annotate_pool \\
                                  --mask-dir  data/masks/01_annotate_pool
python scripts/validate_masks.py --multiclass \\
                                  --mask-dir  data/masks/01_annotate_pool_multiclass
"""
             
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def validate(mask_dir: Path, multiclass: bool = False) -> dict:
    masks = sorted(p for p in mask_dir.iterdir() if p.suffix.lower() == ".png")
    valid_values = {0, 1, 2} if multiclass else {0, 255}
    fg_value = 1 if multiclass else 255

    results: dict = {
        "total": len(masks),
        "ok": 0,
        "unreadable": [],
        "not_binary": [],
        "empty": [],
    }

    for mask_path in masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            results["unreadable"].append(mask_path.name)
            continue

        if not set(np.unique(mask)).issubset(valid_values):
            results["not_binary"].append(mask_path.name)
            continue

        if not np.any(mask == fg_value):
            results["empty"].append(mask_path.name)
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
                        help="Pool folder name under data/masks/")
    parser.add_argument("--mask-dir", type=Path,
                        help="Override mask directory")
    parser.add_argument("--multiclass", action="store_true",
                        help="Validate 3-class masks (0/1/2) instead of binary (0/255)")
    args = parser.parse_args()

    default_pool = f"{args.pool}_multiclass" if args.multiclass else args.pool
    mask_dir: Path = args.mask_dir or (Path("data/masks") / default_pool)

    if not mask_dir.exists():
        sys.exit(f"Mask directory not found: {mask_dir}")

    print(f"Masks: {mask_dir}\n")

    r = validate(mask_dir, args.multiclass)

    pct = r["ok"] / r["total"] * 100 if r["total"] else 0.0
    print(f"Valid: {r['ok']}/{r['total']}  ({pct:.1f}%)")

    label_desc = "values other than 0/1/2" if args.multiclass else "values other than 0/255"
    empty_desc = "no trapped-cell pixels (label 1)" if args.multiclass else "no cell pixels"
    _print_list("Unreadable", r["unreadable"])
    _print_list(f"Not binary ({label_desc})", r["not_binary"])
    _print_list(f"Empty ({empty_desc})", r["empty"])

    n_issues = sum(len(r[k]) for k in ("unreadable", "not_binary", "empty"))
    if n_issues == 0:
        print("\nAll masks valid.")
        sys.exit(0)
    else:
        print(f"\n{n_issues} issue(s) found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
