#!/usr/bin/env python3
"""
Validate the masks that are present in the annotation pool.

Checks performed per mask
--------------------------
1. Readable   — the file is a valid image OpenCV can decode
2. Binary     — pixel values are strictly 0 or 255
3. Non-empty  — at least one cell pixel (255) is present

Exit code 0 if all present masks pass; non-zero otherwise (suitable for CI /
pre-train gate scripts).

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


def validate(mask_dir: Path) -> dict:
    masks = sorted(p for p in mask_dir.iterdir() if p.suffix.lower() == ".png")

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

        if not set(np.unique(mask)).issubset({0, 255}):
            results["not_binary"].append(mask_path.name)
            continue

        if not np.any(mask == 255):
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
    args = parser.parse_args()

    mask_dir: Path = args.mask_dir or (Path("data/masks") / args.pool)

    if not mask_dir.exists():
        sys.exit(f"Mask directory not found: {mask_dir}")

    print(f"Masks: {mask_dir}\n")

    r = validate(mask_dir)

    pct = r["ok"] / r["total"] * 100 if r["total"] else 0.0
    print(f"Valid: {r['ok']}/{r['total']}  ({pct:.1f}%)")

    _print_list("Unreadable", r["unreadable"])
    _print_list("Not binary (values other than 0/255)", r["not_binary"])
    _print_list("Empty (no cell pixels)", r["empty"])

    n_issues = sum(len(r[k]) for k in ("unreadable", "not_binary", "empty"))
    if n_issues == 0:
        print("\nAll masks valid.")
        sys.exit(0)
    else:
        print(f"\n{n_issues} issue(s) found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
