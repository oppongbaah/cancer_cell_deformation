#!/usr/bin/env python3
"""
Validate the masks that are present in the annotation pool.

Checks performed per mask
--------------------------
1. Present    — every image in the frame pool has a corresponding mask file
2. Readable   — the file is a valid image OpenCV can decode
3. Binary     — pixel values are strictly 0 or 255 (--multiclass: 0, 1, or 2)
4. Non-empty  — at least one foreground pixel is present (255 for binary;
                label 1 or label 2 for --multiclass)
5. Has trapped cell — (--multiclass only) at least one label-1 pixel is
                present. Every frame in these experiments contains a trapped
                cell, so a mask with only decoy objects (label 2) means the
                trapped cell was missed or mislabeled. For binary masks this
                is the same as check 4, since foreground = trapped cell.

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
python scripts/annotate_frame.py --multiclass domain_high_cell8_000000 
"""
             
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def validate(mask_dir: Path, multiclass: bool = False, image_dir: Path | None = None) -> dict:
    masks = sorted(p for p in mask_dir.iterdir() if p.suffix.lower() == ".png")
    valid_values = {0, 1, 2} if multiclass else {0, 255}

    results: dict = {
        "total": len(masks),
        "ok": 0,
        "missing": [],
        "unreadable": [],
        "not_binary": [],
        "empty": [],
        "no_trapped_cell": [],
    }

    if image_dir is not None:
        image_stems = sorted(
            p.stem for p in image_dir.iterdir() if p.suffix.lower() in _EXTS
        )
        mask_stems = {p.stem for p in masks}
        results["missing"] = [
            stem + ".png" for stem in image_stems if stem not in mask_stems
        ]
        results["total"] = len(image_stems)

    for mask_path in masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            results["unreadable"].append(mask_path.name)
            continue

        values = set(np.unique(mask))
        if not values.issubset(valid_values):
            results["not_binary"].append(mask_path.name)
            continue

        if values == {0}:
            results["empty"].append(mask_path.name)
            continue

        if multiclass and 1 not in values:
            results["no_trapped_cell"].append(mask_path.name)
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
                        help="Pool folder name under data/masks/ (and data/frames/)")
    parser.add_argument("--mask-dir", type=Path,
                        help="Override mask directory")
    parser.add_argument("--image-dir", type=Path,
                        help="Override frame directory (checked for missing masks); "
                             "pass --no-completeness to skip this check entirely")
    parser.add_argument("--no-completeness", action="store_true",
                        help="Skip checking that every pool image has a mask")
    parser.add_argument("--multiclass", action="store_true",
                        help="Validate 3-class masks (0/1/2) instead of binary (0/255)")
    args = parser.parse_args()

    default_pool = f"{args.pool}_multiclass" if args.multiclass else args.pool
    mask_dir: Path = args.mask_dir or (Path("data/masks") / default_pool)
    image_dir: Path | None = None
    if not args.no_completeness:
        image_dir = args.image_dir or (Path("data/frames") / args.pool)

    if not mask_dir.exists():
        sys.exit(f"Mask directory not found: {mask_dir}")
    if image_dir is not None and not image_dir.exists():
        sys.exit(f"Image directory not found: {image_dir} (pass --no-completeness to skip)")

    print(f"Masks:  {mask_dir}")
    if image_dir is not None:
        print(f"Frames: {image_dir}")
    print()

    r = validate(mask_dir, args.multiclass, image_dir)

    pct = r["ok"] / r["total"] * 100 if r["total"] else 0.0
    print(f"Valid: {r['ok']}/{r['total']}  ({pct:.1f}%)")

    label_desc = "values other than 0/1/2" if args.multiclass else "values other than 0/255"
    empty_desc = "completely blank — no label 1 or 2 pixels" if args.multiclass else "no cell pixels"
    _print_list("Missing (no mask file)", r["missing"])
    _print_list("Unreadable", r["unreadable"])
    _print_list(f"Not binary ({label_desc})", r["not_binary"])
    _print_list(f"Empty ({empty_desc})", r["empty"])
    _print_list("No trapped cell (label 2 only — trapped cell missed or mislabeled)",
                r["no_trapped_cell"])

    n_issues = sum(
        len(r[k]) for k in ("missing", "unreadable", "not_binary", "empty", "no_trapped_cell")
    )
    if n_issues == 0:
        print("\nAll masks valid.")
        sys.exit(0)
    else:
        print(f"\n{n_issues} issue(s) found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
