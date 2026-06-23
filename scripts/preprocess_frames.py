#!/usr/bin/env python3
"""
Apply PreprocessingPipeline to annotated frames and save the outputs to disk.

This script is a **required step before U-Net training**.  Run it once after
annotation is complete.  The training script (train_unet.py) expects already-
preprocessed 256×256 PNGs and will not apply any preprocessing internally.

Produces up to three artefacts:
  1. Preprocessed frames  → data/preprocessed/frames/<pool>/  (256×256 uint8 PNG)
  2. (--masks) Resized masks → data/preprocessed/masks/<pool>/  (256×256 binary PNG,
     nearest-neighbour to preserve 0/255 values)
  3. (--compare) Pipeline stage figures → data/preprocessed/comparisons/<pool>/
     showing Raw → Denoise → CLAHE → Morphology → Final side-by-side

Recommended pre-training workflow
-----------------------------------
  1. Annotate:   python scripts/annotate.py
  2. Validate:   python scripts/validate_masks.py
  3. Preprocess: python scripts/preprocess_frames.py --masks
  4. Train:      python scripts/train_unet.py --config configs/default.yaml

Usage
-----
python scripts/preprocess_frames.py --masks
python scripts/preprocess_frames.py --masks --compare --n-compare 5
python scripts/preprocess_frames.py --pool 02_unet_holdout --masks
python scripts/preprocess_frames.py --image-dir data/frames/01_annotate_pool \\
                                     --output-dir data/preprocessed/frames/01_annotate_pool \\
                                     --mask-dir   data/masks/01_annotate_pool
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from celldform.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline

_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read {path}")
    return img


def _run_pipeline(img: np.ndarray, cfg: PreprocessingConfig) -> np.ndarray:
    """Return float32 [0,1] preprocessed image."""
    return PreprocessingPipeline(cfg)(img)


def _stage_outputs(img: np.ndarray, cfg: PreprocessingConfig) -> dict[str, np.ndarray]:
    """
    Run the pipeline incrementally and capture the output after each stage.
    Returns a dict of stage_name → image (uint8 or float32).
    """
    stages: dict[str, np.ndarray] = {"Raw": img.copy()}

    pipe = PreprocessingPipeline(cfg)

    after_denoise = pipe._denoise(img) if cfg.enabled.get("denoise") else img.copy()
    stages["Denoise"] = after_denoise

    after_clahe = pipe._enhance(after_denoise) if cfg.enabled.get("enhance") else after_denoise.copy()
    stages["CLAHE"] = after_clahe

    after_morph = pipe._morphology(after_clahe) if cfg.enabled.get("morphology") else after_clahe.copy()
    stages["Morphology"] = after_morph

    after_resize = pipe._resize(after_morph) if cfg.enabled.get("resize") else after_morph.copy()
    after_norm = pipe._normalize(after_resize) if cfg.enabled.get("normalize") else after_resize.astype(np.float32) / 255.0
    stages["Final\n(256×256)"] = after_norm

    return stages


def _save_comparison(
    stages: dict[str, np.ndarray],
    out_path: Path,
    image_name: str,
) -> None:
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle(image_name, fontsize=10, y=1.01)

    for ax, (title, img) in zip(axes, stages.items()):
        display = img if img.dtype == np.uint8 else (img * 255).astype(np.uint8)
        ax.imshow(display, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        h, w = img.shape[:2]
        ax.set_xlabel(f"{w}×{h}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# Maps each pool folder to its pipeline stage name
_POOL_STAGE = {
    "01_annotate_pool": "train",
    "02_unet_holdout":  "evaluate",
    "03_mask_factory":  "inference",
    "04_clf_arena":     "inference",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess annotated frames and optionally save pipeline figures"
    )
    parser.add_argument("--pool", default="01_annotate_pool",
                        help="Pool folder name under data/frames/")
    parser.add_argument("--stage", choices=["train", "evaluate", "inference"],
                        help="Pipeline stage (default: auto-detected from --pool)")
    parser.add_argument("--image-dir", type=Path,
                        help="Override source image directory")
    parser.add_argument("--output-dir", type=Path,
                        help="Override output directory for preprocessed frames")
    parser.add_argument("--compare", action="store_true",
                        help="Save pipeline stage comparison figures")
    parser.add_argument("--n-compare", type=int, default=5,
                        help="Number of images to generate comparison figures for (default: 5)")
    parser.add_argument("--compare-dir", type=Path,
                        help="Directory for comparison figures")
    parser.add_argument("--masks", action="store_true",
                        help="Also resize and save binary masks alongside inputs")
    parser.add_argument("--mask-dir", type=Path,
                        help="Source mask directory (default: data/masks/<pool>)")
    parser.add_argument("--mask-output-dir", type=Path,
                        help="Override destination for resized masks")
    args = parser.parse_args()

    stage = args.stage or _POOL_STAGE.get(args.pool)
    if stage is None:
        sys.exit(
            f"Cannot auto-detect stage for pool '{args.pool}'. "
            "Use --stage train|evaluate|inference"
        )

    base = Path("data/preprocessed") / stage / args.pool

    image_dir: Path = args.image_dir or (Path("data/frames") / args.pool)
    output_dir: Path = args.output_dir or (base / "inputs")
    mask_src_dir: Path = args.mask_dir or (Path("data/masks") / args.pool)
    mask_out_dir: Path = args.mask_output_dir or (base / "labels")
    compare_dir: Path = args.compare_dir or (Path("data/preprocessed/comparisons") / args.pool)

    if not image_dir.exists():
        sys.exit(f"Image directory not found: {image_dir}")

    all_images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _EXTS)
    if not all_images:
        sys.exit(f"No images found in {image_dir}")

    if args.masks:
        images = [p for p in all_images if (mask_src_dir / p.with_suffix(".png").name).exists()]
        n_skipped = len(all_images) - len(images)
    else:
        images = all_images
        n_skipped = 0

    if not images:
        sys.exit(f"No images with matching masks found in {mask_src_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = PreprocessingConfig()
    n_compare = min(args.n_compare, len(images)) if args.compare else 0

    print(f"Source frames: {image_dir}  ({len(all_images)} total, {n_skipped} skipped — no mask)")
    print(f"Output frames: {output_dir}")
    if args.masks:
        print(f"Source masks:  {mask_src_dir}")
        print(f"Output masks:  {mask_out_dir}")
    if args.compare:
        print(f"Figures:       {compare_dir}  ({n_compare} images)")
    print()

    if args.masks:
        mask_out_dir.mkdir(parents=True, exist_ok=True)

    target_size = (cfg.output_size[1], cfg.output_size[0])  # cv2 expects (W, H)
    n_masks_saved = 0

    for i, path in enumerate(images):
        img = _load_gray(path)

        preprocessed = _run_pipeline(img, cfg)
        uint8_out = (preprocessed * 255).astype(np.uint8)
        out_path = output_dir / path.with_suffix(".png").name
        cv2.imwrite(str(out_path), uint8_out)

        if args.masks:
            mask_src = mask_src_dir / path.with_suffix(".png").name
            if mask_src.exists():
                mask = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE)
                mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(mask_out_dir / mask_src.name), mask_resized)
                n_masks_saved += 1

        if args.compare and i < n_compare:
            stages = _stage_outputs(img, cfg)
            fig_path = compare_dir / f"{path.stem}_pipeline.png"
            _save_comparison(stages, fig_path, path.name)
            print(f"  [{i + 1}/{len(images)}] {path.name}  → frame + mask + figure")
        else:
            suffix = " + mask" if args.masks and (mask_src_dir / path.with_suffix(".png").name).exists() else ""
            print(f"  [{i + 1}/{len(images)}] {path.name}  → frame{suffix}")

    print(f"\nDone.  {len(images)} frames saved to {output_dir}" + (f"  ({n_skipped} skipped)" if n_skipped else ""))
    if args.masks:
        print(f"       {n_masks_saved} masks saved to {mask_out_dir}")
    if args.compare:
        print(f"       {n_compare} figures saved to {compare_dir}")


if __name__ == "__main__":
    main()
