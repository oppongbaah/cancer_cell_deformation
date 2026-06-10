"""
Console entry points registered in pyproject.toml:

  celldform-extract-frames  → extract_frames()
  celldform-train-unet      → train_unet()
  celldform-infer           → infer()

Run each with --help for usage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# extract_frames
# ---------------------------------------------------------------------------

def extract_frames() -> None:
    """CLI entry point: extract frames from an optical tweezer video."""
    parser = argparse.ArgumentParser(
        prog="celldform-extract-frames",
        description="Extract frames from an optical tweezer video recording.",
    )
    parser.add_argument("--video", required=True,
                        help="Path to input video file (.mp4, .avi, …).")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write frames (overrides config data.frames_dir).")
    parser.add_argument("--every-n", type=int, default=None,
                        help="Keep every N-th frame (overrides config).")
    parser.add_argument("--target-fps", type=float, default=None,
                        help="Resample to this frame rate (mutually exclusive with --every-n).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this many frames.")
    parser.add_argument("--target-size", type=int, nargs=2, metavar=("W", "H"), default=None,
                        help="Resize each frame to W H pixels (overrides config).")
    parser.add_argument("--format", default=None, choices=["png", "tiff"],
                        help="Output image format (overrides config).")
    parser.add_argument("--change-threshold", type=float, default=None,
                        help=(
                            "Minimum trapped-cell centroid displacement (pixels) required to save "
                            "a frame. A frame is skipped when the trapped cell has not moved this "
                            "far from its position in the last saved frame. "
                            "Overrides config acquisition.change_threshold."
                        ))
    parser.add_argument("--dark-percentile", type=float, default=None,
                        help=(
                            "Pixels at or below this intensity percentile (0–100) are treated as "
                            "the trapped cell region — the trapped cell appears darker than "
                            "surrounding cells. Default: 20. Overrides config acquisition.dark_percentile."
                        ))
    args = parser.parse_args()

    if args.every_n is not None and args.target_fps is not None:
        parser.error("--every-n and --target-fps are mutually exclusive.")

    from celldform.acquisition.extractor import FrameExtractor
    from celldform.config import load as load_config

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    conf = load_config(args.config)

    output_dir       = args.output_dir    or conf.data.frames_dir
    every_n          = args.every_n       if args.every_n          is not None else conf.acquisition.every_n
    target_fps       = args.target_fps    if args.target_fps        is not None else conf.acquisition.target_fps
    max_frames       = args.max_frames    if args.max_frames        is not None else conf.acquisition.max_frames
    image_format     = args.format        or conf.acquisition.image_format
    target_size      = tuple(args.target_size) if args.target_size else conf.preprocessing.target_size
    change_threshold = args.change_threshold if args.change_threshold is not None else conf.acquisition.change_threshold
    dark_percentile  = args.dark_percentile  if args.dark_percentile  is not None else conf.acquisition.dark_percentile

    print(f"[celldform] Video            : {video_path}")
    print(f"[celldform] Output dir       : {output_dir}")
    print(f"[celldform] Target size      : {target_size[0]}×{target_size[1]}")
    print(f"[celldform] Format           : {image_format}")
    if target_fps:
        print(f"[celldform] Sampling         : {target_fps} fps")
    else:
        print(f"[celldform] Sampling         : every {every_n} frame(s)")
    if change_threshold is not None:
        print(f"[celldform] Change threshold : {change_threshold} px  (trapped-cell centroid displacement)")
        print(f"[celldform] Dark percentile  : {dark_percentile}%  (intensity cutoff for cell detection)")
    else:
        print(f"[celldform] Change threshold : disabled (saving all candidates)")
    if max_frames:
        print(f"[celldform] Max frames       : {max_frames}")
    print()

    extractor = FrameExtractor(
        video_path=video_path,
        output_dir=output_dir,
        target_size=target_size,
        image_format=image_format,
        change_threshold=change_threshold,
        dark_percentile=dark_percentile,
    )

    frames = extractor.extract(
        every_n=every_n if target_fps is None else None,
        target_fps=target_fps,
        max_frames=max_frames,
    )

    print(f"[celldform] Extracted {len(frames)} frames → {output_dir}")
    print(f"[celldform] Manifest  → {Path(output_dir) / (video_path.stem + '_manifest.csv')}")


# ---------------------------------------------------------------------------
# train_unet
# ---------------------------------------------------------------------------

def train_unet() -> None:
    """CLI entry point: train the U-Net segmentation model."""
    parser = argparse.ArgumentParser(
        prog="celldform-train-unet",
        description="Train U-Net for cancer cell segmentation.",
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs from config.")
    parser.add_argument("--device", default=None,
                        help="Compute device: 'cuda' or 'cpu'.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Override checkpoint output directory.")
    args = parser.parse_args()

    import yaml

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    train_cfg = cfg.get("training", {})
    epochs = args.epochs or train_cfg.get("epochs", 50)
    device = args.device or cfg.get("device", None)
    ckpt_dir = args.checkpoint_dir or cfg.get("checkpoint_dir", "checkpoints")

    print(f"[celldform] Training U-Net  epochs={epochs}  device={device}  ckpt={ckpt_dir}")
    print("[celldform] Initialise your DataLoaders and call SegmentationTrainer.fit().")
    print("            See scripts/train_unet.py for a complete example.")


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

def infer() -> None:
    """CLI entry point: run inference on a video file or image directory."""
    parser = argparse.ArgumentParser(
        prog="celldform-infer",
        description="Run end-to-end inference on optical tweezers video data.",
    )
    parser.add_argument("input", help="Path to input video (.mp4/.avi) or image directory.")
    parser.add_argument("--unet-ckpt", required=True,
                        help="Path to trained U-Net checkpoint (.pt).")
    parser.add_argument("--classifier-ckpt", default=None,
                        help="Path to trained classifier checkpoint (.pkl).")
    parser.add_argument("--output-dir", default="results",
                        help="Directory to write inference results.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--every-n", type=int, default=1,
                        help="Process every N-th frame.")
    args = parser.parse_args()

    from celldform.segmentation.unet import UNet

    device = args.device or ("cuda" if _cuda_available() else "cpu")
    print(f"[celldform] Loading U-Net from {args.unet_ckpt} on {device} ...")

    model = UNet.from_checkpoint(args.unet_ckpt, device=device)
    print(f"[celldform] Running inference on {args.input} ...")
    print(f"            Results will be saved to {args.output_dir}")
    print("            See scripts/infer.py for a complete example.")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
