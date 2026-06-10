#!/usr/bin/env python
"""
Data acquisition — extract frames from an optical tweezer video recording.

Usage:
    python scripts/extract_frames.py --video data/videos/experiment_01.mp4

    # override output directory
    python scripts/extract_frames.py --video experiment.mp4 --output-dir data/frames

    # keep every 5th frame
    python scripts/extract_frames.py --video experiment.mp4 --every-n 5

    # resample to a fixed frame rate
    python scripts/extract_frames.py --video experiment.mp4 --target-fps 10.0

    # stop after 200 frames (useful for quick sanity checks)
    python scripts/extract_frames.py --video experiment.mp4 --max-frames 200

Outputs (written to output_dir):
    frame_000000.png, frame_000001.png, ...   — extracted grayscale frames
    manifest.csv                               — index, timestamp, and path per frame
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from celldform.acquisition.extractor import FrameExtractor
from celldform.config import load as load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="extract-frames",
        description="Extract frames from an optical tweezer video recording.",
    )
    parser.add_argument(
        "--video", required=True,
        help="Path to the input video file (.mp4, .avi, …).",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write extracted frames (overrides config data.frames_dir).",
    )
    parser.add_argument(
        "--every-n", type=int, default=None,
        help="Keep every N-th frame (overrides config acquisition.every_n).",
    )
    parser.add_argument(
        "--target-fps", type=float, default=None,
        help="Resample to this frame rate in Hz (mutually exclusive with --every-n).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop after this many frames (overrides config acquisition.max_frames).",
    )
    parser.add_argument(
        "--target-size", type=int, nargs=2, metavar=("W", "H"), default=None,
        help="Resize each frame to W H pixels (overrides config preprocessing.target_size).",
    )
    parser.add_argument(
        "--format", default=None, choices=["png", "tiff"],
        help="Output image format (overrides config acquisition.image_format).",
    )
    parser.add_argument(
        "--change-threshold", type=float, default=None,
        help=(
            "Minimum trapped-cell centroid displacement (pixels) required to save a frame. "
            "A frame is skipped when the trapped cell has not moved this far from its "
            "position in the last saved frame. Overrides config acquisition.change_threshold."
        ),
    )
    parser.add_argument(
        "--dark-percentile", type=float, default=None,
        help=(
            "Bottom X% of channel pixel intensities treated as candidate cells. "
            "Default: 20. Overrides config acquisition.dark_percentile."
        ),
    )
    parser.add_argument(
        "--channel-width", type=float, default=None,
        help=(
            "Width of the vertical flow channel as a fraction of image width, "
            "centred at the horizontal midpoint (e.g. 0.30 = ±15%% from centre). "
            "Overrides config acquisition.channel_width."
        ),
    )
    parser.add_argument(
        "--min-circularity", type=float, default=None,
        help=(
            "Minimum circularity (4π·area/perimeter², 0–1) for a blob to be accepted "
            "as a cell. Default: 0.60. Overrides config acquisition.min_circularity."
        ),
    )
    args = parser.parse_args()

    if args.every_n is not None and args.target_fps is not None:
        parser.error("--every-n and --target-fps are mutually exclusive.")

    conf = load_config(args.config)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # CLI args override config values where provided.
    output_dir       = args.output_dir       or conf.data.frames_dir
    every_n          = args.every_n          if args.every_n          is not None else conf.acquisition.every_n
    target_fps       = args.target_fps       if args.target_fps       is not None else conf.acquisition.target_fps
    max_frames       = args.max_frames       if args.max_frames       is not None else conf.acquisition.max_frames
    image_format     = args.format           or conf.acquisition.image_format
    target_size      = tuple(args.target_size) if args.target_size    else conf.preprocessing.target_size
    change_threshold = args.change_threshold if args.change_threshold is not None else conf.acquisition.change_threshold
    dark_percentile  = args.dark_percentile  if args.dark_percentile  is not None else conf.acquisition.dark_percentile
    channel_width    = args.channel_width    if args.channel_width    is not None else conf.acquisition.channel_width
    min_circularity  = args.min_circularity  if args.min_circularity  is not None else conf.acquisition.min_circularity

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
        print(f"[celldform] Channel width    : {channel_width:.0%} of image width (centred)")
        print(f"[celldform] Dark percentile  : {dark_percentile}%  (candidate cell threshold)")
        print(f"[celldform] Min circularity  : {min_circularity}")
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
        channel_width=channel_width,
        min_circularity=min_circularity,
    )

    frames = extractor.extract(
        every_n=every_n if target_fps is None else None,
        target_fps=target_fps,
        max_frames=max_frames,
    )

    print(f"[celldform] Extracted {len(frames)} frames → {output_dir}")
    print(f"[celldform] Manifest  → {Path(output_dir) / (video_path.stem + '_manifest.csv')}")


if __name__ == "__main__":
    main()
