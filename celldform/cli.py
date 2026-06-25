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
                            "Bottom X% of channel pixel intensities treated as candidate cells. "
                            "Default: 20. Overrides config acquisition.dark_percentile."
                        ))
    parser.add_argument("--channel-width", type=float, default=None,
                        help=(
                            "Width of the vertical flow channel as a fraction of image width, "
                            "centred at the horizontal midpoint (e.g. 0.30 = ±15%% from centre). "
                            "Overrides config acquisition.channel_width."
                        ))
    parser.add_argument("--min-circularity", type=float, default=None,
                        help=(
                            "Minimum circularity (4π·area/perimeter², 0–1) for a blob to be "
                            "accepted as a cell. Default: 0.60. Overrides config acquisition.min_circularity."
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


# ---------------------------------------------------------------------------
# train_unet
# ---------------------------------------------------------------------------

def train_unet() -> None:
    """CLI entry point: train the U-Net segmentation model."""
    import importlib.util

    script = Path(__file__).parent.parent / "scripts" / "train_unet.py"
    if not script.exists():
        print(f"[ERROR] Training script not found: {script}", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("_train_unet_script", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


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


# ---------------------------------------------------------------------------
# organize_dataset
# ---------------------------------------------------------------------------

def organize_dataset() -> None:
    """CLI entry point: organise raw data into pipeline-ready frame folders."""
    from celldform.acquisition.organiser import main
    main()


def hpc_submit() -> None:
    """CLI entry point: generate and submit a SLURM training job on Anvil."""
    parser = argparse.ArgumentParser(
        prog="celldform-hpc-submit",
        description="Generate a SLURM batch script for U-Net training and submit it via sbatch.",
    )
    parser.add_argument("--account", required=True,
                        help="ACCESS/Anvil allocation account ID (required).")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config YAML path passed to celldform-train-unet (default: configs/default.yaml).")
    parser.add_argument("--partition", default="gpu",
                        help="SLURM partition (default: gpu).")
    parser.add_argument("--time", default="04:00:00",
                        help="Wall-clock time limit HH:MM:SS (default: 04:00:00).")
    parser.add_argument("--cpus", type=int, default=8,
                        help="CPUs per task (default: 8).")
    parser.add_argument("--gpus", type=int, default=1,
                        help="GPUs per node (default: 1).")
    parser.add_argument("--mem", default="32G",
                        help="Memory per node (default: 32G).")
    parser.add_argument("--script-out", default="scripts/hpc_train_unet.sh",
                        help="Where to write the generated SLURM script (default: scripts/hpc_train_unet.sh).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write the script but do not call sbatch.")
    args = parser.parse_args()

    script_path = Path(args.script_out)
    script_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    script_content = f"""\
#!/bin/bash
#SBATCH --job-name=celldform-unet
#SBATCH --partition={args.partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpus}
#SBATCH --gpus-per-node={args.gpus}
#SBATCH --mem={args.mem}
#SBATCH --time={args.time}
#SBATCH --account={args.account}
#SBATCH --output=logs/unet_%j.out
#SBATCH --error=logs/unet_%j.err

set -euo pipefail

echo "[celldform] Job started: $(date)"
echo "[celldform] Node: $SLURMD_NODENAME"
echo "[celldform] Job ID: $SLURM_JOB_ID"
echo ""

module load anaconda
conda activate celldform

# Verify GPU
python -c "import torch; print('[celldform] CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# Train
celldform-train-unet --config {args.config}

echo ""
echo "[celldform] Job finished: $(date)"
"""

    script_path.write_text(script_content)
    script_path.chmod(0o755)
    print(f"[celldform] Script written → {script_path}")

    if args.dry_run:
        print("[celldform] --dry-run: skipping sbatch submission.")
        print(f"[celldform] To submit manually: sbatch {script_path}")
        return

    import subprocess
    result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] sbatch failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    job_id = result.stdout.strip().split()[-1]
    print(f"[celldform] Submitted job {job_id}")
    print(f"[celldform] Monitor : squeue -u $USER")
    print(f"[celldform] Log     : tail -f logs/unet_{job_id}.out")
    print(f"[celldform] Cancel  : scancel {job_id}")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
