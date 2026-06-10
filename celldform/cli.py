"""
Console entry points registered in pyproject.toml:

  celldform-train-unet  → train_unet()
  celldform-train-cnn   → train_cnn()
  celldform-infer       → infer()

Run each with --help for usage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
# train_cnn
# ---------------------------------------------------------------------------

def train_cnn() -> None:
    """CLI entry point: train MegaNet for deformation regression."""
    parser = argparse.ArgumentParser(
        prog="celldform-train-cnn",
        description="Train MegaNet CNN for deformation parameter regression.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-outputs", type=int, default=None,
                        help="Number of regression targets.")
    args = parser.parse_args()

    import yaml

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    train_cfg = cfg.get("training", {})
    epochs = args.epochs or train_cfg.get("epochs", 100)
    n_out = args.n_outputs or cfg.get("meganet", {}).get("n_outputs", 1)
    device = args.device or cfg.get("device", None)

    print(f"[celldform] Training MegaNet  epochs={epochs}  n_outputs={n_out}  device={device}")
    print("            See scripts/train_cnn.py for a complete example.")


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
