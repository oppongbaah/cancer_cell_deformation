#!/usr/bin/env python3
"""
Evaluate a trained U-Net on the holdout set and report segmentation metrics.

Run this AFTER:
  1. U-Net training is complete (best checkpoint saved)
  2. 02_unet_holdout images have been annotated
  3. preprocess_frames.py --masks --pool 02_unet_holdout has been run

The DSC reported here is the honest, thesis-reportable score — these cells
were never seen during training or hyperparameter tuning.

Usage
-----
python scripts/evaluate_unet.py --checkpoint checkpoints/unet_best.pt
python scripts/evaluate_unet.py --checkpoint checkpoints/unet_best.pt \\
                                 --pool 02_unet_holdout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from celldform.config import load as load_config
from celldform.segmentation.trainer import SegmentationTrainer
from celldform.segmentation.unet import UNet

# Reuse CellDataset from train_unet
sys.path.insert(0, str(Path(__file__).parent))
from train_unet import CellDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate U-Net on holdout set")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--pool", default="02_unet_holdout",
                        help="Pool folder name under data/preprocessed/ (default: 02_unet_holdout)")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    conf = load_config(str(args.config))

    inputs_dir = Path("data/preprocessed/evaluate") / args.pool / "inputs"
    labels_dir = Path("data/preprocessed/evaluate") / args.pool / "labels"

    if not inputs_dir.exists():
        sys.exit(
            f"Preprocessed inputs not found: {inputs_dir}\n"
            f"Run: python scripts/preprocess_frames.py --masks --pool {args.pool}"
        )
    if not labels_dir.exists():
        sys.exit(
            f"Preprocessed labels not found: {labels_dir}\n"
            f"Run: python scripts/preprocess_frames.py --masks --pool {args.pool}"
        )

    frame_paths = sorted(inputs_dir.glob("*.png"))
    mask_paths = [labels_dir / p.name for p in frame_paths]
    pairs = [(f, m) for f, m in zip(frame_paths, mask_paths) if m.exists()]

    if not pairs:
        sys.exit(f"No matched frame/mask pairs found in {inputs_dir} / {labels_dir}")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Pool:       {args.pool}  ({len(pairs)} images)\n")

    device = torch.device(
        conf.device if conf.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    unet = UNet(in_channels=conf.unet.in_channels, base_features=conf.unet.base_features)
    state = torch.load(str(args.checkpoint), map_location=device)
    unet.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    unet.to(device)

    fs, ms = zip(*pairs)
    holdout_loader = DataLoader(
        CellDataset(list(fs), list(ms)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=conf.training.num_workers,
    )

    trainer = SegmentationTrainer(
        model=unet,
        train_loader=holdout_loader,
        val_loader=holdout_loader,
        conf=conf,
    )

    metrics = trainer.evaluate(loader=holdout_loader)

    print("Holdout evaluation results")
    print("-" * 30)
    for name, value in metrics.items():
        print(f"  {name:<12} {value:.4f}")


if __name__ == "__main__":
    main()
