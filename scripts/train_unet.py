#!/usr/bin/env python
"""
Train U-Net for cancer cell segmentation.

Usage (local):
    python scripts/train_unet.py --config configs/default.yaml

Usage (HPC / Anvil):
    sbatch scripts/hpc_train_unet.sh   # wraps this script with SLURM directives

The script expects two directories (configured in default.yaml):
    data/frames/  — preprocessed grayscale PNG frames (256×256)
    data/masks/   — corresponding binary PNG masks (same basename)

File pairs are matched by filename stem.  A random 80/20 train/val split
is applied.  The best checkpoint (by val DSC) is written to checkpoints/.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from celldform.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from celldform.segmentation.trainer import SegmentationTrainer
from celldform.segmentation.unet import UNet
from celldform.utils.visualization import Visualizer


# ---------------------------------------------------------------------------
# Minimal dataset
# ---------------------------------------------------------------------------

class CellDataset(Dataset):
    """Loads (image, mask) pairs from paired frame/mask directories."""

    def __init__(self, frame_paths, mask_paths, preprocessor: PreprocessingPipeline) -> None:
        assert len(frame_paths) == len(mask_paths)
        self.frame_paths = frame_paths
        self.mask_paths = mask_paths
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        import cv2

        frame = cv2.imread(str(self.frame_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        frame_proc = self.preprocessor(frame)   # float32 (H, W)
        mask_bin = (mask > 127).astype(np.float32)

        # Add channel dimension: (1, H, W)
        frame_t = torch.from_numpy(frame_proc[np.newaxis])
        mask_t = torch.from_numpy(mask_bin[np.newaxis])
        return frame_t, mask_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train U-Net segmentation model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Data ────────────────────────────────────────────────────────────────
    frames_dir = Path(cfg["data"]["frames_dir"])
    masks_dir = Path(cfg["data"]["masks_dir"])

    frame_paths = sorted(frames_dir.glob("*.png"))
    mask_paths = [masks_dir / p.name for p in frame_paths]

    # Drop pairs where the mask does not exist.
    pairs = [(f, m) for f, m in zip(frame_paths, mask_paths) if m.exists()]
    if not pairs:
        raise FileNotFoundError(
            f"No matched frame/mask pairs found in {frames_dir} / {masks_dir}."
        )

    random.shuffle(pairs)
    split = int(len(pairs) * (1 - cfg["training"]["val_split"]))
    train_pairs, val_pairs = pairs[:split], pairs[split:]

    pre_cfg = PreprocessingConfig(
        output_size=tuple(cfg["preprocessing"]["target_size"]),
        median_kernel=cfg["preprocessing"]["median_kernel"],
        clahe_clip_limit=cfg["preprocessing"]["clahe_clip_limit"],
    )
    preprocessor = PreprocessingPipeline(pre_cfg)

    train_ds = CellDataset(*zip(*train_pairs), preprocessor)
    val_ds = CellDataset(*zip(*val_pairs), preprocessor)

    batch = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=nw)

    # ── Model ────────────────────────────────────────────────────────────────
    unet = UNet(
        in_channels=cfg["unet"]["in_channels"],
        base_features=cfg["unet"]["base_features"],
    )

    trainer = SegmentationTrainer(
        model=unet,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg["training"]["lr"],
        checkpoint_dir=cfg["checkpoint_dir"],
        device=cfg.get("device"),
    )

    # ── Train ────────────────────────────────────────────────────────────────
    history = trainer.fit(epochs=cfg["training"]["epochs"])

    # ── Final evaluation ─────────────────────────────────────────────────────
    metrics = trainer.evaluate()
    print("\nFinal validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Plot training curves ──────────────────────────────────────────────────
    viz = Visualizer()
    fig = viz.plot_training_curves(history, save_path="unet_training_curves.png")
    print("Training curves saved to unet_training_curves.png")


if __name__ == "__main__":
    main()
