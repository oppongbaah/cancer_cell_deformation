#!/usr/bin/env python
"""
Train U-Net for cancer cell segmentation.

Preprocessing is NOT applied here — run preprocess_frames.py --masks first to
produce the 256×256 PNGs that this script expects.

Pre-training workflow:
    1. python scripts/annotate.py
    2. python scripts/validate_masks.py
    3. python scripts/preprocess_frames.py --masks
    4. python scripts/train_unet.py --config configs/default.yaml

Usage (local):
    python scripts/train_unet.py --config configs/default.yaml

Usage (HPC / Anvil):
    sbatch scripts/hpc_train_unet.sh

The script expects two directories configured in the YAML:
    data.frames_dir — preprocessed 256×256 grayscale PNG frames
    data.masks_dir  — corresponding 256×256 binary PNG masks (same basename)

File pairs are matched by filename stem.  The dataset is split into
train / val / test according to data.val_split and data.test_split.
The best checkpoint (by val DSC) and a config snapshot are written to
the checkpoint directory for full reproducibility.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from celldform.config import load as load_config
from celldform.segmentation.trainer import SegmentationTrainer
from celldform.segmentation.unet import UNet
from celldform.utils.visualization import Visualizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CellDataset(Dataset):
    """Loads pre-processed (image, mask) pairs from paired directories.

    Expects frames that have already been processed by preprocess_frames.py:
    256×256 uint8 PNGs.  No preprocessing is applied here.

    When augment=True the same random spatial transform is applied to both
    the frame and the mask so they remain aligned:
      - horizontal flip (p=0.5)
      - vertical flip   (p=0.5)
      - 90-degree rotation: 0 / 90 / 180 / 270 (uniform)
    Intensity augmentation is intentionally omitted — CLAHE in Stage 2 already
    normalises contrast, so brightness jitter would undo that work.
    """

    def __init__(self, frame_paths, mask_paths, augment: bool = False) -> None:
        assert len(frame_paths) == len(mask_paths)
        self.frame_paths = frame_paths
        self.mask_paths = mask_paths
        self.augment = augment

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        import cv2

        frame = cv2.imread(str(self.frame_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        frame_f = frame.astype(np.float32) / 255.0     # normalise to [0, 1]
        mask_bin = (mask > 127).astype(np.float32)     # binarise to 0.0 / 1.0

        if self.augment:
            if random.random() < 0.5:
                frame_f = np.fliplr(frame_f).copy()
                mask_bin = np.fliplr(mask_bin).copy()
            if random.random() < 0.5:
                frame_f = np.flipud(frame_f).copy()
                mask_bin = np.flipud(mask_bin).copy()
            k = random.randint(0, 3)
            if k:
                frame_f = np.rot90(frame_f, k).copy()
                mask_bin = np.rot90(mask_bin, k).copy()

        frame_t = torch.from_numpy(frame_f[np.newaxis])    # (1, H, W)
        mask_t = torch.from_numpy(mask_bin[np.newaxis])
        return frame_t, mask_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train U-Net segmentation model.")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config value).")
    args = parser.parse_args()

    conf = load_config(args.config)

    seed = args.seed if args.seed is not None else conf.training.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ── Data ────────────────────────────────────────────────────────────────
    frames_dir = Path(conf.data.frames_dir)
    masks_dir = Path(conf.data.masks_dir)

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Frames directory not found: {frames_dir}\n"
            "Run: python scripts/preprocess_frames.py --masks"
        )
    if not masks_dir.exists():
        raise FileNotFoundError(
            f"Masks directory not found: {masks_dir}\n"
            "Run: python scripts/preprocess_frames.py --masks"
        )

    frame_paths = sorted(frames_dir.glob("*.png"))
    mask_paths = [masks_dir / p.name for p in frame_paths]

    pairs = [(f, m) for f, m in zip(frame_paths, mask_paths) if m.exists()]
    if not pairs:
        raise FileNotFoundError(
            f"No matched frame/mask pairs found in {frames_dir} / {masks_dir}."
        )

    random.shuffle(pairs)
    n = len(pairs)
    n_test = int(n * conf.data.test_split)
    n_val = int(n * conf.data.val_split)

    test_pairs = pairs[:n_test]
    val_pairs = pairs[n_test:n_test + n_val]
    train_pairs = pairs[n_test + n_val:]

    print(
        f"[celldform] Dataset split — "
        f"train: {len(train_pairs)}  val: {len(val_pairs)}  test: {len(test_pairs)}"
    )

    on_cuda = torch.cuda.is_available()
    n_workers = conf.training.num_workers

    def make_loader(pairs, shuffle, augment=False):
        fs, ms = zip(*pairs)
        ds = CellDataset(list(fs), list(ms), augment=augment)
        return DataLoader(
            ds,
            batch_size=conf.training.batch_size,
            shuffle=shuffle,
            num_workers=n_workers,
            pin_memory=on_cuda,
            persistent_workers=(n_workers > 0),
            prefetch_factor=(4 if n_workers > 0 else None),
        )

    train_loader = make_loader(train_pairs, shuffle=True, augment=True)
    val_loader = make_loader(val_pairs, shuffle=False, augment=False)

    # ── Model ────────────────────────────────────────────────────────────────
    unet = UNet(
        in_channels=conf.unet.in_channels,
        base_features=conf.unet.base_features,
    )

    trainer = SegmentationTrainer(
        model=unet,
        train_loader=train_loader,
        val_loader=val_loader,
        conf=conf,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    history = trainer.fit()

    # ── Final evaluation ─────────────────────────────────────────────────────
    metrics = trainer.evaluate()
    print("\nFinal validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Plot training curves ──────────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    existing = sorted(results_dir.glob("unet_training_curves_v*.png"))
    next_version = len(existing) + 1
    curve_path = results_dir / f"unet_training_curves_v{next_version}.png"

    viz = Visualizer()
    viz.plot_training_curves(history, save_path=str(curve_path))
    print(f"Training curves saved to {curve_path}")


if __name__ == "__main__":
    main()
