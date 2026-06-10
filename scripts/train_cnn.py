#!/usr/bin/env python
"""
Train MegaNet CNN for deformation parameter regression.

Usage:
    python scripts/train_cnn.py --config configs/default.yaml

Expects:
    data/frames/      — preprocessed PNG frames
    data/features.csv — feature CSV produced by MorphologyExtractor.batch()
                        with column headers matching the extractor's _FEATURE_NAMES.

The script regresses the 'deformation_index' column by default (n_outputs=1).
Set meganet.n_outputs > 1 in the config to regress multiple targets.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from celldform.models.meganet import MegaNet, MegaNetTrainer
from celldform.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from celldform.utils.visualization import Visualizer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DeformationDataset(Dataset):
    """Pairs preprocessed frames with scalar regression targets from a CSV."""

    def __init__(self, frame_paths, targets, preprocessor: PreprocessingPipeline) -> None:
        self.frame_paths = list(frame_paths)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        import cv2

        frame = cv2.imread(str(self.frame_paths[idx]), cv2.IMREAD_GRAYSCALE)
        proc = self.preprocessor(frame)
        frame_t = torch.from_numpy(proc[np.newaxis])       # (1, H, W)
        target_t = torch.tensor(self.targets[idx])         # scalar or vector
        return frame_t, target_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MegaNet deformation regression.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    frames_dir = Path(cfg["data"]["frames_dir"])
    features_csv = Path(cfg["data"]["features_csv"])

    if not features_csv.exists():
        raise FileNotFoundError(
            f"Features CSV not found: {features_csv}. "
            "Run MorphologyExtractor.batch() first."
        )

    feat_df = pd.read_csv(features_csv, index_col=0)
    target_col = cfg["biomechanics"]["metric"]   # e.g. "deformation_index"
    n_outputs = cfg["meganet"]["n_outputs"]

    if n_outputs == 1:
        targets = feat_df[target_col].values
    else:
        # Take the first n_outputs feature columns.
        target_cols = feat_df.columns[:n_outputs].tolist()
        targets = feat_df[target_cols].values

    frame_paths = [frames_dir / f"{idx}.png" for idx in feat_df.index]
    valid = [i for i, p in enumerate(frame_paths) if p.exists()]
    frame_paths = [frame_paths[i] for i in valid]
    targets = targets[valid]

    random.seed(args.seed)
    indices = list(range(len(frame_paths)))
    random.shuffle(indices)
    split = int(len(indices) * (1 - cfg["training"]["val_split"]))
    tr_idx, va_idx = indices[:split], indices[split:]

    pre_cfg = PreprocessingConfig(output_size=tuple(cfg["preprocessing"]["target_size"]))
    preprocessor = PreprocessingPipeline(pre_cfg)

    train_ds = DeformationDataset([frame_paths[i] for i in tr_idx],
                                  targets[tr_idx], preprocessor)
    val_ds = DeformationDataset([frame_paths[i] for i in va_idx],
                                targets[va_idx], preprocessor)

    batch = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=nw)

    # ── Model + training ──────────────────────────────────────────────────────
    model = MegaNet(
        in_channels=cfg["meganet"]["in_channels"],
        n_outputs=n_outputs,
        dropout_p=cfg["meganet"]["dropout_p"],
    )

    trainer = MegaNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg["training"]["lr"],
        checkpoint_dir=cfg["checkpoint_dir"],
        device=cfg.get("device"),
    )

    history = trainer.fit(epochs=cfg["training"]["epochs"])

    viz = Visualizer()
    viz.plot_regression_curves(history, save_path="meganet_training_curves.png")
    print("Training curves saved to meganet_training_curves.png")


if __name__ == "__main__":
    main()
