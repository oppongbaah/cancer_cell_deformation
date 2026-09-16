#!/usr/bin/env python3
"""
Evaluate a trained U-Net on the holdout set and report segmentation metrics.

Run this AFTER:
  1. U-Net training is complete (best checkpoint saved)
  2. 02_unet_holdout images have been annotated
  3. preprocess_frames.py --masks --pool 02_unet_holdout has been run

The DSC reported here is the honest, thesis-reportable score — these cells
were never seen during training or hyperparameter tuning.

Every run appends its metrics as one JSON line to a results log (default
results/holdout_eval_log.jsonl) — printing to stdout alone lost every prior
sweep's numbers the moment the terminal scrolled past them. Pass
--no-save-results to skip logging, or --results-log to point elsewhere.

Usage
-----
python scripts/evaluate_unet.py --checkpoint checkpoints/unet_best.pt
python scripts/evaluate_unet.py --checkpoint checkpoints/unet_best.pt \\
                                 --pool 02_unet_holdout
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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
    parser.add_argument("--filter", default=None,
                        help="Only evaluate frames whose filename stem starts with this prefix "
                             "(e.g. --filter low_cell6 to isolate one cell from a multi-cell pool)")
    parser.add_argument("--results-log", type=Path, default=Path("results/holdout_eval_log.jsonl"),
                        help="JSONL file to append this run's metrics to")
    parser.add_argument("--no-save-results", action="store_true",
                        help="Skip writing to --results-log")
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
    if args.filter:
        frame_paths = [p for p in frame_paths if p.stem.startswith(args.filter)]
    mask_paths = [labels_dir / p.name for p in frame_paths]
    pairs = [(f, m) for f, m in zip(frame_paths, mask_paths) if m.exists()]

    if not pairs:
        filter_msg = f" matching --filter '{args.filter}'" if args.filter else ""
        sys.exit(f"No matched frame/mask pairs found in {inputs_dir} / {labels_dir}{filter_msg}")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Pool:       {args.pool}" + (f"  (filter: {args.filter})" if args.filter else "") +
          f"  ({len(pairs)} images)\n")

    device = torch.device(
        conf.device if conf.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    multiclass = conf.unet.out_channels > 1
    unet = UNet(
        in_channels=conf.unet.in_channels,
        base_features=conf.unet.base_features,
        out_channels=conf.unet.out_channels,
    )
    state = torch.load(str(args.checkpoint), map_location=device)
    unet.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    unet.to(device)

    fs, ms = zip(*pairs)
    holdout_loader = DataLoader(
        CellDataset(list(fs), list(ms), multiclass=multiclass),
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
    if multiclass:
        class_names = {0: "background", 1: "trapped_cell"}
        for class_id, class_metrics in metrics.items():
            if class_id == 0:
                continue  # background: trivially near-1.0 from pixel imbalance, not informative; "mean"
                          # (SegmentationMetrics.per_class) already excludes it too
            label = class_names.get(class_id, f"class_{class_id}" if isinstance(class_id, int) else class_id)
            print(f"  [{label}]")
            for name, value in class_metrics.items():
                print(f"    {name:<12} {value:.4f}")
    else:
        for name, value in metrics.items():
            print(f"  {name:<12} {value:.4f}")

    if not args.no_save_results:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "pool": args.pool,
            "filter": args.filter,
            "n_images": len(pairs),
            "multiclass": multiclass,
            "metrics": {str(k): v for k, v in metrics.items()},
        }
        args.results_log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_log, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"\nAppended to {args.results_log}")


if __name__ == "__main__":
    main()
