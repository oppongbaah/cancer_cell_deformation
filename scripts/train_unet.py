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
from collections import defaultdict
from datetime import datetime
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

    When multiclass=True, masks are loaded as raw integer class-id arrays
    (0=background, 1=trapped cell, 2=other/decoy, ...) instead of being
    binarised — used for the experimental multiclass annotation path.
    """

    def __init__(self, frame_paths, mask_paths, augment: bool = False, multiclass: bool = False) -> None:
        assert len(frame_paths) == len(mask_paths)
        self.frame_paths = frame_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.multiclass = multiclass

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        import cv2

        frame = cv2.imread(str(self.frame_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        frame_f = frame.astype(np.float32) / 255.0     # normalise to [0, 1]
        if self.multiclass:
            mask_arr = mask.astype(np.int64)            # raw class ids 0/1/2
        else:
            mask_arr = (mask > 127).astype(np.float32)  # binarise to 0.0 / 1.0

        if self.augment:
            if random.random() < 0.5:
                frame_f = np.fliplr(frame_f).copy()
                mask_arr = np.fliplr(mask_arr).copy()
            if random.random() < 0.5:
                frame_f = np.flipud(frame_f).copy()
                mask_arr = np.flipud(mask_arr).copy()
            k = random.randint(0, 3)
            if k:
                frame_f = np.rot90(frame_f, k).copy()
                mask_arr = np.rot90(mask_arr, k).copy()

        frame_t = torch.from_numpy(frame_f[np.newaxis])    # (1, H, W)
        if self.multiclass:
            mask_t = torch.from_numpy(mask_arr)             # (H, W) long — CrossEntropyLoss target shape
        else:
            mask_t = torch.from_numpy(mask_arr[np.newaxis])  # (1, H, W) float
        return frame_t, mask_t


def _worker_init_fn(worker_id: int) -> None:
    """Reseed each DataLoader worker's random/numpy RNGs deterministically.

    PyTorch's default worker startup only reseeds torch's own RNG per worker
    (base_seed + worker_id) — it leaves Python's `random` module and
    `numpy.random` at whatever state they inherited via fork from the parent
    process. Since CellDataset's augmentation uses `random`, that means
    augmentation decisions were neither reproducible across runs (worker
    process scheduling is non-deterministic) nor independent across workers
    (all workers fork from the same inherited state) whenever num_workers>0.
    torch.initial_seed() inside a worker already returns that worker's
    deterministic, worker-unique seed — reuse it here for random/numpy too.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


_LEGACY_TS_FMT = "%Y%m%d%H%M%S"
_LEGACY_BURST_GAP_S = 5  # gap between legacy frame timestamps that starts a new acquisition burst


def _assign_legacy_bursts(stems: list[str], gap_s: int = _LEGACY_BURST_GAP_S) -> dict[str, str]:
    """Cluster legacy timestamp-named frames into acquisition bursts.

    Legacy frames are named by capture timestamp (YYYYMMDDHHMMSS) and were
    sampled roughly one second apart within a burst — consecutive seconds of
    an essentially static trapped cell, i.e. near-duplicate images. A gap
    larger than gap_s between sorted timestamps marks a new burst.
    """
    ordered = sorted(stems, key=int)
    groups: dict[str, str] = {}
    burst_id = 0
    prev_ts = None
    for stem in ordered:
        ts = datetime.strptime(stem, _LEGACY_TS_FMT)
        if prev_ts is not None and (ts - prev_ts).total_seconds() > gap_s:
            burst_id += 1
        groups[stem] = f"legacy_burst_{burst_id}"
        prev_ts = ts
    return groups


def _build_groups(pairs: list[tuple[Path, Path]]) -> list[str]:
    """Assign a group id per (frame, mask) pair so correlated frames never
    split across train/val/test.

    Two sources of correlation in this dataset:
      - legacy frames: one-second-apart bursts of a near-static cell (see
        _assign_legacy_bursts)
      - domain_<cell>_<frame> frames: multiple samples drawn from the same
        real trapped cell, grouped by cell id
    """
    stems = [f.stem for f, _ in pairs]
    legacy_bursts = _assign_legacy_bursts([s for s in stems if s.isdigit()])

    groups = []
    for stem in stems:
        if stem.startswith("domain_"):
            parts = stem.split("_")
            groups.append("_".join(parts[1:-1]))  # domain_high_cell1_002042 -> high_cell1
        elif stem in legacy_bursts:
            groups.append(legacy_bursts[stem])
        else:
            groups.append(stem)  # unrecognised naming scheme: treat as its own group
    return groups


def _group_split(
    pairs: list[tuple[Path, Path]], groups: list[str], n_test: int, n_val: int
) -> tuple[list, list, list]:
    """Split pairs into train/val/test by whole group, never splitting a
    group across sets. Group order is shuffled (uses the already-seeded
    `random` module), then groups are assigned to test, then val, then train
    until each set's target frame count is reached — so split sizes
    approximate n_test/n_val rather than matching exactly, since groups vary
    in size.
    """
    by_group: dict[str, list] = defaultdict(list)
    for pair, g in zip(pairs, groups):
        by_group[g].append(pair)

    group_ids = list(by_group.keys())
    random.shuffle(group_ids)

    test_pairs, val_pairs, train_pairs = [], [], []
    for g in group_ids:
        items = by_group[g]
        if len(test_pairs) < n_test:
            test_pairs.extend(items)
        elif len(val_pairs) < n_val:
            val_pairs.extend(items)
        else:
            train_pairs.extend(items)

    return train_pairs, val_pairs, test_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train U-Net segmentation model.")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config value).")
    parser.add_argument("--tag", default=None,
                        help="Suffix appended to the saved training-curve filename "
                             "(e.g. --tag auto -> unet_training_curves_v6_auto.png), "
                             "for distinguishing runs at a glance.")
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

    n = len(pairs)
    n_test = int(n * conf.data.test_split)
    n_val = int(n * conf.data.val_split)

    groups = _build_groups(pairs)
    train_pairs, val_pairs, test_pairs = _group_split(pairs, groups, n_test, n_val)

    n_groups = len(set(groups))
    print(
        f"[celldform] Dataset split (group-aware, {n_groups} groups) — "
        f"train: {len(train_pairs)}  val: {len(val_pairs)}  test: {len(test_pairs)}"
    )

    on_cuda = torch.cuda.is_available()
    n_workers = conf.training.num_workers
    multiclass = conf.unet.out_channels > 1

    def make_loader(pairs, shuffle, augment=False):
        fs, ms = zip(*pairs)
        ds = CellDataset(list(fs), list(ms), augment=augment, multiclass=multiclass)
        return DataLoader(
            ds,
            batch_size=conf.training.batch_size,
            shuffle=shuffle,
            num_workers=n_workers,
            pin_memory=on_cuda,
            persistent_workers=(n_workers > 0),
            prefetch_factor=(4 if n_workers > 0 else None),
            worker_init_fn=_worker_init_fn if n_workers > 0 else None,
        )

    train_loader = make_loader(train_pairs, shuffle=True, augment=True)
    val_loader = make_loader(val_pairs, shuffle=False, augment=False)

    # ── Model ────────────────────────────────────────────────────────────────
    unet = UNet(
        in_channels=conf.unet.in_channels,
        base_features=conf.unet.base_features,
        out_channels=conf.unet.out_channels,
    )

    trainer = SegmentationTrainer(
        model=unet,
        train_loader=train_loader,
        val_loader=val_loader,
        conf=conf,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    history = trainer.fit()

    # ── Reload best checkpoint ──────────────────────────────────────────────
    # trainer.model holds the LAST epoch's weights when fit() returns, not
    # necessarily the best one — small datasets like this often overfit past
    # their peak validation DSC well before training ends. Reload the
    # checkpoint fit() already saved whenever val_dsc improved, so the
    # reported metrics and curve reflect the best model, not wherever
    # training happened to stop.
    best_epoch = None
    best_ckpt_path = trainer.checkpoint_dir / "unet_best.pt"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=trainer.device)
        trainer.model.load_state_dict(best_ckpt["model_state_dict"])
        best_epoch = best_ckpt["epoch"]
        print(
            f"\n[celldform] Reloaded best checkpoint for final evaluation: "
            f"epoch {best_epoch}, val_dsc={best_ckpt['val_dsc']:.4f}"
        )

    # ── Final evaluation ─────────────────────────────────────────────────────
    metrics = trainer.evaluate()
    print("\nFinal validation metrics:")
    if multiclass:
        class_names = {0: "background", 1: "trapped_cell"}
        for class_id, class_metrics in metrics.items():
            label = class_names.get(class_id, f"class_{class_id}" if isinstance(class_id, int) else class_id)
            print(f"  [{label}]")
            for k, v in class_metrics.items():
                print(f"    {k}: {v:.4f}")
    else:
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # ── Plot training curves ──────────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    existing = sorted(results_dir.glob("unet_training_curves_v*.png"))
    next_version = len(existing) + 1
    suffix = f"_{args.tag}" if args.tag else ""
    curve_path = results_dir / f"unet_training_curves_v{next_version}{suffix}.png"

    viz = Visualizer()
    viz.plot_training_curves(
        history, final_metrics=metrics, best_epoch=best_epoch, save_path=str(curve_path)
    )
    print(f"Training curves saved to {curve_path}")


if __name__ == "__main__":
    main()
