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
train / val according to data.val_split — val is a monitoring split only
(checkpoint selection, early stopping), not the thesis-reportable score;
that comes from evaluate_unet.py against the external 02_unet_holdout pool,
which a train/val/test split would be redundant with (removed 2026-09-04).
The best checkpoint (by val DSC) and a config snapshot are written to
the checkpoint directory for full reproducibility.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Must be set before CUDA is initialized (i.e. before `import torch` below) —
# required for cuBLAS's deterministic algorithms, used by
# torch.use_deterministic_algorithms() in SegmentationTrainer for reproducible
# training across runs with the same seed.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
    Additive Gaussian noise (noise_std, off by default) is a separate,
    narrower knob: it perturbs per-pixel intensity without touching overall
    contrast, meant to make the model less sensitive to the legacy-vs-real-
    domain appearance gap (see training.augment_noise_std in config.py).
    Applied to the frame only, never the mask.

    Cutout (cutout_prob, off by default): erases one random rectangular patch
    of the frame (filled with the frame's own mean intensity), leaving the
    mask untouched. Targets memorizing per-cell-identity local texture rather
    than boundary shape, by forcing the label to stay correct even when part
    of the evidence for it is missing (see training.augment_cutout_prob).

    Brightness/contrast jitter (brightness_contrast_prob, off by default):
    random global brightness shift (+/-0.15) and contrast scale (0.8-1.2x)
    applied to the frame only. Originally omitted on the reasoning that
    CLAHE in Stage 2 already normalises contrast, so jitter would undo that
    work — revisited 2026-09-04 once the full 2-cell holdout sweep showed
    the model is precision-, not recall-, bottlenecked (it finds the cell
    fine, it predicts too much area around it) and the over-segmentation is
    laser-power-correlated. CLAHE equalises *local* contrast within each
    already-preprocessed frame; it does not erase the *global* exposure
    level a frame was captured at, which is exactly what plausibly tracks
    laser power. The two aren't redundant: this targets a training-time
    invariance CLAHE's offline, per-image normalisation doesn't provide.
    See training.augment_brightness_contrast_prob in config.py. Tried once
    (bf64_bc, 2026-09-04) — no real effect on the real holdout, precision
    barely moved; see CLAUDE.md. Kept as a knob, not removed, but not the
    active hypothesis anymore.

    Scale/zoom jitter (scale_prob, off by default): resizes the frame+mask
    together by a random factor (0.7-1.3x), then center-crops (zoom in) or
    pads with the frame's own mean (zoom out) back to the original size —
    the same transform applied to both, so they stay aligned, unlike
    brightness/contrast which only ever touches the frame. Added 2026-09-04
    after finding both holdout cells' true trapped-cell size (704/886 px^2)
    falls at or below every one of the 11 training cells (825-1549 px^2) —
    a real train/holdout size mismatch, not a laser-power-sampling artifact
    (that was checked and ruled out). The model likely leans on an implicit
    learned "typical size" from training, which doesn't transfer to a
    genuinely smaller novel cell — this forces scale invariance the fixed
    256x256 crop alone doesn't provide. See training.augment_scale_prob.

    When multiclass=True, masks are loaded as raw integer class-id arrays
    (0=background, 1=trapped cell, 2=other/decoy, ...) instead of being
    binarised — used for the experimental multiclass annotation path.
    """

    def __init__(self, frame_paths, mask_paths, augment: bool = False, multiclass: bool = False,
                 noise_std: float = 0.0, cutout_prob: float = 0.0,
                 brightness_contrast_prob: float = 0.0, scale_prob: float = 0.0) -> None:
        assert len(frame_paths) == len(mask_paths)
        self.frame_paths = frame_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.multiclass = multiclass
        self.noise_std = noise_std
        self.cutout_prob = cutout_prob
        self.brightness_contrast_prob = brightness_contrast_prob
        self.scale_prob = scale_prob

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
            if self.scale_prob > 0 and random.random() < self.scale_prob:
                h, w = frame_f.shape
                scale = random.uniform(0.7, 1.3)
                new_h = max(1, int(round(h * scale)))
                new_w = max(1, int(round(w * scale)))
                frame_resized = cv2.resize(frame_f, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(
                    mask_arr.astype(np.uint8) if self.multiclass else mask_arr,
                    (new_w, new_h), interpolation=cv2.INTER_NEAREST,
                )
                if self.multiclass:
                    mask_resized = mask_resized.astype(np.int64)
                if scale >= 1.0:
                    # Zoomed in — center-crop back to (h, w).
                    y0 = (new_h - h) // 2
                    x0 = (new_w - w) // 2
                    frame_f = frame_resized[y0:y0 + h, x0:x0 + w].copy()
                    mask_arr = mask_resized[y0:y0 + h, x0:x0 + w].copy()
                else:
                    # Zoomed out — pad back to (h, w), centered. Frame padded with its own mean
                    # intensity (matches cutout's fill choice below); mask padded with background (0).
                    pad_h, pad_w = h - new_h, w - new_w
                    top, bottom = pad_h // 2, pad_h - pad_h // 2
                    left, right = pad_w // 2, pad_w - pad_w // 2
                    frame_f = cv2.copyMakeBorder(frame_resized, top, bottom, left, right,
                                                  cv2.BORDER_CONSTANT, value=float(frame_f.mean()))
                    mask_arr = cv2.copyMakeBorder(mask_resized, top, bottom, left, right,
                                                   cv2.BORDER_CONSTANT, value=0)
            if self.noise_std > 0:
                frame_f = np.clip(
                    frame_f + np.random.normal(0.0, self.noise_std, frame_f.shape), 0.0, 1.0
                ).astype(np.float32)
            if self.cutout_prob > 0 and random.random() < self.cutout_prob:
                h, w = frame_f.shape
                ph = random.randint(int(0.08 * h), int(0.25 * h))
                pw = random.randint(int(0.08 * w), int(0.25 * w))
                y0 = random.randint(0, h - ph)
                x0 = random.randint(0, w - pw)
                frame_f = frame_f.copy()
                frame_f[y0:y0 + ph, x0:x0 + pw] = float(frame_f.mean())
            if self.brightness_contrast_prob > 0 and random.random() < self.brightness_contrast_prob:
                brightness = random.uniform(-0.15, 0.15)
                contrast = random.uniform(0.8, 1.2)
                mean = float(frame_f.mean())
                frame_f = np.clip((frame_f - mean) * contrast + mean + brightness, 0.0, 1.0).astype(np.float32)

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
    pairs: list[tuple[Path, Path]], groups: list[str], n_val: int,
    split_seed: int | None = None
) -> tuple[list, list]:
    """Split pairs into train/val by whole group, never splitting a group
    across sets. Group order is shuffled (uses the already-seeded `random`
    module), then groups are assigned to val, then train, until val's target
    frame count is reached — so the split size approximates n_val rather
    than matching exactly, since groups vary in size.

    Two-way only (no test split) since 2026-09-04 — a third internal split
    was redundant with the external 02_unet_holdout evaluation (see
    train_unet.py's module docstring and celldform.config.DataConfig).

    `split_seed` (2026-09-17) shuffles through a dedicated random.Random rather
    than the global module, so the split can be held fixed while `seed` varies
    weight initialisation. Passing the same value the global RNG was seeded with
    reproduces the historical split exactly — random.Random(x) and
    random.seed(x) yield the same Mersenne Twister stream — so the default
    (split_seed = seed) changes nothing about any existing run.
    """
    by_group: dict[str, list] = defaultdict(list)
    for pair, g in zip(pairs, groups):
        by_group[g].append(pair)

    group_ids = list(by_group.keys())
    rng = random.Random(split_seed) if split_seed is not None else random
    rng.shuffle(group_ids)

    val_pairs, train_pairs = [], []
    for g in group_ids:
        items = by_group[g]
        if len(val_pairs) < n_val:
            val_pairs.extend(items)
        else:
            train_pairs.extend(items)

    return train_pairs, val_pairs


def _log_split(pairs, groups, train_pairs, val_pairs, seed: int, split_seed: int) -> None:
    """Print which groups landed in train vs. validation, and flag oversized ones.

    Added 2026-09-17. Until now the split was reported as two integers, so a run
    that happened to draw a large cell into validation was indistinguishable in the
    log from one that did not — and with 22 groups the draw varies a lot. Any group
    holding a sizeable share of the pool can, on an unlucky shuffle, take over the
    validation set and simultaneously vanish from training; `high_cell10` (50 frames
    against a ~45-frame validation target) is the live example. Printing the
    assignment makes that visible at a glance instead of being inferred later from
    an unexplained score.
    """
    val_set = set(val_pairs)
    by_group: dict[str, list] = defaultdict(list)
    for pair, g in zip(pairs, groups):
        by_group[g].append(pair)

    val_groups, train_groups = [], []
    for g, items in by_group.items():
        (val_groups if items[0] in val_set else train_groups).append((g, len(items)))
    val_groups.sort(key=lambda kv: -kv[1])
    train_groups.sort(key=lambda kv: -kv[1])

    seed_note = f"seed={seed}" + ("" if split_seed == seed else f", split_seed={split_seed}")
    print(f"[celldform]   split assignment ({seed_note})")
    print("[celldform]     val  : " + ", ".join(f"{g}({n})" for g, n in val_groups))
    cells = [(g, n) for g, n in train_groups if "cell" in g]
    legacy_n = sum(n for g, n in train_groups if "cell" not in g)
    print("[celldform]     train: " + ", ".join(f"{g}({n})" for g, n in cells)
          + (f", legacy({legacy_n})" if legacy_n else ""))

    # A group big enough to dominate validation distorts both sides of the split.
    n_val_target = len(val_pairs)
    for g, n in val_groups:
        # 0.5, not a higher bar: once one group is the majority of validation, the
        # monitoring signal is mostly about that one cell, and checkpoint selection
        # and early stopping are steered by it. Calibrated against the live case —
        # high_cell10 lands at 72% of validation on an unlucky draw, which a 0.8
        # threshold would have let through silently.
        if n_val_target and n / n_val_target >= 0.5:
            print(f"[celldform]   WARNING: '{g}' is {n} of {n_val_target} validation frames "
                  f"({100 * n / n_val_target:.0f}%) — validation is mostly one group, so "
                  f"checkpoint selection and early stopping are steered by that single cell, "
                  f"and it is absent from training.")
    missing = [g for g, n in val_groups if "cell" in g]
    if missing:
        print(f"[celldform]   note: cell group(s) held out of training this run: "
              f"{', '.join(missing)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train U-Net segmentation model.")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config value).")
    parser.add_argument("--domain-oversample-weight", type=float, default=None,
                        help="Sampling weight for domain_-prefixed (real-cell) train frames "
                             "vs. legacy frames (overrides training.domain_oversample_weight). "
                             "1.0 = off/uniform (default); pass 1.0 explicitly to force it off "
                             "when the config has it enabled.")
    parser.add_argument("--tag", default=None,
                        help="Suffix appended to the saved training-curve filename "
                             "(e.g. --tag auto -> unet_training_curves_v6_auto.png), "
                             "for distinguishing runs at a glance.")
    args = parser.parse_args()

    conf = load_config(args.config)
    if args.domain_oversample_weight is not None:
        conf.training.domain_oversample_weight = args.domain_oversample_weight

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
    n_val = int(n * conf.data.val_split)

    groups = _build_groups(pairs)
    split_seed = conf.training.split_seed if conf.training.split_seed is not None else seed
    train_pairs, val_pairs = _group_split(pairs, groups, n_val, split_seed=split_seed)

    n_groups = len(set(groups))
    print(
        f"[celldform] Dataset split (group-aware, {n_groups} groups) — "
        f"train: {len(train_pairs)}  val: {len(val_pairs)}"
    )
    _log_split(pairs, groups, train_pairs, val_pairs, seed, split_seed)

    on_cuda = torch.cuda.is_available()
    n_workers = conf.training.num_workers
    multiclass = conf.unet.out_channels > 1

    def make_loader(pairs, shuffle, augment=False, sampler=None):
        fs, ms = zip(*pairs)
        noise_std = conf.training.augment_noise_std if augment else 0.0
        cutout_prob = conf.training.augment_cutout_prob if augment else 0.0
        brightness_contrast_prob = conf.training.augment_brightness_contrast_prob if augment else 0.0
        scale_prob = conf.training.augment_scale_prob if augment else 0.0
        ds = CellDataset(list(fs), list(ms), augment=augment, multiclass=multiclass,
                          noise_std=noise_std, cutout_prob=cutout_prob,
                          brightness_contrast_prob=brightness_contrast_prob, scale_prob=scale_prob)
        return DataLoader(
            ds,
            batch_size=conf.training.batch_size,
            shuffle=shuffle if sampler is None else False,  # mutually exclusive with sampler
            sampler=sampler,
            num_workers=n_workers,
            pin_memory=on_cuda,
            persistent_workers=(n_workers > 0),
            prefetch_factor=(4 if n_workers > 0 else None),
            worker_init_fn=_worker_init_fn if n_workers > 0 else None,
        )

    train_sampler = None
    domain_weight = conf.training.domain_oversample_weight
    if domain_weight != 1.0:
        is_domain = [f.stem.startswith("domain_") for f, m in train_pairs]
        n_domain = sum(is_domain)
        sample_weights = [domain_weight if d else 1.0 for d in is_domain]
        train_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_pairs), replacement=True
        )
        effective_domain_frac = (n_domain * domain_weight) / (
            n_domain * domain_weight + (len(train_pairs) - n_domain)
        )
        print(
            f"[celldform] Domain oversampling enabled (weight={domain_weight}): "
            f"{n_domain} domain / {len(train_pairs) - n_domain} legacy train frames -> "
            f"~{100 * effective_domain_frac:.1f}% domain per epoch (sampled with replacement)"
        )

    train_loader = make_loader(train_pairs, shuffle=True, augment=True, sampler=train_sampler)
    val_loader = make_loader(val_pairs, shuffle=False, augment=False)

    # ── Model ────────────────────────────────────────────────────────────────
    unet = UNet(
        in_channels=conf.unet.in_channels,
        base_features=conf.unet.base_features,
        out_channels=conf.unet.out_channels,
        dropout=conf.unet.dropout,
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
        # val_dsc intentionally omitted here — it's the per-epoch curve's batch-pooled-then-averaged
        # figure (SegmentationTrainer._validate), not the globally-pooled DSC the "Final validation
        # metrics" table below reports (SegmentationTrainer.evaluate) — quote the table's number instead.
        print(f"\n[celldform] Reloaded best checkpoint for final evaluation: epoch {best_epoch}")

    # ── Final evaluation ─────────────────────────────────────────────────────
    metrics = trainer.evaluate()
    print("\nFinal validation metrics:")
    if multiclass:
        class_names = {0: "background", 1: "trapped_cell"}
        for class_id, class_metrics in metrics.items():
            if class_id == 0:
                continue  # background: trivially near-1.0 from pixel imbalance, not informative; "mean"
                          # (SegmentationMetrics.per_class) already excludes it too
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
        history, final_metrics=metrics, best_epoch=best_epoch, save_path=str(curve_path),
        n_classes=conf.unet.out_channels,
    )
    print(f"Training curves saved to {curve_path}")


if __name__ == "__main__":
    main()
