#!/usr/bin/env python3
"""
Quick sanity check — run a trained U-Net checkpoint on one raw frame and draw
two outlines on top of it: a solid line around each predicted region (the
actual, possibly irregular, area the network segmented) and a dotted line
around the hand-drawn ground truth, when one exists for that frame.

Not part of the pipeline proper; a fast way to eyeball whether a checkpoint
is fitting the trapped cell before committing to a full holdout evaluation.

Reads architecture (in_channels, base_features, out_channels) and
preprocessing settings (target_size, CLAHE/morphology params) from the
config.yaml snapshot saved next to the checkpoint, so binary and multiclass
checkpoints both load correctly without manual flags.

Usage
-----
python scripts/predict_single_frame.py \\
    --checkpoint checkpoints/multiclass_experiment_256/unet_best.pt \\
    --frame data/frames/02_unet_holdout/high_cell9_005000.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from celldform.config import load as load_config
from celldform.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from celldform.segmentation.unet import UNet
from celldform.utils.metrics import SegmentationMetrics

# matplotlib RGB (0-1 floats); trapped cell = green, other/decoy = red
_LABEL_COLORS = {1: (0.20, 0.85, 0.20), 2: (0.95, 0.20, 0.20)}
_LABEL_NAMES = {1: "trapped cell", 2: "other/decoy"}
_GT_COLOR = (1.0, 0.0, 1.0)  # magenta — maximally distinct in hue from the green/red class colors and the gray background


def _blobs(mask: np.ndarray, label: int, min_area: float = 6.0) -> list[dict]:
    """Connected components of ``mask == label`` as a smoothed contour outline.

    Raw per-pixel U-Net masks produce a blocky, staircase-edged contour when
    traced directly — most visible on small blobs. Blurring the binary mask
    before re-thresholding rounds those pixel corners off (a standard
    "smooth contour" trick) so the drawn outline reads as a clean line rather
    than a jagged pixel boundary; it does not change which pixels the model
    predicted — this only affects how the outline is drawn.
    """
    binary = (mask == label).astype(np.uint8) * 255
    smoothed = cv2.GaussianBlur(binary, (5, 5), sigmaX=1.0)
    _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    blobs = _contours_to_blobs(smoothed, min_area)
    if not blobs and binary.any():
        # A very small/sparse prediction can blur below the re-threshold cutoff and vanish
        # entirely — that would silently hide a real (if weak) detection. Fall back to the
        # unsmoothed mask so the outline still shows up, just without the rounded corners.
        blobs = _contours_to_blobs(binary, min_area)
    return blobs


def _contours_to_blobs(binary: np.ndarray, min_area: float) -> list[dict]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        c = cv2.approxPolyDP(c, epsilon=0.8, closed=True)  # thin out near-collinear points for a cleaner line
        blobs.append({"points": c.squeeze(1).astype(float), "area": area})
    return blobs


def _find_ground_truth(frame_path: Path, multiclass: bool) -> Path | None:
    """Best-effort lookup of a hand-drawn mask matching ``frame_path``, if one exists."""
    pool = frame_path.parent.name
    candidates = []
    if multiclass:
        candidates.append(Path("data/masks") / f"{pool}_multiclass" / f"{frame_path.stem}.png")
    candidates.append(Path("data/masks") / pool / f"{frame_path.stem}.png")
    return next((c for c in candidates if c.exists()), None)


def _gt_binary(gt_mask: np.ndarray, label: int, gt_is_multiclass: bool) -> np.ndarray:
    """Ground-truth pixels for ``label``. Plain binary masks (0/255) only ever encode the
    trapped cell (label 1) — they predate/lack decoy annotation, so label 2 is empty there."""
    if gt_is_multiclass:
        return gt_mask == label
    return (gt_mask > 127) if label == 1 else np.zeros_like(gt_mask, dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a unet_best.pt / unet_last.pt checkpoint.")
    parser.add_argument("--frame", required=True, help="Path to a single raw frame image.")
    parser.add_argument("--config", default=None,
                         help="Config YAML to read architecture/preprocessing from "
                              "(default: config.yaml next to the checkpoint).")
    parser.add_argument("--min-area", type=float, default=6.0, help="Ignore predicted regions smaller than this (px^2) as noise.")
    parser.add_argument("--trapped-only", action="store_true",
                         help="Draw only the trapped-cell class (label 1); suppress other/decoy (label 2). "
                              "No effect on binary checkpoints, which only ever predict the trapped cell.")
    parser.add_argument("--gt-mask", default=None,
                         help="Hand-drawn mask to compare against (default: auto-detected from "
                              "data/masks/<pool>[_multiclass]/<frame_stem>.png, if one exists).")
    parser.add_argument("--output", default=None, help="Where to save the figure (default: results/<frame_stem>_<ckpt>_overlay.png).")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else ckpt_path.parent / "config.yaml"
    if not config_path.exists():
        raise SystemExit(f"No config.yaml found next to checkpoint ({config_path}); pass --config explicitly.")
    conf = load_config(config_path)

    frame_path = Path(args.frame)
    raw = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise SystemExit(f"Could not read frame: {frame_path}")

    pre_cfg = PreprocessingConfig(
        median_kernel=conf.preprocessing.median_kernel,
        clahe_clip_limit=conf.preprocessing.clahe_clip_limit,
        clahe_tile_grid=conf.preprocessing.clahe_tile_grid,
        morph_kernel_size=conf.preprocessing.morph_kernel_size,
        output_size=conf.preprocessing.target_size,
    )
    processed = PreprocessingPipeline(pre_cfg)(raw)  # float32 [0, 1], target_size

    multiclass = conf.unet.out_channels > 1
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNet(
        in_channels=conf.unet.in_channels,
        base_features=conf.unet.base_features,
        out_channels=conf.unet.out_channels,
    )
    # SegmentationTrainer._save_checkpoint wraps the state dict with epoch/val_dsc/
    # optimizer/scheduler state (see scripts/evaluate_unet.py for the same pattern);
    # UNet.from_checkpoint expects a bare state dict, so load manually here instead.
    state = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    unet.to(device).eval()
    if "val_dsc" in state:
        print(f"Checkpoint: epoch {state.get('epoch')}, val_dsc={state['val_dsc']:.4f}")
    mask = unet.predict(processed, device=device)
    class_labels = [1] if (not multiclass or args.trapped_only) else [1, 2]

    # ── Ground-truth comparison (only possible if this frame has a hand-drawn mask) ──
    gt_path = Path(args.gt_mask) if args.gt_mask else _find_ground_truth(frame_path, multiclass)
    gt_mask = None
    gt_scores: dict[int, dict[str, float]] = {}
    if gt_path is not None:
        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_raw, conf.preprocessing.target_size[::-1], interpolation=cv2.INTER_NEAREST)
        gt_is_multiclass = "_multiclass" in gt_path.parent.name
        metrics = SegmentationMetrics()
        for label in class_labels:
            pred_binary = mask == label
            gt_scores[label] = {
                "dsc": metrics.dice(pred_binary, _gt_binary(gt_mask, label, gt_is_multiclass)),
                "iou": metrics.iou(pred_binary, _gt_binary(gt_mask, label, gt_is_multiclass)),
            }
        print(f"Ground truth: {gt_path}")
        for label, scores in gt_scores.items():
            print(f"  {_LABEL_NAMES[label]:<14} DSC={scores['dsc']:.4f}  IoU={scores['iou']:.4f}")
    else:
        print("No ground-truth mask found for this frame — skipping DSC/IoU (pass --gt-mask to compare explicitly).")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5), dpi=150)
    # processed is float32 in [0, 1] — PreprocessingPipeline's final min-max normalisation
    # step (celldform/preprocessing/pipeline.py) — so vmin/vmax are fixed rather than
    # auto-scaled to this frame's actual min/max, keeping intensity comparable across frames.
    im = ax.imshow(processed, cmap="gray", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Pixel intensity (normalized [0, 1])")

    legend_handles = []
    for label in class_labels:
        color = _LABEL_COLORS[label]
        blobs = _blobs(mask, label, min_area=args.min_area)
        legend_handles.append(mpatches.Patch(edgecolor=color, facecolor="none", label=_LABEL_NAMES[label]))

        for blob in blobs:
            # Predicted outline — the (smoothed) shape the network segmented.
            ax.add_patch(plt.Polygon(
                blob["points"], closed=True, fill=False,
                edgecolor=color, linewidth=1.4, alpha=0.95,
            ))

    if gt_mask is not None:
        for label in class_labels:
            gt_binary = _gt_binary(gt_mask, label, gt_is_multiclass)
            if not gt_binary.any():
                continue
            gt_label_mask = np.where(gt_binary, label, 0).astype(np.uint8)
            for blob in _blobs(gt_label_mask, label, min_area=args.min_area):
                ax.add_patch(plt.Polygon(
                    blob["points"], closed=True, fill=False,
                    edgecolor=_GT_COLOR, linewidth=1.0, linestyle=":", alpha=0.9,
                ))
        legend_handles.append(mpatches.Patch(edgecolor=_GT_COLOR, facecolor="none", label="ground truth", linestyle=":"))

    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.6)
    if gt_scores:
        title = "   ".join(f"{_LABEL_NAMES[label]} DSC={s['dsc']:.2f}" for label, s in gt_scores.items())
    else:
        title = "DSC: no ground truth available for this frame"
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()

    out_path = Path(args.output) if args.output else Path("results") / f"{frame_path.stem}_{ckpt_path.parent.name}_overlay.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved overlay -> {out_path}")


if __name__ == "__main__":
    main()
