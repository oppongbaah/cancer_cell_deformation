#!/usr/bin/env python
"""
End-to-end inference on optical tweezers video data.

Usage:
    python scripts/infer.py \\
        --video data/videos/experiment_01.mp4 \\
        --unet-ckpt checkpoints/unet_best.pt \\
        --classifier-ckpt checkpoints/svm_clf.pkl \\
        --config configs/default.yaml \\
        --output-dir results/experiment_01

Outputs (per run):
    results/<name>/features.csv   — per-frame morphological features
    results/<name>/labels.csv     — predicted HER2 class labels + probabilities
    results/<name>/masks/         — saved binary masks as PNGs (optional)
    results/<name>/deformation_timeseries.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from celldform.biomechanics.analysis import DeformationAnalyzer
from celldform.classification.classifiers import CellClassifier
from celldform.features.extractor import MorphologyExtractor
from celldform.integration.realtime import RealTimePipeline
from celldform.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from celldform.segmentation.unet import UNet
from celldform.utils.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(description="Run celldform inference on a video.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--unet-ckpt", required=True, help="U-Net checkpoint (.pt).")
    parser.add_argument("--classifier-ckpt", default=None, help="Classifier checkpoint (.pkl).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="results/inference")
    parser.add_argument("--save-masks", action="store_true",
                        help="Write binary masks to disk (warning: large).")
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (out_dir / "masks").mkdir(exist_ok=True)

    device = args.device or cfg.get("device")

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"Loading U-Net from {args.unet_ckpt} ...")
    unet = UNet.from_checkpoint(args.unet_ckpt, device=device)

    if args.classifier_ckpt:
        print(f"Loading classifier from {args.classifier_ckpt} ...")
        classifier = CellClassifier.load(args.classifier_ckpt)
    else:
        # Dummy un-fitted classifier — labels will be 0 (placeholder).
        print("No classifier checkpoint provided — skipping HER2 classification.")
        classifier = None

    # ── Pipeline components ───────────────────────────────────────────────────
    pre_cfg = PreprocessingConfig(output_size=tuple(cfg["preprocessing"]["target_size"]))
    preprocessor = PreprocessingPipeline(pre_cfg)
    extractor = MorphologyExtractor(
        pixel_size_um=cfg["features"].get("pixel_size_um"),
        min_area_px=cfg["features"]["min_area_px"],
    )
    analyzer = DeformationAnalyzer(
        laser_power_mW=cfg["biomechanics"]["laser_power_mW"],
        trap_current_mA=cfg["biomechanics"]["trap_current_mA"],
        metric=cfg["biomechanics"]["metric"],
        window_s=cfg["biomechanics"]["window_s"],
    )

    if classifier is not None:
        pipeline = RealTimePipeline(
            unet=unet,
            preprocessor=preprocessor,
            extractor=extractor,
            classifier=classifier,
            analyzer=analyzer,
            device=device,
        )
    else:
        pipeline = None

    # ── Run inference ─────────────────────────────────────────────────────────
    feature_rows, label_rows = [], []

    from celldform.acquisition.extractor import FrameExtractor
    import cv2

    fe = FrameExtractor(args.video, output_dir=None)

    for frame_idx, timestamp, frame in fe.stream(every_n=args.every_n):
        if args.max_frames and frame_idx // args.every_n >= args.max_frames:
            break

        processed = preprocessor(frame)
        mask = unet.predict(processed, device=device)
        features = extractor.extract(mask)
        features["frame_index"] = frame_idx
        features["timestamp"] = timestamp
        feature_rows.append(features)

        if args.save_masks:
            mask_path = out_dir / "masks" / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(mask_path), mask * 255)

        if classifier is not None:
            feat_vec = np.array(
                [[features[k] for k in sorted(extractor._FEATURE_NAMES)]],
                dtype=np.float32,
            )
            feat_vec = np.nan_to_num(feat_vec, nan=0.0)
            label = int(classifier.predict(feat_vec)[0])
            try:
                proba = float(classifier.predict_proba(feat_vec)[0, 1])
            except Exception:
                proba = float("nan")
            label_rows.append({"frame_index": frame_idx, "timestamp": timestamp,
                                "label": label, "proba_high_HER2": proba})

        if frame_idx % 50 == 0:
            print(f"  Processed frame {frame_idx}  t={timestamp:.2f}s")

    # ── Save outputs ──────────────────────────────────────────────────────────
    feat_df = pd.DataFrame(feature_rows)
    feat_df.to_csv(out_dir / "features.csv", index=False)
    print(f"Features saved → {out_dir / 'features.csv'}")

    if label_rows:
        label_df = pd.DataFrame(label_rows)
        label_df.to_csv(out_dir / "labels.csv", index=False)
        print(f"Labels saved   → {out_dir / 'labels.csv'}")

    # ── Deformation time-series plot ──────────────────────────────────────────
    metric_col = cfg["biomechanics"]["metric"]
    if metric_col in feat_df.columns:
        t = feat_df["timestamp"].values
        v = feat_df[metric_col].values
        rod = analyzer.deformation_rate(v[~np.isnan(v)], t[~np.isnan(v)])

        viz = Visualizer()
        viz.plot_deformation_timeseries(
            t, v, metric_name=metric_col, rod=rod,
            laser_power_mW=cfg["biomechanics"]["laser_power_mW"],
            save_path=out_dir / "deformation_timeseries.png",
        )
        print(f"Time-series plot → {out_dir / 'deformation_timeseries.png'}")
        print(f"Mean deformation rate (RoD) = {rod:.6f} /s")

    print("\nInference complete.")


if __name__ == "__main__":
    main()
