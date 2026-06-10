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
from pathlib import Path

import numpy as np
import pandas as pd

from celldform.biomechanics.analysis import DeformationAnalyzer
from celldform.classification.classifiers import CellClassifier
from celldform.config import load as load_config
from celldform.features.extractor import MorphologyExtractor
from celldform.integration.realtime import RealTimePipeline
from celldform.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from celldform.segmentation.unet import UNet
from celldform.utils.visualization import Visualizer


def main():
    parser = argparse.ArgumentParser(description="Run celldform inference on a video.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--unet-ckpt", required=True, help="U-Net checkpoint (.pt).")
    parser.add_argument("--classifier-ckpt", default=None,
                        help="Classifier checkpoint (.pkl).")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--output-dir", default="results/inference")
    parser.add_argument("--save-masks", action="store_true",
                        help="Write binary masks to disk (warning: large).")
    parser.add_argument("--every-n", type=int, default=None,
                        help="Process every N-th frame (overrides config).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this many frames (overrides config).")
    parser.add_argument("--device", default=None,
                        help="Compute device (overrides config).")
    args = parser.parse_args()

    conf = load_config(args.config)

    # CLI overrides for inference-specific params.
    device = args.device or conf.device
    every_n = args.every_n if args.every_n is not None else conf.acquisition.every_n
    max_frames = args.max_frames if args.max_frames is not None else conf.acquisition.max_frames

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (out_dir / "masks").mkdir(exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"Loading U-Net from {args.unet_ckpt} ...")
    unet = UNet.from_checkpoint(args.unet_ckpt, device=device)

    if args.classifier_ckpt:
        print(f"Loading classifier from {args.classifier_ckpt} ...")
        classifier = CellClassifier.load(args.classifier_ckpt)
    else:
        print("No classifier checkpoint provided — skipping HER2 classification.")
        classifier = None

    # ── Pipeline components ───────────────────────────────────────────────────
    pre_cfg = PreprocessingConfig(output_size=conf.preprocessing.target_size)
    preprocessor = PreprocessingPipeline(pre_cfg)

    extractor = MorphologyExtractor(
        pixel_size_um=conf.features.pixel_size_um,
        min_area_px=conf.features.min_area_px,
    )
    analyzer = DeformationAnalyzer(
        laser_power_mW=conf.biomechanics.laser_power_mW,
        trap_current_mA=conf.biomechanics.trap_current_mA,
        metric=conf.biomechanics.metric,
        window_s=conf.biomechanics.window_s,
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
    import cv2
    from celldform.acquisition.extractor import FrameExtractor

    feature_rows, label_rows = [], []
    fe = FrameExtractor(args.video, output_dir=None)

    for frame_idx, timestamp, frame in fe.stream(every_n=every_n):
        if max_frames and frame_idx // every_n >= max_frames:
            break

        processed = preprocessor(frame)
        mask = unet.predict(processed, device=device)
        features = extractor.extract(mask)
        features["frame_index"] = frame_idx
        features["timestamp"] = timestamp
        feature_rows.append(features)

        if args.save_masks:
            cv2.imwrite(
                str(out_dir / "masks" / f"frame_{frame_idx:06d}.png"),
                mask * 255,
            )

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
            label_rows.append({
                "frame_index": frame_idx,
                "timestamp": timestamp,
                "label": label,
                "proba_high_HER2": proba,
            })

        if frame_idx % 50 == 0:
            print(f"  Processed frame {frame_idx}  t={timestamp:.2f}s")

    # ── Save outputs ──────────────────────────────────────────────────────────
    feat_df = pd.DataFrame(feature_rows)
    feat_df.to_csv(out_dir / "features.csv", index=False)
    print(f"Features saved → {out_dir / 'features.csv'}")

    if label_rows:
        pd.DataFrame(label_rows).to_csv(out_dir / "labels.csv", index=False)
        print(f"Labels saved   → {out_dir / 'labels.csv'}")

    # ── Deformation time-series plot ──────────────────────────────────────────
    metric_col = conf.biomechanics.metric
    if metric_col in feat_df.columns:
        t = feat_df["timestamp"].values
        v = feat_df[metric_col].values
        valid = ~np.isnan(v)
        rod = analyzer.deformation_rate(v[valid], t[valid])

        Visualizer().plot_deformation_timeseries(
            t, v,
            metric_name=metric_col,
            rod=rod,
            laser_power_mW=conf.biomechanics.laser_power_mW,
            save_path=out_dir / "deformation_timeseries.png",
        )
        print(f"Time-series plot → {out_dir / 'deformation_timeseries.png'}")
        print(f"Mean deformation rate (RoD) = {rod:.6f} /s")

    print("\nInference complete.")


if __name__ == "__main__":
    main()
