#!/usr/bin/env python3
"""
Frame-to-mask latency: our U-Net against nnU-Net, measured the same way on the same frames.

    python scripts/evaluation/latency_benchmark.py \
        --unet checkpoints/randswap/experiment_0_cv_baseline/cv_baseline/fold_deploy/unet_deploy.pt \
        --config configs/randswap/experiment_0_cv_baseline/cv_baseline.yaml \
        --crop-center-from results/randswap/experiment_0_cv_baseline/cv_baseline_fold_deploy_split.json \
        --nnunet checkpoints/randswap/experiment_1_nnunet/Dataset501_CellTrapRandswap/nnUNetTrainer_100epochs__nnUNetPlans__2d \
        --fold 0 --out results/randswap/experiment_1_nnunet/latency_ms.json

Timed span, for both: a raw 1228x922 grayscale frame in memory -> the native-resolution
trapped-cell mask -> the shape features (MorphologyExtractor) -- the same span as the
full-pipeline figures in results/postswap/latency_ms.json (preprocess, U-Net, native mask,
features; no classifier). Ours: PreprocessingPipeline (median, CLAHE,
morphology, 384 crop) + forward + probability resampling (feature_mask). nnU-Net: its own
normalisation + sliding-window forward (one tile here: the frame fits its 1024x1280 patch) +
resampling, via nnUNetPredictor.predict_single_npy_array -- once with its default mirroring
test-time augmentation, once without. Median over --n frames of the feature factory after --warmup frames,
GPU otherwise idle, fp32.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from celldform.config import load as load_config
from celldform.features.extractor import MorphologyExtractor
from celldform.integration.realtime import feature_mask
from celldform.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from celldform.segmentation.unet import UNet


def timed(fn, frames, warmup):
    for f in frames[:warmup]:
        fn(f)
    out = []
    for f in frames[warmup:]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(f)
        torch.cuda.synchronize()
        out.append(1000 * (time.perf_counter() - t0))
    return st.median(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--unet", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--crop-center-from", type=Path, required=True, help="a split.json with crop_center")
    ap.add_argument("--nnunet", type=Path, required=True, help="nnU-Net trainer output folder")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--student", type=Path, default=None,
                    help="a MegaNet (distilled student) checkpoint: same network and crop as --unet")
    ap.add_argument("--frames", type=Path, default=Path("data/frames/03_mask_factory"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    paths = sorted(a.frames.glob("*.jpg"))[: a.n + a.warmup]
    frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    h, w = frames[0].shape

    conf = load_config(str(a.config))
    pp = conf.preprocessing
    rec = json.loads(a.crop_center_from.read_text())
    cfg = PreprocessingConfig(output_size=tuple(pp.target_size), median_kernel=pp.median_kernel,
                              clahe_clip_limit=pp.clahe_clip_limit, clahe_tile_grid=tuple(pp.clahe_tile_grid),
                              morph_kernel_size=pp.morph_kernel_size, crop_size=pp.crop_size,
                              crop_center=tuple(rec["crop_center"]))
    pipe = PreprocessingPipeline(cfg)
    unet = UNet.from_checkpoint(a.unet, in_channels=1, out_channels=conf.unet.out_channels,
                                base_features=conf.unet.base_features)

    feats = MorphologyExtractor()

    def ours(img):
        x = pipe(img)
        box = pipe.crop_box(h, w) if cfg.crop_size else None
        return feats.extract(feature_mask(unet, x, (w, h), box).astype(np.uint8))

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    def nn(mirroring):
        p = nnUNetPredictor(use_mirroring=mirroring, device=torch.device("cuda"), allow_tqdm=False)
        p.initialize_from_trained_model_folder(str(a.nnunet), use_folds=(int(a.fold),),
                                               checkpoint_name="checkpoint_final.pth")
        # 2D natural images: shape (c, 1, y, x), dummy z spacing -- as NaturalImage2DIO loads them
        return lambda img: feats.extract((p.predict_single_npy_array(
            img[None, None].astype(np.float32), {"spacing": [999, 1, 1]})[0] == 1).astype(np.uint8))

    def ensemble():
        """Ours + nnU-Net, 0.5/0.5 softmax average at native resolution (Experiment 6), exactly as
        eval_ensemble_cv.py scores it: our crop probabilities pasted into a certain-background frame,
        nnU-Net's probabilities with its default mirroring, averaged, argmax, then the features."""
        p = nnUNetPredictor(use_mirroring=True, device=torch.device("cuda"), allow_tqdm=False)
        p.initialize_from_trained_model_folder(str(a.nnunet), use_folds=(int(a.fold),),
                                               checkpoint_name="checkpoint_final.pth")
        x0, y0, x1, y1 = pipe.crop_box(h, w)

        def run(img):
            po = np.zeros((3, h, w), np.float32); po[0] = 1.0
            po[:, y0:y1, x0:x1] = unet.predict_proba(pipe(img))
            _, pn = p.predict_single_npy_array(img[None, None].astype(np.float32), {"spacing": [999, 1, 1]},
                                               save_or_return_probabilities=True)
            pe = 0.5 * po + 0.5 * np.asarray(pn, np.float32)[:, 0]
            return feats.extract((pe.argmax(0) == 1).astype(np.uint8))
        return run

    def nn_forward_ms(amp: bool, reps: int = 40) -> float:
        """nnU-Net's network alone on one full patch -- the floor any faster wrapper could reach
        (predict_single_npy_array is documented by nnU-Net itself as the slow path)."""
        p = nnUNetPredictor(use_mirroring=False, device=torch.device("cuda"), allow_tqdm=False)
        p.initialize_from_trained_model_folder(str(a.nnunet), use_folds=(int(a.fold),),
                                               checkpoint_name="checkpoint_final.pth")
        net = p.network.to("cuda").eval()
        import json as _j
        patch = _j.loads((a.nnunet / "plans.json").read_text())["configurations"]["2d"]["patch_size"]
        x = torch.randn(1, 1, *patch, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", enabled=amp):
            for _ in range(10):
                net(x)
            ts = []
            for _ in range(reps):
                torch.cuda.synchronize(); t0 = time.perf_counter(); net(x); torch.cuda.synchronize()
                ts.append(1000 * (time.perf_counter() - t0))
        return st.median(ts)

    res = {
        "_what": "Median latency per frame: raw 1228x922 frame in memory -> native trapped-cell mask -> "
                 "shape features (no classifier), RTX 4060 laptop, fp32, GPU otherwise idle. "
                 "scripts/evaluation/latency_benchmark.py",
        "_frames": f"{len(frames) - a.warmup} feature-factory frames after {a.warmup} warm-up",
        "ours_crop384": round(timed(ours, frames, a.warmup), 1),
        "nnunet_no_tta": round(timed(nn(False), frames, a.warmup), 1),
        "nnunet_tta": round(timed(nn(True), frames, a.warmup), 1),
        "nnunet_forward_only_fp32": round(nn_forward_ms(False), 1),
        "nnunet_forward_only_fp16": round(nn_forward_ms(True), 1),
        "ensemble_tta": round(timed(ensemble(), frames, a.warmup), 1),
    }
    if a.student:
        stu = UNet.from_checkpoint(a.student, in_channels=1, out_channels=conf.unet.out_channels,
                                   base_features=conf.unet.base_features)
        box = pipe.crop_box(h, w)
        res["meganet"] = round(timed(lambda img: feats.extract(feature_mask(stu, pipe(img), (w, h), box)
                                                                .astype(np.uint8)), frames, a.warmup), 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
