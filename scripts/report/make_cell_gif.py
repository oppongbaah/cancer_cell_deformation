#!/usr/bin/env python3
"""
Social-media GIF: one trapped cell across the laser-power ramp, raw beside the model's outline.

    python scripts/report/make_cell_gif.py --cell high_cell2 --output results/social/high_cell2.gif
    python scripts/report/make_cell_gif.py --survey high_cell2 low_cell2 ...   # deformation per cell, no GIF

Frames: the cell's feature frames (data/frames/03_mask_factory, every 200th video frame = one per
laser step) -- frames no model was trained on. Masks: through the real pipeline
(PreprocessingPipeline -> feature_mask at native 1228x922), on the CPU so a GPU job can keep running,
from either
  * --round k: that CV round's model, which never saw ANY frame of its test cells -- the script
    refuses a cell that was not one of round k's test cells ("a cell the AI had never seen"), or
  * default: the deployment model (fold_deploy/unet_deploy.pt), which trained on every cell. The view is a fixed square around the
cell's median centroid, upscaled for a phone screen; the outline is drawn, never the mask fill
alone, so the raw cell stays visible under it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from celldform.config import load as load_config
from celldform.features.extractor import MorphologyExtractor
from celldform.integration.realtime import feature_mask
from celldform.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from celldform.segmentation.unet import UNet
from celldform.splits import parse_stem

EXP = "randswap/experiment_0_cv_baseline"
CFG = Path(f"configs/{EXP}/cv_baseline.yaml")


def model_for(cell: str, round_: int | None) -> tuple[Path, Path, str]:
    """(checkpoint, split record, label). A round model is only allowed on its own test cells."""
    if round_ is None:
        return (Path(f"checkpoints/{EXP}/cv_baseline/fold_deploy/unet_deploy.pt"),
                Path(f"results/{EXP}/cv_baseline_fold_deploy_split.json"), "deployment model")
    split = Path(f"results/{EXP}/cv_baseline_fold_{round_}_split.json")
    test = json.loads(split.read_text())["cells"]["test"]
    if cell not in test:
        raise SystemExit(f"{cell} was not a test cell of round {round_} (its test cells: {test}) -- "
                         f"that model has seen it")
    return Path(f"checkpoints/{EXP}/cv_baseline/fold_{round_}/unet_best.pt"), split, f"round-{round_} model"
GREEN = (46, 204, 113)


def masks_for(cell: str, frames_dir: Path, round_: int | None = None):
    ckpt, split, label = model_for(cell, round_)
    print(f"[gif] {cell}: {label} ({ckpt})")
    conf = load_config(str(CFG))
    pp = conf.preprocessing
    cfg = PreprocessingConfig(output_size=tuple(pp.target_size), median_kernel=pp.median_kernel,
                              clahe_clip_limit=pp.clahe_clip_limit, clahe_tile_grid=tuple(pp.clahe_tile_grid),
                              morph_kernel_size=pp.morph_kernel_size, crop_size=pp.crop_size,
                              crop_center=tuple(json.loads(split.read_text())["crop_center"]))
    pipe = PreprocessingPipeline(cfg)
    unet = UNet.from_checkpoint(ckpt, in_channels=1, out_channels=conf.unet.out_channels,
                                base_features=conf.unet.base_features, device="cpu")
    paths = sorted(frames_dir.glob(f"{cell}_*.jpg"), key=lambda p: parse_stem(p.stem)[2])
    out = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        m = feature_mask(unet, pipe(img), (w, h), pipe.crop_box(h, w)).astype(np.uint8)
        out.append((parse_stem(p.stem)[2], img, m))
    return out


def survey(cells, frames_dir, round_=None):
    ex = MorphologyExtractor()
    for c in cells:
        ar = [ex.extract(m).get("aspect_ratio", np.nan) for _, _, m in masks_for(c, frames_dir, round_)]
        ar = np.array(ar, float)
        n = len(ar) // 5
        print(f"{c:<12} aspect ratio: first 5 steps {np.nanmean(ar[:n]):.2f} -> last 5 {np.nanmean(ar[-n:]):.2f}"
              f"  (max {np.nanmax(ar):.2f}, empty masks {int(np.isnan(ar).sum())})")


def font(size):
    for f in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_gif(cell, frames_dir, output, half=70, scale=3, ms=220, hold=1400, step_frames=200, n_steps=50,
             round_=None):
    data = masks_for(cell, frames_dir, round_)
    ex = MorphologyExtractor()
    ars = [ex.extract(m).get("aspect_ratio", float("nan")) for _, _, m in data]
    cents = [np.argwhere(m).mean(0) for _, _, m in data if m.any()]
    cy, cx = np.median(np.array(cents), 0).astype(int)
    side = 2 * half * scale
    f_lab, f_cap = font(22), font(20)
    frames, rgb = [], []
    for k, ((idx, img, m), ar) in enumerate(zip(data, ars), 1):
        step = idx // step_frames + 1          # one laser step = ~200 video frames
        h, w = img.shape
        y0, x0 = min(max(cy - half, 0), h - 2 * half), min(max(cx - half, 0), w - 2 * half)
        raw = cv2.resize(img[y0:y0 + 2 * half, x0:x0 + 2 * half], (side, side), interpolation=cv2.INTER_CUBIC)
        msk = cv2.resize(m[y0:y0 + 2 * half, x0:x0 + 2 * half] * 255, (side, side), interpolation=cv2.INTER_LINEAR)
        msk = cv2.GaussianBlur(msk, (0, 0), scale * 0.9)          # smooth the native-pixel staircase
        left = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        right = left.copy()
        cnts, _ = cv2.findContours((msk > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(right, cnts, -1, GREEN, 3, lineType=cv2.LINE_AA)
        gap, top, bottom = 12, 46, 70
        canvas = Image.new("RGB", (2 * side + gap, side + top + bottom), (18, 18, 18))
        canvas.paste(Image.fromarray(left), (0, top))
        canvas.paste(Image.fromarray(right), (side + gap, top))
        d = ImageDraw.Draw(canvas)
        d.text((14, 12), "What the microscope sees", font=f_lab, fill=(230, 230, 230))
        d.text((side + gap + 14, 12), "What the AI sees", font=f_lab, fill=GREEN)
        # laser-power progress bar
        bx0, bx1, by = 14, 2 * side + gap - 14, side + top + 22
        d.rounded_rectangle((bx0, by, bx1, by + 12), 6, fill=(60, 60, 60))
        d.rounded_rectangle((bx0, by, bx0 + (bx1 - bx0) * step / n_steps, by + 12), 6, fill=(235, 104, 52))
        d.text((bx0, by + 20), f"Laser power  ·  step {step} of {n_steps}", font=f_cap, fill=(230, 230, 230))
        txt = f"Stretch (length ÷ width): {ar:.2f}"
        d.text((bx1 - d.textlength(txt, font=f_cap), by + 20), txt, font=f_cap, fill=GREEN)
        rgb.append(np.asarray(canvas))
        frames.append(canvas.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    output.parent.mkdir(parents=True, exist_ok=True)
    durations = [ms] * (len(frames) - 1) + [hold]
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"-> {output}  ({len(frames)} frames, {output.stat().st_size / 1e6:.1f} MB, {canvas.size[0]}x{canvas.size[1]})")
    # MP4 (H.264): LinkedIn autoplays video; same frames, same timing, the last one held.
    import imageio.v2 as iio
    mp4 = output.with_suffix(".mp4")
    fps = 1000 / ms
    with iio.get_writer(mp4, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
                        macro_block_size=2) as wr:
        for a in rgb:
            wr.append_data(a)
        for _ in range(int(hold / ms)):
            wr.append_data(rgb[-1])
    print(f"-> {mp4}  ({mp4.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cell")
    ap.add_argument("--survey", nargs="+")
    ap.add_argument("--frames", type=Path, default=Path("data/frames/03_mask_factory"))
    ap.add_argument("--output", type=Path)
    ap.add_argument("--half", type=int, default=70, help="half-width of the view, native px")
    ap.add_argument("--round", type=int, default=None, dest="round_",
                    help="use CV round k's model (the cell must be one of its test cells)")
    a = ap.parse_args()
    if a.survey:
        survey(a.survey, a.frames, a.round_)
    else:
        make_gif(a.cell, a.frames, a.output, half=a.half, round_=a.round_)


if __name__ == "__main__":
    main()
