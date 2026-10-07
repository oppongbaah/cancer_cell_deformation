#!/usr/bin/env python3
"""
Distillation data for one randswap round: our round-k training data + nnU-Net (teacher) targets.

    python scripts/data/build_distill_data.py --round 4

For round k, into data/preprocessed/randswap/experiment_3_distill/fold_<k>/:
  train/inputs, train/labels   the round's own CV training data (11 train cells + 1 val cell +
                               legacy), copied unchanged from experiment_0_cv_baseline/fold_<k>, PLUS
                               every --every-th unannotated frame of the round's TRAINING cells (never
                               its validation or test cells), on the offset grid 25, 75, ... so no
                               extra frame can coincide with an annotated (step ~500 / 200) or feature
                               (step 200, offset 100) frame. Extra frames are preprocessed exactly as
                               build_randswap.py did (same pipeline, crop centre, 8-bit PNG); their
                               label is the teacher's argmax.
  train/soft/<stem>.npy        the teacher's class probabilities (3, 384, 384, float16) for EVERY
                               training frame, cropped to the frame's own crop box (legacy frames use
                               the legacy centre).
  test/                        the round's test inputs, copied unchanged (same frames as the CV).
  split.json                   the round's record + annotated_train_stems / extra_stems.

Teacher: nnU-Net fold k-1 (never saw round k's test cells), checkpoint_final.pth, default mirroring
TTA -- the model Experiment 5 scored. Raw frames for the extra cells are cut from the source videos
at 1228x922 with FrameExtractor.seek_extract, the same call the frame pools were built with.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
for var, sub in (("nnUNet_raw", "data/nnunet/raw"), ("nnUNet_preprocessed", "data/nnunet/preprocessed"),
                 ("nnUNet_results", "checkpoints/randswap/experiment_1_nnunet")):
    os.environ.setdefault(var, str(ROOT / sub))

from celldform.acquisition.extractor import FrameExtractor
from celldform.config import load as load_config
from celldform.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from celldform.splits import collect_frames, parse_stem

SRC = "randswap/experiment_0_cv_baseline"
OUT = Path("data/preprocessed/randswap/experiment_3_distill")
POOL = Path("data/frames/04_distill_pool")          # raw extra frames, gitignored like every pool
NN_MODEL = Path(os.environ["nnUNet_results"]) / "Dataset501_CellTrapRandswap" / "nnUNetTrainer_100epochs__nnUNetPlans__2d"


def video_of(cell: str) -> Path:
    her2, n = cell.split("_cell")
    return Path("data/raw") / her2 / f"cell {n} {her2} 50 increments.mp4"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--round", required=True)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--offset", type=int, default=25)
    a = ap.parse_args()
    k = a.round

    conf = load_config(f"configs/{SRC}/cv_baseline.yaml")
    pp = conf.preprocessing
    rec = json.loads(Path(f"results/{SRC}/cv_baseline_fold_{k}_split.json").read_text())
    src = Path(conf.split.data_root) / f"fold_{k}"
    out = OUT / f"fold_{k}"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src / "train", out / "train")
    shutil.copytree(src / "test", out / "test")
    (out / "train" / "soft").mkdir()
    annotated = sorted(p.stem for p in (out / "train" / "inputs").glob("*.png"))

    def pipe_for(cell):
        c = PreprocessingConfig(output_size=tuple(pp.target_size), median_kernel=pp.median_kernel,
                                clahe_clip_limit=pp.clahe_clip_limit, clahe_tile_grid=tuple(pp.clahe_tile_grid),
                                morph_kernel_size=pp.morph_kernel_size, crop_size=pp.crop_size,
                                crop_center=tuple(rec["legacy_crop_center"] if cell is None else rec["crop_center"]))
        return PreprocessingPipeline(c)

    # raw images of the annotated training frames
    raw = {f.stem: f.image for f in collect_frames(conf.split.sources) if f.stem in set(annotated)}
    assert len(raw) == len(annotated), "every copied training frame must have its raw image"

    # extra frames: the round's TRAINING cells only. Frames are compared by (cell, frame INDEX),
    # never by file name: annotated frames carry a "domain_" prefix, so a name check let two low-2
    # frames (2475 and 7425, on its 495-step annotation grid) in twice on 2026-10-01. Annotated
    # frames of every pool and the classifier's feature frames are all excluded.
    taken: dict = {}
    for f in collect_frames(conf.split.sources):
        if f.cell is not None:
            taken.setdefault(f.cell, set()).add(f.index)
    for p in Path("data/frames/03_mask_factory").glob("*.jpg"):
        c, _, i = parse_stem(p.stem)
        taken.setdefault(c, set()).add(i)
    extra = []
    for cell in rec["cells"]["train"]:
        vid = video_of(cell)
        cap = cv2.VideoCapture(str(vid)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        targets = [t for t in range(a.offset, total, a.every) if t not in taken.get(cell, set())]
        d = POOL / cell
        d.mkdir(parents=True, exist_ok=True)
        todo = [t for t in targets if not (d / f"{cell}_{t:06d}.jpg").exists()]
        if todo:
            FrameExtractor(vid, output_dir=d, target_size=(1228, 922), image_format="jpg",
                           change_threshold=None).seek_extract(todo, file_stem=cell)
        for t in targets:
            p = d / f"{cell}_{t:06d}.jpg"
            if p.exists():
                stem = p.stem
                assert stem not in raw and parse_stem(stem)[0] == cell
                raw[stem] = p
                extra.append(stem)
        print(f"  {cell}: {sum(1 for s in extra if s.startswith(cell + '_'))} extra frames", flush=True)
    forbidden = set(rec["cells"]["test"]) | set(rec["cells"]["val"])
    assert not any(parse_stem(s)[0] in forbidden for s in extra)
    assert not any(parse_stem(s)[2] in taken.get(parse_stem(s)[0], set()) for s in extra), "frame reused"

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    teacher = nnUNetPredictor(device=torch.device("cuda"), allow_tqdm=False)       # default: mirroring TTA
    teacher.initialize_from_trained_model_folder(str(NN_MODEL), use_folds=(int(k) - 1,),
                                                 checkpoint_name="checkpoint_final.pth")
    pipes = {None: pipe_for(None), "cell": pipe_for("cell")}
    agree = []
    for n, stem in enumerate(annotated + extra, 1):
        cell = parse_stem(stem)[0]
        pipe = pipes[None if cell is None else "cell"]
        img = cv2.imread(str(raw[stem]), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        _, probs = teacher.predict_single_npy_array(img[None, None].astype(np.float32), {"spacing": [999, 1, 1]},
                                                    save_or_return_probabilities=True)
        x0, y0, x1, y1 = pipe.crop_box(h, w)
        soft = np.asarray(probs, np.float32)[:, 0, y0:y1, x0:x1]
        np.save(out / "train" / "soft" / f"{stem}.npy", soft.astype(np.float16))
        if stem in extra:
            cv2.imwrite(str(out / "train" / "inputs" / f"{stem}.png"), (pipe(img) * 255).astype(np.uint8))
            cv2.imwrite(str(out / "train" / "labels" / f"{stem}.png"), soft.argmax(0).astype(np.uint8))
        else:
            gt = cv2.imread(str(out / "train" / "labels" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            p1, g1 = soft.argmax(0) == 1, gt == 1
            if g1.any():
                agree.append(2 * (p1 & g1).sum() / (p1.sum() + g1.sum()))
        if n % 250 == 0:
            print(f"  teacher {n}/{len(annotated) + len(extra)}", flush=True)

    rec = dict(rec, annotated_train_stems=annotated, extra_stems=extra, teacher=f"nnU-Net fold {int(k) - 1}",
               extra_grid=f"range({a.offset}, total, {a.every})")
    (out / "split.json").write_text(json.dumps(rec, indent=1))
    print(f"-> {out}: {len(annotated)} annotated + {len(extra)} teacher-labelled training frames; "
          f"teacher vs hand masks on the annotated frames: mean Dice {np.mean(agree):.3f}")


if __name__ == "__main__":
    main()
