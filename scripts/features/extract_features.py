  #!/usr/bin/env python3
"""
Classifier features from the feature frames -- segment, validate every segmentation, then extract.

Every cell is segmented by the cross-validation round model that never saw it, so no cell's
features come from a model trained on that cell. Two protocols supply those models:

  5-fold     5 checkpoints, each segments its round's 3 test cells (Dataset501, fold k-1 = round k)
  LOOCV      15 checkpoints, each segments the one cell it held out (Dataset502, fold r-1 = round r;
             which cell each round held out is read from its eval log). Since 2026-10-21 the default
             (user decision): high-10's round-14 model over-predicts it and is to be investigated and
             rerun later -- the 5-fold had its own weak cell (high-1).

Two sets of frames go through the same code:

  test     the 20 annotated test frames per cell (hand-drawn masks exist) -- used to CALIBRATE
           the segmentation checks: do the frames that pass really have good masks and accurate
           features?
  frames   the feature frames -- the classifier's input. Since 2026-10-22 up to 6 per laser step: the step
           frame (data/frames/03_mask_factory, at offset 0 or 100 of the step) and the frames at offsets 20,
           60, 100, 140, 180 (data/frames/05_dense_frames; a frame already in the factory or hand-masked is not
           cut again). Both folders are ONE set: checked together (the neighbouring frames that set a frame's
           typical size and local trap come from both) and written to one table, column `offset`. A frame's
           features are kept only if its segmentation passes.

Steps (each reads the previous one's files, so any can be rerun alone):

  segment    nnU-Net prediction of both sets: labels (0 bg, 1 cell, 2 decoy) and the per-pixel
             top-class probability, at native 1228x922.
  calibrate  the checks on the TEST set against the hand-drawn masks: Dice and feature error of
             passing vs failing frames, per check and per HER2 class.
  extract    the FEATURE frames: checks, pass/fail with reasons, the 15 shape and 7 deformation features
             of passing frames, image quality (focus, illumination, contrast, noise) of every frame, the
             per-cell deformation summaries, and a high-vs-low comparison of the image measurements
             (recording-session check).
  (cut       cuts the frames between the step frames from the videos -- once, before segment)

The trapped cell is the predicted region nearest the video's trap (median cell position over its own
frames), not the largest region -- in high-10 the untrapped neighbour is larger. Its distance is then
checked against a LOCAL trap, the median cell position over the neighbouring laser steps (+-3), so a cell
that sits a little off-centre for a while is followed, and only a sudden jump (a different object, a
cell not yet trapped) is flagged.

    python scripts/features/extract_features.py segment   --model nnunet1000_loocv
    python scripts/features/extract_features.py calibrate --model nnunet1000_loocv
    python scripts/features/extract_features.py extract   --model nnunet1000_loocv
    python scripts/features/extract_features.py all       --model nnunet1000_loocv

The checkpoints of the 1000-epoch models are on the cluster: run the segment step there
(scripts/hpc/evaluation/extract_features.sh), bring back data/features/<model>/, then calibrate and
extract anywhere (CPU only).

Outputs
  data/features/inputs/<protocol>_fold<f>/{feature,test}/    frames as nnU-Net input, per checkpoint
  data/features/<model>/<set>/{labels,maxprob}/<stem>.png    predictions
  results/features/<model>/
      checks_<set>.csv        one row per frame: checks, pass, reasons (+ Dice, GT features on test)
      calibration.json        how well the checks sort good from bad masks (test set)
      features_frames.csv     the feature frames, one row per frame (offset = frames into its laser step);
                              features NaN where the frame failed
      features_cells.csv      per cell: frames passed, median of each feature, deformation summaries
      image_quality_session.csv   each image measurement, high vs low HER2 (cell means)
      rejected_frames.png     every rejected feature frame: the mask on the raw frame, with the reason
      multi_region_{test,feature}.csv/.png   every frame with more than one trapped-cell region: all regions,
                              their distance from the trap and which one was kept
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from celldform.acquisition.extractor import FrameExtractor
from celldform.features.deformation import (FRAME_FEATURES, cell_summary,
                                            frame_features, within_step)
from celldform.features.extractor import MorphologyExtractor
from celldform.features.quality import (Gates, apply_gates, image_quality, regions, segmentation_checks,
                                        select_trapped_region, trap_position)
from celldform.features.segmenters import DATASET_5FOLD as DATASET
from celldform.features.segmenters import MODELS, five_fold_records, plan
from celldform.analysis_config import dump as dump_config, load as load_config
from celldform.utils.runlog import start_run_log

FACTORY = ROOT / "data" / "frames" / "03_mask_factory"
DENSE = ROOT / "data" / "frames" / "05_dense_frames"   # the feature frames between the step frames (2026-10-22):
SETS = {"feature": ("feature", "dense")}   # the feature frames are segmented from two folders, checked as one set
# The settings below come from configs/analysis/features.yaml (celldform.analysis_config; --config for another
# file). What each one is for:
#   DENSE_OFFSETS      frames into a step cut besides the step frame: within-step behaviour, a steadier step value
#   FRAMES_PER_STEP    one laser step = 200 video frames
#   GOOD_DSC           a hand-masked frame with Dice below this counts as a bad mask in calibration
#   AREA_WINDOW        "typical area" = median over frames within +-window_steps laser steps (2026-10-21: a
#                      whole-video median rejected the real shrinking of the cell at high laser power)
#   TRAP_WINDOW        the trap a frame is checked against = median cell position within the same window
#                      (2026-10-21: one position per video dropped high-4's correctly segmented early frames)
#   MIN_PASS_FRACTION  a cell with fewer of its feature frames passing is flagged
#   CELL_MIN_DSC, CELL_SIZE_BIAS  cell-level validation (2026-10-21): a frame check cannot see a bias that is
#                      the same on every frame, so each cell must also be segmented accurately on its own
#                      hand-masked frames, by the same held-out model


def configure(cfg) -> None:
    """Set the module's settings from a FeaturesConfig."""
    global CFG, GATES, DENSE_OFFSETS, FRAMES_PER_STEP, GOOD_DSC, AREA_WINDOW, TRAP_WINDOW, MIN_PASS_FRACTION
    global CELL_MIN_DSC, CELL_SIZE_BIAS
    CFG = cfg
    fc, cv = cfg.frame_checks, cfg.cell_validation
    GATES = Gates(max_trap_dist=fc.max_trap_dist, min_solidity=fc.min_solidity,
                  area_ratio_range=tuple(fc.area_ratio_range), min_boundary_conf=fc.min_boundary_conf,
                  allow_border=fc.allow_border)
    DENSE_OFFSETS = tuple(cfg.dense_offsets)
    FRAMES_PER_STEP = cfg.frames_per_step
    GOOD_DSC = cfg.good_dice
    AREA_WINDOW = fc.window_steps * FRAMES_PER_STEP + FRAMES_PER_STEP // 2
    TRAP_WINDOW = AREA_WINDOW
    MIN_PASS_FRACTION = cv.min_pass_fraction
    CELL_MIN_DSC = cv.min_dice
    CELL_SIZE_BIAS = tuple(cv.size_bias_range)


configure(load_config("features"))
SHAPE = ["area", "perimeter", "major_axis", "minor_axis", "aspect_ratio", "eccentricity",
         "circularity", "solidity"]


def cell_of(stem: str) -> str:
    return re.search(r"(high|low)_cell\d+", stem).group(0)


def index_of(stem: str) -> int:
    return int(re.search(r"_(\d+)$", stem).group(1))


def raw_path(which: str, stem: str) -> Path:
    if which == "test":
        return ROOT / "data" / "nnunet" / "raw" / DATASET / "imagesTr" / f"{stem}_0000.png"
    return (DENSE if which == "dense" else FACTORY) / f"{stem}.jpg"


def located(model: str, which: str) -> dict[str, tuple[Path, Path, Path]]:
    """stem -> (labels, top-class probability, raw frame) for every segmented frame of a set. The feature set
    reads both of its folders; a frame in both (cut twice before 2026-10-22) is taken from the factory."""
    out: dict[str, tuple[Path, Path, Path]] = {}
    for sub in SETS.get(which, (which,)):
        lab, mp = out_dirs(model, sub)
        for p in sorted(lab.glob("*.png")):
            out.setdefault(p.stem, (p, mp / p.name, raw_path(sub, p.stem)))
    return out


def dense_frames() -> None:
    """Cut DENSE_OFFSETS frames per laser step from each cell's video, exactly as the feature frames were cut
    (FrameExtractor.seek_extract: grayscale, INTER_AREA resize to 1228x922, JPEG), skipping hand-masked frames."""
    import glob
    DENSE.mkdir(parents=True, exist_ok=True)
    hand = {(cell_of(s), index_of(s)) for r in five_fold_records().values() for s in r["test_masks_native"]}
    for man in sorted(glob.glob(str(FACTORY / "00_cell *_manifest.csv"))):
        m = pd.read_csv(man)
        video = ROOT / m.source.iloc[0]
        if not video.exists():          # the raw videos were regrouped into high/ low/ after the factory was cut
            video = next((ROOT / "data" / "raw").rglob(video.name))
        cell = cell_of(Path(m.output_path.iloc[0]).stem)
        steps = sorted({int(i) // FRAMES_PER_STEP for i in m.frame_index})
        total = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT))
        targets = [s * FRAMES_PER_STEP + o for s in steps for o in DENSE_OFFSETS]
        have = {int(i) for i in m.frame_index}          # already a feature frame (the step frame)
        targets = [t for t in targets if t < total and (cell, t) not in hand and t not in have
                   and not (DENSE / f"{cell}_{t:06d}.jpg").exists()]
        if targets:
            FrameExtractor(video, output_dir=DENSE, target_size=(1228, 922), image_format="jpg",
                           change_threshold=None).seek_extract(targets, file_stem=cell)
        print(f"  {cell}: {len(list(DENSE.glob(f'{cell}_*.jpg')))} dense frames", flush=True)


def out_dirs(model: str, which: str) -> tuple[Path, Path]:
    base = ROOT / "data" / "features" / model / which
    return base / "labels", base / "maxprob"


# ── segment ──────────────────────────────────────────────────────────────────

def fold_inputs(protocol: str, f: int, cells: list[str]) -> dict[str, Path]:
    """The frames one checkpoint segments, as nnU-Net input (<stem>_0000.png, grayscale -- exactly how
    scripts/data/build_nnunet_dataset.py wrote the training frames):
      feature  its cells' feature frames (data/frames/03_mask_factory)
      test     its cells' hand-masked test frames (the 5-fold dataset's imagesTs_round*)"""
    base = ROOT / "data" / "features" / "inputs" / f"{protocol}_fold{f}"
    raw = ROOT / "data" / "nnunet" / "raw" / DATASET
    srcs = {"feature": [p for c in cells for p in sorted(FACTORY.glob(f"{c}_*.jpg"))],
            "test": [p for c in cells for p in sorted(raw.glob(f"imagesTs_round*/*{c}_*_0000.png"))
                     if cell_of(p.name) == c],
            "dense": [p for c in cells for p in sorted(DENSE.glob(f"{c}_*.jpg"))]}
    if not srcs["dense"]:
        del srcs["dense"]             # the dense frames are optional (dense_frames step)
    for which, files in srcs.items():
        if not files:
            raise SystemExit(f"{protocol} fold {f}: no {which} frames for {cells}")
        d = base / which
        d.mkdir(parents=True, exist_ok=True)
        for src in files:
            dst = d / (src.name if src.suffix == ".png" else f"{src.stem}_0000.png")
            if dst.exists():
                continue
            if src.suffix == ".png":
                shutil.copy2(src, dst)
            else:
                cv2.imwrite(str(dst), cv2.imread(str(src), cv2.IMREAD_GRAYSCALE))
    return {w: base / w for w in srcs}


def segment(model: str) -> None:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    seg = MODELS[model]
    for var, sub in (("nnUNet_raw", "data/nnunet/raw"), ("nnUNet_preprocessed", "data/nnunet/preprocessed"),
                     ("nnUNet_results", f"checkpoints/{seg.exp}")):
        os.environ.setdefault(var, str(ROOT / sub))
    model_dir = seg.model_dir
    rounds = plan(model)
    missing = [r["fold"] for r in rounds if not (model_dir / f"fold_{r['fold']}" / "checkpoint_final.pth").exists()]
    if missing:
        raise SystemExit(f"{model_dir.relative_to(ROOT)}: no checkpoint_final.pth for folds {missing} -- "
                         f"are this model's checkpoints on this machine?")
    print(f"{seg.label}: {len(rounds)} checkpoints, {model_dir.relative_to(ROOT)}")
    for r in rounds:
        f, cells = r["fold"], r["test"]
        ckpt = model_dir / f"fold_{f}" / "checkpoint_final.pth"
        for which, src in fold_inputs(seg.protocol, f, cells).items():
            lab, mp = out_dirs(model, which)
            stems = [p.name[:-9] for p in sorted(src.glob("*_0000.png"))]
            if all((lab / f"{s}.png").exists() and (mp / f"{s}.png").exists() for s in stems):
                print(f"fold {f} {which}: {len(stems)} frames already segmented")
                continue
            tmp = lab.parent / f"_tmp_fold{f}"
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"fold {f} {which} ({', '.join(cells)}): segmenting {len(stems)} frames with "
                  f"{ckpt.relative_to(ROOT)}", flush=True)
            # TTA on, as in the CV scoring. The sliding-window logits are accumulated on the CPU, not the GPU
            # (2026-10-28): on the 8 GB laptop GPU the on-device attempt ran out of memory and, under WSL, the
            # fallback died with "CUDA error: unknown error" twice. The network still runs on the GPU.
            p = nnUNetPredictor(device=torch.device("cuda"), perform_everything_on_device=False, allow_tqdm=False)
            p.initialize_from_trained_model_folder(str(model_dir), use_folds=(f,),
                                                   checkpoint_name="checkpoint_final.pth")
            p.predict_from_files(str(src), str(tmp), save_probabilities=True,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2)
            del p
            torch.cuda.empty_cache()    # release this round's network before the next one loads
            lab.mkdir(parents=True, exist_ok=True)
            mp.mkdir(parents=True, exist_ok=True)
            for s in stems:
                shutil.move(str(tmp / f"{s}.png"), lab / f"{s}.png")
                prob = np.load(tmp / f"{s}.npz")["probabilities"].squeeze()          # (classes, H, W)
                cv2.imwrite(str(mp / f"{s}.png"), np.round(prob.max(0) * 255).astype(np.uint8))
            shutil.rmtree(tmp)          # the full probability maps are ~7 MB a frame; only the top-class map is kept


# ── checks (both sets) ───────────────────────────────────────────────────────

def check_frames(model: str, which: str, gates: Gates, ex: MorphologyExtractor) -> pd.DataFrame:
    files = located(model, which)
    rs = five_fold_records()
    her2 = next(iter(rs.values()))["her2"]
    gt_path = {s: ROOT / m for r in rs.values() for s, m in r.get("test_masks_native", {}).items()}
    stems = sorted(files)
    if not stems:
        raise SystemExit(f"no {which} predictions for {model} -- run the segment step first")
    rows = []
    for cell in sorted({cell_of(s) for s in stems}):
        cs = [s for s in stems if cell_of(s) == cell]
        labels = {s: cv2.imread(str(files[s][0]), cv2.IMREAD_UNCHANGED) for s in cs}
        # trap position from this video's own predictions: frames with exactly one cell region
        cents = {s: rr[0]["centroid"] for s, rr in ((s, regions(labels[s] == 1)) for s in cs) if len(rr) == 1}
        if len(cents) < 3:
            cents = {s: max(rr, key=lambda x: x["area"])["centroid"]
                     for s, rr in ((s, regions(labels[s] == 1)) for s in cs) if rr}
        trap = trap_position(list(cents.values()))       # the whole video's: reported, and the fallback
        if trap:   # every frame's cell position: the region nearest the video's trap (multi-region frames too)
            cents = {s: r["centroid"] for s in cs if (r := select_trapped_region(labels[s] == 1, trap)[0])}

        def local_trap(s):  # median cell position over the NEIGHBOURING laser steps (the frame itself excluded)
            near = [c for t, c in cents.items() if t != s and abs(index_of(t) - index_of(s)) <= TRAP_WINDOW]
            return trap_position(near) if len(near) >= 2 else trap
        traps = {s: local_trap(s) if trap else None for s in cs}
        # the region is CHOSEN by the video's trap (stable: a shifted local estimate can sit between the cell
        # and a neighbour); its distance is CHECKED against the local trap (follows a slow shift of the cell)
        picked = {s: select_trapped_region(labels[s] == 1, trap) if trap else (None, 0) for s in cs}
        for s, (reg, _) in picked.items():
            if reg is not None:
                reg["trap_dist"] = float(np.hypot(reg["centroid"][0] - traps[s][0], reg["centroid"][1] - traps[s][1]))
        # typical area from neighbours whose OWN mask is sound (near the trap, not merged, not cut) --
        # a merged neighbour would otherwise inflate it and reject the good frames next to it
        def sound(s, p):
            if p[0] is None:
                return False
            c0 = segmentation_checks(p[0], labels[s], None, None, p[1])
            return c0["trap_dist"] <= gates.max_trap_dist and c0["solidity"] >= gates.min_solidity \
                and not c0["touches_border"]
        area = {s: p[0]["area"] for s, p in picked.items() if sound(s, p)}
        whole = float(np.median(list(area.values()))) if area else np.nan

        def typical(s):     # median area of the NEIGHBOURING frames (the frame itself excluded)
            near = [a for t, a in area.items() if t != s and abs(index_of(t) - index_of(s)) <= AREA_WINDOW]
            return float(np.median(near)) if len(near) >= 2 else whole
        for s in cs:
            reg, n = picked[s]
            mp = cv2.imread(str(files[s][1]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
            c = segmentation_checks(reg, labels[s], mp, typical(s), n)
            ok, why = apply_gates(c, gates)
            raw = cv2.imread(str(files[s][2]), cv2.IMREAD_GRAYSCALE)
            row = {"stem": s, "cell": cell, "her2": her2[cell], "frame_index": index_of(s),
                   "laser_step": index_of(s) // FRAMES_PER_STEP, "offset": index_of(s) % FRAMES_PER_STEP,
                   "trap_x": trap[0] if trap else np.nan, "trap_y": trap[1] if trap else np.nan,   # chooses
                   "local_trap_x": traps[s][0] if traps[s] else np.nan,                          # checks
                   "local_trap_y": traps[s][1] if traps[s] else np.nan, **c, "passed": ok, "reasons": "; ".join(why),
                   **image_quality(raw, reg["mask"] if reg else None)}
            feats = ex.extract(reg["mask"].astype(np.uint8)) if reg else {k: np.nan for k in ex._FEATURE_NAMES}
            row.update(feats)
            row.update(frame_features(reg["mask"] if reg else None))
            if which == "test":
                gt = cv2.imread(str(gt_path[s]), cv2.IMREAD_UNCHANGED) == 1
                pr = reg["mask"] if reg else np.zeros_like(gt)
                row["dsc"] = 2 * (gt & pr).sum() / (gt.sum() + pr.sum() + 1e-9)
                row.update({f"gt_{k}": v for k, v in ex.extract(gt.astype(np.uint8)).items()})
                row.update({f"gt_{k}": v for k, v in frame_features(gt).items()})
            rows.append(row)
        print(f"  {which:7s} {cell:12s} trap ({trap[0]:.0f}, {trap[1]:.0f})  passed "
              f"{sum(r['passed'] for r in rows if r['cell'] == cell)}/{len(cs)}" if trap else f"  {cell}: no trap found")
    return pd.DataFrame(rows)


def limits_from(good: pd.DataFrame) -> Gates:
    """The rule the limits were set by: just outside the GOOD masks -- their 99th percentile of distance,
    1st of solidity and outline confidence, 1st-99th of the size ratio."""
    return Gates(max_trap_dist=float(good.trap_dist.quantile(0.99)), min_solidity=float(good.solidity.quantile(0.01)),
                 area_ratio_range=(float(good.area_ratio.quantile(0.01)), float(good.area_ratio.quantile(0.99))),
                 min_boundary_conf=float(good.boundary_conf.quantile(0.01)))


def cross_check(df: pd.DataFrame, gates: Gates) -> dict:
    """How the frame checks do on frames that did NOT set their limits. For each cell, the limits are set by
    the rule above from the other cells' good masks only, then applied to this cell's hand-masked frames.
    Compared with the limits used (fixed, set on all frames) and with the rule applied to all frames."""
    good = df.dsc >= GOOD_DSC

    def score(passed: pd.Series) -> dict:
        return {"good_dropped": int((good & ~passed).sum()), "good": int(good.sum()),
                "bad_caught": int((~good & ~passed).sum()), "bad": int((~good).sum())}

    def run(g: Gates, rows: pd.DataFrame) -> pd.Series:
        return pd.Series([apply_gates(r, g)[0] for r in rows.to_dict("records")], index=rows.index)

    held = pd.Series(False, index=df.index)
    per_cell = {}
    for cell in df.cell.unique():
        g = limits_from(df[good & (df.cell != cell)])
        rows = df[df.cell == cell]
        held.loc[rows.index] = run(g, rows)
        per_cell[cell] = {"max_trap_dist": round(g.max_trap_dist, 1), "min_solidity": round(g.min_solidity, 3),
                          "area_ratio_range": [round(v, 3) for v in g.area_ratio_range],
                          "min_boundary_conf": round(g.min_boundary_conf, 3)}
    return {"rule": "limits = 99th pct distance, 1st pct solidity and confidence, 1st-99th pct size ratio, of the good masks",
            "used": score(df.passed.astype(bool)), "rule_on_all_frames": score(run(limits_from(df[good]), df)),
            "rule_without_the_cell": score(held), "limits_without_the_cell": per_cell}


def cell_feature_check(df: pd.DataFrame, cv: pd.DataFrame, res: Path) -> None:
    """The per-cell deformation features computed twice on each cell's hand-masked frames -- from the predicted
    masks (passing frames) and from the ground-truth masks (all frames) -- to see whether the segmentation
    preserves them. ~20 frames per cell, so these are coarser than on the feature frames."""
    rows = []
    for cell, g in df.groupby("cell"):
        truth = g[["laser_step"]].assign(passed=True, **{k: g[f"gt_{k}"] for k in FRAME_FEATURES})
        pr, gt = cell_summary(g), cell_summary(truth)
        rows.append({"cell": cell, "her2": g.her2.iloc[0], "validated": bool(cv.loc[cell, "validated"]),
                     **{f"pred_{k}": v for k, v in pr.items()}, **{f"gt_{k}": v for k, v in gt.items()}})
    pd.DataFrame(rows).to_csv(res / "cell_feature_check.csv", index=False)


def calibrate(model: str, gates: Gates) -> None:
    ex = MorphologyExtractor()
    df = check_frames(model, "test", gates, ex)
    res = ROOT / "results" / "features" / model
    res.mkdir(parents=True, exist_ok=True)
    df.to_csv(res / "checks_test.csv", index=False)
    multi_region_report(model, "test", df, res, gates)
    bad = df["dsc"] < GOOD_DSC
    p = df["passed"]
    # relative error, except eccentricity (0 for a circle): absolute difference there
    rel = {f: ((df[f] - df[f"gt_{f}"]).abs() / (1 if f == "eccentricity" else df[f"gt_{f}"].abs())) for f in SHAPE}
    df["size_vs_gt"] = df["area"] / df["gt_area"]
    cv = df[p].groupby("cell").agg(her2=("her2", "first"), frames=("stem", "size"), dsc=("dsc", "mean"),
                                   size_bias=("size_vs_gt", "median"), dsc_min=("dsc", "min"))
    cv["validated"] = (cv.dsc >= CELL_MIN_DSC) & cv.size_bias.between(*CELL_SIZE_BIAS)
    cv.to_csv(res / "cell_validation.csv")
    cell_feature_check(df, cv, res)
    summary = {
        "gates": gates.as_dict(), "good_dsc": GOOD_DSC, "frames": int(len(df)),
        "passed": int(p.sum()), "bad_masks": int(bad.sum()),
        "bad_masks_caught": int((bad & ~p).sum()), "bad_masks_passed": int((bad & p).sum()),
        "good_masks_rejected": int((~bad & ~p).sum()),
        "dsc_passed_mean": float(df.loc[p, "dsc"].mean()), "dsc_failed_mean": float(df.loc[~p, "dsc"].mean()),
        "dsc_passed_min": float(df.loc[p, "dsc"].min()),
        "median_rel_error_passed": {f: float(rel[f][p].median()) for f in SHAPE},
        "p90_rel_error_passed": {f: float(rel[f][p].quantile(0.9)) for f in SHAPE},
        "by_her2": {h: {"passed": f"{int((p & (df.her2 == h)).sum())}/{int((df.her2 == h).sum())}",
                        "dsc_passed_mean": float(df.loc[p & (df.her2 == h), "dsc"].mean()),
                        "area_rel_error_median": float(rel['area'][p & (df.her2 == h)].median())}
                    for h in ("high", "low")},
        "per_cell": {c: {"passed": f"{int(g.passed.sum())}/{len(g)}", "dsc_passed": round(float(g.loc[g.passed, 'dsc'].mean()), 3)
                         if g.passed.any() else None, "dsc_all": round(float(g.dsc.mean()), 3)}
                     for c, g in df.groupby("cell")},
    }
    summary["cross_check"] = cross_check(df, gates)
    (res / "calibration.json").write_text(json.dumps(summary, indent=1))
    cc = summary["cross_check"]
    print("  CROSS-CHECK (limits set without the cell they are applied to):")
    for k in ("used", "rule_on_all_frames", "rule_without_the_cell"):
        v = cc[k]
        print(f"    {k:22s} good masks dropped {v['good_dropped']}/{v['good']}, bad masks caught {v['bad_caught']}/{v['bad']}")
    print(f"\nCALIBRATION on {len(df)} annotated test frames ({model}); bad mask = Dice < {GOOD_DSC}")
    print(f"  passed {p.sum()}/{len(df)}   bad masks {bad.sum()}: caught {summary['bad_masks_caught']}, "
          f"slipped through {summary['bad_masks_passed']};   good masks rejected {summary['good_masks_rejected']}")
    print(f"  Dice of passed frames {summary['dsc_passed_mean']:.3f} (min {summary['dsc_passed_min']:.3f}), "
          f"of failed frames {summary['dsc_failed_mean']:.3f}")
    print("  feature error of passed frames vs the hand-drawn mask (median / 90th pct):")
    for f in SHAPE:
        u = "" if f == "eccentricity" else "%"
        k = 1 if f == "eccentricity" else 100
        print(f"    {f:13s} {k * summary['median_rel_error_passed'][f]:5.3g}{u} / {k * summary['p90_rel_error_passed'][f]:5.3g}{u}"
              + ("   (absolute)" if f == "eccentricity" else ""))
    for h, v in summary["by_her2"].items():
        print(f"  {h:4s} HER2: passed {v['passed']}, Dice {v['dsc_passed_mean']:.3f}, area error {100 * v['area_rel_error_median']:.1f}%")
    print(f"\n  CELL VALIDATION (annotated frames, same model): Dice >= {CELL_MIN_DSC}, size bias "
          f"{CELL_SIZE_BIAS[0]}-{CELL_SIZE_BIAS[1]}")
    print(cv.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"  -> {int(cv.validated.sum())}/{len(cv)} cells validated")
    reasons = df.loc[~p, "reasons"].str.split("; ").explode().str.replace(r"[\d.]+", "#", regex=True).value_counts()
    print("  why frames failed:\n" + "\n".join(f"    {n:4d}  {r}" for r, n in reasons.items()))


# ── extract (feature set) ────────────────────────────────────────────────────

def frame_view(img: np.ndarray, masks: list[np.ndarray], trap: tuple[float, float], r_lim: float,
               half: int) -> tuple[np.ndarray, int]:
    """A square crop holding every mask and the trap's limit circle (at least 2*half px), zero-padded at the
    frame edge, so a region far from the trap is still in view."""
    x, y = int(trap[0]), int(trap[1])
    ys, xs = np.nonzero(np.any(masks, axis=0)) if masks else (np.array([y]), np.array([x]))
    x0, x1 = min(xs.min(), x - r_lim) - 12, max(xs.max(), x + r_lim) + 12
    y0, y1 = min(ys.min(), y - r_lim) - 12, max(ys.max(), y + r_lim) + 12
    side = int(max(2 * half, x1 - x0, y1 - y0))
    cx, cy = int((x0 + x1) // 2), int((y0 + y1) // 2)
    pad = cv2.copyMakeBorder(img, side, side, side, side, cv2.BORDER_CONSTANT)
    return pad[cy - side // 2 + side:cy - side // 2 + 2 * side, cx - side // 2 + side:cx - side // 2 + 2 * side], side


def label_lines(tile: np.ndarray, y0: int, head: str, head_col: tuple, reasons: list[str], col: tuple) -> None:
    """The tile's caption: a header line, then the reasons wrapped at ~38 characters (at most 3 lines)."""
    cv2.putText(tile, head, (4, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.42, head_col, 1)
    lines: list[str] = []
    for w in reasons:
        if lines and len(lines[-1]) + len(w) < 38:
            lines[-1] += f"; {w}"
        else:
            lines.append(w)
    for k, line in enumerate(lines[:3]):
        cv2.putText(tile, line, (4, y0 + 18 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)


def review_sheet(model: str, df: pd.DataFrame, out: Path, half: int = 70, cols: int = 6, which: str = "feature") -> None:
    """Every rejected feature frame on its raw frame, the PREDICTED mask outlined in the report's colours --
    selected region green, other trapped-cell-class regions white, other-object class grey -- the trap
    (yellow cross) and its distance limit (circle), with the failed checks underneath. The view grows to
    show a region that lies far from the trap."""
    files = located(model, which)
    r_lim = GATES.max_trap_dist
    tiles = []
    for _, r in df[~df.passed].sort_values(["cell", "frame_index"]).iterrows():
        img = cv2.cvtColor(cv2.imread(str(files[r.stem][2]), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        lab = cv2.imread(str(files[r.stem][0]), cv2.IMREAD_UNCHANGED)
        rs = regions(lab == 1)
        kept, _ = select_trapped_region(lab == 1, (r.trap_x, r.trap_y)) if rs else (None, 0)
        for m, col in [((lab == 2).astype(np.uint8), (110, 110, 110))] + \
                [(reg["mask"].astype(np.uint8), (0, 200, 0) if kept and reg["label"] == kept["label"] else (255, 255, 255))
                 for reg in rs]:
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnts, -1, col, 1)
        x, y = int(r.local_trap_x), int(r.local_trap_y)
        cv2.drawMarker(img, (x, y), (0, 200, 255), cv2.MARKER_CROSS, 10, 1)
        cv2.circle(img, (x, y), int(r_lim), (0, 200, 255), 1, cv2.LINE_AA)
        crop, side = frame_view(img, [reg["mask"] for reg in rs], (x, y), r_lim, half)
        crop = cv2.resize(crop, (280, 280), interpolation=cv2.INTER_AREA if side > 280 else cv2.INTER_NEAREST)
        cv2.putText(crop, f"{side} px", (4, 274), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        tile = cv2.copyMakeBorder(crop, 0, 70, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        label_lines(tile, 296, f"{r.cell.replace('_cell', '-')} step {r.laser_step}  DROPPED", (30, 30, 200),
                    str(r.reasons).split("; "), (30, 30, 200))
        tiles.append(tile)
    if not tiles:
        return
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % cols)
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def multi_region_report(model: str, which: str, df: pd.DataFrame, out_dir: Path, gates: Gates,
                        half: int = 70, cols: int = 6) -> None:
    """Every frame where the model marks MORE THAN ONE trapped-cell region: one CSV row per region (area,
    centre, distance from the trap, which one the extraction kept; on the test set also its Dice with
    the ground truth) and a review sheet, each region outlined and numbered (kept one green, others red)."""
    files = located(model, which)
    gt_path = {s: ROOT / m for r in five_fold_records().values() for s, m in r.get("test_masks_native", {}).items()}
    rows, tiles = [], []
    for _, r in df[df.n_regions > 1].sort_values(["cell", "frame_index"]).iterrows():
        lab = cv2.imread(str(files[r.stem][0]), cv2.IMREAD_UNCHANGED)
        rs = regions(lab == 1)
        kept, _ = select_trapped_region(lab == 1, (r.trap_x, r.trap_y))
        gt = cv2.imread(str(gt_path[r.stem]), cv2.IMREAD_UNCHANGED) == 1 if which == "test" else None
        img = cv2.cvtColor(cv2.imread(str(files[r.stem][2]), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        for k, reg in enumerate(sorted(rs, key=lambda x: x["area"], reverse=True), 1):
            is_kept = reg["label"] == kept["label"]
            row = {"stem": r.stem, "cell": r.cell, "laser_step": r.laser_step, "region": k, "area": reg["area"],
                   "x": round(reg["centroid"][0], 1), "y": round(reg["centroid"][1], 1),
                   "trap_dist": round(float(np.hypot(reg["centroid"][0] - r.local_trap_x, reg["centroid"][1] - r.local_trap_y)), 1),
                   "kept": is_kept, "frame_passed": bool(r.passed)}
            if gt is not None:
                row["dsc_with_gt"] = round(2 * (gt & reg["mask"]).sum() / (gt.sum() + reg["mask"].sum()), 3)
            rows.append(row)
            col = (0, 200, 0) if is_kept else (40, 40, 230)
            cnts, _ = cv2.findContours(reg["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cnts, -1, col, 1)
            cx, cy = (int(v) for v in reg["centroid"])
            cv2.putText(img, str(k), (cx - 4, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        x, y = int(r.local_trap_x), int(r.local_trap_y)
        cv2.drawMarker(img, (x, y), (0, 200, 255), cv2.MARKER_CROSS, 10, 1)
        cv2.circle(img, (x, y), int(gates.max_trap_dist), (0, 200, 255), 1, cv2.LINE_AA)   # the distance limit
        # a square view that holds every region and the trap's limit circle (at least 2*half px wide)
        ys, xs = np.nonzero(np.any([reg["mask"] for reg in rs], axis=0))
        r_lim = int(gates.max_trap_dist)
        x0, x1 = min(xs.min(), x - r_lim) - 12, max(xs.max(), x + r_lim) + 12
        y0, y1 = min(ys.min(), y - r_lim) - 12, max(ys.max(), y + r_lim) + 12
        side = int(max(2 * half, x1 - x0, y1 - y0))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        padded = cv2.copyMakeBorder(img, side, side, side, side, cv2.BORDER_CONSTANT)
        crop = padded[cy - side // 2 + side:cy - side // 2 + 2 * side, cx - side // 2 + side:cx - side // 2 + 2 * side]
        cv2.putText(crop, f"{side} px", (4, crop.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * side / 140,
                    (255, 255, 255), 1)
        tile = cv2.copyMakeBorder(cv2.resize(crop, (280, 280), interpolation=cv2.INTER_NEAREST), 0, 66, 0, 0,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, f"{r.cell.replace('_cell', '-')} step {r.laser_step}  {len(rs)} regions  "
                    f"{'kept' if r.passed else 'DROPPED'}", (4, 296), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0) if r.passed else (30, 30, 200), 1)
        why = ["all checks passed"] if r.passed else str(r.reasons).split("; ")   # failed checks of the kept region
        lines: list[str] = []
        for w in why:                    # wrap the reasons at ~40 characters a line
            if lines and len(lines[-1]) + len(w) < 38:
                lines[-1] += f"; {w}"
            else:
                lines.append(w)
        for k, line in enumerate(lines[:3]):
            cv2.putText(tile, line, (4, 314 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (60, 120, 60) if r.passed else (30, 30, 200), 1)
        tiles.append(tile)
    pd.DataFrame(rows).to_csv(out_dir / f"multi_region_{which}.csv", index=False)
    if tiles:
        tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % cols)
        cv2.imwrite(str(out_dir / f"multi_region_{which}.png"),
                    np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))
    n = df.n_regions.gt(1).sum()
    print(f"  {which}: {n} frames with more than one trapped-cell region -> multi_region_{which}.csv/.png")


def extract(model: str, gates: Gates) -> None:
    from scipy.stats import mannwhitneyu
    ex = MorphologyExtractor()
    df = check_frames(model, "feature", gates, ex)
    res = ROOT / "results" / "features" / model
    res.mkdir(parents=True, exist_ok=True)
    review_sheet(model, df, res / "rejected_frames.png")
    multi_region_report(model, "feature", df, res, gates)
    df.loc[~df.passed, ex._FEATURE_NAMES + FRAME_FEATURES] = np.nan   # a failed frame contributes no features
    df.to_csv(res / "features_frames.csv", index=False)

    cells = df.groupby("cell").agg(her2=("her2", "first"), frames=("stem", "size"), passed=("passed", "sum"),
                                   **{f"{f}_median": (f, "median") for f in ex._FEATURE_NAMES})
    cv_path = res / "cell_validation.csv"
    if not cv_path.exists():
        raise SystemExit("run the calibrate step first: cells must be validated on their annotated frames")
    cv = pd.read_csv(cv_path, index_col="cell")
    cells["cell_validated"] = cv["validated"].reindex(cells.index).fillna(False).astype(bool)
    cells["size_bias_annotated"] = cv["size_bias"].reindex(cells.index)
    cells["usable"] = (cells["passed"] >= MIN_PASS_FRACTION * cells["frames"]) & cells["cell_validated"]
    # deformation over the laser steps: one value per step, the median of the step's passing frames; and the
    # behaviour within a step, from the frames' offsets into it
    for cell, frames in df.groupby("cell"):
        for k, v in {**cell_summary(frames), **within_step(frames)}.items():
            cells.loc[cell, k] = v
    cells.to_csv(res / "features_cells.csv")

    iq = ["focus_frame", "focus_cell", "illum_mean", "illum_gradient", "contrast_rms", "contrast_cell", "noise_sigma"]
    per_cell = df.groupby("cell").agg(her2=("her2", "first"), **{m: (m, "mean") for m in iq})
    rows = []
    for m in iq:
        hi, lo = per_cell.loc[per_cell.her2 == "high", m], per_cell.loc[per_cell.her2 == "low", m]
        pooled = np.sqrt((hi.var() * (len(hi) - 1) + lo.var() * (len(lo) - 1)) / (len(hi) + len(lo) - 2))
        u = mannwhitneyu(hi, lo, alternative="two-sided")
        rows.append({"measure": m, "high_mean": hi.mean(), "low_mean": lo.mean(), "cohens_d": (hi.mean() - lo.mean()) / pooled,
                     "auc_high_vs_low": u.statistic / (len(hi) * len(lo)), "p_mannwhitney": u.pvalue})
    sess = pd.DataFrame(rows)
    sess.to_csv(res / "image_quality_session.csv", index=False)

    print(f"\nEXTRACTION ({model}): {int(df.passed.sum())}/{len(df)} feature frames passed")
    print(cells[["her2", "frames", "passed", "cell_validated", "size_bias_annotated", "usable"]]
          .to_string(float_format=lambda v: f"{v:.2f}"))
    print("\nreasons frames failed:")
    print(df.loc[~df.passed, "reasons"].str.split("; ").explode().str.replace(r"[\d.]+", "#", regex=True)
          .value_counts().to_string())
    print("\nIMAGE QUALITY, high vs low HER2 (per-cell means; AUC 0.5 = no separation, 0 or 1 = complete):")
    print(sess.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["cut", "segment", "calibrate", "extract", "all"])
    ap.add_argument("--model", choices=list(MODELS), default="nnunet1000_loocv")
    ap.add_argument("--config", type=Path, default=None,
                    help="settings file (default configs/analysis/features.yaml)")
    a = ap.parse_args()
    start_run_log(f"features/{a.model}", f"extract_features_{a.step}")
    configure(load_config("features", a.config))
    gates = GATES
    if a.step in ("calibrate", "extract", "all"):
        dump_config(CFG, ROOT / "results" / "features" / a.model / f"config_{a.step}.yaml")
    if a.step in ("cut", "all"):
        dense_frames()
    if a.step in ("segment", "all"):
        segment(a.model)
    if a.step in ("calibrate", "all"):
        calibrate(a.model, gates)
    if a.step in ("extract", "all"):
        extract(a.model, gates)


if __name__ == "__main__":
    main()
