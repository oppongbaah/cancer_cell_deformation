#!/usr/bin/env python3
"""
Ensemble test (report: Pending 1, "the gate"): our U-Net + nnU-Net, averaged, on the randswap rounds.

    python scripts/training/eval_ensemble_cv.py                 # all 5 rounds, writes the log + verdict

Fixed BEFORE looking at any ensemble result (progress report 10/1, Pending 1):
  * weights 0.5 / 0.5 on the per-class probabilities, at native 1228x922 -- never tuned;
  * each round's OWN models on that round's OWN test frames: our fold_k/unet_best.pt and nnU-Net
    fold k-1 (checkpoint_final.pth, default mirroring TTA -- exactly as the benchmark was scored),
    so neither member ever saw the cells it is scored on;
  * prediction = argmax of the averaged 3-class probabilities, trapped cell = class 1; scored
    like every other model here (pixel counts per test cell, summed);
  * PASS only if all four hold against the better single model: (a) higher per-cell mean DSC,
    (b) higher Pooled DSC, (c) a better worst cell, (d) better on at least 8 of the 15 cells.

Our model's probabilities exist only inside its 384 crop; outside it they are set to
certain background -- the same contract as feature_mask(), which pastes the crop back into an
all-background frame.

Sanity check built in: each member is re-scored ALONE through this same path and compared with
its logged benchmark score; the run stops if either differs by more than 0.002 Pooled DSC.

Writes results/randswap/experiment_2_ensemble/{eval_log_native.jsonl, ensemble_fold_<k>_split.json,
gate.json}, then summarize_cv.py -> ensemble_cv_*.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics as st
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
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

from celldform.config import load as load_config
from celldform.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from celldform.segmentation.unet import UNet
from celldform.splits import parse_stem

OURS_EXP, OURS_RUN = "randswap/experiment_0_cv_baseline", "cv_baseline"
NN_EXP = "randswap/experiment_1_nnunet"
EXP, RUN = "randswap/experiment_2_ensemble", "ensemble"
NN_MODEL = Path(os.environ["nnUNet_results"]) / "Dataset501_CellTrapRandswap" / "nnUNetTrainer_100epochs__nnUNetPlans__2d"
W = 0.5                       # weight of OUR model; nnU-Net gets 1 - W. Fixed in advance.


def counts(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.array([(pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()], dtype=np.int64)


def dsc(c) -> dict:
    tp, fp, fn = (int(x) for x in c)
    return {"dsc": 2 * tp / max(2 * tp + fp + fn, 1), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn}


def logged(exp: str, fold: str) -> float:
    """The member's own logged Pooled DSC for this round (eval_native / eval_nnunet)."""
    best = None
    for line in (ROOT / "results" / exp / "eval_log_native.jsonl").read_text().splitlines():
        r = json.loads(line)
        f = r.get("fold") or (r["checkpoint"].split("/fold_")[1].split("/")[0] if "/fold_" in r["checkpoint"] else None)
        if f == fold and r["filter"] is None:
            best = r["metrics"]["1"]["dsc"]
    return best


def main() -> None:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    conf = load_config(f"configs/{OURS_EXP}/{OURS_RUN}.yaml")
    pp = conf.preprocessing
    tests = json.loads((Path(os.environ["nnUNet_raw"]) / "Dataset501_CellTrapRandswap" / "test_frames.json").read_text())
    res = ROOT / "results" / EXP
    res.mkdir(parents=True, exist_ok=True)
    log = res / "eval_log_native.jsonl"
    log.unlink(missing_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for k in sorted(tests, key=int):
        rec = json.loads((ROOT / "results" / OURS_EXP / f"{OURS_RUN}_fold_{k}_split.json").read_text())
        shutil.copy(ROOT / "results" / OURS_EXP / f"{OURS_RUN}_fold_{k}_split.json", res / f"{RUN}_fold_{k}_split.json")
        cfg = PreprocessingConfig(output_size=tuple(pp.target_size), median_kernel=pp.median_kernel,
                                  clahe_clip_limit=pp.clahe_clip_limit, clahe_tile_grid=tuple(pp.clahe_tile_grid),
                                  morph_kernel_size=pp.morph_kernel_size, crop_size=pp.crop_size,
                                  crop_center=tuple(rec["crop_center"]))
        pipe = PreprocessingPipeline(cfg)
        ours = UNet.from_checkpoint(ROOT / f"checkpoints/{OURS_EXP}/{OURS_RUN}/fold_{k}/unet_best.pt", in_channels=1,
                                    out_channels=conf.unet.out_channels, base_features=conf.unet.base_features)
        nn = nnUNetPredictor(device=torch.device("cuda"), allow_tqdm=False)            # default: mirroring TTA on
        nn.initialize_from_trained_model_folder(str(NN_MODEL), use_folds=(int(k) - 1,), checkpoint_name="checkpoint_final.pth")

        acc = {m: defaultdict(lambda: np.zeros(3, np.int64)) for m in ("ours", "nnunet", "ensemble")}
        t = tests[k]
        for s in t["stems"]:
            img = cv2.imread(str(Path(os.environ["nnUNet_raw"]) / "Dataset501_CellTrapRandswap" / "imagesTr" / f"{s}_0000.png"),
                             cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(t["masks"][s], cv2.IMREAD_GRAYSCALE) == 1
            h, w = img.shape
            # ours: crop probabilities pasted into a certain-background full frame
            p_ours = np.zeros((3, h, w), np.float32); p_ours[0] = 1.0
            x0, y0, x1, y1 = pipe.crop_box(h, w)
            # 8-bit round trip: the CV scored (and trained) our model on preprocessed frames stored
            # as uint8 PNGs (build_randswap.py); feeding float32 instead moves high-4 by 0.006.
            x = (pipe(img) * 255).astype(np.uint8).astype(np.float32) / 255.0
            pc = ours.predict_proba(x)                                             # (3, 384, 384), 1:1 crop
            if pc.shape[1:] != (y1 - y0, x1 - x0):
                pc = np.stack([cv2.resize(c, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) for c in pc])
            p_ours[:, y0:y1, x0:x1] = pc
            # nnU-Net: (3, 1, h, w) probabilities, same call as the benchmark's predictions
            _, p_nn = nn.predict_single_npy_array(img[None, None].astype(np.float32), {"spacing": [999, 1, 1]},
                                                  save_or_return_probabilities=True)
            p_nn = np.asarray(p_nn, np.float32)[:, 0]
            p_ens = W * p_ours + (1 - W) * p_nn
            cell = parse_stem(s)[0]
            for name, p in (("ours", p_ours), ("nnunet", p_nn), ("ensemble", p_ens)):
                c = counts(p.argmax(0) == 1, gt)
                acc[name][cell] += c
                acc[name]["ALL"] += c

        line = f"round {k}:"
        for name, exp in (("ours", OURS_EXP), ("nnunet", NN_EXP)):
            here, ref = dsc(acc[name]["ALL"])["dsc"], logged(exp, k)
            line += f"  {name} {here:.3f} (logged {ref:.3f})"
            if abs(here - ref) > 0.002:
                raise SystemExit(f"{line}\n  re-scored {name} differs from its logged score -- not trusting this path")
        e = dsc(acc["ensemble"]["ALL"])
        print(f"{line}  ->  ENSEMBLE {e['dsc']:.3f}", flush=True)
        cells = sorted(c for c in acc["ensemble"] if c != "ALL")
        mean_cell = st.mean(dsc(acc["ensemble"][c])["dsc"] for c in cells)
        with log.open("a") as fh:
            for scope in [*cells, "ALL"]:
                fh.write(json.dumps({"timestamp": stamp, "checkpoint": f"ensemble ours fold_{k} + nnU-Net fold {int(k) - 1}",
                                     "fold": k, "members": [f"{OURS_EXP} fold_{k}", f"{NN_EXP} fold {int(k) - 1}"],
                                     "weights": [W, 1 - W], "config": "0.5/0.5 softmax average, argmax",
                                     "resolution": "native", "input": f"{w}x{h}",
                                     "filter": None if scope == "ALL" else scope, "metrics": {"1": dsc(acc['ensemble'][scope])},
                                     "per_cell_mean_dsc": mean_cell if scope == "ALL" else None}) + "\n")
        del ours, nn
        torch.cuda.empty_cache()

    subprocess.run([sys.executable, "scripts/training/summarize_cv.py", "--exp", EXP, "--run", RUN], check=True,
                   stdout=subprocess.DEVNULL)
    gate()


def _cells(exp: str, run: str) -> dict:
    import csv
    return {r["cell"]: r for r in csv.DictReader((ROOT / "results" / exp / f"{run}_cv_cells.csv").open())}


def gate() -> None:
    summ = lambda e, r: json.loads((ROOT / "results" / e / f"{r}_cv_summary.json").read_text())
    S = {"ours": summ(OURS_EXP, OURS_RUN), "nnunet": summ(NN_EXP, "nnunet"), "ensemble": summ(EXP, RUN)}
    C = {"ours": _cells(OURS_EXP, OURS_RUN), "nnunet": _cells(NN_EXP, "nnunet"), "ensemble": _cells(EXP, RUN)}
    best = max(("ours", "nnunet"), key=lambda m: S[m]["pooled_dsc"])
    e, b = S["ensemble"], S[best]
    wins = sum(float(C["ensemble"][c]["dsc"]) > float(C[best][c]["dsc"]) for c in C[best])
    checks = {
        "a_per_cell_mean": [e["per_cell_mean_dsc"], b["per_cell_mean_dsc"], e["per_cell_mean_dsc"] > b["per_cell_mean_dsc"]],
        "b_pooled_dsc": [e["pooled_dsc"], b["pooled_dsc"], e["pooled_dsc"] > b["pooled_dsc"]],
        "c_worst_cell": [e["worst_dsc"], b["worst_dsc"], e["worst_dsc"] > b["worst_dsc"]],
        "d_cells_better": [wins, len(C[best]), wins >= 8],
    }
    verdict = {"against": best, "weights": [W, 1 - W], "checks": checks,
               "passed": all(v[2] for v in checks.values()),
               "ensemble_worst_cell": e["worst_cell"], "against_worst_cell": b["worst_cell"]}
    (ROOT / "results" / EXP / "gate.json").write_text(json.dumps(verdict, indent=1))
    print(f"\nGATE against the better single model ({best}):")
    for name, (x, y, ok) in checks.items():
        print(f"  {name:<16} ensemble {x:.4f}  vs  {y:.4f}  {'PASS' if ok else 'FAIL'}" if name != "d_cells_better"
              else f"  {name:<16} better on {x} of {y} cells (need 8)  {'PASS' if ok else 'FAIL'}")
    print(f"  => {'PASSED' if verdict['passed'] else 'FAILED'}")


def gate_table(path: Path) -> None:
    """LaTeX table of the gate, from gate.json + the three summaries (report Pending section)."""
    g = json.loads((ROOT / "results" / EXP / "gate.json").read_text())
    summ = lambda e, r: json.loads((ROOT / "results" / e / f"{r}_cv_summary.json").read_text())
    S = {"ours": summ(OURS_EXP, OURS_RUN), "nnunet": summ(NN_EXP, "nnunet"), "ensemble": summ(EXP, RUN)}
    short = lambda c: c.replace("_cell", "-")
    mark = lambda ok: r"\textcolor{nsugreen}{\ding{51}}" if ok else r"\textcolor{warnred}{\ding{55}}"
    c = g["checks"]
    rows = [
        ("(a) per-cell mean DSC", S["ours"]["per_cell_mean_dsc"], S["nnunet"]["per_cell_mean_dsc"], S["ensemble"]["per_cell_mean_dsc"], c["a_per_cell_mean"][2]),
        ("(b) Pooled DSC", S["ours"]["pooled_dsc"], S["nnunet"]["pooled_dsc"], S["ensemble"]["pooled_dsc"], c["b_pooled_dsc"][2]),
    ]
    lines = [r"% GENERATED by scripts/training/eval_ensemble_cv.py --gate-only --tables -- do not edit by hand.",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"\textbf{Condition} & \textbf{U-Net} & \textbf{nnU-Net} & \textbf{Ensemble} & \\", r"\midrule"]
    for name, o, n, e, ok in rows:
        lines.append(f"{name} & {o:.3f} & {n:.3f} & \\textbf{{{e:.3f}}} & {mark(ok)} \\\\")
    lines.append(f"(c) worst cell & {short(S['ours']['worst_cell'])} {S['ours']['worst_dsc']:.3f} & "
                 f"{short(S['nnunet']['worst_cell'])} {S['nnunet']['worst_dsc']:.3f} & "
                 f"\\textbf{{{short(S['ensemble']['worst_cell'])} {S['ensemble']['worst_dsc']:.3f}}} & {mark(c['c_worst_cell'][2])} \\\\")
    lines.append(f"(d) better than {'nnU-Net' if g['against'] == 'nnunet' else 'the U-Net'} on $\\geq$8 cells & & & "
                 f"\\textbf{{{c['d_cells_better'][0]} of {c['d_cells_better'][1]}}} & {mark(c['d_cells_better'][2])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"-> {path}")


if __name__ == "__main__":
    if "--gate-only" in sys.argv:
        gate()
        if "--tables" in sys.argv:
            gate_table(Path(sys.argv[sys.argv.index("--tables") + 1]) / "ensemble_gate.tex")
    else:
        main()
