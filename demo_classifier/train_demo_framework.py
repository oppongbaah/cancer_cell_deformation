"""
Demo-only HER2 framework on held-out U-Net masks (NOT a thesis result; the thesis classifier stays on nnU-Net masks).

Same model as the thesis one -- step LR + LR on PC1 of the four selected features, equal-weight mean -- fitted on the cells
usable from results/features/unet_loocv (extracted with configs/analysis/features_unet.yaml, pass fraction 0.6). The
feature set is NOT re-selected: the four features are the thesis ones (celldform.demo.framework.SELECTED).
Leave-one-cell-out score is stored in the file; no permutation test is run.

    python demo_classifier/train_demo_framework.py

Everything this task created (U-Net segmentations, features, logs, the limits file) was moved under demo_classifier/. To rerun
the extraction, `extract_features.py --model unet_loocv --config demo_classifier/configs/analysis/features_unet.yaml` reads
and writes the normal data/features and results/features paths: move the folders back first, or it starts from scratch.
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from celldform import final_models  # noqa: E402
from celldform.demo import framework as fw  # noqa: E402

spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts/classification/benchmark_cells.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
DEMO = ROOT / "demo_classifier"      # the U-Net feature run lives here: demo_classifier/{data,results,logs,configs}/...
bench.ROOT = DEMO                   # bench.load reads <ROOT>/results/features/<model>
MODEL = "unet_loocv"


def main() -> None:
    warnings.simplefilter("ignore")
    cells = bench.load(MODEL)
    cells["y"] = (cells.her2 == "high").astype(int)
    fr = pd.read_csv(DEMO / "results/features" / MODEL / "features_frames.csv")
    fr = fr[fr.passed & fr.cell.isin(cells.cell)]
    st = fr.groupby(["cell", "laser_step"])[fw.STEP_FEATURES].median().reset_index()
    st["y"] = st.cell.map(dict(zip(cells.cell, cells.y)))
    lr = bench.baseline_lr()
    rows = []
    for c in cells.cell:                                    # leave one cell out
        m = fw.fit(st[st.cell != c], cells[cells.cell != c], lr)
        r = m.p_high(st[st.cell == c], cells[cells.cell == c].iloc[0][fw.SELECTED].to_dict())
        rows.append({"cell": c, "y": int(cells.set_index("cell").y[c]), **r})
    d = pd.DataFrame(rows)
    d["pred"] = (d.p_high >= .5).astype(int)
    correct, auc = int((d.pred == d.y).sum()), float(roc_auc_score(d.y, d.p_high))
    print(d.round(3).to_string(index=False))
    print(f"\nLOOCV: {correct}/{len(d)} correct, AUC {auc:.2f}  ({int(cells.y.sum())} high, {int((1 - cells.y).sum())} low)")
    meta = final_models.info(MODEL, cells.cell, cells.her2, step_rows=len(st), loocv_correct=correct, loocv_auc=auc,
                             selected=fw.SELECTED, purpose="demo only; U-Net held-out masks; no permutation test")
    out = ROOT / "demo_classifier" / "her2_framework_demo_unet.joblib"
    fw.save(fw.fit(st, cells, lr, meta), out)
    d.to_csv(ROOT / "demo_classifier" / "demo_loocv_cells.csv", index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
