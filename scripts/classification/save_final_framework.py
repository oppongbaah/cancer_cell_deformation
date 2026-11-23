"""
Fit the framework's HER2 classifier (step LR + LR on PC1 of the selected 4) once on all 15 usable cells and save it.

    python scripts/classification/save_final_framework.py --model nnunet1000_loocv

Same inputs and settings as combine_models.py / pca_models.py (their leave-one-cell-out score, 14/15, is stored with the
model); nothing is tuned here. Output: checkpoints/classification/<model>/framework/her2_framework.joblib
"""
import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from celldform import final_models  # noqa: E402
from celldform.demo import framework as fw  # noqa: E402

spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts/classification/benchmark_cells.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="nnunet1000_loocv")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    res = ROOT / "results" / "classification" / a.model
    cells = bench.load(a.model)
    sel = pd.read_csv(res / "feature_selection.csv")
    chosen = sel[sel.selected].sort_values("rank").feature.tolist()
    assert sorted(chosen) == sorted(fw.SELECTED), f"selected features changed: {chosen}"
    cells["y"] = (cells.her2 == "high").astype(int)
    fr = pd.read_csv(ROOT / "results/features" / a.model / "features_frames.csv")
    fr = fr[fr.passed & fr.cell.isin(cells.cell)]
    st = fr.groupby(["cell", "laser_step"])[fw.STEP_FEATURES].median().reset_index()
    st["y"] = st.cell.map(dict(zip(cells.cell, cells.y)))
    lr = bench.baseline_lr()
    met = pd.read_csv(res / "pca_metrics.csv").set_index("model").loc["Step LR + LR, selected 4, PC1"]
    meta = final_models.info(a.model, cells.cell, cells.her2, step_rows=len(st),
                             loocv_correct=int(met.correct), loocv_auc=float(met.auc),
                             loocv_p_perm=float(met.p_balanced_acc), selected=chosen)
    m = fw.fit(st, cells, lr, meta)
    out = final_models.ckpt_dir("classification", a.model, "framework") / "her2_framework.joblib"
    fw.save(m, out)
    print(f"saved {out}: {len(cells)} cells, {len(st)} step rows; LOOCV {meta['loocv_correct']}/15, AUC {meta['loocv_auc']:.2f}")


if __name__ == "__main__":
    main()
