"""
What chance looks like: the permutation null distributions behind the classification benchmark's p-values.

    python scripts/report/plot_permutation.py --model nnunet1000_loocv --img <dir>

The benchmark (benchmark_cells.py) keeps only each model's p-value. This redraws the shuffles for two models -- the
primary classifier (logistic regression, selected features) and the image-only baseline -- with the benchmark's own
seed and procedure (labels shuffled across cells, the whole leave-one-cell-out rerun, balanced accuracy), and checks
that the p-values come out the same as in benchmark_cells.csv before drawing them.

Output: cls_permutation.png (and .csv: every shuffle's balanced accuracy).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from celldform.analysis_config import load as load_config  # noqa: E402

_spec = importlib.util.spec_from_file_location("benchmark_cells", ROOT / "scripts" / "classification" / "benchmark_cells.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
GOOD, SESSION, NULL = "#1b7a3a", "#7a4fd6", "#b0aea8"


def null_scores(est, X: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Balanced accuracy of every shuffle -- the same shuffles, in the same order, as benchmark_cells.perm_p."""
    rng = np.random.default_rng(bench.SEED)
    shuffles = [rng.permutation(y) for _ in range(n)]

    def one(yp):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return balanced_accuracy_score(yp, bench.loocv(est, X, yp)[0])

    return np.array(Parallel(n_jobs=bench.CFG.n_jobs)(delayed(one)(yp) for yp in shuffles))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nnunet1000_loocv")
    ap.add_argument("--img", type=Path, required=True)
    a = ap.parse_args()
    cfg = load_config("classifiers")
    bench.configure(cfg)
    res = ROOT / "results" / "classification" / a.model
    cells = bench.load(a.model)
    y = (cells.her2 == "high").to_numpy().astype(int)
    sel = pd.read_csv(res / "feature_selection.csv")
    chosen = sel[sel.selected].sort_values("rank").feature.tolist()
    table = pd.read_csv(res / "benchmark_cells.csv").set_index(["features", "classifier"])
    panels = [("selected", "Logistic regression", chosen, "(a) Logistic regression, 7 selected features", GOOD),
              ("image", "Logistic regression", bench.IMAGE, "(b) Image only (the recording session)", SESSION)]
    n = cfg.permutations
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
    rows = []
    for ax, (fs, clf, cols, title, col) in zip(axes, panels):
        X = cells[cols].to_numpy(float)
        est = bench.models()[clf] if fs == "selected" else bench.baseline_lr()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            observed = balanced_accuracy_score(y, bench.loocv(est, X, y)[0])
        null = null_scores(est, X, y, n)
        p = (1 + (null >= observed - 1e-12).sum()) / (1 + n)
        saved = table.loc[(fs, clf), "p_perm"]
        if not np.isclose(p, saved):
            raise RuntimeError(f"{fs}/{clf}: p {p:.4f} here, {saved:.4f} in benchmark_cells.csv -- not the same shuffles")
        rows += [{"features": fs, "classifier": clf, "shuffle": i, "balanced_acc": v} for i, v in enumerate(null)]
        vals, counts = np.unique(np.round(null, 4), return_counts=True)
        ax.bar(vals, counts / n, width=0.018, color=[col if v >= observed - 1e-9 else NULL for v in vals])
        ax.axvline(observed, color=col, lw=2)
        ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
        ax.text(observed + 0.015, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] else 0.2,
                f"real labels: {observed:.2f}\np = {p:.3f}", fontsize=8, color=col, va="top")
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.set_xlabel(f"balanced accuracy, {n:,} label shuffles", fontsize=8, color=INK2)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7.5, colors=INK2)
        print(f"{fs}/{clf}: observed {observed:.3f}, p {p:.4f} (benchmark {saved:.4f}), "
              f"null median {np.median(null):.2f}, 95th pct {np.percentile(null, 95):.2f}")
    axes[0].set_ylabel("share of shuffles", fontsize=8, color=INK2)
    fig.tight_layout()
    a.img.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.img / "cls_permutation.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(res / "permutation_null.csv", index=False)
    print(f"wrote {a.img / 'cls_permutation.png'}, {res / 'permutation_null.csv'}")


if __name__ == "__main__":
    main()
