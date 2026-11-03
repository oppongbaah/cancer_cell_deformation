"""
Hyperparameter tuning of the classical classifiers by nested leave-one-cell-out cross-validation.

    python scripts/classification/tune_cells.py --model nnunet1000_loocv [--config <yaml>] [--perms N]

Outer loop: each of the 14 cells is the test cell once. Inner loop, on the other 13 cells only: every point of the
classifier's grid (configs/analysis/classifiers.yaml, `tuning.grids`) is scored by leave-one-cell-out over the 13,
by the Brier score of their held-out P(high) (smoother than counting 13 calls; ties go to the earlier grid point).
The best point is refitted on the 13 and the test cell is scored once. The test cell never influences the choice,
so the outer scores are an honest estimate of a TUNED classifier -- comparable with the untuned benchmark
(benchmark_cells.py) on the same folds and the same selected features.

Permutation p: the cell labels shuffled and the WHOLE nested procedure rerun (`tuning.permutations`; the slow
tree ensembles are skipped). Outputs in results/classification/<model>/: tuning_cells.csv (per classifier),
tuning_choices.csv (the grid point chosen in every outer fold), tuning_scores.csv (held-out P(high)),
config_tuning.yaml.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from celldform.analysis_config import dump as dump_config, load as load_config  # noqa: E402
from celldform.utils.runlog import start_run_log  # noqa: E402

_spec = importlib.util.spec_from_file_location("benchmark_cells", Path(__file__).with_name("benchmark_cells.py"))
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def grid_points(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def loo_probs(name: str, params: dict, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Held-out P(high) of every row, leave-one-out, with fixed settings."""
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        m = bench.pipe(bench.build(name, params)).fit(X[tr], y[tr])
        p[i] = bench.score_of(m, X[i:i + 1])[0]
    return p


def nested(name: str, grid: dict, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Outer held-out P(high) of every cell, and the grid point chosen in each outer fold."""
    points = grid_points(grid)
    probs, chosen = np.zeros(len(y)), []
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        Xi, yi = X[tr], y[tr]
        if len(set(yi)) < 2:
            raise ValueError("a training set with one class")
        brier = [np.mean((loo_probs(name, pt, Xi, yi) - yi) ** 2) for pt in points]
        best = points[int(np.argmin(brier))]
        m = bench.pipe(bench.build(name, best)).fit(Xi, yi)
        probs[i] = bench.score_of(m, X[i:i + 1])[0]
        chosen.append(best)
    return probs, chosen


def perm_p(name: str, grid: dict, X: np.ndarray, y: np.ndarray, observed: float, n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    shuffles = [rng.permutation(y) for _ in range(n)]

    def one(yp):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                p, _ = nested(name, grid, X, yp)
            except ValueError:
                return 0.0
            return balanced_accuracy_score(yp, (p >= 0.5).astype(int))

    null = Parallel(n_jobs=bench.CFG.n_jobs)(delayed(one)(yp) for yp in shuffles)
    return (1 + sum(b >= observed - 1e-12 for b in null)) / (1 + n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nnunet1000_loocv", help="feature run in results/features/")
    ap.add_argument("--config", type=Path, default=None, help="settings (default configs/analysis/classifiers.yaml)")
    ap.add_argument("--perms", type=int, default=None, help="label shuffles (default: the config's tuning.permutations)")
    ap.add_argument("--only", nargs="+", default=None, help="tune only these classifiers and merge into the saved results")
    a = ap.parse_args()
    start_run_log(f"classification/{a.model}", "tune_cells")
    cfg = load_config("classifiers", a.config)
    bench.configure(cfg)
    tun = cfg.tuning
    perms = tun.get("permutations", 200) if a.perms is None else a.perms
    out = ROOT / "results" / "classification" / a.model
    dump_config(cfg, out / "config_tuning.yaml")

    cells = bench.load(a.model)
    sel = pd.read_csv(out / "feature_selection.csv")
    feats = sel[sel.selected].sort_values("rank").feature.tolist()
    X = cells[feats].to_numpy(float)
    y = (cells.her2 == "high").to_numpy().astype(int)
    untuned = pd.read_csv(out / "benchmark_cells.csv")
    untuned = untuned[untuned.features == "selected"].set_index("classifier")
    print(f"{len(cells)} cells, selected features: {', '.join(feats)}")

    rows, choices, scores = [], [], []
    for name, grid in tun["grids"].items():
        if a.only and name not in a.only:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, chosen = nested(name, grid, X, y)
        pred = (p >= 0.5).astype(int)
        bacc = balanced_accuracy_score(y, pred)
        pv = np.nan if name in tun.get("skip_permutations", []) or perms == 0 else \
            perm_p(name, grid, X, y, bacc, perms, cfg.seed)
        modal = pd.Series([str(c) for c in chosen]).value_counts()
        rows.append({"classifier": name, "grid_points": len(grid_points(grid)), "correct": int((pred == y).sum()),
                     "balanced_acc": bacc, "auc": roc_auc_score(y, p), "p_perm": pv,
                     "untuned_correct": int(untuned.loc[name, "correct"]) if name in untuned.index else np.nan,
                     "untuned_balanced_acc": untuned.loc[name, "balanced_acc"] if name in untuned.index else np.nan,
                     "untuned_p_perm": untuned.loc[name, "p_perm"] if name in untuned.index else np.nan,
                     "modal_choice": modal.index[0], "modal_share": modal.iloc[0] / len(chosen),
                     "distinct_choices": len(modal),
                     "wrong": " ".join(c.replace("_cell", "-") for c in cells.cell[pred != y])})
        choices += [{"classifier": name, "test_cell": c, **ch} for c, ch in zip(cells.cell, chosen)]
        scores.append({"classifier": name, **dict(zip(cells.cell, p))})
        r = rows[-1]
        print(f"{name:20s} tuned {r['correct']:2d}/14 (untuned {r['untuned_correct']})  bal.acc {bacc:.2f}  "
              f"AUC {r['auc']:.2f}  p {pv:.3f}  choice {r['modal_choice']} in {r['modal_share']:.0%} of folds "
              f"({r['distinct_choices']} distinct)  wrong: {r['wrong']}", flush=True)
    order = list(tun["grids"])
    for new, f in ((pd.DataFrame(rows), "tuning_cells.csv"), (pd.DataFrame(choices), "tuning_choices.csv"),
                   (pd.DataFrame(scores), "tuning_scores.csv")):
        if a.only and (out / f).exists():     # merge into the saved results, in the config's order
            old = pd.read_csv(out / f)
            new = pd.concat([old[~old.classifier.isin(a.only)], new], ignore_index=True)
            new = new.iloc[np.argsort([order.index(c) if c in order else len(order) for c in new.classifier],
                                      kind="stable")]
        new.to_csv(out / f, index=False)
    print(f"wrote {out}/tuning_{{cells,choices,scores}}.csv")


if __name__ == "__main__":
    main()
