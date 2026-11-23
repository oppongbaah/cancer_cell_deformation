"""
PCA in place of feature selection: does compressing the features beat the selected 4? Leave-one-cell-out.

    python scripts/classification/pca_models.py --model nnunet1000_loocv [--perms N]

Run after benchmark_cells.py and combine_models.py (it reads feature_selection.csv and the step-level LR's held-out
P(high) from instance_cells.csv).

What is fitted
    Three starting sets -- all 29, the 9 that pass reliability (rules 1-2), the selected 4 -- each standardised and
    reduced to its first k principal components (k = 1..6, below the set size), then a logistic regression (the
    benchmark's settings). PCA, imputation and scaling are fitted on the 14 training cells of each fold only, so the
    held-out cell never shapes the components. Each setting is scored alone and as the best model's ensemble
    (equal-weight mean with the step-level LR). The raw sets (no PCA) are in the grid as k = 0.

Statistics
    p            permutation of the whole procedure (labels shuffled across cells, step LR and every setting rerun)
    p_counted    the same shuffles, scored against the BEST balanced accuracy of the whole grid in each shuffle --
                 the p of "pick the best of the 36 settings", the choice actually made after seeing the results
    parallel analysis: the eigenvalues against those of 15 x n_features standard-normal data (5 000 draws); a
                 component is real only above the noise 95th percentile
    C sweep      the finalist and the current best model with both LRs at C = 0.1, 0.3, 1, 3, 10

Outputs in results/classification/<model>/: pca_metrics.csv (every setting, per-class metrics, AUC interval, p,
p_counted), pca_cells.csv (held-out P(high)), pca_null.csv (every shuffle's score), pca_variance.csv, pca_loadings.csv, pca_csweep.csv.
The figure is cls_pca.png in scripts/report/plot_classification_thesis.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from celldform.utils.runlog import start_run_log  # noqa: E402

spec = importlib.util.spec_from_file_location("combine", ROOT / "scripts/classification/combine_models.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)
bench = cm.bench

KS = (1, 2, 3, 4, 5, 6)
CS = (0.1, 0.3, 1.0, 3.0, 10.0)


def held_out(X: np.ndarray, y: np.ndarray, k: int, C: float | None = None) -> np.ndarray:
    """Held-out P(high) per cell: impute, scale, PCA(k) (k = 0: none) and LR, all fitted on the other cells."""
    lr = bench.models()["Logistic regression"] if C is None else bench.build("Logistic regression", {"C": C})
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        steps = [SimpleImputer(strategy="median"), StandardScaler()] + ([PCA(k)] if k else []) + [clone(lr)]
        p[i] = make_pipeline(*steps).fit(X[tr], y[tr]).predict_proba(X[i:i + 1])[0, 1]
    return p


def step_p(st: pd.DataFrame, g: np.ndarray, y: np.ndarray, C: float | None = None) -> np.ndarray:
    """The step-level LR (combine_models.instance_p) with an optional C."""
    if C is None:
        return cm.instance_p(st, g, y)
    X, yi, p = st[cm.FRAME_FEATS].to_numpy(float), y[g], np.zeros(len(y))
    for i in range(len(y)):
        m = bench.pipe(bench.build("Logistic regression", {"C": C})).fit(X[g != i], yi[g != i])
        p[i] = m.predict_proba(X[g == i])[:, 1].mean()
    return p


def grid(Xs: dict, y: np.ndarray, step: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for s, X in Xs.items():
        for k in (0,) + tuple(k for k in KS if k < X.shape[1]):
            p = held_out(X, y, k)
            lab = f"{s}, {'raw' if k == 0 else 'PC1' if k == 1 else f'PC1-{k}'}"
            out[f"LR, {lab}"] = p
            out[f"Step LR + LR, {lab}"] = (p + step) / 2
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nnunet1000_loocv")
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    res = ROOT / "results/classification" / a.model
    start_run_log(f"classification/{a.model}", "pca_models")
    warnings.simplefilter("ignore")
    cells = bench.load(a.model)
    names = cells.cell.tolist()
    y = (cells.her2 == "high").to_numpy().astype(int)
    sel = pd.read_csv(res / "feature_selection.csv")
    sets = {"all 29": bench.SHAPE + bench.DEFORMATION, "reliable 9": sel[sel.passes].feature.tolist(),
            "selected 4": sel[sel.selected].sort_values("rank").feature.tolist()}
    Xs = {s: cells[f].to_numpy(float) for s, f in sets.items()}
    step = pd.read_csv(res / "instance_cells.csv", index_col=0)["LR every laser step"].reindex(names).to_numpy()
    fr = pd.read_csv(ROOT / "results/features" / a.model / "features_frames.csv")
    fr = fr[fr.passed & fr.cell.isin(names)].reset_index(drop=True)
    st = fr.groupby(["cell", "laser_step"])[cm.FRAME_FEATS].median().reset_index()
    g = np.array([names.index(c) for c in st.cell])

    # variance, loadings and parallel analysis (no labels)
    rng = np.random.default_rng(bench.SEED)
    var, load = [], []
    for s, X in Xs.items():
        Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
        pca = PCA().fit(Z)
        ev = pca.explained_variance_ * (len(y) - 1) / len(y)          # eigenvalues of the correlation matrix
        noise = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal(Z.shape).T)))[::-1][:len(ev)]
                          for _ in range(5000)])
        for k in range(len(ev)):
            var.append({"set": s, "component": k + 1, "eigenvalue": ev[k],
                        "explained": pca.explained_variance_ratio_[k],
                        "cumulative": pca.explained_variance_ratio_[:k + 1].sum(), "noise_mean": noise[:, k].mean(),
                        "noise_p5": np.percentile(noise[:, k], 5), "noise_p95": np.percentile(noise[:, k], 95),
                        "above_noise": ev[k] > np.percentile(noise[:, k], 95)})
            load += [{"set": s, "component": k + 1, "feature": f, "weight": w}
                     for f, w in zip(sets[s], pca.components_[k])]
    pd.DataFrame(var).to_csv(res / "pca_variance.csv", index=False)
    pd.DataFrame(load).to_csv(res / "pca_loadings.csv", index=False)

    P = grid(Xs, y, step)
    pd.DataFrame(P, index=names).to_csv(res / "pca_cells.csv")

    null = []
    if a.perms:
        rng = np.random.default_rng(bench.SEED)          # the same shuffles as combine_models.py
        shuffles = [rng.permutation(y) for _ in range(a.perms)]

        def one(yp):
            warnings.simplefilter("ignore")
            Q = grid(Xs, yp, step_p(st, g, yp))
            return {k: (balanced_accuracy_score(yp, q >= .5), roc_auc_score(yp, q)) for k, q in Q.items()}
        null = Parallel(n_jobs=bench.CFG.n_jobs)(delayed(one)(yp) for yp in shuffles)
        pd.DataFrame([{"shuffle": i, "model": k, "balanced_acc": v[0], "auc": v[1]}
                      for i, d in enumerate(null) for k, v in d.items()]).to_csv(res / "pca_null.csv", index=False)
    rows = []
    best = np.array([max(v[0] for v in d.values()) for d in null]) if null else None
    for k, p in P.items():
        r = {"model": k, **cm.metrics(y, p, names)}
        r["auc_lo"], r["auc_hi"] = cm.boot_auc(y, p)
        if null:
            nb, na = np.array([d[k][0] for d in null]), np.array([d[k][1] for d in null])
            r["p_balanced_acc"] = (1 + (nb >= r["balanced_acc"] - 1e-12).sum()) / (1 + len(null))
            r["p_auc"] = (1 + (na >= r["auc"] - 1e-12).sum()) / (1 + len(null))
            r["p_counted"] = (1 + (best >= r["balanced_acc"] - 1e-12).sum()) / (1 + len(null))
        rows.append(r)
        print(f"{k:34s} {r['correct']:2d}/15 F1 {r['f1_high']:.2f}/{r['f1_low']:.2f} AUC {r['auc']:.2f} "
              f"p {r.get('p_balanced_acc', np.nan):.3f} counted {r.get('p_counted', np.nan):.3f}  {r['missed']}")
    pd.DataFrame(rows).to_csv(res / "pca_metrics.csv", index=False)

    # C sweep: the PCA finalist against the current best model
    sw = []
    for C in CS:
        sp = step_p(st, g, y, C)
        for lab, k in (("Step LR + LR, selected 4, PC1", 1), ("Step LR + LR, selected 4, raw", 0)):
            q = (held_out(Xs["selected 4"], y, k, C) + sp) / 2
            sw.append({"model": lab, "C": C, **cm.metrics(y, q, names)})
    sw = pd.DataFrame(sw)
    sw.to_csv(res / "pca_csweep.csv", index=False)
    print(sw[["model", "C", "correct", "auc", "missed"]].to_string(index=False))


if __name__ == "__main__":
    main()
