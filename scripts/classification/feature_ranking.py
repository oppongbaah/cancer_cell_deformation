"""
Which features, chosen how? Six feature rankings x seven feature counts x the eight classifiers, leave-one-cell-out.

    python scripts/classification/feature_ranking.py --model nnunet1000_loocv [--perms N] [--nested-perms N]

The label-free selection (benchmark_cells.py) fixes 7 features before any classifier. This asks whether other
choices -- rankings that DO use the labels, and other feature counts -- do better. A ranking that uses the labels is
refitted inside every fold on the training cells only, so the held-out cell never influences which features it gets.

Candidates: the 29 per-cell features (15 shape medians + 14 deformation summaries; benchmark_cells.SHAPE + DEFORMATION).
Missing values are filled with the training cells' median and the features standardised before ranking.

Rankings (each orders the 29 features; the top k are used)
    label-free   the fixed label-free selection's order (feature_selection.csv), then the rest by its score -- the
                 same in every fold
    univariate   |AUC - 0.5| of each feature alone on the training cells (Mann-Whitney)
    mRMR         minimum redundancy, maximum relevance: greedily the feature with the highest univariate relevance
                 minus its mean |Spearman rho| with the features already chosen
    L1 path      L1 logistic regression on a falling penalty: the order in which coefficients become non-zero
    forest       random-forest impurity importance (300 trees)
    RFE          recursive feature elimination with logistic regression: drop the smallest |coefficient|, refit
Feature counts k: 1, 2, 3, 5, 7, 10, 29 (29 = every feature; the same for every ranking).
Classifiers: the eight of configs/analysis/classifiers.yaml, settings fixed (benchmark_cells.models()).

Three readings, from least to most honest
    grid         every (ranking, k, classifier): its leave-one-cell-out score. Exploratory: the best of 42 settings
                 per classifier is optimistic by construction.
    best of grid p   the permutation null of the BEST of the 42 settings (labels shuffled across cells, every ranking
                 and every fit rerun, the maximum AUC / balanced accuracy over the grid kept): the chance of a grid
                 maximum this high. Fast classifiers only (the linear models, RBF SVM, k-NN).
    nested       the ranking and k chosen INSIDE each fold, by leave-one-out over its training cells (lowest Brier
                 score), then the test cell scored once: an honest estimate of "pick the best features" as a method.
                 Logistic regression and k-NN, with a permutation p of the whole nested procedure.

Outputs in results/classification/<model>/: rank_grid.csv (every setting), rank_cells.csv (held-out P(high) of every
setting and the nested runs), rank_freq.csv (how often each feature is in the top 5, per ranking), rank_nested.csv,
rank_null.npz. The thesis figures are drawn by scripts/report/plot_classification_thesis.py.
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
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from celldform.utils.runlog import start_run_log  # noqa: E402

spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts/classification/benchmark_cells.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

RANKERS = ["label-free", "univariate", "mRMR", "L1 path", "forest", "RFE"]
KS = [1, 2, 3, 5, 7, 10, 29]
FAST = ["Logistic regression", "Linear SVM", "Shrinkage LDA", "RBF SVM", "k-NN (k=3)"]   # cheap enough to permute
NESTED = ["Logistic regression", "k-NN (k=3)"]
HIGH, LOW, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"


def prep(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill with the training median, standardise with the training mean and SD (constant features -> 0)."""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr, Xte = np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def relevance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|AUC - 0.5| of every column alone (ties ranked as half)."""
    from scipy.stats import rankdata
    n1, n0 = y.sum(), len(y) - y.sum()
    r = np.apply_along_axis(rankdata, 0, X)
    auc = (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return np.abs(auc - 0.5)


def rank(name: str, X: np.ndarray, y: np.ndarray, fixed: list[int], seed: int) -> list[int]:
    """Column order (best first) of one ranking, fitted on the training cells X, y (already prepared)."""
    p = X.shape[1]
    if name == "label-free":
        return fixed
    if name == "univariate":
        return list(np.argsort(-relevance(X, y), kind="stable"))
    if name == "mRMR":
        rel = relevance(X, y)
        rho = np.abs(spearmanr(X).statistic) if p > 2 else np.ones((p, p))
        rho = np.nan_to_num(rho, nan=0.0)
        chosen = [int(np.argmax(rel))]
        while len(chosen) < p:
            rest = [j for j in range(p) if j not in chosen]
            score = [rel[j] - rho[j, chosen].mean() for j in rest]
            chosen.append(rest[int(np.argmax(score))])
        return chosen
    if name == "L1 path":
        entry = np.full(p, np.inf)
        for i, C in enumerate(np.logspace(-2.5, 1.5, 25)):
            m = LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced", max_iter=5000)
            nz = np.abs(m.fit(X, y).coef_[0]) > 1e-8
            entry[nz & ~np.isfinite(entry)] = i
        size = np.abs(LogisticRegression(class_weight="balanced", max_iter=5000).fit(X, y).coef_[0])
        return list(np.lexsort((-size, entry)))               # by entry point, ties by the ridge coefficient
    if name == "forest":
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed,
                                   n_jobs=1).fit(X, y)
        return list(np.argsort(-m.feature_importances_, kind="stable"))
    if name == "RFE":
        left, dropped = list(range(p)), []
        while len(left) > 1:
            c = np.abs(LogisticRegression(class_weight="balanced", max_iter=5000).fit(X[:, left], y).coef_[0])
            dropped.append(left.pop(int(np.argmin(c))))
        return left + dropped[::-1]
    raise KeyError(name)


def fold_rankings(X: np.ndarray, y: np.ndarray, i: int, fixed: list[int], seed: int):
    """Prepared train/test rows of fold i and every ranking's order, from the training cells only."""
    tr = np.arange(len(y)) != i
    Xtr, Xte = prep(X[tr], X[i:i + 1])
    return Xtr, Xte, y[tr], {r: rank(r, Xtr, y[tr], fixed, seed) for r in RANKERS}


def score_one(est, Xtr, ytr, Xte) -> float:
    from sklearn.base import clone
    m = clone(est).fit(Xtr, ytr)
    return float(bench.score_of(m, Xte)[0])


def grid_scores(X: np.ndarray, y: np.ndarray, fixed: list[int], models: dict, seed: int) -> dict:
    """Held-out P(high) of every (ranking, k, classifier): {(r, k, m): array over cells}."""
    out = {(r, k, m): np.zeros(len(y)) for r in RANKERS for k in KS for m in models}
    for i in range(len(y)):
        Xtr, Xte, ytr, orders = fold_rankings(X, y, i, fixed, seed)
        for r, order in orders.items():
            for k in KS:
                cols = order[:k]
                for m, est in models.items():
                    out[(r, k, m)][i] = score_one(est, Xtr[:, cols], ytr, Xte[:, cols])
    return out


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    hi, lo = y == 1, y == 0
    return {"correct": int((pred == y).sum()), "acc_high": (pred[hi] == 1).mean(), "acc_low": (pred[lo] == 0).mean(),
            "f1_high": f1_score(y, pred, pos_label=1, zero_division=0),
            "f1_low": f1_score(y, pred, pos_label=0, zero_division=0),
            "balanced_acc": balanced_accuracy_score(y, pred), "auc": roc_auc_score(y, p)}


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def nested(X: np.ndarray, y: np.ndarray, fixed: list[int], est, seed: int) -> tuple[np.ndarray, list]:
    """Ranking and k chosen inside every outer fold by leave-one-out over its training cells (Brier score)."""
    p_out, choices = np.zeros(len(y)), []
    for i in range(len(y)):
        tr = np.flatnonzero(np.arange(len(y)) != i)
        inner = {(r, k): np.zeros(len(tr)) for r in RANKERS for k in KS}
        for j in range(len(tr)):
            Xa, Xb, ya, orders = fold_rankings(X[tr], y[tr], j, fixed, seed)
            for r, order in orders.items():
                for k in KS:
                    inner[(r, k)][j] = score_one(est, Xa[:, order[:k]], ya, Xb[:, order[:k]])
        best = min(inner, key=lambda key: (brier(y[tr], inner[key]), KS.index(key[1]), RANKERS.index(key[0])))
        Xtr, Xte, ytr, orders = fold_rankings(X, y, i, fixed, seed)
        cols = orders[best[0]][:best[1]]
        p_out[i] = score_one(est, Xtr[:, cols], ytr, Xte[:, cols])
        choices.append(best)
    return p_out, choices


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nnunet1000_loocv")
    ap.add_argument("--perms", type=int, default=1000, help="shuffles for the best-of-grid null (0 = skip)")
    ap.add_argument("--nested-perms", type=int, default=100, help="shuffles of the nested procedure (0 = skip)")
    a = ap.parse_args()
    res = ROOT / "results" / "classification" / a.model
    start_run_log(f"classification/{a.model}", "feature_ranking")
    warnings.simplefilter("ignore")
    cells = bench.load(a.model)
    feats = bench.SHAPE + bench.DEFORMATION
    X, y = cells[feats].to_numpy(float), (cells.her2 == "high").to_numpy().astype(int)
    sel = pd.read_csv(res / "feature_selection.csv")
    sel_order = sel.sort_values(["rank", "score"], ascending=[True, False], na_position="last").feature.tolist()
    fixed = [feats.index(f) for f in sel_order if f in feats]
    models, seed = bench.models(), bench.SEED
    print(f"{len(y)} cells ({y.sum()} high), {len(feats)} features, {len(RANKERS)} rankings x {len(KS)} k x "
          f"{len(models)} classifiers", flush=True)

    P = grid_scores(X, y, fixed, models, seed)
    rows = [{"ranking": r, "k": k, "classifier": m, **metrics(y, p)} for (r, k, m), p in P.items()]
    grid = pd.DataFrame(rows)
    cell_rows = {f"{r} | {k} | {m}": p for (r, k, m), p in P.items()}

    # how often each feature is in the top 5, per ranking, over the folds
    freq = {r: np.zeros(len(feats)) for r in RANKERS}
    for i in range(len(y)):
        _, _, _, orders = fold_rankings(X, y, i, fixed, seed)
        for r, order in orders.items():
            freq[r][order[:5]] += 1
    pd.DataFrame({"feature": feats, **{r: v / len(y) for r, v in freq.items()}}).to_csv(res / "rank_freq.csv",
                                                                                        index=False)

    # permutation null of the grid maximum, fast classifiers
    if a.perms:
        rng = np.random.default_rng(seed)
        shuffles = [rng.permutation(y) for _ in range(a.perms)]
        fast = {m: models[m] for m in FAST if m in models}

        def one(yp):
            warnings.simplefilter("ignore")
            Q = grid_scores(X, yp, fixed, fast, seed)
            auc = np.array([[roc_auc_score(yp, Q[(r, k, m)]) for r in RANKERS for k in KS] for m in fast])
            bal = np.array([[balanced_accuracy_score(yp, Q[(r, k, m)] >= .5) for r in RANKERS for k in KS] for m in fast])
            return auc, bal
        out = Parallel(n_jobs=bench.CFG.n_jobs)(delayed(one)(yp) for yp in shuffles)
        null_auc, null_bal = np.stack([o[0] for o in out]), np.stack([o[1] for o in out])   # perm x model x setting
        np.savez_compressed(res / "rank_null.npz", auc=null_auc, bal=null_bal, models=np.array(list(fast)),
                            settings=np.array([f"{r} | {k}" for r in RANKERS for k in KS]))
        for mi, m in enumerate(fast):
            g = grid.classifier == m
            for col, null in (("auc", null_auc), ("balanced_acc", null_bal)):
                obs = grid.loc[g, col].to_numpy()
                order_ = [RANKERS.index(r) * len(KS) + KS.index(k) for r, k in zip(grid.loc[g, "ranking"], grid.loc[g, "k"])]
                per = null[:, mi, :][:, order_]
                grid.loc[g, f"p_{col}"] = [(1 + (per[:, j] >= o - 1e-12).sum()) / (1 + a.perms) for j, o in enumerate(obs)]
                grid.loc[g, f"p_{col}_bestofgrid"] = [(1 + (null[:, mi, :].max(1) >= o - 1e-12).sum()) / (1 + a.perms)
                                                       for o in obs]
            print(f"{m}: best-of-grid AUC {grid.loc[g, 'auc'].max():.2f}, p "
                  f"{grid.loc[g, 'p_auc_bestofgrid'].min():.3f}", flush=True)
    grid.to_csv(res / "rank_grid.csv", index=False)

    # nested choice
    nrows = []
    for m in NESTED:
        p, choices = nested(X, y, fixed, models[m], seed)
        cell_rows[f"nested | {m}"] = p
        r = {"classifier": m, **metrics(y, p),
             "choices": "; ".join(f"{c[0]}/{c[1]}" for c in choices)}
        if a.nested_perms:
            rng = np.random.default_rng(seed + 1)
            shuffles = [rng.permutation(y) for _ in range(a.nested_perms)]

            def one(yp, est=models[m]):
                warnings.simplefilter("ignore")
                q, _ = nested(X, yp, fixed, est, seed)
                return balanced_accuracy_score(yp, q >= .5), roc_auc_score(yp, q)
            null = np.array(Parallel(n_jobs=bench.CFG.n_jobs)(delayed(one)(yp) for yp in shuffles))
            r["p_balanced_acc"] = (1 + (null[:, 0] >= r["balanced_acc"] - 1e-12).sum()) / (1 + a.nested_perms)
            r["p_auc"] = (1 + (null[:, 1] >= r["auc"] - 1e-12).sum()) / (1 + a.nested_perms)
        nrows.append(r)
        print(f"nested {m}: {r['correct']}/{len(y)} AUC {r['auc']:.2f} p_auc {r.get('p_auc', np.nan):.3f}", flush=True)
    pd.DataFrame(nrows).to_csv(res / "rank_nested.csv", index=False)
    pd.DataFrame(cell_rows, index=cells.cell).T.to_csv(res / "rank_cells.csv")


if __name__ == "__main__":
    main()
