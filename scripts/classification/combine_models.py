"""
Frame- and step-level logistic regression, and combinations of the deep and classical models, leave-one-cell-out.

    python scripts/classification/combine_models.py --model nnunet1000_loocv [--perms N]

Run after benchmark_cells.py, frames_cnn.py finetune (3 seeds), sequence_net.py and feature_ranking.py: it reads their
held-out P(high) per cell and fits nothing of theirs again.

Instance-level logistic regression (fitted here)
    every frame   one row per passing feature frame, the 15 shape measures + 5 per-frame deformation measures
                  (L_par, L_perp, volume, elongation f2, roughness); a held-out cell's P(high) = mean over its frames
    every step    the same, one row per laser step (median of the step's passing frames)
    Settings as the benchmark's logistic regression; scaling inside the fold; cells weighted by their rows. Permutation
    p: labels shuffled across CELLS (every frame of a cell keeps its cell's label), the whole thing rerun.

Combinations: the equal-weight mean of the members' held-out P(high) -- no weights are fitted, so nothing is learnt
from the test cell and a combination needs no extra validation. Called high at >= 0.5. Members: the CNN (fine-tuned,
cell crop, mean of 3 seeds), the eight classifiers on the selected features, LR and the ANN on all 29, the nested feature-ranking runs, the instance
LRs. A combination of classical models only has a permutation p (every member rerun under the same shuffle); one with
the CNN has none (one shuffle = 45 CNN trainings) -- it has a 2 000-resample cell-bootstrap interval on the AUC.

Outputs in results/classification/<model>/: instance_cells.csv, combine_cells.csv (every model's and combination's
held-out P(high)), combine_metrics.csv, combine_null.csv. The thesis figures are drawn by
scripts/report/plot_classification_thesis.py.
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
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from celldform.utils.runlog import start_run_log  # noqa: E402

spec = importlib.util.spec_from_file_location("bench", ROOT / "scripts/classification/benchmark_cells.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

FRAME_FEATS = bench.SHAPE + ["len_par", "len_perp", "volume", "ellip_f2", "roughness"]
CNN = "CNN (cell crop)"
SHORT = {"Logistic regression": "LR", "k-NN (k=3)": "k-NN", "Linear SVM": "linear SVM", "Shrinkage LDA": "LDA",
         "RBF SVM": "RBF SVM", "Random forest": "random forest", "Gradient boosting": "gradient boosting",
         "MLP (ANN)": "ANN"}
HIGH, LOW, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
CNN_COL, ML_COL, MIX_COL, BASE_COL = "#b02a2a", "#1b7a3a", "#4a3aa7", "#8a8985"


def instance_p(rows: pd.DataFrame, g: np.ndarray, y_cell: np.ndarray) -> np.ndarray:
    """Held-out P(high) per cell from a logistic regression on the rows (frames or steps) of the other cells."""
    X, yi = rows[FRAME_FEATS].to_numpy(float), y_cell[g]
    p = np.zeros(len(y_cell))
    for i in range(len(y_cell)):
        m = bench.pipe(bench.baseline_lr()).fit(X[g != i], yi[g != i])
        p[i] = m.predict_proba(X[g == i])[:, 1].mean()
    return p


def metrics(y: np.ndarray, p: np.ndarray, names: list[str]) -> dict:
    pred = (p >= 0.5).astype(int)
    hi, lo = y == 1, y == 0
    return {"correct": int((pred == y).sum()), "acc_high": (pred[hi] == 1).mean(), "acc_low": (pred[lo] == 0).mean(),
            "f1_high": f1_score(y, pred, pos_label=1, zero_division=0),
            "f1_low": f1_score(y, pred, pos_label=0, zero_division=0),
            "balanced_acc": balanced_accuracy_score(y, pred), "auc": roc_auc_score(y, p),
            "missed": " ".join(n.replace("_cell", "-") for n, a, b in zip(names, pred, y) if a != b)}


def boot_auc(y: np.ndarray, p: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng, out = np.random.default_rng(seed), []
    while len(out) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            out.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nnunet1000_loocv")
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    res = ROOT / "results" / "classification" / a.model
    start_run_log(f"classification/{a.model}", "combine_models")
    warnings.simplefilter("ignore")
    cells = bench.load(a.model)
    names = cells.cell.tolist()
    y = (cells.her2 == "high").to_numpy().astype(int)

    # instance-level LR
    fr = pd.read_csv(ROOT / "results/features" / a.model / "features_frames.csv")
    fr = fr[fr.passed & fr.cell.isin(names)].reset_index(drop=True)
    st = fr.groupby(["cell", "laser_step"])[FRAME_FEATS].median().reset_index()
    inst = {"LR every frame": (fr, np.array([names.index(c) for c in fr.cell])),
            "LR every laser step": (st, np.array([names.index(c) for c in st.cell]))}
    P = {k: instance_p(d, g, y) for k, (d, g) in inst.items()}

    # the other runs' held-out P(high)
    sel = pd.read_csv(res / "feature_selection.csv")
    chosen = sel[sel.selected].sort_values("rank").feature.tolist()
    LF = f"selected {len(chosen)}"
    sc = pd.read_csv(res / "scores_cells.csv")
    for f, label in (("selected", LF), ("all", "all 29"), ("size", None), ("image", None)):
        for _, r in sc[sc.features == f].iterrows():
            key = {"size": "Size only", "image": "Image only (session)"}.get(f) or f"{SHORT[r.classifier]}, {label}"
            if f != "all" or r.classifier in ("Logistic regression", "MLP (ANN)"):
                P[key] = r[names].to_numpy(float)
    seeds = [res / f"cnn_frames_finetune_masked{'' if s == 0 else f'_seed{s}'}.csv" for s in (0, 1, 2)]
    P[CNN] = np.mean([pd.read_csv(f).groupby("cell").p_high.mean().reindex(names).to_numpy() for f in seeds], axis=0)
    seq = pd.read_csv(res / "seq_cells.csv")
    for arch, g in seq.groupby("arch"):
        P[{"cnn1d": "1-D CNN, strain curves", "gru": "GRU, strain curves"}[arch]] = g[names].mean().to_numpy(float)
    rk_path = res / "rank_cells.csv"
    rk = pd.read_csv(rk_path, index_col=0) if rk_path.exists() else pd.DataFrame()
    for m in ("Logistic regression", "k-NN (k=3)"):
        if f"nested | {m}" in rk.index:            # feature_ranking.py's nested choice, when it has run
            P[f"{SHORT[m]}, nested ranking"] = rk.loc[f"nested | {m}", names].to_numpy(float)
    pd.DataFrame(P, index=names).to_csv(res / "instance_cells.csv")

    ml8 = [f"{SHORT[m]}, {LF}" for m in bench.models()]
    combos = {   # name -> members, fixed before the results of the combinations were seen
        f"CNN + LR, {LF}": [CNN, f"LR, {LF}"],
        "CNN + LR, all 29": [CNN, "LR, all 29"],
        "CNN + ANN, all 29": [CNN, "ANN, all 29"],
        "CNN + LR every laser step": [CNN, "LR every laser step"],
        "CNN + LR every frame": [CNN, "LR every frame"],
        "CNN + LR, nested ranking": [CNN, "LR, nested ranking"],
        f"CNN + LR step + LR, {LF}": [CNN, "LR every laser step", f"LR, {LF}"],
        f"CNN + all 8 classifiers, {LF}": [CNN] + ml8,
        f"LR every laser step + LR, {LF}": ["LR every laser step", f"LR, {LF}"],
        "LR every laser step + ANN, all 29": ["LR every laser step", "ANN, all 29"],
        f"Mean of the 8 classifiers, {LF}": ml8,
    }
    combos = {k: m for k, m in combos.items() if all(x in P for x in m)}
    for k, members in combos.items():
        P[k] = np.mean([P[m] for m in members], axis=0)

    # permutation p for the instance LRs and the classical-only combinations (fast members only)
    fast = {"LR every frame", "LR every laser step", f"LR, {LF}", "LR, all 29", "ANN, all 29"}
    permutable = [k for k in P if k in ("LR every frame", "LR every laser step")] + \
                 [k for k, m in combos.items() if set(m) <= fast]
    pvals: dict[str, tuple[float, float]] = {}
    if a.perms:
        Xsel, Xall = cells[chosen].to_numpy(float), cells[bench.SHAPE + bench.DEFORMATION].to_numpy(float)
        models = bench.models()
        rng = np.random.default_rng(bench.SEED)
        shuffles = [rng.permutation(y) for _ in range(a.perms)]

        def one(yp):
            warnings.simplefilter("ignore")
            Q = {k: instance_p(d, g, yp) for k, (d, g) in inst.items()}
            Q[f"LR, {LF}"] = bench.loocv(models["Logistic regression"], Xsel, yp)[1]
            Q["LR, all 29"] = bench.loocv(models["Logistic regression"], Xall, yp)[1]
            Q["ANN, all 29"] = bench.loocv(models["MLP (ANN)"], Xall, yp)[1]
            out = {}
            for k in permutable:
                q = Q[k] if k in Q else np.mean([Q[m] for m in combos[k]], axis=0)
                out[k] = (balanced_accuracy_score(yp, q >= .5), roc_auc_score(yp, q))
            return out
        null = Parallel(n_jobs=bench.CFG.n_jobs)(delayed(one)(yp) for yp in shuffles)
        pd.DataFrame([{"shuffle": i, "model": k, "balanced_acc": d[k][0], "auc": d[k][1]}
                      for i, d in enumerate(null) for k in permutable]).to_csv(res / "combine_null.csv", index=False)
        for k in permutable:
            obs = metrics(y, P[k], names)
            nb = np.array([d[k][0] for d in null])
            na = np.array([d[k][1] for d in null])
            pvals[k] = ((1 + (nb >= obs["balanced_acc"] - 1e-12).sum()) / (1 + a.perms),
                        (1 + (na >= obs["auc"] - 1e-12).sum()) / (1 + a.perms))
            print(f"{k}: p_bal {pvals[k][0]:.3f} p_auc {pvals[k][1]:.3f}", flush=True)

    rows = []
    for k, p in P.items():
        lo, hi = boot_auc(y, p)
        kind = ("baseline" if k in ("Size only", "Image only (session)") else "combination" if k in combos
                else "deep" if k == CNN or "strain curves" in k else "classical")
        rows.append({"model": k, "kind": kind, "members": " + ".join(combos.get(k, [k])), **metrics(y, p, names),
                     "auc_lo": lo, "auc_hi": hi, "p_balanced_acc": pvals.get(k, (np.nan,))[0],
                     "p_auc": pvals.get(k, (np.nan, np.nan))[1]})
        r = rows[-1]
        print(f"{k:34s} {r['correct']:2d}/{len(y)} acc {r['acc_high']:.2f}/{r['acc_low']:.2f} "
              f"F1 {r['f1_high']:.2f}/{r['f1_low']:.2f} AUC {r['auc']:.2f} [{lo:.2f}, {hi:.2f}]  missed: {r['missed']}")
    pd.DataFrame(rows).to_csv(res / "combine_metrics.csv", index=False)
    pd.DataFrame(P, index=names).to_csv(res / "combine_cells.csv")


if __name__ == "__main__":
    main()
