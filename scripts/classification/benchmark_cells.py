"""
Per-cell HER2 benchmark: seven classical classifiers on the per-cell features, leave-one-cell-out.

    python scripts/classification/benchmark_cells.py --model nnunet1000_loocv [--perms 1000]

One row per usable cell (features_cells.csv, usable == True), label high = 1. Every model sees the same
14 folds; scaling and missing-value filling are fitted inside each fold. Hyperparameters are FIXED in
advance (no tuning: with 13 training cells an inner search would mostly fit noise).

Feature sets
    selected  chosen from the 29 by a label-free selection, once, before any classifier (main):
                1. agreement  the per-cell value from the predicted masks against the same from the hand masks,
                              on the hand-masked frames: Spearman rho across the cells >= 0.8 (where testable)
                2. split-half the per-cell value from the even laser steps against the same from the odd ones,
                              on the feature frames: Spearman rho >= 0.8
                3. groups     the candidates are grouped by the correlation matrix (average-linkage clustering on
                              1 - |Spearman rho| across the cells, cut at |rho| = 0.7, as AutoSpearman); from each
                              group only its highest-ranked passing feature is kept
              ranked by the weaker of the two rho; every group's best passing feature is kept (no cap). The HER2 labels are never
              used, so the selection cannot leak the held-out cell.  -> feature_selection.csv / .tex
    all       every per-cell shape and deformation feature, no selection                       (sensitivity)
Baselines
    size      baseline length along the beam only
    image     per-cell medians of the whole-frame image measures (illumination, unevenness, contrast,
              noise). These track the recording session, not the cell: a classifier that does well on
              them is reading the session. Its score is the bar the cell features must clear.

Scores: correct cells, balanced accuracy, AUC of the held-out P(high), and a permutation p-value
(labels shuffled across cells, the whole leave-one-cell-out rerun, balanced accuracy).

Outputs in results/classification/<model>/:  feature_selection.csv / .tex, benchmark_cells.csv, scores_cells.csv
(every held-out cell's P(high) per model), benchmark_cells.png, benchmark_cells.tex.

Final models: after the run, every classifier on each feature set (and the size baseline) is fitted once on ALL
usable cells and saved with its leave-one-cell-out score -- the models a new cell is classified with
(celldform.final_models). --final does only this, from the saved selection and scores.
    -> checkpoints/classification/<model>/classical/<set>__<classifier>.joblib, manifest.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from celldform.features.deformation import cell_summary, within_step  # noqa: E402
from celldform.analysis_config import dump as dump_config, load as load_config  # noqa: E402
from celldform import final_models  # noqa: E402
from celldform.utils.runlog import start_run_log  # noqa: E402


def configure(cfg) -> None:
    """Settings from configs/analysis/classifiers.yaml (--config for another file). The selection rules were
    fixed on 2026-10-28, before the selected-set run."""
    global CFG, SEED, RHO_MIN, GROUP_RHO, MAX_SELECTED, MIN_CELLS
    fs = cfg.feature_selection
    CFG, SEED = cfg, cfg.seed
    RHO_MIN, GROUP_RHO, MAX_SELECTED, MIN_CELLS = fs.rho_min, fs.group_rho, fs.max_selected, fs.min_cells


configure(load_config("classifiers"))
SHAPE = ["area", "perimeter", "major_axis", "minor_axis", "aspect_ratio", "eccentricity", "circularity",
         "solidity"] + [f"hu_{i}" for i in range(7)]
DEFORMATION = ["len_par_base", "len_perp_base", "compliance_par", "compliance_perp", "poisson", "nonlinearity",
               "onset_step", "volume_change", "ellip_change", "roughness_median", "orient_spread",
               "fluctuation", "creep", "creep_tau"]
IMAGE = ["illum_mean", "illum_gradient", "contrast_rms", "noise_sigma"]
NAMES = {"area": "Area", "perimeter": "Perimeter", "major_axis": "Major axis", "minor_axis": "Minor axis",
         "aspect_ratio": "Aspect ratio", "eccentricity": "Eccentricity", "circularity": "Circularity",
         "solidity": "Solidity", **{f"hu_{i}": f"Hu moment {i}" for i in range(7)},
         "len_par_base": "Baseline length along the beam", "len_perp_base": "Baseline length across",
         "compliance_par": "Compliance along the beam", "compliance_perp": "Compliance across",
         "poisson": "Transverse ratio $\\nu$", "nonlinearity": "Nonlinearity", "onset_step": "Onset step",
         "volume_change": "Volume change", "ellip_change": "Elongation change", "roughness_median": "Roughness",
         "orient_spread": "Orientation spread", "fluctuation": "Fluctuation", "creep": "Creep amplitude",
         "creep_tau": "Creep time $\\tau$"}


def build(name: str, params: dict):
    """One classifier from its config entry. Fixed, not tunable: class_weight balances 8 high vs 6 low where the
    model has it; the tree ensembles get the seed (and boosting no subsampling: with 14 rows a reshuffle of the
    rows alone moved it by a cell)."""
    p = dict(params)
    makers = {
        "Logistic regression": lambda: LogisticRegression(class_weight="balanced", max_iter=5000, **p),
        "Linear SVM": lambda: SVC(kernel="linear", class_weight="balanced", **p),
        "Shrinkage LDA": lambda: LinearDiscriminantAnalysis(solver="lsqr", **p),
        "RBF SVM": lambda: SVC(kernel="rbf", class_weight="balanced", **p),
        "Random forest": lambda: RandomForestClassifier(class_weight="balanced", random_state=SEED, **p),
        "Gradient boosting": lambda: GradientBoostingClassifier(random_state=SEED, **p),
    }
    if name.startswith("k-NN"):
        return KNeighborsClassifier(**p)
    if name.startswith("MLP"):          # a small feed-forward neural network (ANN); no class weights in sklearn
        if "hidden_layer_sizes" in p:
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        return MLPClassifier(random_state=SEED, **{"max_iter": 5000, **p})
    if name not in makers:
        raise KeyError(f"unknown classifier {name!r} in the config; known: {sorted(makers) + ['k-NN (...)', 'MLP (...)']}")
    return makers[name]()


def models() -> dict[str, object]:
    """The classifiers listed in the config, in its order, with its settings (no tuning inside the folds)."""
    return {name: build(name, params) for name, params in CFG.models.items()}


def baseline_lr():
    """The size-only and image-only baselines: logistic regression with the config's setting (default C = 1)."""
    return build("Logistic regression", CFG.models.get("Logistic regression", {"C": 1.0}))


def pipe(est) -> object:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clone(est))


def score_of(fitted, X: np.ndarray) -> np.ndarray:
    """P(high) where the model has it; for the SVMs the margin through a sigmoid (0.5 = on the boundary). That
    keeps the ranking, so the AUC is the same, but it is a score, not a calibrated probability."""
    if hasattr(fitted, "predict_proba"):
        return fitted.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-fitted.decision_function(X)))


def loocv(est, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Held-out prediction and score of every cell, each from a model trained on the other cells."""
    pred, score = np.zeros(len(y), int), np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        m = pipe(est).fit(X[tr], y[tr])
        pred[i], score[i] = m.predict(X[i:i + 1])[0], score_of(m, X[i:i + 1])[0]
    return pred, score


def perm_p(est, X: np.ndarray, y: np.ndarray, observed: float, n: int) -> float:
    """Share of label shuffles (across cells) whose leave-one-cell-out balanced accuracy reaches the observed."""
    rng = np.random.default_rng(SEED)
    shuffles = [rng.permutation(y) for _ in range(n)]

    def one(yp):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return balanced_accuracy_score(yp, loocv(est, X, yp)[0])

    null = Parallel(n_jobs=CFG.n_jobs)(delayed(one)(yp) for yp in shuffles)
    return (1 + sum(b >= observed - 1e-12 for b in null)) / (1 + n)


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rho over the cells where both are defined; NaN if fewer than MIN_CELLS are."""
    ok = a.notna() & b.notna()
    return float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() >= MIN_CELLS else np.nan


def half_values(frames: pd.DataFrame) -> pd.DataFrame:
    """Every per-cell feature computed on one half of each cell's laser steps (the same calculation as
    features_cells.csv: shape = median of the passing frames, deformation over the steps and within them)."""
    rows = {}
    for cell, f in frames.groupby("cell"):
        ok = f[f.passed]
        rows[cell] = {**{c: ok[c].median() for c in SHAPE}, **cell_summary(f), **within_step(f)}
    return pd.DataFrame(rows).T


def groups(cells: pd.DataFrame, feats: list[str]) -> list[int]:
    """Group label per feature: average-linkage clustering of the Spearman correlation matrix across the cells
    (distance 1 - |rho|), cut at |rho| = GROUP_RHO. A pair with fewer than MIN_CELLS shared values counts as
    uncorrelated. GROUP_RHO None: every feature its own group."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    if GROUP_RHO is None:
        return list(range(1, len(feats) + 1))
    v = cells.set_index("cell")
    C = np.array([[1.0 if a == b else abs(np.nan_to_num(spearman(v[a], v[b]))) for b in feats] for a in feats])
    return fcluster(linkage(squareform(1 - C, checks=False), "average"), 1 - GROUP_RHO, "distance").tolist()


def select_features(model: str, cells: pd.DataFrame) -> pd.DataFrame:
    """The label-free selection (module docstring): one row per candidate feature with both checks, the
    redundancy decision, the rank and whether it is selected."""
    res = ROOT / "results" / "features" / model
    usable = set(cells.cell)
    # 1. agreement with the hand masks, on the hand-masked frames of the usable cells
    test = pd.read_csv(res / "checks_test.csv")
    test = test[test.passed & test.cell.isin(usable)].groupby("cell")
    deform = pd.read_csv(res / "cell_feature_check.csv").set_index("cell").loc[sorted(usable)]
    agree = {c: spearman(test[c].median(), test[f"gt_{c}"].median()) for c in SHAPE}
    agree |= {c: spearman(deform[f"pred_{c}"], deform[f"gt_{c}"]) if f"pred_{c}" in deform else np.nan
              for c in DEFORMATION}
    # 2. split-half: even against odd laser steps, on the feature frames
    frames = pd.read_csv(res / "features_frames.csv")
    frames = frames[frames.cell.isin(usable)]
    even, odd = (half_values(frames[frames.laser_step % 2 == k]) for k in (0, 1))
    split = {c: spearman(even[c].astype(float), odd[c].astype(float)) for c in SHAPE + DEFORMATION}

    t = pd.DataFrame({"feature": SHAPE + DEFORMATION})
    t["agreement_rho"] = t.feature.map(agree)
    t["split_half_rho"] = t.feature.map(split)
    t["testable_on_hand_masks"] = t.agreement_rho.notna() | t.feature.isin(
        [c for c in DEFORMATION if f"pred_{c}" in deform])
    # the weaker of the two checks; a feature the hand masks cannot test is judged on the split-half alone
    t["score"] = np.where(t.testable_on_hand_masks, t[["agreement_rho", "split_half_rho"]].min(axis=1, skipna=False),
                          t.split_half_rho)
    t["passes"] = t.score >= RHO_MIN
    t = t.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
    # 3. groups of the correlation matrix, over every candidate with values in enough cells
    t["group"], t["redundant_with"] = groups(cells, t.feature.tolist()), ""
    kept = []
    for i, r in t[t.passes].iterrows():          # t is in rank order: the first passing member of a group is kept
        twin = next((k for k in kept if GROUP_RHO is not None and t.set_index("feature").group[k] == r.group), None)
        if twin:
            t.loc[i, "redundant_with"] = twin
        else:
            kept.append(r.feature)
    t["rank"] = t.feature.map({f: i + 1 for i, f in enumerate(kept)})
    t["selected"] = t.feature.isin(kept if MAX_SELECTED is None else kept[:MAX_SELECTED])
    return t


def selection_latex(t: pd.DataFrame, out: Path) -> None:
    fmt = lambda v: "--" if pd.isna(v) else f"{v:.2f}"   # noqa: E731
    rows = []
    for _, r in t.iterrows():
        if r.selected:
            verdict = r"\textcolor{tpcol}{\textbf{selected}} (%d)" % r["rank"]
        elif r.redundant_with:
            verdict = r"same group as %s" % NAMES[r.redundant_with]
        elif r.passes:
            verdict = "passes, beyond the 7"
        else:
            verdict = r"\textcolor{warnred}{no}"
        rows.append(f"{NAMES[r.feature]} & {fmt(r.agreement_rho)} & {fmt(r.split_half_rho)} & {verdict} \\\\")
    out.write_text("\\begin{tabular}{l c c l}\n\\toprule\n"
                   "Feature & Agreement $\\rho$ & Split-half $\\rho$ & Selected \\\\\n\\midrule\n"
                   + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def load(model: str) -> pd.DataFrame:
    res = ROOT / "results" / "features" / model
    cells = pd.read_csv(res / "features_cells.csv")
    cells = cells[cells.usable].copy()
    for c in SHAPE:
        cells[c] = cells[f"{c}_median"]
    frames = pd.read_csv(res / "features_frames.csv")
    image = frames[frames.passed].groupby("cell")[IMAGE].median()
    cells = cells.join(image, on="cell")
    cells["_n"] = cells.cell.str.extract(r"(\d+)$")[0].astype(int)
    return cells.sort_values(["her2", "_n"]).drop(columns="_n").reset_index(drop=True)


def latex(table: pd.DataFrame, n_cells: int, out: Path) -> None:
    rows = []
    for _, r in table.iterrows():
        p = "--" if pd.isna(r.p_perm) else f"{r.p_perm:.3f}"
        auc = "--" if pd.isna(r.auc) else f"{r.auc:.2f}"
        name = r.classifier + ("$^\\ddagger$" if r.classifier in ("Random forest", "Gradient boosting") else "")
        rows.append(f"{r.features} & {name} & {r.correct}/{n_cells} & {r.balanced_acc:.2f} & {auc} & {p} \\\\")
        if r.classifier == "Gradient boosting" or r.features in ("majority",):
            rows.append("\\midrule")
    if rows[-1] == "\\midrule":
        rows.pop()
    out.write_text("\\begin{tabular}{llcccc}\n\\toprule\n"
                   "Features & Classifier & Correct & Bal.\\ acc. & AUC & $p_{\\text{perm}}$ \\\\\n\\midrule\n"
                   + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


HIGH, LOW = "#2a78d6", "#eb6834"   # the report's HER2 colours


def plot(scores: pd.DataFrame, table: pd.DataFrame, cells: pd.DataFrame, out: Path) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    main_set = table.features.iloc[0]              # the main feature set, first in the table
    main_ = table[table.features == main_set].classifier.tolist()
    base = [("size", "Logistic regression"), ("image", "Logistic regression")]
    cols = [(main_set, m) for m in main_] + base
    names = [m + ("$^\\dagger$" if "SVM" in m else "") for m in main_] + ["Size only (LR)", "Image only (LR)"]
    row = lambda f, m: scores[(scores.features == f) & (scores.classifier == m)].iloc[0]
    P = np.array([[row(f, m)[c] for c in cells.cell] for f, m in cols])
    y = (cells.her2 == "high").to_numpy().astype(int)
    cmap = LinearSegmentedColormap.from_list("her2", [LOW, "#ffffff", HIGH])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [2.4, 1]})
    im = ax.imshow(P, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, (f, m) in enumerate(cols):
        r = row(f, m)
        for j, c in enumerate(cells.cell):
            if r[f"{c}__pred"] != y[j]:
                ax.text(j, i, "\u2715", ha="center", va="center", fontsize=9, color="#111111")
    ax.set_xticks(range(len(cells)), [c.replace("_cell", "-") for c in cells.cell], rotation=60, fontsize=8)
    for lab, h in zip(ax.get_xticklabels(), y):
        lab.set_color(HIGH if h else LOW)
        lab.set_fontweight("bold")
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.axvline(y.sum() - .5, color="k", lw=1)
    ax.axhline(len(main_) - .5, color="k", lw=1)
    ax.set_title(f"(a) Held-out P(high) of every cell, {main_set} features  (\u2715 = wrong call)", fontsize=9,
                 loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("P(high)   ($^\\dagger$ SVM margin through a sigmoid)", fontsize=7)

    t = pd.concat([table[table.features == main_set]] +
                  [table[(table.features == f) & (table.classifier == m)] for f, m in base])
    labels = main_ + ["Size only (LR)", "Image only (LR)"]
    bx.barh(range(len(t)), t.balanced_acc, color=["#3f7a4d"] * len(main_) + ["#8a8985", "#b02a2a"])
    for i, (b, pv, k) in enumerate(zip(t.balanced_acc, t.p_perm, t.correct)):
        bx.text(b + .01, i, f"{b:.2f} ({k}/{len(y)})" + ("" if pd.isna(pv) else f"  p={pv:.3f}"), va="center",
                fontsize=7)
    bx.axvline(0.5, color="k", ls=":", lw=1)
    bx.set_yticks(range(len(t)), labels, fontsize=8)
    bx.invert_yaxis()
    bx.set_xlim(0, 1.45)
    bx.set_xticks([0, .25, .5, .75, 1])
    bx.spines[["top", "right"]].set_visible(False)
    bx.set_xlabel("balanced accuracy (leave-one-cell-out)", fontsize=8)
    bx.set_title("(b) Balanced accuracy (cells correct)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def save_final(model: str, cells: pd.DataFrame, sets: dict[str, list[str]], table: pd.DataFrame,
               only: list[str] | None = None) -> None:
    """Every classifier on each feature set, and the size baseline, fitted once on all usable cells and saved with
    its leave-one-cell-out score from `table`. Same settings as the cross-validation (no tuning)."""
    y = (cells.her2 == "high").astype(int).to_numpy()
    folder = final_models.ckpt_dir("classification", model, "classical")
    runs = [(f, name, est, cols) for f, cols in sets.items() for name, est in models().items()]
    runs.append(("size", "Logistic regression", baseline_lr(), ["len_par_base"]))
    if only:
        runs = [r for r in runs if r[1] in only and r[0] != "size"]
    score = table.set_index(["features", "classifier"])
    rows = []
    for f, name, est, cols in runs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = pipe(est).fit(cells[cols].to_numpy(float), y)
        loo = score.loc[(f, name)] if (f, name) in score.index else None
        loo = {} if loo is None else {"loocv_correct": int(loo.correct), "loocv_balanced_acc": float(loo.balanced_acc),
                                      "loocv_auc": float(loo.auc), "loocv_p_perm": float(loo.p_perm)}
        params = CFG.models.get(name, {}) if f != "size" else CFG.models.get("Logistic regression", {"C": 1.0})
        meta = final_models.info(model, cells.cell, cells.her2, classifier=name, feature_set=f, params=params,
                                 seed=SEED, **loo)
        file = f"{f}__{final_models.slug(name)}.joblib"
        final_models.save_classifier(final_models.FinalClassifier(fitted, list(cols), meta), folder / file)
        rows.append({"file": file, "feature_set": f, "classifier": name, "n_features": len(cols), **loo,
                     "created": meta["created"]})
    final_models.write_manifest(folder, rows)
    print(f"final models ({len(rows)}, fitted on all {len(y)} cells) -> {folder}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nnunet1000_loocv", help="feature run in results/features/")
    ap.add_argument("--config", type=Path, default=None,
                    help="settings file (default configs/analysis/classifiers.yaml)")
    ap.add_argument("--perms", type=int, default=None, help="label shuffles for the permutation p (0 = skip; "
                                                            "default: the config's)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default results/classification/<model>)")
    ap.add_argument("--replot", action="store_true", help="redraw the figure and table from the saved CSVs only")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these classifiers (config names) and merge them into the saved results")
    ap.add_argument("--selection-only", action="store_true",
                    help="only run the feature selection and write feature_selection.csv (no classifiers)")
    ap.add_argument("--final", action="store_true",
                    help="only fit the final models on all cells and save them (uses the saved selection and scores)")
    a = ap.parse_args()
    if not (a.replot or a.out):          # log real runs only, not redraws or runs into a scratch folder
        start_run_log(f"classification/{a.model}", "benchmark_cells" + ("_final" if a.final else ""))
    configure(load_config("classifiers", a.config))
    a.perms = CFG.permutations if a.perms is None else a.perms
    out = a.out or ROOT / "results" / "classification" / a.model
    out.mkdir(parents=True, exist_ok=True)
    if not (a.replot or a.final):
        from dataclasses import replace
        dump_config(replace(CFG, permutations=a.perms), out / "config_classifiers.yaml")

    cells = load(a.model)
    if a.final:
        sel = pd.read_csv(out / "feature_selection.csv")
        chosen = sel[sel.selected].sort_values("rank").feature.tolist()
        save_final(a.model, cells, {"selected": chosen, "all": SHAPE + DEFORMATION},
                   pd.read_csv(out / "benchmark_cells.csv"), a.only)
        return
    if a.replot:
        table = pd.read_csv(out / "benchmark_cells.csv")
        sel_path = out / "feature_selection.csv"
        publish(table, pd.read_csv(out / "scores_cells.csv"), cells, out,
                pd.read_csv(sel_path).fillna({"redundant_with": ""}) if sel_path.exists() else None)
        return
    y = (cells.her2 == "high").astype(int).to_numpy()
    print(f"{len(cells)} cells: {y.sum()} high, {len(y) - y.sum()} low")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selection = select_features(a.model, cells)
    selection.to_csv(out / "feature_selection.csv", index=False)
    print("\nFEATURE SELECTION (without the labels):")
    print(selection.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    chosen = selection[selection.selected].sort_values("rank").feature.tolist()
    print(f"selected: {', '.join(chosen)}\n")
    if a.selection_only:
        return
    sets = {"selected": chosen, "all": SHAPE + DEFORMATION}
    runs = [(f, name, est, sets[f]) for f in sets for name, est in models().items()]
    runs += [("size", "Logistic regression", baseline_lr(), ["len_par_base"]),
             ("image", "Logistic regression", baseline_lr(), IMAGE)]
    order = [(f, n) for f, n, _, _ in runs]
    if a.only:
        runs = [r for r in runs if r[0] in sets and r[1] in a.only]

    rows, score_rows = [], []
    for f, name, est, cols in runs:
        X = cells[cols].to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred, score = loocv(est, X, y)
        bacc = balanced_accuracy_score(y, pred)
        auc = roc_auc_score(y, score)
        p = np.nan if a.perms == 0 else perm_p(est, X, y, bacc, a.perms)
        rows.append({"features": f, "classifier": name, "n_features": len(cols), "correct": int((pred == y).sum()),
                     "balanced_acc": bacc, "auc": auc, "p_perm": p,                      "wrong": " ".join(c.replace("_cell", "-") for c in cells.cell[pred != y])})
        score_rows.append({"features": f, "classifier": name,
                           **{c: s for c, s in zip(cells.cell, score)},
                           **{f"{c}__pred": int(q) for c, q in zip(cells.cell, pred)}})
        r = rows[-1]
        print(f"{f:9s} {name:20s} {r['correct']:2d}/{len(y)}  bal.acc {bacc:.2f}  AUC {auc:.2f}  p {p:.3f}  "
              f"wrong: {r['wrong']}")

    table, scores = pd.DataFrame(rows), pd.DataFrame(score_rows)
    if a.only:          # merge into the saved results, in the config's order
        def merged(new: pd.DataFrame, old_path: Path) -> pd.DataFrame:
            old = pd.read_csv(old_path)
            keep = [(f, c) not in set(zip(new.features, new.classifier)) for f, c in zip(old.features, old.classifier)]
            both = pd.concat([old[keep], new], ignore_index=True)
            rank = {k: i for i, k in enumerate(order)}
            both["_r"] = [rank.get(k, len(rank)) for k in zip(both.features, both.classifier)]
            return both.sort_values("_r", kind="stable").drop(columns="_r").reset_index(drop=True)
        table, scores = merged(table, out / "benchmark_cells.csv"), merged(scores, out / "scores_cells.csv")
    table.to_csv(out / "benchmark_cells.csv", index=False)
    scores.to_csv(out / "scores_cells.csv", index=False)
    publish(table, scores, cells, out, selection)
    if not a.out:                        # a run into a scratch folder keeps no models
        save_final(a.model, cells, sets, table, a.only)


def publish(table: pd.DataFrame, scores: pd.DataFrame, cells: pd.DataFrame, out: Path,
            selection: pd.DataFrame | None = None) -> None:
    table = table[table.features != "majority"]          # dropped 2026-10-29 (older runs still carry the row)
    scores = scores[scores.features != "majority"]
    latex(table, len(cells), out / "benchmark_cells.tex")
    if selection is not None:
        selection_latex(selection, out / "feature_selection.tex")   # results only: the report it fed is retired
    plot(scores, table, cells, out / "benchmark_cells.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
