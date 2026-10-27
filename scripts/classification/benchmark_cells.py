"""
Per-cell HER2 benchmark: seven classical classifiers on the per-cell features, leave-one-cell-out.

    python scripts/classification/benchmark_cells.py --model nnunet100_5fold [--perms 1000]

One row per usable cell (features_cells.csv, usable == True), label high = 1. Every model sees the same
14 folds; scaling and missing-value filling are fitted inside each fold. Hyperparameters are FIXED in
advance (no tuning: with 13 training cells an inner search would mostly fit noise).

Feature sets
    trusted   the per-cell features whose predicted value tracks the hand-mask value (rho >= 0.8):
              baseline lengths along / across the beam, compliance along / across, volume change   (main)
    all       every per-cell shape and deformation feature                                     (sensitivity)
Baselines
    majority  always the training majority
    size      baseline length along the beam only
    image     per-cell medians of the whole-frame image measures (illumination, unevenness, contrast,
              noise). These track the recording session, not the cell: a classifier that does well on
              them is reading the session. Its score is the bar the cell features must clear.

Scores: correct cells, balanced accuracy, AUC of the held-out P(high), and a permutation p-value
(labels shuffled across cells, the whole leave-one-cell-out rerun, balanced accuracy).

Outputs in results/classification/<model>/:  benchmark_cells.csv, scores_cells.csv (every held-out cell's
P(high) per model), benchmark_cells.png, benchmark_cells.tex.
"""

from __future__ import annotations

import argparse
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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent.parent
SEED = 0

TRUSTED = ["len_par_base", "len_perp_base", "compliance_par", "compliance_perp", "volume_change"]
SHAPE = ["area", "perimeter", "major_axis", "minor_axis", "aspect_ratio", "eccentricity", "circularity",
         "solidity"] + [f"hu_{i}" for i in range(7)]
DEFORMATION = ["len_par_base", "len_perp_base", "compliance_par", "compliance_perp", "poisson", "nonlinearity",
               "onset_step", "volume_change", "ellip_change", "roughness_median", "orient_spread",
               "fluctuation", "creep", "creep_tau"]
IMAGE = ["illum_mean", "illum_gradient", "contrast_rms", "noise_sigma"]


def models() -> dict[str, object]:
    """The seven classifiers, fixed settings. class_weight balances 8 high vs 6 low where the model has it. The
    tree ensembles are made stable (1000 trees; boosting without subsampling): with 14 rows, a reshuffle of the
    rows alone moved them by a cell."""
    return {
        "Logistic regression": LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000),
        "Linear SVM": SVC(kernel="linear", C=1.0, class_weight="balanced"),
        "Shrinkage LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "RBF SVM": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
        "k-NN (k=3)": KNeighborsClassifier(n_neighbors=3),
        "Random forest": RandomForestClassifier(n_estimators=1000, min_samples_leaf=2, class_weight="balanced",
                                                random_state=SEED),
        "Gradient boosting": GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.1,
                                                        random_state=SEED),   # no subsampling: row-order free
    }


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

    null = Parallel(n_jobs=-1)(delayed(one)(yp) for yp in shuffles)
    return (1 + sum(b >= observed - 1e-12 for b in null)) / (1 + n)


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
    main_ = table[table.features == "trusted"].classifier.tolist()
    base = [("size", "Logistic regression"), ("image", "Logistic regression")]
    cols = [("trusted", m) for m in main_] + base
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
    ax.set_title("(a) Held-out P(high) of every cell, trusted features  (\u2715 = wrong call)", fontsize=9, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("P(high)   ($^\\dagger$ SVM margin through a sigmoid)", fontsize=7)

    t = pd.concat([table[table.features == "trusted"]] +
                  [table[(table.features == f) & (table.classifier == m)] for f, m in base] +
                  [table[table.features == "majority"]])
    labels = main_ + ["Size only (LR)", "Image only (LR)", "Majority"]
    bx.barh(range(len(t)), t.balanced_acc, color=["#3f7a4d"] * len(main_) + ["#8a8985", "#b02a2a", "#c9c8c4"])
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nnunet100_5fold", help="feature run in results/features/")
    ap.add_argument("--perms", type=int, default=1000, help="label shuffles for the permutation p (0 = skip)")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default results/classification/<model>)")
    ap.add_argument("--replot", action="store_true", help="redraw the figure and table from the saved CSVs only")
    ap.add_argument("--img", type=Path, default=None, help="also write the figure here (report images folder)")
    ap.add_argument("--tables", type=Path, default=None, help="also write the LaTeX table here")
    a = ap.parse_args()
    out = a.out or ROOT / "results" / "classification" / a.model
    out.mkdir(parents=True, exist_ok=True)

    cells = load(a.model)
    if a.replot:
        table = pd.read_csv(out / "benchmark_cells.csv")
        publish(table, pd.read_csv(out / "scores_cells.csv"), cells, out, a.img, a.tables)
        return
    y = (cells.her2 == "high").astype(int).to_numpy()
    print(f"{len(cells)} cells: {y.sum()} high, {len(y) - y.sum()} low")
    sets = {"trusted": TRUSTED, "all": SHAPE + DEFORMATION}
    runs = [(f, name, est, sets[f]) for f in sets for name, est in models().items()]
    runs += [("size", "Logistic regression", models()["Logistic regression"], ["len_par_base"]),
             ("image", "Logistic regression", models()["Logistic regression"], IMAGE),
             ("majority", "Majority class", DummyClassifier(strategy="most_frequent"), ["len_par_base"])]

    rows, score_rows = [], []
    for f, name, est, cols in runs:
        X = cells[cols].to_numpy(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred, score = loocv(est, X, y)
        bacc = balanced_accuracy_score(y, pred)
        auc = np.nan if f == "majority" else roc_auc_score(y, score)
        p = np.nan if f == "majority" or a.perms == 0 else perm_p(est, X, y, bacc, a.perms)
        rows.append({"features": f, "classifier": name, "n_features": len(cols), "correct": int((pred == y).sum()),
                     "balanced_acc": bacc, "auc": auc, "p_perm": p,                      "wrong": " ".join(c.replace("_cell", "-") for c in cells.cell[pred != y])})
        score_rows.append({"features": f, "classifier": name,
                           **{c: s for c, s in zip(cells.cell, score)},
                           **{f"{c}__pred": int(q) for c, q in zip(cells.cell, pred)}})
        r = rows[-1]
        print(f"{f:9s} {name:20s} {r['correct']:2d}/{len(y)}  bal.acc {bacc:.2f}  AUC {auc:.2f}  p {p:.3f}  "
              f"wrong: {r['wrong']}")

    table, scores = pd.DataFrame(rows), pd.DataFrame(score_rows)
    table.to_csv(out / "benchmark_cells.csv", index=False)
    scores.to_csv(out / "scores_cells.csv", index=False)
    publish(table, scores, cells, out, a.img, a.tables)


def publish(table: pd.DataFrame, scores: pd.DataFrame, cells: pd.DataFrame, out: Path, img: Path | None,
            tables: Path | None) -> None:
    latex(table, len(cells), out / "benchmark_cells.tex")
    plot(scores, table, cells, out / "benchmark_cells.png")
    if tables:
        latex(table, len(cells), tables / "classify_cells.tex")
    if img:
        plot(scores, table, cells, img / "classify_cells.png")
    print(f"wrote {out}" + (f", {img}" if img else "") + (f", {tables}" if tables else ""))


if __name__ == "__main__":
    main()
