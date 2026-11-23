"""The saved final models: a classical classifier round-trips through joblib, and the RoD model estimates a new
cell's rate of deformation from its frames."""
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from celldform import final_models
from celldform.biomechanics.rod import estimate, load_model


def test_classifier_round_trip_reads_median_columns(tmp_path):
    rng = np.random.default_rng(0)
    cells = pd.DataFrame({"len_par_base": rng.normal(size=20), "area_median": rng.normal(size=20)})
    y = (cells.len_par_base > 0).astype(int)
    X = cells.rename(columns={"area_median": "area"})[["len_par_base", "area"]].to_numpy()
    pipe = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression()).fit(X, y)
    clf = final_models.FinalClassifier(pipe, ["len_par_base", "area"], {"classifier": "Logistic regression"})
    final_models.save_classifier(clf, tmp_path / "m.joblib")
    back = final_models.load_classifier(tmp_path / "m.joblib")
    assert back.info == clf.info
    # `area` is missing from the table and read from `area_median`, as in features_cells.csv
    assert np.allclose(back.p_high(cells), pipe.predict_proba(X)[:, 1])


def test_manifest_replaces_rows_of_the_same_file(tmp_path):
    final_models.write_manifest(tmp_path, [{"file": "a", "v": 1}, {"file": "b", "v": 1}])
    final_models.write_manifest(tmp_path, [{"file": "a", "v": 2}])
    m = pd.read_csv(tmp_path / "manifest.csv").set_index("file").v
    assert m.to_dict() == {"b": 1, "a": 2}


def test_slug():
    assert final_models.slug("k-NN (k=3)") == "k_nn_k_3"
    assert final_models.slug("MLP (ANN)") == "mlp_ann"


def test_rod_estimate_uses_the_class_prior(tmp_path):
    steps = np.repeat(np.arange(30), 3)
    noise = np.random.default_rng(1).normal(0, 0.2, len(steps))       # ~1 % strain per frame
    frames = pd.DataFrame({"laser_step": steps, "passed": True, "len_par": 20 * (1 + 0.002 * steps) + noise})
    prior = {"mean": 0.0, "sd": 0.1, "n_cells": 6}
    model = {"measures": ["len_par"], "base_steps": 10, "hac_lags": 3, "step_seconds": 200 / 22.0,
             "priors": {"len_par": {"pooled": prior, "high": {**prior, "mean": 0.5}, "low": prior}}}
    (tmp_path / "rod_model.json").write_text(json.dumps(model))
    m = load_model(tmp_path / "rod_model.json")
    own = estimate(frames, m)
    assert own["n_steps"] == 30 and own["prior"] == "pooled"
    assert 0.15 < own["rod"] < 0.25                  # 0.2 % per step, against the first 10 steps' median
    high = estimate(frames, m, her2="high")
    assert high["rod"] == own["rod"] and high["shrunk"] > own["shrunk"]   # pulled toward the class mean 0.5
