"""Tests for celldform.classification.CellClassifier."""

import numpy as np
import pytest

from celldform.classification.classifiers import CellClassifier


def _make_data(n=60, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)  # linearly separable
    return X[:40], y[:40], X[40:], y[40:]


@pytest.mark.parametrize("clf_name", ["svm", "rf", "dt"])
class TestCellClassifier:

    def test_fit_predict_shape(self, clf_name):
        X_tr, y_tr, X_te, _ = _make_data()
        clf = CellClassifier(classifier=clf_name)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        assert preds.shape == (20,)

    def test_predict_labels_binary(self, clf_name):
        X_tr, y_tr, X_te, _ = _make_data()
        clf = CellClassifier(classifier=clf_name)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_evaluate_returns_required_keys(self, clf_name):
        X_tr, y_tr, X_te, y_te = _make_data()
        clf = CellClassifier(classifier=clf_name)
        clf.fit(X_tr, y_tr)
        result = clf.evaluate(X_te, y_te)
        for key in ("accuracy", "classification_report", "auc_roc"):
            assert key in result

    def test_save_and_load(self, clf_name, tmp_path):
        X_tr, y_tr, X_te, _ = _make_data()
        clf = CellClassifier(classifier=clf_name)
        clf.fit(X_tr, y_tr)
        path = tmp_path / f"{clf_name}_clf.pkl"
        clf.save(path)
        loaded = CellClassifier.load(path)
        preds_orig = clf.predict(X_te)
        preds_loaded = loaded.predict(X_te)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_predict_before_fit_raises(self, clf_name):
        clf = CellClassifier(classifier=clf_name)
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(np.zeros((5, 5), dtype=np.float32))
