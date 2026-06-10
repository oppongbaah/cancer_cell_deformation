"""Tests for celldform.classification.CellClassifier (SVM)."""

import numpy as np
import pytest

from celldform.classification.classifiers import CellClassifier


def _make_data(n=60, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)  # linearly separable
    return X[:40], y[:40], X[40:], y[40:]


class TestCellClassifier:

    def test_fit_predict_shape(self):
        X_tr, y_tr, X_te, _ = _make_data()
        clf = CellClassifier()
        clf.fit(X_tr, y_tr)
        assert clf.predict(X_te).shape == (20,)

    def test_predict_labels_binary(self):
        X_tr, y_tr, X_te, _ = _make_data()
        clf = CellClassifier()
        clf.fit(X_tr, y_tr)
        assert set(np.unique(clf.predict(X_te))).issubset({0, 1})

    def test_evaluate_returns_required_keys(self):
        X_tr, y_tr, X_te, y_te = _make_data()
        clf = CellClassifier()
        clf.fit(X_tr, y_tr)
        result = clf.evaluate(X_te, y_te)
        for key in ("accuracy", "classification_report", "auc_roc"):
            assert key in result

    def test_save_and_load(self, tmp_path):
        X_tr, y_tr, X_te, _ = _make_data()
        clf = CellClassifier()
        clf.fit(X_tr, y_tr)
        path = tmp_path / "svm_clf.pkl"
        clf.save(path)
        loaded = CellClassifier.load(path)
        np.testing.assert_array_equal(clf.predict(X_te), loaded.predict(X_te))

    def test_predict_before_fit_raises(self):
        clf = CellClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(np.zeros((5, 5), dtype=np.float32))
