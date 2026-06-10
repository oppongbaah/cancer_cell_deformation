"""
Cell expression-level classifier — low vs. high HER2 expression (SVM).

Trains an SVM (RBF kernel) on the morphological feature vectors extracted by
:class:`celldform.features.MorphologyExtractor` and predicts one of two labels:
  0 = low expression  (less deformable, mechanically stiffer)
  1 = high expression (more deformable, lower Young's modulus)

Checkpointing: the classifier is persisted with pickle.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class CellClassifier:
    """SVM classifier (RBF kernel) for low / high HER2 expression.

    Parameters
    ----------
    scale_features:
        Z-score normalise features before fitting / predicting.
    random_state:
        Seed for reproducibility.
    **kwargs:
        Additional keyword arguments forwarded to :class:`sklearn.svm.SVC`.
    """

    def __init__(
        self,
        scale_features: bool = True,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self._model = CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state, **kwargs)
        )
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: "pd.DataFrame | np.ndarray",
        y: np.ndarray,
    ) -> "CellClassifier":
        """Train on feature matrix *X* and binary labels *y* (0 or 1)."""
        X_arr = self._to_array(X)
        if self.scaler is not None:
            X_arr = self.scaler.fit_transform(X_arr)
        self._model.fit(X_arr, y)
        self._is_fitted = True
        return self

    def predict(self, X: "pd.DataFrame | np.ndarray") -> np.ndarray:
        """Predict class labels (0 = low, 1 = high) for *X*."""
        self._check_fitted()
        return self._model.predict(self._transform(X))

    def predict_proba(self, X: "pd.DataFrame | np.ndarray") -> np.ndarray:
        """Return class probabilities, shape (n_samples, 2)."""
        self._check_fitted()
        return self._model.predict_proba(self._transform(X))

    def evaluate(
        self,
        X: "pd.DataFrame | np.ndarray",
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Return accuracy, classification report, and AUC-ROC."""
        preds = self.predict(X)
        report = classification_report(y, preds, output_dict=True)
        proba = self.predict_proba(X)[:, 1]
        return {
            "accuracy": report["accuracy"],
            "classification_report": report,
            "auc_roc": roc_auc_score(y, proba),
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: "str | Path") -> None:
        """Persist the fitted classifier and scaler to *path*."""
        payload = {
            "model": self._model,
            "scaler": self.scaler,
            "is_fitted": self._is_fitted,
        }
        with open(Path(path), "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: "str | Path") -> "CellClassifier":
        """Restore a serialised classifier from *path*."""
        with open(Path(path), "rb") as fh:
            payload = pickle.load(fh)
        obj = cls.__new__(cls)
        obj._model = payload["model"]
        obj.scaler = payload["scaler"]
        obj.scale_features = obj.scaler is not None
        obj._is_fitted = payload["is_fitted"]
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _transform(self, X: Any) -> np.ndarray:
        arr = self._to_array(X)
        if self.scaler is not None:
            arr = self.scaler.transform(arr)
        return arr

    @staticmethod
    def _to_array(X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
