"""
Cell expression-level classifier — low vs. high HER2 expression.

Wraps three scikit-learn classifiers so they share a uniform API:
  - Support Vector Machine  (SVM  — RBF kernel, probability=True)
  - Random Forest           (RF   — 100 estimators)
  - Decision Tree           (DT   — Gini criterion)

Each classifier is trained on the morphological feature vectors extracted by
:class:`celldform.features.MorphologyExtractor` and predicts one of two labels:
  0 = low expression  (less deformable, mechanically stiffer)
  1 = high expression (more deformable, lower Young's modulus)

The comparative analysis in the thesis will select the best-performing
classifier based on accuracy, precision, recall, and F1-score on a held-out
test set.

Checkpointing: classifiers are persisted with joblib (pickle-compatible format).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Allowed classifier identifiers.
_CLASSIFIERS = Literal["svm", "rf", "dt"]


class CellClassifier:
    """Unified wrapper around SVM, Random Forest, and Decision Tree classifiers.

    Parameters
    ----------
    classifier:
        Which classifier to use: ``"svm"``, ``"rf"``, or ``"dt"``.
    scale_features:
        If *True*, features are z-score normalised before fitting / predicting.
        Always recommended for SVM; less critical for tree-based models.
    random_state:
        Seed for reproducibility.
    **kwargs:
        Additional keyword arguments forwarded to the underlying scikit-learn
        estimator constructor.
    """

    def __init__(
        self,
        classifier: _CLASSIFIERS = "svm",
        scale_features: bool = True,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.classifier_name = classifier
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self._model = self._build(classifier, random_state, **kwargs)
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
    ) -> "CellClassifier":
        """Train the classifier on feature matrix *X* and labels *y*.

        Parameters
        ----------
        X:
            Feature matrix of shape (n_samples, n_features).
        y:
            Integer labels — 0 (low expression) or 1 (high expression).
        """
        X_arr = self._to_array(X)
        if self.scaler is not None:
            X_arr = self.scaler.fit_transform(X_arr)
        self._model.fit(X_arr, y)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels (0 or 1) for *X*."""
        self._check_fitted()
        X_arr = self._transform(X)
        return self._model.predict(X_arr)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (n_samples, 2)."""
        self._check_fitted()
        X_arr = self._transform(X)
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X_arr)
        raise NotImplementedError(f"{self.classifier_name} does not support predict_proba.")

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Return a dict with accuracy, classification report, and AUC-ROC."""
        preds = self.predict(X)
        report = classification_report(y, preds, output_dict=True)

        result: Dict[str, Any] = {
            "classifier": self.classifier_name,
            "accuracy": report["accuracy"],
            "classification_report": report,
        }

        try:
            proba = self.predict_proba(X)[:, 1]
            result["auc_roc"] = roc_auc_score(y, proba)
        except (NotImplementedError, ValueError):
            result["auc_roc"] = None

        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the fitted classifier (and scaler) to *path*."""
        payload = {
            "classifier_name": self.classifier_name,
            "model": self._model,
            "scaler": self.scaler,
            "is_fitted": self._is_fitted,
        }
        with open(Path(path), "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: str | Path) -> "CellClassifier":
        """Restore a serialised classifier from *path*."""
        with open(Path(path), "rb") as fh:
            payload = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.classifier_name = payload["classifier_name"]
        obj._model = payload["model"]
        obj.scaler = payload["scaler"]
        obj.scale_features = obj.scaler is not None
        obj._is_fitted = payload["is_fitted"]
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build(name: str, random_state: int, **kwargs: Any) -> Any:
        """Instantiate the requested scikit-learn estimator."""
        if name == "svm":
            return CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state, **kwargs)
            )
        if name == "rf":
            return RandomForestClassifier(
                n_estimators=100, random_state=random_state, n_jobs=-1, **kwargs
            )
        if name == "dt":
            return DecisionTreeClassifier(
                criterion="gini", random_state=random_state, **kwargs
            )
        raise ValueError(f"Unknown classifier '{name}'. Choose from: svm, rf, dt.")

    def _transform(self, X: Any) -> np.ndarray:
        """Drop NaN rows, scale, and return array."""
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
