"""
The HER2 classifier of the framework: the model scripts/classification/pca_models.py scored 14/15 (AUC 0.94)
leave-one-cell-out -- the equal-weight mean of

    step model   logistic regression on one row per laser step (20 measures: 15 shape + 5 per-frame deformation,
                 the median of the step's passing frames); a cell's P(high) = mean over its steps
    cell model   PC1 of the selected 4 per-cell features (baseline length, fluctuation, volume change, orientation
                 spread), then logistic regression

Imputation and scaling are inside each pipeline. :func:`fit` is called by scripts/classification/save_final_framework.py
on all 15 cells; the demo loads the result with :func:`load`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SHAPE = ["area", "perimeter", "major_axis", "minor_axis", "aspect_ratio", "eccentricity", "circularity",
         "solidity"] + [f"hu_{i}" for i in range(7)]
STEP_FEATURES = SHAPE + ["len_par", "len_perp", "volume", "ellip_f2", "roughness"]
SELECTED = ["len_par_base", "fluctuation", "volume_change", "orient_spread"]
MIN_STEPS = 10            # cell_summary needs 10 steps (and 3 baseline steps) before it returns anything


@dataclass
class HER2Framework:
    step_model: object
    cell_model: object
    step_features: List[str] = field(default_factory=lambda: list(STEP_FEATURES))
    cell_features: List[str] = field(default_factory=lambda: list(SELECTED))
    info: Dict = field(default_factory=dict)

    def p_step(self, steps: pd.DataFrame) -> float:
        """Mean P(high) over a cell's laser-step rows."""
        return float(self.step_model.predict_proba(steps[self.step_features].to_numpy(float))[:, 1].mean())

    def p_cell(self, cell: Dict[str, float]) -> float:
        """P(high) from the per-cell features; NaN if a selected feature is missing and the imputer cannot say."""
        x = np.array([[cell.get(f, np.nan) for f in self.cell_features]], float)
        return float(self.cell_model.predict_proba(x)[0, 1])

    def p_high(self, steps: pd.DataFrame, cell: Dict[str, float]) -> Dict[str, float]:
        a, b = self.p_step(steps), self.p_cell(cell)
        return {"p_step": a, "p_cell": b, "p_high": (a + b) / 2}


def fit(steps: pd.DataFrame, cells: pd.DataFrame, lr, info: Dict | None = None) -> HER2Framework:
    """Fit both models. ``steps``: one row per (cell, laser step) with STEP_FEATURES and ``y``; ``cells``: one row per
    cell with SELECTED and ``y``. ``lr`` is the logistic-regression estimator (the benchmark's settings)."""
    step = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clone(lr))
    step.fit(steps[STEP_FEATURES].to_numpy(float), steps.y.to_numpy(int))
    cell = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(1), clone(lr))
    cell.fit(cells[SELECTED].to_numpy(float), cells.y.to_numpy(int))
    return HER2Framework(step, cell, info=info or {})


def save(model: HER2Framework, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


# What a saved framework is allowed to contain. A joblib file is a pickle, and a pickle can run any code when it is
# opened, so the loader refuses every class that is not one of these (they are exactly what `fit` builds).
ALLOWED_GLOBALS = frozenset({
    ("celldform.demo.framework", "HER2Framework"),
    ("sklearn.pipeline", "Pipeline"),
    ("sklearn.impute._base", "SimpleImputer"),
    ("sklearn.preprocessing._data", "StandardScaler"),
    ("sklearn.decomposition._pca", "PCA"),
    ("sklearn.linear_model._logistic", "LogisticRegression"),
    ("joblib.numpy_pickle", "NumpyArrayWrapper"),
    ("numpy", "dtype"), ("numpy", "ndarray"),
    ("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
})


class UnsafeFile(ValueError):
    """The file asks for something a HER2 classifier never contains, so it was not opened."""


def _safe_joblib_load(path: Path):
    """``joblib.load`` with the unpickler restricted to :data:`ALLOWED_GLOBALS`. Fails closed: if this joblib has no
    way to be restricted, the file is refused rather than opened unrestricted."""
    import pickle

    from joblib import numpy_pickle as npk
    try:
        base, validate = npk.NumpyUnpickler, npk._validate_fileobject_and_memmap
    except AttributeError as e:
        raise UnsafeFile("this version of joblib cannot be restricted, so the file was not opened") from e

    class Restricted(base):
        def find_class(self, module, name):
            if (module, name) not in ALLOWED_GLOBALS:
                raise pickle.UnpicklingError(f"{module}.{name}")
            return super().find_class(module, name)

    with open(path, "rb") as f:
        with validate(f, str(path), None) as (fobj, mmap_mode):
            if isinstance(fobj, str):
                raise UnsafeFile("a joblib file from before version 0.10 cannot be opened safely")
            return Restricted(str(path), fobj, True, mmap_mode=mmap_mode).load()


def load(path: Path | str, trusted: bool = False) -> HER2Framework:
    """Load a saved framework. Safe by default: the file may only contain what ``fit`` builds (see ALLOWED_GLOBALS),
    so a file from someone else cannot run code when it is opened. ``trusted=True`` is plain ``joblib.load`` for a
    file you made yourself that holds other estimators."""
    import pickle
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"classifier file not found: {path}")
    try:
        obj = joblib.load(path) if trusted else _safe_joblib_load(path)
    except UnsafeFile:
        raise
    except pickle.UnpicklingError as e:
        if trusted:
            raise ValueError(f"{path.name} could not be read as a joblib file ({type(e).__name__}: {str(e)[:120]})") from e
        raise UnsafeFile(f"{path.name} contains {str(e)[:100]}, which is not part of a HER2 classifier; "
                         "the file was not opened") from e
    except Exception as e:
        raise ValueError(f"{path.name} could not be read as a joblib file ({type(e).__name__}: {str(e)[:120]})") from e
    if not isinstance(obj, HER2Framework):
        raise ValueError(f"{path.name} holds a {type(obj).__name__}, not a HER2 framework "
                         "(make one with celldform-save-final-framework)")
    return obj
