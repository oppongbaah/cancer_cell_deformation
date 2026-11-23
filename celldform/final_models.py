"""
The final models: each method fitted ONCE on all usable cells and saved, so a new cell can be classified and its
rate of deformation estimated without retraining.

The leave-one-cell-out runs measure how well a method does on a cell it has not seen; they keep no model (14
throwaway fits per method). The final model is the one the method would be used with: the same settings, trained
on all 14 cells. Its expected accuracy is the leave-one-cell-out estimate, which is stored with it.

    checkpoints/classification/<features model>/classical/<set>__<classifier>.joblib   benchmark_cells.py
    checkpoints/classification/<features model>/cnn/finetune_<variant>_seed<k>.pt      frames_cnn.py final
    checkpoints/biomechanics/<features model>/rod_model.json                         rate_of_deformation.py

Each folder has a manifest.csv (one row per saved model). Every saved model carries `info`: the training cells,
the settings, the leave-one-cell-out score, the date and the git commit.

    from celldform.final_models import load_classifier
    clf = load_classifier("checkpoints/classification/nnunet1000_loocv/classical/selected__logistic_regression.joblib")
    clf.p_high(features_cells_rows)        # a DataFrame with the feature columns, one row per cell
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def ckpt_dir(stage: str, model: str, *sub: str) -> Path:
    """checkpoints/<stage>/<features model>/<sub...>, created."""
    d = ROOT / "checkpoints" / stage / model / Path(*sub) if sub else ROOT / "checkpoints" / stage / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def slug(name: str) -> str:
    """'k-NN (k=3)' -> 'k_nn_k_3'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def info(model: str, train_cells: list[str], her2: list[str], **extra: Any) -> dict:
    """The metadata every final model carries."""
    from celldform.utils.runlog import _git_commit
    return {"features_model": model, "train_cells": list(train_cells),
            "n_high": int(sum(h == "high" for h in her2)), "n_low": int(sum(h == "low" for h in her2)),
            "created": datetime.now().isoformat(timespec="seconds"), "git": _git_commit(), **extra}


def write_manifest(folder: Path, rows: list[dict]) -> None:
    """manifest.csv of a checkpoint folder; rows replace earlier ones with the same file name."""
    p = folder / "manifest.csv"
    new = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        new = pd.concat([old[~old.file.isin(new.file)], new], ignore_index=True)
    new.to_csv(p, index=False)


def to_json(obj: dict, path: Path) -> None:
    def plain(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        return v
    path.write_text(json.dumps(plain(obj), indent=2) + "\n")


# ── classical classifiers ────────────────────────────────────────────────────

@dataclass
class FinalClassifier:
    pipeline: Any                     # imputer + scaler + classifier, fitted on all cells
    features: list[str]               # the columns it expects, in this order
    info: dict

    def p_high(self, cells: pd.DataFrame) -> np.ndarray:
        """P(high) per row (the SVMs: their margin through a sigmoid, as in the benchmark). Rows of
        features_cells.csv work as they are: a shape feature missing as `area` is read from `area_median`."""
        X = pd.DataFrame({f: cells[f] if f in cells else cells[f"{f}_median"] for f in self.features}).to_numpy(float)
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)[:, 1]
        return 1 / (1 + np.exp(-self.pipeline.decision_function(X)))


def save_classifier(clf: FinalClassifier, path: Path) -> None:
    import joblib
    joblib.dump({"pipeline": clf.pipeline, "features": clf.features, "info": clf.info}, path)


def load_classifier(path: str | Path) -> FinalClassifier:
    import joblib
    d = joblib.load(path)
    return FinalClassifier(d["pipeline"], d["features"], d["info"])
