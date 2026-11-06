"""
Settings of the analysis stages -- feature extraction, the CNN, the classical classifiers and the deformation
regression -- read from YAML, so they can be changed without touching code.

    from celldform.analysis_config import load
    cfg = load("features")                         # configs/analysis/features.yaml
    cfg = load("cnn", "configs/analysis/my_cnn.yaml")

Every field has a default equal to the value the reported results were produced with, so an empty or missing file
reproduces them. A key that does not exist raises an error (a typo must not be silently ignored).
`dump(cfg, path)` writes the settings actually used next to a run's results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs" / "analysis"


# ── feature extraction ───────────────────────────────────────────────────────

@dataclass
class FrameChecks:
    max_trap_dist: float = 45.0                    # px from the local trap
    min_solidity: float = 0.95
    area_ratio_range: tuple = (0.85, 1.20)         # area / median area of the sound frames within +-window_steps
    min_boundary_conf: float = 0.95                # mean top-class probability along the outline
    allow_border: bool = False
    window_steps: int = 3                          # local trap and typical area: median over +-this many steps


@dataclass
class CellValidation:
    min_dice: float = 0.85                         # mean Dice of the cell's passing hand-masked frames (= good_dice)
    size_bias_range: tuple = (0.80, 1.25)          # median predicted / hand-drawn area (good masks: 0.82-1.23)
    min_pass_fraction: float = 0.8                 # of its feature frames, for the cell to be usable


@dataclass
class FeaturesConfig:
    frames_per_step: int = 200
    dense_offsets: tuple = (20, 60, 100, 140, 180)  # frames into a step cut besides the step frame
    good_dice: float = 0.85                        # calibration: a hand-masked frame below this is a bad mask
    frame_checks: FrameChecks = field(default_factory=FrameChecks)
    cell_validation: CellValidation = field(default_factory=CellValidation)


# ── CNN ──────────────────────────────────────────────────────────────────────

@dataclass
class Augment:
    flips: bool = True
    rot90: bool = True
    intensity_scale: tuple = (0.9, 1.1)            # masked crop only, inside the cell
    intensity_noise: float = 0.1


@dataclass
class CNNConfig:
    crop_px: int = 96                              # window at frame resolution
    grow_px: int = 2                               # mask grown by this before the background is cut away
    input_px: int = 128                            # the crop is resized to this for the network
    epochs: int = 8                                # fine-tuning epochs per fold (fixed; no validation set)
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2
    dropout: float = 0.3
    trainable_blocks: tuple = ("layer4",)          # ResNet-18 blocks fine-tuned; the rest stay frozen
    frozen_lr_C: float = 0.01                      # frozen CNN: logistic regression on its 512 features
    seed: int = 0
    variants: tuple = ("masked", "mask")
    augment: Augment = field(default_factory=Augment)
    tuning: dict = field(default_factory=lambda: {      # nested CV (frames_cnn.py tune): inner folds of cells
        "inner_folds": 4,
        "grid": {"epochs": [4, 8, 16], "lr": [1e-4, 3e-4]},
    })


# ── classical classifiers ────────────────────────────────────────────────────

@dataclass
class FeatureSelection:
    rho_min: float = 0.8                           # agreement with the hand masks and split-half reliability
    group_rho: Any = 0.7                           # correlation-matrix groups (average linkage on 1 - |rho|, cut
                                                   # here); one feature per group; None = no grouping
    max_selected: Any = None                       # at most this many; None = every feature that passes
    min_cells: int = 12                            # a feature must have a value in this many cells


@dataclass
class ClassifiersConfig:
    seed: int = 0
    permutations: int = 1000
    n_jobs: int = 8                                # parallel workers of the permutation tests (memory: -1 = every core)
    feature_selection: FeatureSelection = field(default_factory=FeatureSelection)
    models: dict = field(default_factory=lambda: {
        "Logistic regression": {"C": 1.0},
        "Linear SVM": {"C": 1.0},
        "Shrinkage LDA": {"shrinkage": "auto"},
        "RBF SVM": {"C": 1.0, "gamma": "scale"},
        "k-NN (k=3)": {"n_neighbors": 3},
        "Random forest": {"n_estimators": 1000, "min_samples_leaf": 2},
        "Gradient boosting": {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.1},
        "MLP (ANN)": {"hidden_layer_sizes": [8], "alpha": 1.0, "learning_rate_init": 0.01},
    })
    tuning: dict = field(default_factory=lambda: {      # nested CV (tune_cells.py)
        "permutations": 200,                             # label shuffles of the whole nested procedure
        "skip_permutations": ["Random forest", "Gradient boosting"],   # too slow to permute
        "grids": {
            "Logistic regression": {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "Linear SVM": {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "Shrinkage LDA": {"shrinkage": ["auto", 0.1, 0.5, 0.9]},
            "RBF SVM": {"C": [0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.01, 0.1, 1.0]},
            "k-NN (k=3)": {"n_neighbors": [1, 3, 5, 7]},
            "Random forest": {"min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", None]},
            "Gradient boosting": {"n_estimators": [50, 100, 200], "max_depth": [1, 2, 3]},
            "MLP (ANN)": {"hidden_layer_sizes": [[4], [8], [16]], "alpha": [0.1, 1.0, 10.0]},
        },
    })


# ── deformation regression / rate of deformation ─────────────────────────────

@dataclass
class RegressionConfig:
    measures: tuple = ("len_par", "len_perp", "volume")    # strain measures of the mixed model
    rod_measures: tuple = ("len_par", "len_perp")          # measures of the rate of deformation
    base_steps: int = 10                                   # baseline = median over steps 0..base_steps-1
    random_slope: bool = True                              # each cell its own slope (False: own intercept only)
    size_adjusted: bool = True                             # also fit with the baseline length as a covariate
    hac_lags: int = 3                                      # Newey-West lags of the per-cell RoD interval
    realtime_steps: tuple = (5, 10, 15, 20, 30, 40, 50)    # k for the real-time / early RoD
    fps: float = 22.0


# ── sequence network on the deformation curves ───────────────────────────────

@dataclass
class SequenceConfig:
    measures: tuple = ("len_par", "len_perp", "volume")   # strain curves (one input channel each)
    steps: int = 50                                       # curve length; shorter cells are padded with their last value
    base_steps: int = 10                                  # strain baseline = median over steps 0..base_steps-1
    archs: tuple = ("cnn1d", "gru")                       # models run: a 1-D CNN and a GRU (recurrent)
    channels: int = 8                                     # 1-D CNN filters per layer / GRU hidden units
    kernel: int = 5                                       # 1-D CNN kernel (steps)
    dropout: float = 0.3
    epochs: int = 300
    lr: float = 0.003
    weight_decay: float = 0.01
    seeds: tuple = (0, 1, 2)                              # repeats; their P(high) is averaged per cell
    crop: int = 40                                        # training: a random window of this many steps
    jitter: float = 0.5                                   # training: Gaussian noise added to the strain (%)
    scale: tuple = (0.8, 1.2)                             # training: random amplitude scaling of the strain
    permutations: int = 100                               # label shuffles (first seed only) for the permutation p
    n_jobs: int = 8                                       # parallel workers of the permutation test (each loads torch)


SECTIONS = {"features": FeaturesConfig, "cnn": CNNConfig, "classifiers": ClassifiersConfig,
            "regression": RegressionConfig, "sequence": SequenceConfig}


def _build(cls, data: dict, where: str):
    names = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(names)
    if unknown:
        raise KeyError(f"{where}: unknown setting(s) {sorted(unknown)}; allowed: {sorted(names)}")
    kw = {}
    for k, v in data.items():
        default = getattr(cls(), k)
        if is_dataclass(default):
            kw[k] = _build(type(default), v or {}, f"{where}.{k}")
        elif isinstance(default, tuple):
            kw[k] = tuple(v)
        elif isinstance(default, dict):
            kw[k] = dict(v)
        else:
            kw[k] = v
    return cls(**kw)


def load(section: str, path: str | Path | None = None):
    """The settings of one stage: defaults, overridden by the YAML file (configs/analysis/<section>.yaml)."""
    if section not in SECTIONS:
        raise KeyError(f"unknown section {section!r}; one of {sorted(SECTIONS)}")
    p = Path(path) if path else CONFIG_DIR / f"{section}.yaml"
    data = (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}
    if path and not p.exists():
        raise FileNotFoundError(p)
    return _build(SECTIONS[section], data, str(p.name))


def as_dict(cfg: Any) -> dict:
    def plain(v):
        if isinstance(v, tuple):
            return [plain(x) for x in v]
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        return v
    return {k: plain(v) for k, v in asdict(cfg).items()}


def dump(cfg: Any, path: Path) -> None:
    """Write the settings a run used, next to its results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(as_dict(cfg), sort_keys=False))
