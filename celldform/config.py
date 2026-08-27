"""
Typed configuration for celldform.

All training, validation, and inference settings live here.
A single YAML file drives the entire pipeline — load it once with
:func:`load`, pass the resulting :class:`CelldformConfig` object to
trainers and scripts.

Usage::

    from celldform.config import load
    conf = load("configs/default.yaml")
    print(conf.training.lr, conf.unet.base_features)

Environment variables in path strings are expanded automatically, so
configs can use ``$HOME``, ``$WORK``, ``$SCRATCH``, etc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    frames_dir: str
    masks_dir: str
    features_csv: str = "data/features.csv"
    labels_csv: str = "data/labels.csv"
    val_split: float = 0.15     # fraction held out for validation
    test_split: float = 0.10    # fraction held out for final evaluation (never seen during training)


@dataclass
class PreprocessingCfg:
    target_size: Tuple[int, int] = (256, 256)
    median_kernel: int = 3
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    morph_kernel_size: int = 3
    denoise: bool = True
    enhance: bool = True
    morphology: bool = True
    resize: bool = True
    normalize: bool = True


@dataclass
class UNetCfg:
    in_channels: int = 1
    base_features: int = 64
    out_channels: int = 1  # 1 = binary (sigmoid); >1 = multiclass (softmax), e.g. 3 for background/trapped/other


@dataclass
class TrainingCfg:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"              # "adamw" | "adam" | "sgd"
    scheduler: str = "reduce_on_plateau"  # "reduce_on_plateau" | "cosine" | "none"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    num_workers: int = 4
    seed: int = 42
    loss_alpha: float = 0.3        # BCE weight in combined Dice+BCE loss (Dice = 1 - alpha)
    loss_pos_weight: float = 100.0 # pos_weight for BCEWithLogitsLoss; compensates cell/background imbalance
    use_amp: bool = True           # Automatic Mixed Precision — uses A100 BF16/TF32 units; auto-disabled on CPU
    class_weights: Optional[List[float]] = None  # per-class weight for CrossEntropyLoss (multiclass only); null = uniform
    domain_oversample_weight: float = 1.0  # sampling weight for domain_-prefixed (real-cell) train frames vs.
                                            # legacy frames; 1.0 = no oversampling (uniform, default). See
                                            # scripts/train_unet.py's WeightedRandomSampler for how this is applied.


@dataclass
class ResumeCfg:
    checkpoint: Optional[str] = None   # path to .pt file; null = start fresh
    load_optimizer: bool = True
    load_scheduler: bool = True


@dataclass
class FeaturesCfg:
    pixel_size_um: Optional[float] = None  # physical pixel size for unit conversion
    min_area_px: int = 50


@dataclass
class ClassificationCfg:
    scale_features: bool = True
    random_state: int = 42


@dataclass
class BiomechanicsCfg:
    metric: str = "aspect_ratio"
    window_s: Optional[float] = None
    laser_power_mW: float = 0.0
    trap_current_mA: float = 0.0


@dataclass
class AcquisitionCfg:
    target_fps: Optional[float] = None
    every_n: int = 1
    max_frames: Optional[int] = None
    image_format: str = "png"
    change_threshold: Optional[float] = 2.0  # min trapped-cell centroid displacement (px); null = disabled
    dark_percentile: float = 20.0             # bottom X% of channel intensities = candidate cells
    channel_width: float = 0.30               # vertical channel width as fraction of image width (centred)
    min_circularity: float = 0.60             # min 4π·area/perimeter² score to accept a blob as a cell
    domain_adapt_n: int = 5                   # frames per cell copied into 01_annotate_pool by the
                                              # dataset organiser (celldform-organize-dataset)


@dataclass
class CelldformConfig:
    """Top-level configuration object for the entire celldform pipeline."""
    data: DataConfig
    unet: UNetCfg
    training: TrainingCfg
    preprocessing: PreprocessingCfg
    features: FeaturesCfg
    classification: ClassificationCfg
    biomechanics: BiomechanicsCfg
    acquisition: AcquisitionCfg
    resume: ResumeCfg
    checkpoint_dir: str = "checkpoints"
    keep_n_checkpoints: int = 3
    device: Optional[str] = None   # null → auto-detect (cuda if available, else cpu)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load(path: str | Path) -> CelldformConfig:
    """Load and validate a celldform YAML config file.

    Parameters
    ----------
    path:
        Path to the YAML config file.

    Returns
    -------
    A fully populated :class:`CelldformConfig` with all optional fields
    resolved to their defaults.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If a required field is missing or a value is out of range.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file is empty or not a YAML mapping: {path}")

    raw = _expand_env_vars(raw)
    _require_sections(raw, ("data", "unet", "training"))

    return CelldformConfig(
        data=_parse_data(raw["data"]),
        preprocessing=_parse_preprocessing(raw.get("preprocessing", {})),
        unet=_parse_unet(raw.get("unet", {})),
        training=_parse_training(raw["training"]),
        resume=_parse_resume(raw.get("resume", {})),
        features=_parse_features(raw.get("features", {})),
        classification=_parse_classification(raw.get("classification", {})),
        biomechanics=_parse_biomechanics(raw.get("biomechanics", {})),
        acquisition=_parse_acquisition(raw.get("acquisition", {})),
        checkpoint_dir=str(raw.get("checkpoint_dir", "checkpoints")),
        keep_n_checkpoints=int(raw.get("keep_n_checkpoints", 3)),
        device=raw.get("device", None),
    )


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------

def _parse_data(d: dict) -> DataConfig:
    for key in ("frames_dir", "masks_dir"):
        if key not in d:
            raise ValueError(f"Config 'data' section is missing required key: '{key}'")
    val_split = float(d.get("val_split", 0.15))
    test_split = float(d.get("test_split", 0.10))
    if not (0 < val_split < 1):
        raise ValueError(f"data.val_split must be in (0, 1), got {val_split}")
    if not (0 <= test_split < 1):
        raise ValueError(f"data.test_split must be in [0, 1), got {test_split}")
    if val_split + test_split >= 1.0:
        raise ValueError(
            f"val_split ({val_split}) + test_split ({test_split}) must be < 1.0"
        )
    return DataConfig(
        frames_dir=d["frames_dir"],
        masks_dir=d["masks_dir"],
        features_csv=d.get("features_csv", "data/features.csv"),
        labels_csv=d.get("labels_csv", "data/labels.csv"),
        val_split=val_split,
        test_split=test_split,
    )


def _parse_preprocessing(p: dict) -> PreprocessingCfg:
    enabled = p.get("enabled", {})
    target_size = tuple(p.get("target_size", [256, 256]))
    clahe_tile = tuple(p.get("clahe_tile_grid", [8, 8]))
    return PreprocessingCfg(
        target_size=target_size,
        median_kernel=int(p.get("median_kernel", 3)),
        clahe_clip_limit=float(p.get("clahe_clip_limit", 2.0)),
        clahe_tile_grid=clahe_tile,
        morph_kernel_size=int(p.get("morph_kernel_size", 3)),
        denoise=bool(enabled.get("denoise", True)),
        enhance=bool(enabled.get("enhance", True)),
        morphology=bool(enabled.get("morphology", True)),
        resize=bool(enabled.get("resize", True)),
        normalize=bool(enabled.get("normalize", True)),
    )


def _parse_unet(u: dict) -> UNetCfg:
    return UNetCfg(
        in_channels=int(u.get("in_channels", 1)),
        base_features=int(u.get("base_features", 64)),
        out_channels=int(u.get("out_channels", 1)),
    )


def _parse_training(t: dict) -> TrainingCfg:
    optimizer = str(t.get("optimizer", "adamw")).lower()
    if optimizer not in ("adamw", "adam", "sgd"):
        raise ValueError(f"training.optimizer must be 'adamw', 'adam', or 'sgd', got '{optimizer}'")
    scheduler = str(t.get("scheduler", "reduce_on_plateau")).lower()
    if scheduler not in ("reduce_on_plateau", "cosine", "none"):
        raise ValueError(
            f"training.scheduler must be 'reduce_on_plateau', 'cosine', or 'none', got '{scheduler}'"
        )
    # Accept legacy key names for scheduler params.
    patience = int(t.get("scheduler_patience", t.get("lr_scheduler_patience", 5)))
    factor = float(t.get("scheduler_factor", t.get("lr_scheduler_factor", 0.5)))
    class_weights = t.get("class_weights", None)
    if class_weights is not None:
        class_weights = [float(w) for w in class_weights]
    return TrainingCfg(
        epochs=int(t.get("epochs", 50)),
        batch_size=int(t.get("batch_size", 16)),
        lr=float(t.get("lr", 1e-4)),
        weight_decay=float(t.get("weight_decay", 1e-4)),
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_patience=patience,
        scheduler_factor=factor,
        num_workers=int(t.get("num_workers", 4)),
        seed=int(t.get("seed", 42)),
        loss_alpha=float(t.get("loss_alpha", 0.3)),
        loss_pos_weight=float(t.get("loss_pos_weight", 100.0)),
        use_amp=bool(t.get("use_amp", True)),
        class_weights=class_weights,
        domain_oversample_weight=float(t.get("domain_oversample_weight", 1.0)),
    )


def _parse_resume(r: dict) -> ResumeCfg:
    ckpt = r.get("checkpoint", None)
    if ckpt is not None and not Path(ckpt).exists():
        raise FileNotFoundError(f"resume.checkpoint not found: {ckpt}")
    return ResumeCfg(
        checkpoint=ckpt,
        load_optimizer=bool(r.get("load_optimizer", True)),
        load_scheduler=bool(r.get("load_scheduler", True)),
    )


def _parse_features(fe: dict) -> FeaturesCfg:
    return FeaturesCfg(
        pixel_size_um=fe.get("pixel_size_um", None),
        min_area_px=int(fe.get("min_area_px", 50)),
    )


def _parse_classification(cl: dict) -> ClassificationCfg:
    return ClassificationCfg(
        scale_features=bool(cl.get("scale_features", True)),
        random_state=int(cl.get("random_state", 42)),
    )


def _parse_biomechanics(bm: dict) -> BiomechanicsCfg:
    return BiomechanicsCfg(
        metric=str(bm.get("metric", "aspect_ratio")),
        window_s=bm.get("window_s", None),
        laser_power_mW=float(bm.get("laser_power_mW", 0.0)),
        trap_current_mA=float(bm.get("trap_current_mA", 0.0)),
    )


def _parse_acquisition(ac: dict) -> AcquisitionCfg:
    return AcquisitionCfg(
        target_fps=ac.get("target_fps", None),
        every_n=int(ac.get("every_n", 1)),
        max_frames=ac.get("max_frames", None),
        image_format=str(ac.get("image_format", "png")),
        change_threshold=ac.get("change_threshold", 2.0),
        dark_percentile=float(ac.get("dark_percentile", 20.0)),
        channel_width=float(ac.get("channel_width", 0.30)),
        min_circularity=float(ac.get("min_circularity", 0.60)),
        domain_adapt_n=int(ac.get("domain_adapt_n", 5)),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_sections(raw: dict, sections: tuple) -> None:
    for s in sections:
        if s not in raw:
            raise ValueError(f"Config is missing required section: '{s}'")


def _expand_env_vars(obj):
    """Recursively expand $VAR environment variables in all string values."""
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj
