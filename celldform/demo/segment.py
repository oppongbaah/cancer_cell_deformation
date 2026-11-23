"""
Segmenter for the demo: a gray 1228x922 frame in, trapped-cell labels out, at full-frame size.

The deployment U-Net (384 px native crop, 3 classes) is loaded from a checkpoint folder holding ``config.yaml``; the
crop window follows the config unless ``crop_center`` is given -- the centre is a calibration of the trap position of
THE SETUP, so a recording from a different setup needs its own (see :func:`estimate_crop_center`).

Output per frame (:class:`Seg`): ``labels`` (0 background, 1 trapped cell, 2 decoy; full frame), ``maxprob`` (top-class
probability per pixel, full frame; 1.0 outside the crop, where nothing is predicted), ``crop_box``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from celldform.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline

DEPLOY_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints/randswap/experiment_0_cv_baseline/cv_baseline/fold_deploy"


@dataclass
class Seg:
    labels: np.ndarray                   # (H, W) uint8, 0/1/2
    maxprob: np.ndarray                  # (H, W) float32
    crop_box: Optional[Tuple[int, int, int, int]]

    @property
    def cell(self) -> np.ndarray:
        return (self.labels == 1).astype(np.uint8)


def probs_to_seg(probs: np.ndarray, size: Tuple[int, int], crop_box: Optional[Tuple[int, int, int, int]]) -> Seg:
    """(C, h, w) class probabilities -> :class:`Seg` at ``size`` (w, h); the prediction is pasted into ``crop_box``."""
    x0, y0, x1, y1 = crop_box if crop_box else (0, 0, size[0], size[1])
    w, h = x1 - x0, y1 - y0
    up = np.stack([cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) if p.shape != (h, w) else p for p in probs])
    labels = np.zeros((size[1], size[0]), np.uint8)
    maxprob = np.ones((size[1], size[0]), np.float32)
    labels[y0:y1, x0:x1] = up.argmax(0).astype(np.uint8)
    maxprob[y0:y1, x0:x1] = up.max(0)
    return Seg(labels, maxprob, crop_box)


def read_checkpoint(path: Path | str) -> dict:
    """The weights of a U-Net checkpoint file (a bare state dict or a training checkpoint wrapping one).

    Loaded with ``weights_only=True``: a checkpoint holds tensors and plain numbers, and refusing arbitrary pickled code
    means an unfamiliar file cannot run anything when it is opened.
    """
    import torch
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise ValueError(f"{path.name} is not a PyTorch checkpoint of weights ({type(e).__name__}: {str(e)[:120]})") from e
    sd = obj.get("model_state_dict", obj) if isinstance(obj, dict) else None
    if not isinstance(sd, dict) or not all(hasattr(v, "shape") for v in sd.values()):
        raise ValueError(f"{path.name} does not hold a U-Net state dict")
    return sd


def inspect_checkpoint(path: Path | str) -> dict:
    """``{in_channels, base_features, out_channels}`` of a U-Net checkpoint, read from its weight shapes."""
    sd = read_checkpoint(path)
    try:
        first, last = sd["enc1.block.0.weight"], sd["out_conv.weight"]
    except KeyError as e:
        raise ValueError(f"{Path(path).name} is not a celldform U-Net (no {e.args[0]!r} weight)") from None
    info = {"in_channels": int(first.shape[1]), "base_features": int(first.shape[0]), "out_channels": int(last.shape[0])}
    if info["out_channels"] < 2:
        raise ValueError(f"{Path(path).name} is a binary (1-channel) U-Net; the demo needs a 2- or 3-class one "
                         "(background, trapped cell[, decoy])")
    return info


class UNetSegmenter:
    """A U-Net checkpoint (the deployment one by default). ``segment(gray)`` -> :class:`Seg`.

    The architecture (input channels, width, classes) is read from the weights themselves. The pre-processing (median,
    CLAHE, morphology, crop size and centre) comes from ``config`` -- default ``<folder>/config.yaml``; a checkpoint
    without one is run with the deployment pre-processing (``config_source`` says which).
    """

    def __init__(self, folder: Path | str = DEPLOY_DIR, ckpt: Optional[str] = None, device: Optional[str] = None,
                 crop_center: Optional[Tuple[int, int]] = None, size: Tuple[int, int] = (1228, 922),
                 config: Optional[Path | str] = None) -> None:
        import torch
        from celldform.segmentation.unet import UNet
        folder = Path(folder)
        ckpt = ckpt or next(p.name for p in sorted(folder.glob("*.pt")))
        from celldform.demo.models import load_unet_settings
        cfg_path = Path(config) if config else folder / "config.yaml"
        if cfg_path.is_file():
            pp, self.config_source = load_unet_settings(cfg_path), str(cfg_path)
        elif (DEPLOY_DIR / "config.yaml").is_file():           # only on the machine the repository lives on
            pp, self.config_source = load_unet_settings(DEPLOY_DIR / "config.yaml"), "deployment settings"
        else:
            raise ValueError("this U-Net needs its settings file (the config.yaml saved with it): it says how frames are "
                             "prepared and where to crop. Load it together with the weights.")
        arch = inspect_checkpoint(folder / ckpt)
        self.size = size
        self.pre = PreprocessingPipeline(PreprocessingConfig(
            median_kernel=pp["median_kernel"], clahe_clip_limit=pp["clahe_clip_limit"],
            clahe_tile_grid=tuple(pp["clahe_tile_grid"]), morph_kernel_size=pp["morph_kernel_size"],
            output_size=tuple(pp["target_size"]), crop_size=pp.get("crop_size"),
            crop_center=tuple(crop_center or pp.get("crop_center") or (size[0] // 2, size[1] // 2))))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = UNet.from_checkpoint(folder / ckpt, device=self.device, **arch)
        except Exception as e:
            raise ValueError(f"{ckpt}: the weights do not fit the U-Net architecture ({str(e)[:160]})") from e
        self.checkpoint, self.arch = str(folder / ckpt), arch

    @classmethod
    def from_file(cls, path: Path | str, config: Optional[Path | str] = None, **kw) -> "UNetSegmenter":
        """A segmenter from one ``.pt`` file. ``config`` defaults to a ``config.yaml`` beside it, else the deployment
        pre-processing is used."""
        path = Path(path)
        return cls(path.parent, path.name, config=config, **kw)

    def describe(self) -> str:
        a = self.arch
        return (f"{Path(self.checkpoint).name}: U-Net, {a['out_channels']} classes, base {a['base_features']}, "
                f"pre-processing: {'deployment settings' if self.config_source == 'deployment settings' else Path(self.config_source).name}")

    @property
    def crop_box(self) -> Optional[Tuple[int, int, int, int]]:
        return self.pre.crop_box(self.size[1], self.size[0]) if self.pre.cfg.crop_size else None

    def segment(self, gray: np.ndarray) -> Seg:
        if gray.shape != (self.size[1], self.size[0]):
            raise ValueError(f"expected a {self.size[0]}x{self.size[1]} frame, got {gray.shape[::-1]}")
        probs = self.model.predict_proba(self.pre(gray))
        return probs_to_seg(probs, self.size, self.crop_box)


NNUNET_DIR = (Path(__file__).resolve().parent.parent.parent /
              "checkpoints/randswap/experiment_4_loocv/nnunet/Dataset502_CellTrapLOOCV/nnUNetTrainer__nnUNetPlans__2d")


class NNUNetSegmenter:
    """nnU-Net (2D, 1000 epochs, trained on all 15 cells): the model the classifier's features were measured on.

    Whole frame in (no crop, no pre-processing of ours -- nnU-Net normalises itself, as it did in training), mirroring
    test-time augmentation on. Slower than the U-Net (~0.4 s a frame on a laptop GPU), more accurate (0.949 vs 0.897).
    """

    def __init__(self, folder: Path | str = NNUNET_DIR, fold: str = "all", device: Optional[str] = None,
                 size: Tuple[int, int] = (1228, 922)) -> None:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        folder = Path(folder)
        ckpt = folder / f"fold_{fold}" / "checkpoint_final.pth"
        if not ckpt.is_file():
            raise FileNotFoundError(f"nnU-Net checkpoint not found: {ckpt}")
        self.size, self.checkpoint = size, str(ckpt)
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = str(dev)
        # The sliding-window logits are accumulated on the CPU: on an 8 GB GPU the on-device version ran out of memory.
        self.predictor = nnUNetPredictor(use_mirroring=True, device=dev, perform_everything_on_device=False,
                                         allow_tqdm=False)
        self.predictor.initialize_from_trained_model_folder(str(folder), use_folds=(fold if fold == "all" else int(fold),),
                                                            checkpoint_name="checkpoint_final.pth")

    @property
    def crop_box(self):
        return None

    def segment(self, gray: np.ndarray) -> Seg:
        if gray.shape != (self.size[1], self.size[0]):
            raise ValueError(f"expected a {self.size[0]}x{self.size[1]} frame, got {gray.shape[::-1]}")
        _, probs = self.predictor.predict_single_npy_array(gray[None, None].astype(np.float32),
                                                           {"spacing": [999, 1, 1]}, save_or_return_probabilities=True)
        return probs_to_seg(np.asarray(probs, np.float32)[:, 0], self.size, None)


SEGMENTERS = {"unet": "U-Net (fast, ~40 ms a frame)", "nnunet": "nnU-Net (accurate, ~0.4 s a frame)"}


def make_segmenter(name: str, crop_center: Optional[Tuple[int, int]] = None):
    """``"unet"`` or ``"nnunet"`` -> a segmenter with ``segment(gray) -> Seg``."""
    if name == "unet":
        return UNetSegmenter(crop_center=crop_center)
    if name == "nnunet":
        return NNUNetSegmenter()
    raise ValueError(f"unknown segmenter {name!r}; choose from {sorted(SEGMENTERS)}")


def estimate_crop_center(centroids, size: Tuple[int, int] = (1228, 922), crop: int = 384) -> Tuple[int, int]:
    """Crop centre from the trapped-cell centroids of a few frames (their median), clamped so the window fits."""
    a = np.asarray(centroids, float)
    cx, cy = np.median(a[:, 0]), np.median(a[:, 1])
    return (int(min(max(cx, crop // 2), size[0] - crop // 2)), int(min(max(cy, crop // 2), size[1] - crop // 2)))
