"""
Frame-by-frame inference helper (legacy).

This module holds two things:

* :func:`feature_mask` and :data:`FEATURE_SIZE` -- the way a U-Net prediction is turned into the mask that features are
  measured on (class probabilities resampled to the native 1228x922 frame, the trapped cell only). The evaluation
  scripts rely on this contract.
* :class:`RealTimePipeline` -- a per-frame loop (preprocess -> segment -> features -> classifier) from before the
  project settled on classifying a whole recording. **It is not real-time** (the measured speed is only a reference),
  and it calls the original per-frame ``CellClassifier`` interface, not the final HER2 framework. It is kept because
  ``scripts/inference/infer.py`` still uses it. To classify a recording, use the desktop app / ``celldform.demo``
  (``celldform-demo``), which applies the validated checks and the final classifier.

Data flow of the per-frame loop::

  FrameExtractor.stream() -> PreprocessingPipeline -> feature_mask() -> MorphologyExtractor.extract()
      -> CellClassifier.predict()  (low / high HER2 label for that frame)
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import cv2
import numpy as np

from celldform.acquisition.extractor import FrameExtractor
from celldform.biomechanics.analysis import DeformationAnalyzer
from celldform.classification.classifiers import CellClassifier
from celldform.features.extractor import MorphologyExtractor
from celldform.preprocessing.pipeline import PreprocessingPipeline
from celldform.segmentation.unet import UNet

#: ``(width, height)`` at which masks are handed to MorphologyExtractor: the size every
#: frame pool was extracted at, and the size ``min_area_px`` is written in. The network
#: sees 256x256, reached by a non-uniform stretch (x / 4.80, y / 3.60), so features taken
#: at 256 describe a cell squashed ~1.33:1 along the beam axis -- the axis the trap
#: stretches it along. Measured 2026-09-30 on the mask factory: that squash inverts the
#: high/low HER2 ordering of aspect ratio, and the 50 px floor rejects 74 of 555 frames
#: at 256 against 0 here.
FEATURE_SIZE: Tuple[int, int] = (1228, 922)


def feature_mask(model: Any, processed: np.ndarray,
                 size: Tuple[int, int] = FEATURE_SIZE,
                 crop_box: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Trapped-cell mask at *size* ``(width, height)`` for feature extraction.

    Resamples the model's per-class **probabilities** (bilinear) and argmaxes at the
    target size, rather than upsampling the argmaxed 256 mask, which would leave a
    4.8 x 3.6 px staircase on the boundary and bias perimeter and circularity. Works
    with :class:`UNet` and :class:`~celldform.segmentation.ensemble.EnsembleUNet`.
    Decoy pixels (class 2) are never foreground, the same contract as
    ``predict(binary_output=True)``.

    ``crop_box`` (x0, y0, x1, y1), in full-frame pixels, is for models fed a crop of the
    frame (preprocessing ``crop_size``): the prediction is resampled to the box and pasted
    back, and everything outside it is background.
    """
    probs = model.predict_proba(processed)                          # (C, h, w)
    if crop_box is None:
        return _fg_from_probs(probs, size)
    x0, y0, x1, y1 = crop_box
    full = np.zeros((size[1], size[0]), dtype=np.uint8)
    full[y0:y1, x0:x1] = _fg_from_probs(probs, (x1 - x0, y1 - y0))
    return full


def _fg_from_probs(probs: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Trapped-cell foreground at *size* (w, h) from (C, h, w) class probabilities."""
    if probs.shape[0] == 1:
        up = cv2.resize(probs[0], size, interpolation=cv2.INTER_LINEAR)
        return (up >= 0.5).astype(np.uint8)
    # "class 1 wins the argmax" == "p1 - pk > 0 for every other k". Bilinear resampling is
    # linear, so each margin p1 - pk can be resampled in place of p1 and pk separately --
    # the min over k must come AFTER resampling (min is not linear). Skips a full-size
    # argmax (15 ms at 1228x922, measured). Identical to argmax(resized) == 1 up to ties.
    # At a 1:1 crop the resize is the identity.
    fg = None
    for k in range(probs.shape[0]):
        if k == 1:
            continue
        m = cv2.resize(probs[1] - probs[k], size, interpolation=cv2.INTER_LINEAR) > 0
        fg = m if fg is None else fg & m
    return fg.astype(np.uint8)


class RealTimePipeline:
    """Per-frame inference loop (legacy; not real-time, and not the final HER2 classifier -- see the module docstring).

    Parameters
    ----------
    unet:
        Loaded (and eval-mode) UNet segmentation model.
    preprocessor:
        Configured PreprocessingPipeline instance.
    extractor:
        MorphologyExtractor with physical pixel size set if calibrated.
    classifier:
        Fitted CellClassifier (SVM / RF / DT).
    analyzer:
        DeformationAnalyzer holding the current experimental conditions.
    meganet:
        Optional MegaNet CNN for direct deformation regression from the raw
        frame (bypasses the morphology extractor path).  When provided, its
        predictions are included in the result under ``"meganet_preds"``.
    device:
        PyTorch device string.  Auto-detected if *None*.
    on_result:
        Optional callback invoked with the result dict after each frame.
        Use this to stream results to the C++ application via shared memory
        or a socket (without blocking the Python loop).
    feature_size:
        ``(width, height)`` of the mask features are measured on -- see
        :data:`FEATURE_SIZE`. Video is also streamed at this size, so preprocessing
        (median blur, CLAHE) runs at the resolution the training frames saw it at.
    """

    def __init__(
        self,
        unet: UNet,
        preprocessor: PreprocessingPipeline,
        extractor: MorphologyExtractor,
        classifier: CellClassifier,
        analyzer: Optional[DeformationAnalyzer] = None,
        meganet: Optional[Any] = None,  # MegaNet, avoids circular import
        device: Optional[str] = None,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
        feature_size: Tuple[int, int] = FEATURE_SIZE,
    ) -> None:
        warnings.warn("RealTimePipeline is a legacy per-frame helper (not real-time, not the final HER2 framework); "
                      "to classify a recording use celldform.demo / celldform-demo", DeprecationWarning, stacklevel=2)
        self.unet = unet
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.classifier = classifier
        self.analyzer = analyzer
        self.meganet = meganet
        self.on_result = on_result
        self.feature_size = feature_size

        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> Dict[str, Any]:
        """Run the full pipeline on a single raw camera frame.

        Parameters
        ----------
        frame:
            Raw BGR or grayscale uint8 frame (H × W) or (H × W × C).
        frame_index:
            Monotonically increasing frame counter (for time-series bookkeeping).
        timestamp:
            Capture time in seconds since experiment start.

        Returns
        -------
        Dict with keys:
          frame_index, timestamp, mask (uint8 at ``feature_size``), features (Dict),
          cell_found (bool), label (int, or None when no cell was found),
          label_proba (float), latency_ms (float), and optionally meganet_preds.

        When the mask holds no region of ``min_area_px`` the features are NaN, the
        classifier is **not** called and ``cell_found`` is False -- a missing cell is
        reported as missing, never imputed into a feature vector.
        """
        t_start = time.perf_counter()

        processed = self.preprocessor(frame)
        box = (self.preprocessor.crop_box(self.feature_size[1], self.feature_size[0])
               if self.preprocessor.cfg.crop_size else None)
        mask = feature_mask(self.unet, processed, self.feature_size, box)
        features = self.extractor.extract(mask)
        cell_found = not np.isnan(features["area"])

        label: Optional[int] = None
        proba = float("nan")
        if cell_found:
            x = np.array([[features[k] for k in sorted(features)]], dtype=np.float32)
            label = int(self.classifier.predict(x)[0])
            try:
                proba = float(self.classifier.predict_proba(x)[0, 1])
            except Exception:
                pass

        result: Dict[str, Any] = {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "mask": mask,
            "features": features,
            "cell_found": cell_found,
            "label": label,                  # 0 = low HER2, 1 = high HER2, None = no cell
            "label_proba": proba,
            "latency_ms": (time.perf_counter() - t_start) * 1000.0,
        }

        if self.meganet is not None:
            result["meganet_preds"] = self.meganet.predict(processed, device=self.device)

        if self.on_result is not None:
            self.on_result(result)

        return result

    def run_on_video(
        self,
        video_path: Union[str, Path],
        every_n: int = 1,
        max_frames: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream results frame-by-frame from a video file.

        Parameters
        ----------
        video_path:
            Path to the MP4 / AVI video captured during the experiment.
        every_n:
            Process every n-th frame (1 = all frames).
        max_frames:
            Stop after this many processed frames.

        Yields
        ------
        Result dict for each processed frame (same schema as :meth:`process_frame`).
        """
        extractor = FrameExtractor(video_path, output_dir=None,  # stream mode
                                   target_size=self.feature_size)
        n_processed = 0

        for frame_index, timestamp, frame in extractor.stream(every_n=every_n):
            yield self.process_frame(frame, frame_index=frame_index, timestamp=timestamp)
            n_processed += 1
            if max_frames is not None and n_processed >= max_frames:
                break

    # ------------------------------------------------------------------
    # Convenience: REST-style dict for C++ JSON bridge
    # ------------------------------------------------------------------

    @staticmethod
    def to_json_safe(result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a result dict to JSON-serialisable types.

        Numpy arrays and tensors are converted to Python lists / scalars so
        the dict can be passed directly to ``json.dumps`` or a REST response.
        """
        safe: Dict[str, Any] = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                safe[k] = v.tolist()
            elif isinstance(v, dict):
                safe[k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                           for kk, vv in v.items()}
            elif isinstance(v, (np.floating, np.integer)):
                safe[k] = v.item()
            else:
                safe[k] = v
        return safe
