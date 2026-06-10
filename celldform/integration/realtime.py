"""
Real-time integration pipeline — bridges the Python analysis stack to the
C++ optical tweezers control application.

Architecture position
---------------------

  C++ application
       │
       ▼  (video stream or shared-memory frame buffer)
  FrameExtractor.stream()
       │
       ▼
  PreprocessingPipeline.__call__()
       │
       ▼
  UNet.predict()  ← segmentation mask
       │
       ▼
  MorphologyExtractor.extract()  ← feature dict
       │
       ├── MegaNet.predict()  ← regression outputs (optional)
       │
       └── CellClassifier.predict()  ← low / high HER2 label
       │
       ▼
  result dict / REST API response / callback to C++

HPC note
--------
For HPC batch runs (Anvil/Purdue), `device="cuda"` and `num_workers > 0`
should be passed.  The pipeline is stateless across frames so it is safe
to share across processes without locking.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Union

import numpy as np

from celldform.acquisition.extractor import FrameExtractor
from celldform.biomechanics.analysis import DeformationAnalyzer
from celldform.classification.classifiers import CellClassifier
from celldform.features.extractor import MorphologyExtractor
from celldform.preprocessing.pipeline import PreprocessingPipeline
from celldform.segmentation.unet import UNet


class RealTimePipeline:
    """End-to-end inference pipeline for live optical tweezers experiments.

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
    ) -> None:
        self.unet = unet
        self.preprocessor = preprocessor
        self.extractor = extractor
        self.classifier = classifier
        self.analyzer = analyzer
        self.meganet = meganet
        self.on_result = on_result

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
          frame_index, timestamp, mask (H×W uint8), features (Dict),
          label (int), label_proba (float), latency_ms (float),
          and optionally meganet_preds (array).
        """
        t_start = time.perf_counter()

        processed = self.preprocessor(frame)
        mask = self.unet.predict(processed, device=self.device)
        features = self.extractor.extract(mask)

        label_arr = self.classifier.predict(
            np.array([[features[k] for k in sorted(features)]], dtype=np.float32)
        )
        label = int(label_arr[0])

        try:
            proba = float(
                self.classifier.predict_proba(
                    np.array([[features[k] for k in sorted(features)]], dtype=np.float32)
                )[0, 1]
            )
        except Exception:
            proba = float("nan")

        result: Dict[str, Any] = {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "mask": mask,
            "features": features,
            "label": label,                  # 0 = low HER2, 1 = high HER2
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
        extractor = FrameExtractor(video_path, output_dir=None)  # stream mode
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
