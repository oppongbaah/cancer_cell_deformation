"""
Data preprocessing pipeline for optical tweezer cell images.

The pipeline is implemented as a callable that applies a fixed sequence of
operations to a single grayscale frame:

  1. Denoising      — median filter to suppress salt-and-pepper noise
  2. Enhancement    — CLAHE (adaptive histogram equalisation) for contrast
  3. Morphology     — opening/closing to remove artefacts and fill gaps
  4. Normalisation  — zero-mean, unit-variance; rescale to [0, 1] float32
  5. Resize         — final spatial resolution check before model input

Each step can be individually enabled / disabled via the config dict, making
the pipeline easy to ablate in experiments.

HPC note: PreprocessingPipeline is stateless after construction and safe to
use across multiple worker processes without copying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessingConfig:
    """Hyper-parameters for each preprocessing stage."""

    # Denoising
    median_kernel: int = 3          # must be odd

    # CLAHE contrast enhancement
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    # Morphological operations
    morph_kernel_size: int = 3
    morph_open_iters: int = 1       # removes small bright specks
    morph_close_iters: int = 1      # fills small dark holes

    # Output spatial size (H, W) — must match model input
    output_size: Tuple[int, int] = (256, 256)

    # Toggle individual stages
    enabled: Dict[str, bool] = field(
        default_factory=lambda: {
            "denoise": True,
            "enhance": True,
            "morphology": True,
            "normalize": True,
            "resize": True,
        }
    )


class PreprocessingPipeline:
    """Applies the full preprocessing pipeline to a single cell image.

    Parameters
    ----------
    config:
        A :class:`PreprocessingConfig` instance.  If *None*, defaults are used.

    Examples
    --------
    >>> pipeline = PreprocessingPipeline()
    >>> out = pipeline(raw_frame)          # ndarray float32, shape (256, 256)
    >>> out_batch = pipeline.batch(frames) # list of processed frames
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.cfg = config or PreprocessingConfig()

        # Build the CLAHE object once; reuse across calls (thread-safe read).
        self._clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip_limit,
            tileGridSize=self.cfg.clahe_tile_grid,
        )

        # Structuring element for morphological operations.
        k = self.cfg.morph_kernel_size
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k, k)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Process a single image through the full pipeline.

        Parameters
        ----------
        image:
            Grayscale uint8 array (H × W) or colour BGR array (H × W × 3).

        Returns
        -------
        Float32 array in [0, 1] with shape ``(output_size[0], output_size[1])``.
        """
        img = self._to_gray(image)

        if self.cfg.enabled.get("denoise", True):
            img = self._denoise(img)

        if self.cfg.enabled.get("enhance", True):
            img = self._enhance(img)

        if self.cfg.enabled.get("morphology", True):
            img = self._morphology(img)

        if self.cfg.enabled.get("resize", True):
            img = self._resize(img)

        if self.cfg.enabled.get("normalize", True):
            img = self._normalize(img)

        return img

    def batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Apply the pipeline to a list of images."""
        return [self(img) for img in images]

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        """Convert BGR → grayscale if the input has 3 channels."""
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Median filter — robust against impulse noise in microscopy images."""
        return cv2.medianBlur(img, self.cfg.median_kernel)

    def _enhance(self, img: np.ndarray) -> np.ndarray:
        """CLAHE — improves visibility of cell boundaries under uneven lighting."""
        return self._clahe.apply(img)

    def _morphology(self, img: np.ndarray) -> np.ndarray:
        """Opening removes isolated bright noise; closing fills small dark gaps."""
        opened = cv2.morphologyEx(
            img, cv2.MORPH_OPEN, self._morph_kernel,
            iterations=self.cfg.morph_open_iters,
        )
        closed = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE, self._morph_kernel,
            iterations=self.cfg.morph_close_iters,
        )
        return closed

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Resize to the model's expected spatial input dimensions."""
        h, w = self.cfg.output_size
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        """Rescale pixel values to [0, 1] float32."""
        img_f = img.astype(np.float32)
        min_v, max_v = img_f.min(), img_f.max()
        if max_v > min_v:
            img_f = (img_f - min_v) / (max_v - min_v)
        return img_f
