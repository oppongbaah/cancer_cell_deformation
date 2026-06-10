"""
Morphological feature extraction from U-Net segmentation masks.

Features computed per mask (all derived from the binary cell region):

  Geometric
  ---------
  area          — pixel count of the segmented cell region  (A)
  perimeter     — boundary length of the region             (P)
  major_axis    — length of the major ellipse axis          (L_maj)
  minor_axis    — length of the minor ellipse axis          (L_min)

  Shape descriptors
  -----------------
  aspect_ratio  — L_maj / L_min  (AR)
  eccentricity  — elongation of the fitted ellipse          (0 = circle)
  circularity   — 4π·area / perimeter²                     (1 = perfect circle)
  solidity      — area / convex-hull area

  Moment invariants
  -----------------
  hu_0 … hu_6   — 7 Hu moment invariants (rotation/scale/translation-invariant)

Reference
---------
Proposal §3.5, §2.4, and Table 1.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import cv2
from skimage import measure


class MorphologyExtractor:
    """Compute morphological descriptors from binary segmentation masks.

    Parameters
    ----------
    pixel_size_um:
        Physical size of one pixel in micrometres.  When provided, area and
        perimeter are converted to μm² and μm respectively.  Pass *None* to
        keep pixel units.
    min_area_px:
        Regions smaller than this (pixels) are discarded as noise.
    """

    _FEATURE_NAMES = [
        "area", "perimeter",
        "major_axis", "minor_axis",
        "aspect_ratio", "eccentricity",
        "circularity", "solidity",
        "hu_0", "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6",
    ]

    def __init__(
        self,
        pixel_size_um: Optional[float] = None,
        min_area_px: int = 50,
    ) -> None:
        self.pixel_size_um = pixel_size_um
        self.min_area_px = min_area_px

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, mask: np.ndarray) -> Dict[str, float]:
        """Extract features from a single binary mask.

        Parameters
        ----------
        mask:
            uint8 array (H × W) with cell pixels = 1, background = 0.

        Returns
        -------
        Dict mapping feature name → float value.
        Returns a dict of NaN values if no valid region is found.
        """
        region = self._largest_region(mask)
        if region is None:
            return {k: float("nan") for k in self._FEATURE_NAMES}

        return self._compute_features(region)

    def batch(
        self,
        masks: List[np.ndarray],
        frame_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Extract features from a list of masks and return a DataFrame.

        Parameters
        ----------
        masks:
            List of binary uint8 masks.
        frame_ids:
            Optional list of integer frame indices used as the DataFrame index.

        Returns
        -------
        pandas DataFrame with one row per mask and one column per feature.
        """
        rows = [self.extract(m) for m in masks]
        df = pd.DataFrame(rows, columns=self._FEATURE_NAMES)
        if frame_ids is not None:
            df.index = frame_ids
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _largest_region(self, mask: np.ndarray):
        """Return the skimage RegionProperties of the largest valid region."""
        labeled = measure.label(mask > 0)
        regions = measure.regionprops(labeled)

        # Filter tiny noise blobs.
        valid = [r for r in regions if r.area >= self.min_area_px]
        if not valid:
            return None

        # Select the region with the largest area (the target cell).
        return max(valid, key=lambda r: r.area)

    def _compute_features(self, region) -> Dict[str, float]:
        """Derive all features from a skimage RegionProperties object."""
        area = float(region.area)
        perimeter = float(region.perimeter)

        # Axis lengths from the fitted ellipse.
        major = float(region.axis_major_length)
        minor = float(region.axis_minor_length) + 1e-8  # avoid division by zero

        aspect_ratio = major / minor
        eccentricity = float(region.eccentricity)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
        solidity = float(region.solidity)

        # Hu moment invariants (7 values, log-scaled for numerical stability).
        mask_uint8 = (region.image.astype(np.uint8)) * 255
        moments = cv2.moments(mask_uint8)
        hu_raw = cv2.HuMoments(moments).flatten()
        hu = np.sign(hu_raw) * np.log10(np.abs(hu_raw) + 1e-30)

        # Optionally convert pixel units to physical units.
        if self.pixel_size_um is not None:
            area *= self.pixel_size_um ** 2
            perimeter *= self.pixel_size_um
            major *= self.pixel_size_um
            minor *= self.pixel_size_um

        return {
            "area": area,
            "perimeter": perimeter,
            "major_axis": major,
            "minor_axis": minor,
            "aspect_ratio": aspect_ratio,
            "eccentricity": eccentricity,
            "circularity": circularity,
            "solidity": solidity,
            **{f"hu_{i}": float(hu[i]) for i in range(7)},
        }
