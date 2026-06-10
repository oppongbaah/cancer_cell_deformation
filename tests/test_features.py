"""Tests for celldform.features.MorphologyExtractor."""

import numpy as np
import pytest

from celldform.features.extractor import MorphologyExtractor


def _circle_mask(radius=20, size=64) -> np.ndarray:
    """Synthetic binary mask with a filled circle."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    mask[(x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2] = 1
    return mask


def _ellipse_mask(a=25, b=10, size=64) -> np.ndarray:
    """Synthetic binary mask with a horizontal ellipse."""
    mask = np.zeros((size, size), dtype=np.uint8)
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    mask[((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1] = 1
    return mask


class TestMorphologyExtractor:

    def test_circle_features_are_finite(self):
        ext = MorphologyExtractor()
        feats = ext.extract(_circle_mask())
        for k, v in feats.items():
            assert np.isfinite(v), f"{k} = {v} is not finite"

    def test_deformation_index_near_zero_for_circle(self):
        ext = MorphologyExtractor()
        feats = ext.extract(_circle_mask())
        assert abs(feats["deformation_index"]) < 0.1

    def test_aspect_ratio_gt_one_for_ellipse(self):
        ext = MorphologyExtractor()
        feats = ext.extract(_ellipse_mask())
        assert feats["aspect_ratio"] > 1.0

    def test_empty_mask_returns_nans(self):
        ext = MorphologyExtractor()
        empty = np.zeros((64, 64), dtype=np.uint8)
        feats = ext.extract(empty)
        assert all(np.isnan(v) for v in feats.values())

    def test_batch_returns_dataframe_with_correct_shape(self):
        ext = MorphologyExtractor()
        masks = [_circle_mask(), _ellipse_mask()]
        df = ext.batch(masks)
        assert df.shape == (2, len(ext._FEATURE_NAMES))

    def test_pixel_size_conversion(self):
        ext_px = MorphologyExtractor(pixel_size_um=None)
        ext_um = MorphologyExtractor(pixel_size_um=0.5)
        mask = _circle_mask()
        f_px = ext_px.extract(mask)
        f_um = ext_um.extract(mask)
        assert f_um["area"] == pytest.approx(f_px["area"] * 0.25, rel=1e-3)
