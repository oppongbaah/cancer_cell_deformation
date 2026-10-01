"""Tests for celldform.integration.realtime.feature_mask."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from celldform.integration.realtime import feature_mask


class _Fixed:
    """Stands in for a UNet: returns a fixed probability stack."""

    def __init__(self, probs):
        self.probs = probs

    def predict_proba(self, _image):
        return self.probs


def _random_probs(c, rng, h=32, w=32):
    logits = rng.normal(size=(c, h, w)).astype(np.float32)
    e = np.exp(logits - logits.max(0))
    return e / e.sum(0)


def test_matches_argmax_of_resampled_probabilities():
    rng = np.random.default_rng(0)
    size = (123, 77)                                  # (w, h), anisotropic on purpose
    for _ in range(5):
        p = _random_probs(3, rng)
        ref = np.stack([cv2.resize(c, size, interpolation=cv2.INTER_LINEAR)
                        for c in p]).argmax(0) == 1
        got = feature_mask(_Fixed(p), None, size)
        assert got.shape == (77, 123) and got.dtype == np.uint8
        assert np.array_equal(got.astype(bool), ref)


def test_decoy_is_never_foreground():
    p = np.zeros((3, 8, 8), np.float32)
    p[2] = 1.0                                        # every pixel is decoy
    assert feature_mask(_Fixed(p), None, (16, 16)).sum() == 0


def test_crop_box_pastes_prediction_back_at_the_right_place():
    p = np.zeros((3, 4, 4), np.float32)
    p[0] = 1.0
    p[:, 1, 2] = [0.0, 1.0, 0.0]                      # one trapped-cell pixel in the crop
    full = feature_mask(_Fixed(p), None, (10, 8), crop_box=(5, 3, 9, 7))
    assert full.shape == (8, 10) and full.sum() == 1 and full[3 + 1, 5 + 2] == 1


def test_preprocessing_crop_window_is_clamped_inside_the_frame():
    from celldform.preprocessing.pipeline import crop_box
    assert crop_box((579, 510), 512, 922, 1228) == (323, 254, 835, 766)
    assert crop_box((5, 5), 100, 922, 1228) == (0, 0, 100, 100)          # clamped
    assert crop_box((1225, 920), 100, 922, 1228) == (1128, 822, 1228, 922)
