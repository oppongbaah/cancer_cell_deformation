"""Tests for celldform.preprocessing.PreprocessingPipeline."""

import numpy as np
import pytest

from celldform.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline


class TestPreprocessingPipeline:

    def _gray_frame(self, h=64, w=64) -> np.ndarray:
        return np.random.randint(0, 256, (h, w), dtype=np.uint8)

    def _bgr_frame(self, h=64, w=64) -> np.ndarray:
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    def test_output_is_float32_in_unit_range(self):
        pipe = PreprocessingPipeline()
        result = pipe(self._bgr_frame())
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_shape_is_target_size(self):
        cfg = PreprocessingConfig(output_size=(32, 32))
        pipe = PreprocessingPipeline(cfg)
        result = pipe(self._bgr_frame(128, 128))
        assert result.shape == (32, 32)

    def test_grayscale_input_accepted(self):
        pipe = PreprocessingPipeline()
        result = pipe(self._gray_frame())
        assert result.ndim == 2

    def test_batch_returns_list_of_same_length(self):
        pipe = PreprocessingPipeline()
        frames = [self._bgr_frame() for _ in range(4)]
        out = pipe.batch(frames)
        assert len(out) == 4

    def test_disabled_stages_are_skipped(self):
        cfg = PreprocessingConfig(
            enabled={
                "denoise": False,
                "enhance": False,
                "morphology": False,
                "resize": True,
                "normalize": True,
            }
        )
        pipe = PreprocessingPipeline(cfg)
        result = pipe(self._bgr_frame())
        assert result.dtype == np.float32
