"""Tests for celldform.acquisition.FrameExtractor."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from celldform.acquisition.extractor import FrameExtractor


class TestFrameExtractor:
    """Unit tests using a synthetic in-memory video written via OpenCV."""

    @pytest.fixture
    def dummy_video(self, tmp_path):
        """Create a minimal 5-frame grayscale AVI for testing."""
        cv2 = pytest.importorskip("cv2")
        video_path = tmp_path / "test_video.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 64))
        for i in range(5):
            frame = np.full((64, 64, 3), i * 50, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return video_path

    def test_stream_yields_tuples(self, dummy_video, tmp_path):
        fe = FrameExtractor(dummy_video, output_dir=tmp_path / "frames")
        results = list(fe.stream(every_n=1))
        assert len(results) == 5
        idx, ts, frame = results[0]
        assert isinstance(idx, int)
        assert isinstance(ts, float)
        assert isinstance(frame, np.ndarray)

    def test_extract_saves_frames_and_manifest(self, dummy_video, tmp_path):
        fe = FrameExtractor(dummy_video, output_dir=tmp_path / "frames", target_size=(32, 32))
        frames = fe.extract(every_n=1)
        assert len(frames) == 5
        assert (tmp_path / "frames" / "manifest.csv").exists()
        assert frames[0].shape == (32, 32)

    def test_compute_step_every_n(self):
        step = FrameExtractor._compute_step(every_n=3, native_fps=30.0, target_fps=None)
        assert step == 3

    def test_compute_step_target_fps(self):
        step = FrameExtractor._compute_step(every_n=None, native_fps=30.0, target_fps=10.0)
        assert step == 3

    def test_compute_step_defaults_to_one(self):
        step = FrameExtractor._compute_step(every_n=None, native_fps=30.0, target_fps=None)
        assert step == 1
