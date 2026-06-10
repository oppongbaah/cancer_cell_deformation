"""Tests for celldform.acquisition.FrameExtractor."""

import csv

import numpy as np
import pytest

from celldform.acquisition.extractor import (
    FrameExtractor,
    _centroid_displacement,
    _find_trapped_cell_centroid,
)


class TestFrameExtractor:
    """Unit tests using synthetic in-memory videos written via OpenCV."""

    @pytest.fixture
    def dummy_video(self, tmp_path):
        """5-frame AVI where each frame is a uniform grey, brightening by 50 per frame."""
        cv2 = pytest.importorskip("cv2")
        video_path = tmp_path / "test_video.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 64))
        for i in range(5):
            frame = np.full((64, 64, 3), i * 50, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return video_path

    @pytest.fixture
    def static_video(self, tmp_path):
        """5-frame AVI where every frame is identical."""
        cv2 = pytest.importorskip("cv2")
        video_path = tmp_path / "static_video.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 64))
        for _ in range(5):
            frame = np.full((64, 64, 3), 180, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return video_path

    @pytest.fixture
    def moving_cell_video(self, tmp_path):
        """5-frame video with a large dark blob (radius=20) on a bright background.

        The blob moves diagonally by 20 px per frame (≈28 px Euclidean).
        128×128 frames keep the blob at ~4 % of total pixels so that
        dark_percentile=2 falls inside the blob even after MJPG compression.
        """
        cv2 = pytest.importorskip("cv2")
        video_path = tmp_path / "moving_cell.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (128, 128))
        for i in range(5):
            frame = np.full((128, 128, 3), 210, dtype=np.uint8)
            cx, cy = 20 + i * 20, 20 + i * 20
            cv2.circle(frame, (cx, cy), 20, (10, 10, 10), -1)
            writer.write(frame)
        writer.release()
        return video_path

    # ------------------------------------------------------------------
    # Basic extraction
    # ------------------------------------------------------------------

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
        assert (tmp_path / "frames" / "test_video_manifest.csv").exists()
        assert frames[0].shape == (32, 32)

    def test_frames_named_with_video_stem(self, dummy_video, tmp_path):
        fe = FrameExtractor(dummy_video, output_dir=tmp_path / "frames", target_size=(32, 32))
        fe.extract(every_n=1)
        saved = list((tmp_path / "frames").glob("test_video_*.png"))
        assert len(saved) == 5
        assert all(f.name.startswith("test_video_") for f in saved)

    # ------------------------------------------------------------------
    # Trapped-cell change detection
    # ------------------------------------------------------------------

    def test_change_detection_skips_static_frames(self, static_video, tmp_path):
        """Identical frames share the same centroid — only the first is saved."""
        fe = FrameExtractor(
            static_video,
            output_dir=tmp_path / "frames",
            target_size=(32, 32),
            change_threshold=1.0,
        )
        frames = fe.extract(every_n=1)
        assert len(frames) == 1

    def test_change_detection_saves_all_when_cell_moves(self, moving_cell_video, tmp_path):
        """Blob moving ~28 px per frame with threshold=5 px saves all 5 frames."""
        fe = FrameExtractor(
            moving_cell_video,
            output_dir=tmp_path / "frames",
            target_size=(128, 128),
            change_threshold=5.0,
            dark_percentile=2.0,
        )
        frames = fe.extract(every_n=1)
        assert len(frames) == 5

    def test_change_detection_disabled_saves_all(self, static_video, tmp_path):
        """change_threshold=None disables tracking — all frames are saved."""
        fe = FrameExtractor(
            static_video,
            output_dir=tmp_path / "frames",
            target_size=(32, 32),
            change_threshold=None,
        )
        frames = fe.extract(every_n=1)
        assert len(frames) == 5

    def test_high_threshold_saves_only_large_movements(self, moving_cell_video, tmp_path):
        """Threshold of 200 px is larger than any actual movement — only first frame saved."""
        fe = FrameExtractor(
            moving_cell_video,
            output_dir=tmp_path / "frames",
            target_size=(128, 128),
            change_threshold=200.0,
            dark_percentile=2.0,
        )
        frames = fe.extract(every_n=1)
        assert len(frames) == 1

    # ------------------------------------------------------------------
    # Manifest columns
    # ------------------------------------------------------------------

    def test_manifest_contains_centroid_columns(self, moving_cell_video, tmp_path):
        fe = FrameExtractor(
            moving_cell_video,
            output_dir=tmp_path / "frames",
            target_size=(128, 128),
            change_threshold=5.0,
            dark_percentile=2.0,
        )
        fe.extract(every_n=1)
        manifest_path = tmp_path / "frames" / "moving_cell_manifest.csv"
        with open(manifest_path) as fh:
            rows = list(csv.DictReader(fh))
        assert "centroid_x" in rows[0]
        assert "centroid_y" in rows[0]
        assert "displacement_px" in rows[0]
        # First saved frame has no prior reference — csv.DictWriter writes None as "".
        assert rows[0]["displacement_px"] == ""
        # Subsequent frames have a numeric displacement.
        if len(rows) > 1:
            assert float(rows[1]["displacement_px"]) > 0

    # ------------------------------------------------------------------
    # Step computation
    # ------------------------------------------------------------------

    def test_compute_step_every_n(self):
        assert FrameExtractor._compute_step(every_n=3, target_fps=None, native_fps=30.0) == 3

    def test_compute_step_target_fps(self):
        assert FrameExtractor._compute_step(every_n=None, target_fps=10.0, native_fps=30.0) == 3

    def test_compute_step_defaults_to_one(self):
        assert FrameExtractor._compute_step(every_n=None, target_fps=None, native_fps=30.0) == 1


# ---------------------------------------------------------------------------
# Unit tests for module-level helpers (no video I/O, pure numpy)
# ---------------------------------------------------------------------------

class TestCentroidHelpers:
    def test_find_centroid_returns_tuple_for_dark_blob(self):
        # 64×64 frame, 200 background, dark 10×10 square starting at (20, 20).
        # The square is 100/4096 = 2.4 % of the frame.
        # dark_percentile=2 (82 pixels) falls inside the square, so the
        # centroid should be near the square's centre (24.5, 24.5).
        frame = np.full((64, 64), 200, dtype=np.uint8)
        frame[20:30, 20:30] = 10
        cx, cy = _find_trapped_cell_centroid(frame, dark_percentile=2.0)
        assert abs(cx - 24.5) < 3
        assert abs(cy - 24.5) < 3

    def test_centroid_displacement_correct(self):
        d = _centroid_displacement((0.0, 0.0), (3.0, 4.0))
        assert abs(d - 5.0) < 1e-6

    def test_centroid_displacement_none_reference_returns_inf(self):
        assert _centroid_displacement((10.0, 10.0), None) == float("inf")

    def test_centroid_displacement_none_current_returns_inf(self):
        assert _centroid_displacement(None, (10.0, 10.0)) == float("inf")
