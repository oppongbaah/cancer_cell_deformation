"""Tests for celldform.acquisition.FrameExtractor."""

import csv

import numpy as np
import pytest

from celldform.acquisition.extractor import (
    FrameExtractor,
    _centroid_displacement,
    _find_trapped_cell_centroid,
)

# Detection settings calibrated for synthetic 128×128 test videos:
# The dark circular blob (radius=15) placed at x=64 (channel centre) occupies
# ~14.5% of the 128-wide channel strip → dark_percentile=10 reliably thresholds
# inside the blob after MJPG compression.
_DETECT_KWARGS = dict(dark_percentile=10.0, channel_width=0.30, min_circularity=0.60)


class TestFrameExtractor:
    """Unit tests using synthetic in-memory videos written via OpenCV."""

    @pytest.fixture
    def dummy_video(self, tmp_path):
        """5-frame AVI of uniform brightening frames — no cells, used for I/O tests only."""
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
    def static_cell_video(self, tmp_path):
        """5-frame AVI: dark circular cell at a fixed position in the channel.

        The cell is at the horizontal centre (x=64) and vertical centre (y=64)
        of a 128×128 frame and does not move.  With change detection enabled,
        only the first frame should be saved.
        """
        cv2 = pytest.importorskip("cv2")
        video_path = tmp_path / "static_cell.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (128, 128))
        for _ in range(5):
            frame = np.full((128, 128, 3), 200, dtype=np.uint8)
            cv2.circle(frame, (64, 64), 15, (10, 10, 10), -1)
            writer.write(frame)
        writer.release()
        return video_path

    @pytest.fixture
    def moving_cell_video(self, tmp_path):
        """5-frame AVI: dark circular cell (radius=15) in the channel, moving 20 px downward
        per frame (y-direction only, always at x=64 — channel centre).

        Euclidean displacement between consecutive frames = 20 px, well above
        the 5 px threshold used in the movement tests.
        """
        cv2 = pytest.importorskip("cv2")
        video_path = tmp_path / "moving_cell.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (128, 128))
        for i in range(5):
            frame = np.full((128, 128, 3), 200, dtype=np.uint8)
            cy = 15 + i * 20   # moves from y=15 to y=95 across 5 frames
            cv2.circle(frame, (64, cy), 15, (10, 10, 10), -1)
            writer.write(frame)
        writer.release()
        return video_path

    # ------------------------------------------------------------------
    # Basic extraction (no change detection)
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
        fe = FrameExtractor(
            dummy_video, output_dir=tmp_path / "frames",
            target_size=(32, 32), change_threshold=None,
        )
        frames = fe.extract(every_n=1)
        assert len(frames) == 5
        assert (tmp_path / "frames" / "test_video_manifest.csv").exists()
        assert frames[0].shape == (32, 32)

    def test_frames_named_with_video_stem(self, dummy_video, tmp_path):
        fe = FrameExtractor(
            dummy_video, output_dir=tmp_path / "frames",
            target_size=(32, 32), change_threshold=None,
        )
        fe.extract(every_n=1)
        saved = list((tmp_path / "frames").glob("test_video_*.png"))
        assert len(saved) == 5
        assert all(f.name.startswith("test_video_") for f in saved)

    # ------------------------------------------------------------------
    # Trapped-cell change detection
    # ------------------------------------------------------------------

    def test_change_detection_skips_static_cell(self, static_cell_video, tmp_path):
        """Static cell → centroid doesn't move → only first frame saved."""
        fe = FrameExtractor(
            static_cell_video,
            output_dir=tmp_path / "frames",
            target_size=(128, 128),
            change_threshold=1.0,
            **_DETECT_KWARGS,
        )
        frames = fe.extract(every_n=1)
        assert len(frames) == 1

    def test_change_detection_saves_all_when_cell_moves(self, moving_cell_video, tmp_path):
        """Cell moving 20 px/frame with threshold=5 px → all 5 frames saved."""
        fe = FrameExtractor(
            moving_cell_video,
            output_dir=tmp_path / "frames",
            target_size=(128, 128),
            change_threshold=5.0,
            **_DETECT_KWARGS,
        )
        frames = fe.extract(every_n=1)
        assert len(frames) == 5

    def test_change_detection_disabled_saves_all(self, static_cell_video, tmp_path):
        """change_threshold=None disables tracking — all frames are saved."""
        fe = FrameExtractor(
            static_cell_video,
            output_dir=tmp_path / "frames",
            target_size=(128, 128),
            change_threshold=None,
        )
        frames = fe.extract(every_n=1)
        assert len(frames) == 5

    def test_high_threshold_saves_only_large_movements(self, moving_cell_video, tmp_path):
        """Threshold of 200 px is larger than the 20 px movement → only first frame saved."""
        fe = FrameExtractor(
            moving_cell_video,
            output_dir=tmp_path / "frames",
            target_size=(128, 128),
            change_threshold=200.0,
            **_DETECT_KWARGS,
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
            **_DETECT_KWARGS,
        )
        fe.extract(every_n=1)
        manifest_path = tmp_path / "frames" / "moving_cell_manifest.csv"
        with open(manifest_path) as fh:
            rows = list(csv.DictReader(fh))
        assert "centroid_x" in rows[0]
        assert "centroid_y" in rows[0]
        assert "displacement_px" in rows[0]
        # First frame has no prior reference — csv.DictWriter writes None as "".
        assert rows[0]["displacement_px"] == ""
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
# Unit tests for module-level helpers (pure numpy, no video I/O)
# ---------------------------------------------------------------------------

class TestCentroidHelpers:
    def test_find_centroid_of_circular_blob_in_channel(self):
        """Dark circle in the channel centre → centroid close to circle centre."""
        frame = np.full((128, 128), 200, dtype=np.uint8)
        # Draw a dark circle at (64, 64) — squarely in the 30% channel (x: 51–77).
        import cv2
        cv2.circle(frame, (64, 64), 15, 10, -1)
        result = _find_trapped_cell_centroid(
            frame,
            dark_percentile=10.0,
            channel_width=0.30,
            min_circularity=0.60,
        )
        assert result is not None, "Should detect the dark circular cell in the channel"
        cx, cy = result
        assert abs(cx - 64) < 5, f"Expected cx≈64, got {cx:.1f}"
        assert abs(cy - 64) < 5, f"Expected cy≈64, got {cy:.1f}"

    def test_blob_outside_channel_not_detected(self):
        """Dark circle well outside the channel should not be returned."""
        frame = np.full((128, 128), 200, dtype=np.uint8)
        import cv2
        # Place cell far left (x=5), outside the 30% channel (x: 51–77).
        cv2.circle(frame, (5, 64), 10, 10, -1)
        result = _find_trapped_cell_centroid(
            frame,
            dark_percentile=10.0,
            channel_width=0.30,
            min_circularity=0.60,
        )
        # ROI for the channel doesn't include x=5, so nothing should be found.
        assert result is None

    def test_non_circular_blob_rejected(self):
        """A rectangular blob should be filtered out by the circularity check."""
        frame = np.full((128, 128), 200, dtype=np.uint8)
        # Draw a thin horizontal bar in the channel — very low circularity.
        frame[60:65, 45:83] = 10   # 5×38 bar, circularity ≈ 0.17
        result = _find_trapped_cell_centroid(
            frame,
            dark_percentile=2.0,
            channel_width=0.30,
            min_circularity=0.60,
        )
        assert result is None, "Thin bar should be rejected by circularity filter"

    def test_darkest_of_two_circles_selected(self):
        """When two circular cells are in the channel, the darker one is returned."""
        frame = np.full((128, 128), 200, dtype=np.uint8)
        import cv2
        cv2.circle(frame, (64, 30), 12, 80, -1)   # lighter cell (intensity 80)
        cv2.circle(frame, (64, 90), 12, 10, -1)   # darker cell (intensity 10) — trapped
        result = _find_trapped_cell_centroid(
            frame,
            dark_percentile=15.0,
            channel_width=0.30,
            min_circularity=0.60,
        )
        assert result is not None
        cx, cy = result
        # Should pick the darker cell near (64, 90), not the lighter one near (64, 30).
        assert abs(cy - 90) < 8, f"Expected cy≈90 (dark cell), got {cy:.1f}"

    def test_centroid_displacement_correct(self):
        d = _centroid_displacement((0.0, 0.0), (3.0, 4.0))
        assert abs(d - 5.0) < 1e-6

    def test_centroid_displacement_none_reference_returns_inf(self):
        assert _centroid_displacement((10.0, 10.0), None) == float("inf")

    def test_centroid_displacement_none_current_returns_inf(self):
        assert _centroid_displacement(None, (10.0, 10.0)) == float("inf")
