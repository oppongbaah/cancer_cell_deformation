"""
Frame extraction from optical tweezer video recordings.

Supports two temporal sampling modes:
  - every_n    : keep every n-th frame (coarse pre-filter)
  - target_fps : resample to a uniform frame rate

Trapped-cell-aware extraction:
  When ``change_threshold`` is set, each candidate frame is analysed to locate
  the trapped cell — identified as the largest dark blob in the image (the
  trapped cell appears darker than surrounding cells because it absorbs /
  scatters more laser light).

  The centroid of that dark blob is computed and compared to the centroid from
  the last saved frame.  A new frame is saved only when the Euclidean
  displacement of the centroid exceeds ``change_threshold`` pixels:

      displacement = sqrt((cx - cx_prev)² + (cy - cy_prev)²)

  This targets the trapped cell specifically and ignores global background
  changes, making it sensitive to even small movements while avoiding
  redundant near-duplicate frames.

  ``dark_percentile`` controls how dark a pixel must be to be considered part
  of a cell.  The bottom 20 % of pixel intensities (default) reliably captures
  the trapped cell in typical optical tweezer recordings.

Each extracted frame is saved as ``<video_stem>_<index>.<format>``.  A
per-video CSV manifest records index, timestamp, centroid position, and
displacement for full traceability.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np


class FrameExtractor:
    """Extracts frames from a single optical tweezer video file.

    Parameters
    ----------
    video_path:
        Path to the source video (AVI, MP4, …).
    output_dir:
        Directory where extracted frames will be saved.
    target_size:
        ``(width, height)`` to resize each frame.  Default ``(256, 256)``
        matches the U-Net input resolution.
    image_format:
        ``"png"`` (lossless) or ``"tiff"``.
    change_threshold:
        Minimum Euclidean displacement (in pixels) of the trapped cell centroid
        between the current frame and the last saved frame required to save a
        new frame.  ``None`` disables tracking and saves every candidate frame.
    dark_percentile:
        Pixels at or below this intensity percentile are treated as the trapped
        cell region.  Default ``20`` (bottom 20 % of frame intensities).
    """

    def __init__(
        self,
        video_path: str | os.PathLike,
        output_dir: str | os.PathLike = "data/frames",
        target_size: Tuple[int, int] = (256, 256),
        image_format: str = "png",
        change_threshold: Optional[float] = None,
        dark_percentile: float = 20.0,
    ) -> None:
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.image_format = image_format.lstrip(".")
        self.change_threshold = change_threshold
        self.dark_percentile = dark_percentile

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        every_n: Optional[int] = None,
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Extract frames to ``output_dir``.

        Temporal sampling (``every_n`` / ``target_fps``) is a coarse
        pre-filter.  When ``change_threshold`` is also set, each candidate is
        further tested — only frames where the trapped cell has moved are saved.

        Parameters
        ----------
        every_n:
            Keep every n-th frame.  Mutually exclusive with *target_fps*.
        target_fps:
            Resample to this frame rate.  Mutually exclusive with *every_n*.
        max_frames:
            Hard upper limit on saved frames.

        Returns
        -------
        List of saved frames as uint8 NumPy arrays (H × W).
        """
        if every_n is not None and target_fps is not None:
            raise ValueError("Specify either every_n or target_fps, not both.")

        frames: List[np.ndarray] = []
        manifest_rows: List[dict] = []

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = self._compute_step(every_n, target_fps, native_fps)

        frame_idx = 0
        saved_idx = 0
        last_centroid: Optional[Tuple[float, float]] = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                processed = self._process_frame(frame)
                centroid = _find_trapped_cell_centroid(processed, self.dark_percentile)
                displacement = _centroid_displacement(centroid, last_centroid)

                # Save if: first frame, tracking disabled, or cell has moved enough.
                if (
                    last_centroid is None
                    or self.change_threshold is None
                    or displacement > self.change_threshold
                ):
                    timestamp_s = frame_idx / native_fps
                    out_path = (
                        self.output_dir
                        / f"{self.video_path.stem}_{saved_idx:06d}.{self.image_format}"
                    )
                    cv2.imwrite(str(out_path), processed)
                    last_centroid = centroid

                    frames.append(processed)
                    manifest_rows.append({
                        "source": str(self.video_path),
                        "frame_index": frame_idx,
                        "saved_index": saved_idx,
                        "timestamp_s": round(timestamp_s, 6),
                        "centroid_x": round(centroid[0], 2) if centroid else None,
                        "centroid_y": round(centroid[1], 2) if centroid else None,
                        "displacement_px": round(displacement, 3) if displacement != float("inf") else None,
                        "output_path": str(out_path),
                        "width": processed.shape[1],
                        "height": processed.shape[0],
                    })
                    saved_idx += 1

                    if max_frames is not None and saved_idx >= max_frames:
                        break

            frame_idx += 1

        cap.release()
        self._write_manifest(manifest_rows)
        return frames

    def stream(
        self,
        every_n: int = 1,
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Yield ``(frame_index, timestamp_s, frame)`` without saving to disk.

        All candidate frames are yielded — change detection is not applied in
        stream mode so the downstream pipeline receives every frame.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n == 0:
                yield frame_idx, frame_idx / native_fps, self._process_frame(frame)

            frame_idx += 1

        cap.release()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale and resize to target_size."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _compute_step(
        every_n: Optional[int],
        target_fps: Optional[float],
        native_fps: float,
    ) -> int:
        if every_n is not None:
            return max(1, int(every_n))
        if target_fps is not None:
            return max(1, int(round(native_fps / target_fps)))
        return 1

    def _write_manifest(self, rows: List[dict]) -> None:
        if not rows:
            return
        manifest_path = self.output_dir / f"{self.video_path.stem}_manifest.csv"
        with manifest_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Trapped-cell detection helpers
# ---------------------------------------------------------------------------

def _find_trapped_cell_centroid(
    frame: np.ndarray,
    dark_percentile: float = 20.0,
) -> Optional[Tuple[float, float]]:
    """Locate the trapped cell and return its centroid.

    The trapped cell appears darker than surrounding cells. We threshold at
    the given intensity percentile to isolate dark pixels, apply a small
    morphological open to remove noise, then return the centroid of the
    largest remaining dark blob.

    Returns ``None`` if no dark region is found.
    """
    threshold = float(np.percentile(frame, dark_percentile))
    dark_mask = (frame <= threshold).astype(np.uint8)

    # Remove isolated noise pixels.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(dark_mask)

    if num_labels < 2:   # only background label found
        return None

    # Skip background (label 0); find the largest dark blob.
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    cx, cy = centroids[largest]
    return float(cx), float(cy)


def _centroid_displacement(
    current: Optional[Tuple[float, float]],
    reference: Optional[Tuple[float, float]],
) -> float:
    """Euclidean distance between two centroids in pixels.

    Returns ``inf`` when either centroid is unavailable so the frame is
    always saved when the trapped cell cannot be located.
    """
    if current is None or reference is None:
        return float("inf")
    return float(np.sqrt((current[0] - reference[0]) ** 2 + (current[1] - reference[1]) ** 2))
