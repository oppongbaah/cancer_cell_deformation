"""
Frame extraction from optical tweezer video recordings.

Supports two temporal sampling modes:
  - every_n    : keep every n-th frame (coarse pre-filter)
  - target_fps : resample to a uniform frame rate

Trapped-cell-aware extraction:
  When ``change_threshold`` is set, each candidate frame is analysed to locate
  the trapped cell and track its position.  A new frame is saved only when the
  trapped cell centroid has moved by more than ``change_threshold`` pixels:

      displacement = sqrt((cx - cx_prev)² + (cy - cy_prev)²)

  Cell detection uses three cues derived from the physics of the experiment:

  1. **Spatial channel filter** — cells flow in a narrow vertical channel that
     runs through the horizontal centre of the image.  ``channel_width``
     (fraction of image width, default 0.30) defines how wide this strip is.
     Only blobs whose centroids fall inside this strip are considered.

  2. **Circularity filter** — all cells are approximately circular.  Each dark
     blob found inside the channel is tested:
         circularity = 4π × area / perimeter²  ∈ [0, 1]
     Blobs below ``min_circularity`` (default 0.60) are rejected.

  3. **Intensity selection** — among the remaining circular blobs, the trapped
     cell is the darkest one (lowest mean pixel intensity inside the contour).
     The trapped cell absorbs / scatters more laser light and therefore appears
     darker than free-flowing cells.

  When no qualifying blob is found the frame is always saved (safe fallback).

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
        Minimum Euclidean displacement (pixels) of the trapped cell centroid
        between the current and the last saved frame required to save a new
        frame.  ``None`` disables tracking and saves every candidate frame.
    dark_percentile:
        Pixels at or below this intensity percentile *within the channel* are
        treated as cell candidates.  Default ``20`` (bottom 20 %).
    channel_width:
        Width of the vertical flow channel as a fraction of the image width,
        centred horizontally.  Default ``0.30`` (±15 % around the midpoint).
    min_circularity:
        Minimum circularity score (``4π·area/perimeter²``, 0–1) for a blob to
        be considered a cell.  Default ``0.60``.
    """

    def __init__(
        self,
        video_path: str | os.PathLike,
        output_dir: str | os.PathLike = "data/frames",
        target_size: Tuple[int, int] = (256, 256),
        image_format: str = "png",
        change_threshold: Optional[float] = None,
        dark_percentile: float = 20.0,
        channel_width: float = 0.30,
        min_circularity: float = 0.60,
    ) -> None:
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.image_format = image_format.lstrip(".")
        self.change_threshold = change_threshold
        self.dark_percentile = dark_percentile
        self.channel_width = channel_width
        self.min_circularity = min_circularity

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
                centroid = _find_trapped_cell_centroid(
                    processed,
                    dark_percentile=self.dark_percentile,
                    channel_width=self.channel_width,
                    min_circularity=self.min_circularity,
                )
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
    channel_width: float = 0.30,
    min_circularity: float = 0.60,
) -> Optional[Tuple[float, float]]:
    """Locate the trapped cell and return its centroid in full-frame coordinates.

    Detection uses three constraints that reflect the physics of the experiment:

    1. **Channel mask** — cells travel in a vertical channel centred on the
       horizontal midpoint of the image.  Only the strip spanning
       ``channel_width`` × image_width around the centre is searched.

    2. **Circularity** — all cells are approximately circular.
       Blobs with ``4π·area / perimeter²`` below *min_circularity* are rejected.

    3. **Darkest wins** — among the remaining circular blobs the trapped cell
       is the one with the lowest mean pixel intensity.

    Returns ``None`` when no qualifying circular blob is found inside the
    channel, which causes the frame to be saved as a safe fallback.
    """
    h, w = frame.shape[:2]
    x0 = int(w * (0.5 - channel_width / 2))
    x1 = int(w * (0.5 + channel_width / 2))
    roi = frame[:, x0:x1]

    # Gaussian blur before thresholding — smooths JPEG/codec compression
    # artefacts on cell boundaries so that circularity measurements are stable.
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    threshold = float(np.percentile(blurred, dark_percentile))
    dark_mask = (blurred <= threshold).astype(np.uint8)

    # Open: remove isolated noise pixels.
    # Close: fill small gaps caused by codec artefacts on the cell boundary.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_centroid: Optional[Tuple[float, float]] = None
    best_mean_intensity = float("inf")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        perimeter = cv2.arcLength(cnt, closed=True)
        if perimeter == 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        # Mean intensity inside this contour (lower = darker = more likely trapped).
        blob_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        mean_intensity = float(cv2.mean(roi, mask=blob_mask)[0])

        if mean_intensity < best_mean_intensity:
            best_mean_intensity = mean_intensity
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            # Convert from ROI-local x to full-frame x.
            cx = M["m10"] / M["m00"] + x0
            cy = M["m01"] / M["m00"]
            best_centroid = (float(cx), float(cy))

    return best_centroid


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
