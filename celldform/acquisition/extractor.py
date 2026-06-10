"""
Frame extraction from optical tweezer video recordings.

Supports two temporal sampling modes:
  - every_n  : keep every n-th frame (reduces redundancy in high-FPS recordings)
  - target_fps: resample to a uniform frame rate aligned to the experiment clock

Each extracted frame is resized to a target spatial resolution and written to
disk in PNG or TIFF format. A CSV manifest records frame index, timestamp,
source path, and output path for full traceability in downstream analysis.

HPC note: the extractor is stateless and safe to parallelise across multiple
video files using Python's multiprocessing or MPI via celldform.integration.
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
        Absolute path to the source video (AVI, MP4, TIFF stack …).
    output_dir:
        Directory where extracted frames will be saved.
    target_size:
        (width, height) to resize each frame before saving.
        Default ``(256, 256)`` matches the U-Net input resolution.
    image_format:
        Output image format — ``"png"`` (lossless) or ``"tiff"``.
    """

    def __init__(
        self,
        video_path: str | os.PathLike,
        output_dir: str | os.PathLike = "frames",
        target_size: Tuple[int, int] = (256, 256),
        image_format: str = "png",
    ) -> None:
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.image_format = image_format.lstrip(".")

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
        """Extract frames and persist them to ``output_dir``.

        Parameters
        ----------
        every_n:
            Keep every n-th frame.  Mutually exclusive with *target_fps*.
        target_fps:
            Resample to this frame rate.  Mutually exclusive with *every_n*.
        max_frames:
            Hard upper limit on frames to extract (useful for quick tests).

        Returns
        -------
        List of extracted frames as uint8 NumPy arrays (H × W × C).
        """
        if every_n is not None and target_fps is not None:
            raise ValueError("Specify either every_n or target_fps, not both.")

        frames: List[np.ndarray] = []
        manifest_rows: List[dict] = []

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Derive the frame-step from whichever sampling mode was chosen.
        step = self._compute_step(every_n, target_fps, native_fps)

        frame_idx = 0
        saved_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                timestamp_s = frame_idx / native_fps
                processed = self._process_frame(frame)
                out_path = self.output_dir / f"frame_{saved_idx:06d}.{self.image_format}"
                cv2.imwrite(str(out_path), processed)

                frames.append(processed)
                manifest_rows.append(
                    {
                        "source": str(self.video_path),
                        "frame_index": frame_idx,
                        "saved_index": saved_idx,
                        "timestamp_s": round(timestamp_s, 6),
                        "output_path": str(out_path),
                        "width": processed.shape[1],
                        "height": processed.shape[0],
                    }
                )
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
        """Yield (frame_index, timestamp_s, frame) without saving to disk.

        Suitable for real-time inference where frames are consumed immediately.
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
                ts = frame_idx / native_fps
                yield frame_idx, ts, self._process_frame(frame)

            frame_idx += 1

        cap.release()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize to target_size and convert to grayscale (single-channel)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
        return resized

    @staticmethod
    def _compute_step(
        every_n: Optional[int],
        target_fps: Optional[float],
        native_fps: float,
    ) -> int:
        if every_n is not None:
            return max(1, int(every_n))
        if target_fps is not None:
            # Round down to nearest integer step; at least 1.
            return max(1, int(round(native_fps / target_fps)))
        return 1  # default: keep all frames

    def _write_manifest(self, rows: List[dict]) -> None:
        """Persist frame metadata to a CSV file alongside the extracted frames."""
        if not rows:
            return
        manifest_path = self.output_dir / "manifest.csv"
        with manifest_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
