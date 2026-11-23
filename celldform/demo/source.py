"""
A recording on disk: the video, its notes, and the frames that can actually be decoded.

New recordings (2026-11) come as a folder ``Test_<n>_<date>/`` holding ``video.avi``, ``SKBR3.txt`` (cell label, start time,
camera), ``frames.csv`` (per-frame laser log, may be empty) and ``stills*/``. A copy can be cut short, so a video that
says it holds 11 480 frames may decode 24: :class:`Recording` reports how many frames it really yields, and never raises
on a damaged tail.

Frames are returned grayscale, uint8, at ``size`` (width, height) -- the form every training frame had -- after the
burned-in overlay text (laser current, power, pressure) has been inpainted: it is not in the training frames and a
segmenter would otherwise see it.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

FEATURE_SIZE: Tuple[int, int] = (1228, 922)


def remove_overlay(frame: np.ndarray) -> np.ndarray:
    """Inpaint the yellow overlay text of a colour frame (BGR). Gray frames are returned unchanged.

    The text is saturated yellow (high R and G, low B); the mask is dilated a little to cover anti-aliased edges. Warm
    background pixels are not that saturated, so they are left alone.
    """
    if frame.ndim == 2 or frame.shape[2] == 1:
        return frame
    b, g, r = (frame[..., i].astype(np.int16) for i in range(3))
    text = (r > 200) & (g > 200) & (b < 90) & ((np.minimum(r, g) - b) > 140)
    if text.sum() == 0:
        return frame
    text = cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8))
    return cv2.inpaint(frame, text, 3, cv2.INPAINT_TELEA)


def to_gray(frame: np.ndarray, size: Tuple[int, int] = FEATURE_SIZE) -> np.ndarray:
    """Overlay removed, grayscale, INTER_AREA-resized to ``size`` (w, h) as the frame extractor did."""
    g = frame if frame.ndim == 2 else cv2.cvtColor(remove_overlay(frame), cv2.COLOR_BGR2GRAY)
    if (g.shape[1], g.shape[0]) != tuple(size):
        g = cv2.resize(g, tuple(size), interpolation=cv2.INTER_AREA)
    return g


def read_notes(path: Path) -> dict:
    """``Key: value`` lines of SKBR3.txt as a dict (empty if the file is missing)."""
    out = {}
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            m = re.match(r"\s*([^:]+):\s*(.*)$", line)
            if m:
                out[m.group(1).strip()] = m.group(2).strip()
    return out


def read_frame_log(path: Path) -> Optional[pd.DataFrame]:
    """The per-frame laser log, or None if the file is missing or empty (an empty file is a failed copy, not a log)."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SOURCES = ("video", "images")


def find_images(folder: Path) -> Tuple[Path, "dict[int, Path]"]:
    """The image folder of a recording and its ``{frame index: file}``.

    ``folder`` is a recording folder (its ``stills*/`` subfolder is used), or a folder of images itself. The index is the
    video frame the still was taken at (``stills.csv``: column ``video_frame``, 1-based, so index = video_frame - 1) --
    which keeps laser steps and times identical to the video's. Without that table the sorted order is the index.
    """
    folder = Path(folder)
    sub = sorted(p for p in folder.iterdir() if p.is_dir() and p.name.lower().startswith("stills"))
    img_dir = sub[0] if sub else folder
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no images in {img_dir}")
    table = folder / "stills.csv"
    if table.is_file() and table.stat().st_size and {"still_file", "video_frame"} <= set(pd.read_csv(table, nrows=1).columns):
        t = pd.read_csv(table, usecols=["still_file", "video_frame"])
        names = {p.name: p for p in files}
        index = {int(v) - 1: names[n] for n, v in zip(t.still_file, t.video_frame) if n in names}
        if index:
            return img_dir, index
    return img_dir, dict(enumerate(files))


VIDEO_SUFFIXES = (".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv")


def natural_key(p: Path):
    """Sort key so that img2 comes before img10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", Path(p).name.lower())]


def is_video(p: Path) -> bool:
    return Path(p).suffix.lower() in VIDEO_SUFFIXES


def is_image(p: Path) -> bool:
    return Path(p).suffix.lower() in IMAGE_SUFFIXES


def videos_in(folder: Path) -> list:
    return sorted((p for p in Path(folder).iterdir() if p.is_file() and is_video(p)), key=natural_key)


@dataclass
class Recording:
    """A recording: a video or still images, plus its notes and laser log.

    ``path`` is a recording folder, one video file, or (with ``image_files``) nothing. What the frames are read from:

    * ``source="video"``: the video (a given file, else the first one in the folder);
    * ``source="images"``: the ``stills*/`` folder of the recording, a folder of images, or the explicit
      ``image_files`` (a hand-picked set; ``source`` is then forced to images).

    Frame indices, times and the laser log mean the same for both. Sidecar files (``frames.csv``, the notes ``*.txt``,
    ``stills.csv``) belong to a recording folder with one video; where a folder holds several videos, a video only
    gets sidecars named after it (``<video name>.csv`` for the laser log, ``<video name>.txt`` for the notes).
    """
    path: Optional[Path] = None
    size: Tuple[int, int] = FEATURE_SIZE
    source: str = "video"
    notes: dict = field(default_factory=dict)
    frame_log: Optional[pd.DataFrame] = None
    video: Optional[Path] = None
    image_files: Optional[Sequence] = None
    images: "dict[int, Path]" = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        explicit = [Path(f) for f in self.image_files] if self.image_files else None
        if explicit:
            bad = [f.name for f in explicit if not is_image(f)]
            if bad:
                raise ValueError(f"not an image file: {', '.join(bad[:3])}")
            missing = [f for f in explicit if not f.is_file()]
            if missing:
                raise FileNotFoundError(missing[0])
            self.source, self.path = "images", Path(self.path) if self.path else explicit[0].parent
        elif self.path is None:
            raise ValueError("a recording needs a path or image files")
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        # the recording folder, where the sidecar files live (a stills*/ folder belongs to the folder above it)
        folder = self.path if self.path.is_dir() else self.path.parent
        if folder.name.lower().startswith("stills") and (folder.parent / "stills.csv").is_file():
            folder = folder.parent
        vids = videos_in(folder)
        if self.video is not None:
            self.video = Path(self.video)
        elif self.path.is_file() and is_video(self.path):
            self.video = self.path
        elif self.path.is_file() and not explicit:
            raise ValueError(f"{self.path.name} is not a video file")
        elif vids:
            self.video = vids[0]
        self.folder = folder
        single = len(vids) <= 1 or self.video is None
        stem = self.video.stem if self.video is not None else ""
        self.notes = self.notes or read_notes(self._sidecar(single, [stem + ".txt"], "*.txt", "SKBR3.txt"))
        if self.frame_log is None:
            self.frame_log = read_frame_log(self._sidecar(single, [stem + ".csv", stem + "_frames.csv"], None, "frames.csv"))
        self.fps = 22.0
        if self.source == "images":
            self.images = self._explicit_images(explicit) if explicit else find_images(self.path if self.path.is_dir() else folder)[1]
            self.declared_frames = max(self.images) + 1
        elif self.video is None:
            raise FileNotFoundError(f"no video in {self.path}")
        if self.video is not None:
            cap = cv2.VideoCapture(str(self.video))
            try:
                if not cap.isOpened():
                    if self.source == "video":
                        raise ValueError(f"cannot open {self.video}")
                else:
                    if self.source == "video":
                        self.declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 22.0
            finally:
                cap.release()

    def _sidecar(self, generic_ok: bool, named: list, pattern: Optional[str], default: str) -> Path:
        """The file of this recording: one named after the video, else (when the folder holds a single video) the
        generic one. Returns a path that may not exist (readers treat that as 'none')."""
        for n in named:
            if n and (self.folder / n).is_file():
                return self.folder / n
        if generic_ok:
            if pattern:
                hit = next(iter(sorted(self.folder.glob(pattern))), None)
                if hit:
                    return hit
            return self.folder / default
        return self.folder / "\0none"

    def _explicit_images(self, files: list) -> "dict[int, Path]":
        """Index a hand-picked set of images: by the video frame each was taken at (``stills.csv``) when the table
        knows all of them, else by their order (natural sort of the names)."""
        for d in (files[0].parent, files[0].parent.parent):
            table = d / "stills.csv"
            if table.is_file() and table.stat().st_size:
                try:
                    t = pd.read_csv(table, usecols=["still_file", "video_frame"])
                except (ValueError, pd.errors.EmptyDataError):
                    continue
                frame_of = dict(zip(t.still_file, t.video_frame))
                if all(f.name in frame_of for f in files):
                    return {int(frame_of[f.name]) - 1: f for f in files}
        return dict(enumerate(sorted(files, key=natural_key)))

    @classmethod
    def from_spec(cls, spec, **kw) -> "Recording":
        """From what the window and the command line pass around: a path (folder or video file) or a list of
        image files."""
        if isinstance(spec, (list, tuple)):
            return cls(None, image_files=list(spec), **kw)
        return cls(spec, **kw)

    @property
    def name(self) -> str:
        if self.image_files:
            return f"{len(self.image_files)} images from {Path(self.image_files[0]).parent.name}"
        return self.video.name if (self.video is not None and self.path.is_file()) else self.path.name

    @property
    def cell_label(self) -> str:
        return self.notes.get("Cell label", "")

    def frames(self, every_n: int = 1, start: int = 0, stop: Optional[int] = None,
               select: Optional[Callable[[int], bool]] = None) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yield ``(index, time_s, gray frame)`` for every ``every_n``-th frame, or for the indices ``select`` accepts;
        stops silently at the first frame that cannot be decoded. Images: only the stills that exist are visited (a
        recording has a few frames without one), and unselected ones are never read."""
        if self.source == "images":
            for i in sorted(self.images):
                if i < start or (stop is not None and i >= stop):
                    continue
                if not (select(i) if select else (i - start) % every_n == 0):
                    continue
                f = cv2.imread(str(self.images[i]), cv2.IMREAD_COLOR)
                if f is None:
                    continue
                yield i, i / self.fps, to_gray(f, self.size)
            return
        cap = cv2.VideoCapture(str(self.video))
        try:
            if start:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            i = start
            while stop is None or i < stop:
                if not (select(i) if select else (i - start) % every_n == 0):
                    if not cap.grab():              # skipped frames are not colour-converted (3x faster)
                        return
                    i += 1
                    continue
                ok, f = cap.read()
                if not ok or f is None:
                    return
                yield i, i / self.fps, to_gray(f, self.size)
                i += 1
        finally:
            cap.release()

    def usable_frames(self) -> int:
        """Frames that really decode (a full pass; cheap relative to segmentation)."""
        return sum(1 for _ in self.frames())

    def complete(self) -> bool:
        return self.declared_frames > 0 and self.usable_frames() >= self.declared_frames - 1


class FrameReader:
    """Random access to a recording's frames for browsing: the original colour frames, no analysis.

    Video: sequential reads are cheap, short forward jumps skip with ``grab()``, long jumps and going back seek.
    Images: the still at an index, or the nearest earlier one when that frame has none. ``remove_text`` inpaints the
    burned-in overlay (the analysis always does; browsing shows the frame as recorded unless asked).
    """

    def __init__(self, rec: "Recording", remove_text: bool = False) -> None:
        self.rec, self.remove_text = rec, remove_text
        self._cap, self._pos, self._keys = None, 0, []
        if rec.source == "images":
            self._keys = sorted(rec.images)
            self.count = self._keys[-1] + 1
        else:
            self._cap = cv2.VideoCapture(str(rec.video))
            self.count = int(rec.declared_frames)

    def read(self, i: int) -> Optional[np.ndarray]:
        """BGR frame ``i`` (clamped into range), or None when it cannot be decoded."""
        i = min(max(int(i), 0), self.count - 1)
        if self.rec.source == "images":
            k = bisect.bisect_right(self._keys, i) - 1
            f = cv2.imread(str(self.rec.images[self._keys[max(k, 0)]]), cv2.IMREAD_COLOR)
        else:
            gap = i - self._pos
            if gap < 0 or gap > 60 or self._pos < 0:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            else:
                for _ in range(gap):
                    self._cap.grab()
            ok, f = self._cap.read()
            self._pos = i + 1 if ok else -1
            f = f if ok else None
        if f is None:
            return None
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        return remove_overlay(f) if self.remove_text else f

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
