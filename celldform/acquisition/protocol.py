"""
The laser protocol of the recordings: which drive current and optical power a video frame was taken at.

The recordings do not log the laser settings frame by frame. They follow a fixed protocol (Arkorful 2025,
Sec. 3.2.3; confirmed by Y. Sefa-Boateng, who recorded them, 2026-11-05): both lasers start at 500 mA and the
current rises by 50 mA every 30 s up to 1200 mA -- 15 power levels, 450 s. Each frame's level, current and
power are therefore *nominal* values computed from its time in the video:

    t      = frame_index / fps
    level  = floor(t / hold_s), 0 .. n_levels-1
    I      = start_mA + level * step_mA                      (each laser, mA)
    P      = calibration curve at I                          (each beam at the fiber output, mW)

Caveats (state them wherever the values are reported):
- the video is assumed to start when the lasers reach the first level; the 15 videos last 448-471 s against
  a nominal 450 s, so a frame's level can be off near level boundaries (by < 1 level);
- frames after the last nominal level (t >= n_levels * hold_s) keep the last level and are flagged
  `beyond_ramp`;
- the power is the calibrated output of each fiber, not the power reaching the cell.

    from celldform.acquisition.protocol import LaserProtocol, add_laser_columns
    frames = add_laser_columns(frames, LaserProtocol())     # adds time_s, power_level, current_mA, ...
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LASER_COLUMNS = ("time_s", "power_level", "current_mA", "power_mW", "beyond_ramp")


@dataclass(frozen=True)
class LaserProtocol:
    fps: float = 22.0                       # camera frame rate
    hold_s: float = 30.0                    # time at each power level
    start_mA: float = 500.0                 # drive current of the first level (each laser)
    step_mA: float = 50.0                   # increase per level
    n_levels: int = 15                      # 500 -> 1200 mA
    # power of each beam at the fiber output against drive current, interpolated linearly between the points
    # (Arkorful 2025: ~270 mW at 500 mA, ~685 mW at 1200 mA, room temperature; slope 0.59 mW/mA, as
    # Sefa-Boateng 2025 measured, 0.58-0.59 mW/mA). Replace with the full calibration table when available.
    calibration_mA: tuple = (500.0, 1200.0)
    calibration_mW: tuple = (270.0, 685.0)

    @property
    def frames_per_level(self) -> float:
        return self.fps * self.hold_s

    @property
    def currents_mA(self) -> np.ndarray:
        return self.start_mA + self.step_mA * np.arange(self.n_levels)

    def power_mW(self, current_mA) -> np.ndarray:
        """Nominal power of each beam at the given drive current(s), from the calibration curve."""
        xs, ys = np.asarray(self.calibration_mA, float), np.asarray(self.calibration_mW, float)
        if len(xs) != len(ys) or len(xs) < 2 or np.any(np.diff(xs) <= 0):
            raise ValueError("calibration_mA must increase and match calibration_mW (at least two points)")
        i = np.asarray(current_mA, float)
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])            # linear beyond the last point, like within
        lo_slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return np.where(i > xs[-1], ys[-1] + slope * (i - xs[-1]),
                        np.where(i < xs[0], ys[0] + lo_slope * (i - xs[0]), np.interp(i, xs, ys)))


def laser_settings(frame_index, protocol: LaserProtocol | None = None) -> pd.DataFrame:
    """Nominal laser settings of each frame: one row per frame index, columns LASER_COLUMNS."""
    p = protocol or LaserProtocol()
    idx = np.asarray(frame_index, float)
    if np.any(idx < 0):
        raise ValueError("frame indices must be >= 0")
    t = idx / p.fps
    raw = np.floor(t / p.hold_s).astype(int)
    level = np.minimum(raw, p.n_levels - 1)
    current = p.start_mA + p.step_mA * level
    return pd.DataFrame({"time_s": t, "power_level": level, "current_mA": current,
                         "power_mW": p.power_mW(current), "beyond_ramp": raw >= p.n_levels})


def add_laser_columns(frames: pd.DataFrame, protocol: LaserProtocol | None = None,
                      index_col: str = "frame_index") -> pd.DataFrame:
    """A copy of `frames` with the nominal laser settings of each row's frame (columns LASER_COLUMNS),
    placed right after `index_col`. Existing laser columns are replaced."""
    out = frames.drop(columns=[c for c in LASER_COLUMNS if c in frames.columns])
    settings = laser_settings(out[index_col].to_numpy(), protocol)
    pos = out.columns.get_loc(index_col) + 1
    for k, c in enumerate(LASER_COLUMNS):
        out.insert(pos + k, c, settings[c].to_numpy())
    return out
