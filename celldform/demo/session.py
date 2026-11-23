"""
One cell's recording as it arrives: frames in, laser steps accumulated, a verdict only when there is enough.

Sampling follows the feature step (configs/analysis/features.yaml): in every ``frames_per_step`` = 200 frames the step
frame (offset 0) and the ``dense_offsets`` are measured. Frames are assigned to laser steps by ``frame_index // 200``.

The verdict is withheld, with the reason, when the recording cannot support one:
    - fewer than MIN_STEPS laser steps with a passing frame (the per-cell features need 10; baseline = first 10 steps),
    - fewer than ``min_pass_fraction`` (0.8) of the sampled frames pass the frame checks (the cell is then not
      "usable" by the thesis' own rule).
Otherwise ``p_high`` is the framework's equal-weight mean, called high at >= 0.5. ``caveats`` lists what the framework
was NOT validated for (e.g. a laser protocol other than 500->1200 mA / 30 s steps, or a segmenter other than the one
the classifier's features came from).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from celldform.analysis_config import load as load_config
from celldform.demo.analyze import FrameAnalyzer, FrameResult
from celldform.demo.framework import MIN_STEPS, SELECTED, STEP_FEATURES, HER2Framework
from celldform.demo.segment import Seg
from celldform.features.deformation import FRAME_FEATURES, cell_summary, within_step
from celldform.features.quality import Gates


@dataclass
class Verdict:
    status: str                         # "ok" (validated) | "provisional" (computed, NOT validated) | "insufficient"
    reason: str = ""                    # why there is no verdict, or why a provisional one is not validated
    label: Optional[str] = None         # "high" | "low" HER2 (None when insufficient)
    p_high: float = float("nan")
    p_step: float = float("nan")
    p_cell: float = float("nan")
    n_steps: int = 0
    pass_fraction: float = float("nan")
    caveats: List[str] = field(default_factory=list)

    @property
    def validated(self) -> bool:
        return self.status == "ok"

    def headline(self) -> str:
        """One line that cannot be mistaken: a provisional verdict always says so."""
        if self.status == "ok":
            return f"HER2 {self.label} (P high {self.p_high:.2f})"
        if self.status == "provisional":
            return f"PROVISIONAL, NOT VALIDATED: HER2 {self.label} (P high {self.p_high:.2f})"
        return "no verdict - " + self.reason


class CellSession:
    def __init__(self, framework: Optional[HER2Framework] = None, fps: float = 22.0,
                 caveats: Optional[List[str]] = None, trap_xy=None, provisional: bool = False) -> None:
        cfg = load_config("features")
        self.frames_per_step = cfg.frames_per_step
        self.offsets = (0, *cfg.dense_offsets)
        self.min_pass_fraction = cfg.cell_validation.min_pass_fraction
        fc = cfg.frame_checks
        self.analyzer = FrameAnalyzer(Gates(max_trap_dist=fc.max_trap_dist, min_solidity=fc.min_solidity,
                                            area_ratio_range=tuple(fc.area_ratio_range),
                                            min_boundary_conf=fc.min_boundary_conf, allow_border=fc.allow_border),
                                       trap_xy=trap_xy)
        self.framework, self.fps, self.caveats = framework, fps, list(caveats or [])
        self.provisional = provisional      # also compute a verdict when the checks fail; it is labelled NOT VALIDATED
        self.rows: List[dict] = []
        self.crops: Dict[int, np.ndarray] = {}            # laser step -> picture of the cell (first passing frame)

    # -- sampling ----------------------------------------------------------------------------------------------
    def wants(self, frame_index: int) -> bool:
        """Is this frame one of the measured ones."""
        return frame_index % self.frames_per_step in self.offsets

    def step_of(self, frame_index: int) -> int:
        return frame_index // self.frames_per_step

    # -- intake ------------------------------------------------------------------------------------------------
    def add(self, seg: Seg, frame_index: int, gray: Optional[np.ndarray] = None) -> FrameResult:
        """Measure one frame. With ``gray``, the first passing frame of each laser step also leaves a picture."""
        step = self.step_of(frame_index)
        r = self.analyzer.analyze(seg, frame_index, step)
        row = dict(r.row, time_s=frame_index / self.fps)
        self.rows.append(row)
        if gray is not None and r.passed and step not in self.crops:
            from celldform.demo.dashboard import make_crop
            c = make_crop(gray, r.region)
            if c is not None:
                self.crops[step] = c
        return r

    @property
    def frames(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    # -- per-step table and verdict ----------------------------------------------------------------------------
    def steps(self) -> pd.DataFrame:
        f = self.frames
        if f.empty:
            return pd.DataFrame(columns=["laser_step", *STEP_FEATURES])
        ok = f[f.passed.astype(bool)]
        return ok.groupby("laser_step")[STEP_FEATURES].median().reset_index()

    def cell_features(self) -> Dict[str, float]:
        f = self.frames
        if f.empty:
            return {}
        out = cell_summary(f)
        out.update(within_step(f, self.fps, self.frames_per_step))
        return out

    def verdict(self) -> Verdict:
        f = self.frames
        if f.empty:
            return Verdict("insufficient", "no frames measured yet")
        pf = float(f.passed.astype(bool).mean())
        st = self.steps()
        v = Verdict("insufficient", n_steps=len(st), pass_fraction=pf, caveats=list(self.caveats))
        fails = []
        if len(st) < MIN_STEPS:
            fails.append(f"{len(st)} laser steps measured; the classifier needs {MIN_STEPS} "
                         f"({MIN_STEPS * self.frames_per_step} frames = {MIN_STEPS * self.frames_per_step / self.fps:.0f} s)")
        if pf < self.min_pass_fraction:
            fails.append(f"only {pf:.0%} of measured frames pass the checks (need {self.min_pass_fraction:.0%})")
        if self.framework is None:
            v.reason = "no classifier loaded"
        elif fails and not (self.provisional and len(st) >= 1):
            v.reason = fails[0]
        else:
            cf = self.cell_features()
            p = self.framework.p_high(st, cf)
            if not np.isfinite(p["p_high"]):
                v.reason = "a per-cell feature could not be computed"
            else:
                v.status, v.p_high, v.p_step, v.p_cell = "ok", p["p_high"], p["p_step"], p["p_cell"]
                v.label = "high" if p["p_high"] >= 0.5 else "low"
                if fails:                                    # computed anyway, on request: say what is missing
                    missing = [k for k in SELECTED if not np.isfinite(cf.get(k, np.nan))]
                    if missing:
                        fails.append(f"{len(missing)} of {len(SELECTED)} cell-level measures could not be computed and were "
                                     f"replaced by the training cells' median ({', '.join(missing)})")
                    v.status = "provisional"
                    v.reason = "; ".join(fails) + ". The classifier was validated only on cells that pass these checks"
        return v
