"""
One frame -> the trapped cell, its checks and its features, working CAUSALLY (only frames already seen).

The offline feature step (scripts/features/extract_features.py) knows the whole video: the trap is the median cell
position over every frame, the "typical" cell area comes from the neighbouring laser steps on both sides. A demo
sees frames one at a time, so here:

    trap position   median centroid of the cell regions seen so far (the largest region until ``min_trap_frames``
                    centroids exist); the trapped cell is the region nearest it
    typical area    median area of the last ``area_memory`` SOUND frames (near the trap, not merged, not cut by
                    the frame edge); unknown (check skipped) until two exist
    gates           the same five frame checks and limits (celldform.features.quality.Gates)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from celldform.demo.segment import Seg
from celldform.features.deformation import frame_features
from celldform.features.extractor import MorphologyExtractor
from celldform.features.quality import (Gates, apply_gates, regions, segmentation_checks, select_trapped_region,
                                        trap_position)


@dataclass
class FrameResult:
    row: Dict[str, object]                    # one table row: checks, passed, reasons, shape + deformation features
    region: Optional[np.ndarray]              # bool mask of the trapped cell (None if no cell found)

    @property
    def passed(self) -> bool:
        return bool(self.row["passed"])


class FrameAnalyzer:
    def __init__(self, gates: Optional[Gates] = None, extractor: Optional[MorphologyExtractor] = None,
                 min_trap_frames: int = 3, area_memory: int = 5, min_area_px: int = 50,
                 trap_xy: Optional[Tuple[float, float]] = None) -> None:
        self.gates = gates or Gates()
        self.ex = extractor or MorphologyExtractor(min_area_px=min_area_px)
        self.min_trap_frames, self.area_memory, self.min_area_px = min_trap_frames, area_memory, min_area_px
        self.fixed_trap = trap_xy                            # set by the user (a click on the cell): never learned over
        self.centroids: List[Tuple[float, float]] = []
        self.areas: Deque[float] = deque(maxlen=area_memory)

    @property
    def trap(self) -> Optional[Tuple[float, float]]:
        return self.fixed_trap or trap_position(self.centroids)

    def typical_area(self) -> Optional[float]:
        return float(np.median(self.areas)) if len(self.areas) >= 2 else None

    def analyze(self, seg: Seg, frame_index: int = 0, laser_step: Optional[int] = None) -> FrameResult:
        cell = seg.labels == 1
        rs = regions(cell, self.min_area_px)
        trap = self.trap if (self.fixed_trap or len(self.centroids) >= self.min_trap_frames) else None
        if rs and trap is None:                              # not enough history: the biggest region stands in
            big = max(rs, key=lambda r: r["area"])
            trap_for_pick = big["centroid"]
        else:
            trap_for_pick = trap
        region, n = select_trapped_region(cell, trap_for_pick, self.min_area_px) if trap_for_pick else (None, 0)
        if region is not None:
            self.centroids.append(region["centroid"])        # the trap estimate learns from every selected region
            if trap is not None:
                region["trap_dist"] = float(np.hypot(region["centroid"][0] - trap[0], region["centroid"][1] - trap[1]))
        checks = segmentation_checks(region, seg.labels, seg.maxprob, self.typical_area(), n)
        ok, why = apply_gates(checks, self.gates)
        if (region is not None and checks["trap_dist"] <= self.gates.max_trap_dist
                and checks["solidity"] >= self.gates.min_solidity and not checks["touches_border"]):
            self.areas.append(region["area"])                # a sound frame teaches the typical area
        row: Dict[str, object] = {"frame_index": frame_index, "laser_step": laser_step,
                                  "trap_x": trap[0] if trap else np.nan, "trap_y": trap[1] if trap else np.nan,
                                  **checks, "passed": ok, "reasons": "; ".join(why)}
        mask = region["mask"] if region is not None else None
        row.update(self.ex.extract(mask.astype(np.uint8)) if mask is not None
                   else {k: np.nan for k in self.ex._FEATURE_NAMES})
        row.update(frame_features(mask))
        return FrameResult(row, mask)
