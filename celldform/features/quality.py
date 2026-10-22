"""
Segmentation checks and image-quality measurements for feature extraction.

Two jobs, both run before a frame's shape features are trusted:

1. **Which region is the trapped cell, and is its mask believable?**
   ``select_trapped_region`` keeps the predicted cell region nearest the video's trap position
   (not the largest -- in high-10 the untrapped neighbour is larger than the trapped cell), and
   ``segmentation_checks`` measures what can go wrong with it: no region near the trap, a region
   cut by the frame edge, a merged or ragged outline (solidity), an area far from that cell's
   typical area, a boundary the model was unsure of, a neighbour touching it.
   ``apply_gates`` turns the measurements into pass/fail with the reasons.

2. **What did the camera see?** ``image_quality`` measures focus, illumination, contrast and noise
   on the raw frame, so differences between recording sessions can be checked rather than assumed.

Everything here is pure numpy/OpenCV on one frame; the per-video steps (trap position, typical
area) are computed by the caller across a cell's frames.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── which region is the trapped cell ─────────────────────────────────────────

def regions(cell_mask: np.ndarray, min_area_px: int = 50) -> List[dict]:
    """Connected regions of a binary cell mask, >= min_area_px, with area and centroid (x, y)."""
    n, lab, stats, cents = cv2.connectedComponentsWithStats(cell_mask.astype(np.uint8), connectivity=8)
    return [{"label": i, "area": int(stats[i, cv2.CC_STAT_AREA]), "centroid": (float(cents[i][0]), float(cents[i][1])),
             "mask": lab == i}
            for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area_px]


def trap_position(centroids: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """A video's trap position: the median of its per-frame cell centroids (robust to a few wrong frames)."""
    if not centroids:
        return None
    a = np.asarray(centroids, dtype=float)
    return float(np.median(a[:, 0])), float(np.median(a[:, 1]))


def select_trapped_region(cell_mask: np.ndarray, trap_xy: Tuple[float, float],
                          min_area_px: int = 50) -> Tuple[Optional[dict], int]:
    """The region nearest the trap, and how many candidate regions there were."""
    rs = regions(cell_mask, min_area_px)
    if not rs:
        return None, 0
    d = [np.hypot(r["centroid"][0] - trap_xy[0], r["centroid"][1] - trap_xy[1]) for r in rs]
    best = rs[int(np.argmin(d))]
    best["trap_dist"] = float(min(d))
    return best, len(rs)


# ── is the mask believable ───────────────────────────────────────────────────

def segmentation_checks(region: Optional[dict], labels: np.ndarray, maxprob: Optional[np.ndarray],
                        typical_area: Optional[float], n_regions: int, band_px: int = 3) -> Dict[str, float]:
    """Measurements on the selected region. ``labels`` is the full prediction (0 bg, 1 cell, 2 decoy);
    ``maxprob`` the model's per-pixel top-class probability in [0, 1] (None if not saved)."""
    nan = float("nan")
    out = {"found": region is not None, "n_regions": n_regions, "trap_dist": nan, "area": nan,
           "area_ratio": nan, "solidity": nan, "touches_border": False, "touches_other": False,
           "boundary_conf": nan, "uncertain_frac": nan}
    if region is None:
        return out
    m = region["mask"]
    area = float(m.sum())
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    grown = cv2.dilate(m.astype(np.uint8), k) > 0
    band = grown & ~(cv2.erode(m.astype(np.uint8), k) > 0)        # the boundary, both sides
    ring = grown & ~m                                              # just outside the region
    out.update({
        "trap_dist": region.get("trap_dist", nan),
        "area": area,
        "area_ratio": area / typical_area if typical_area else nan,
        "solidity": area / hull if hull > 0 else nan,
        "touches_border": bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()),
        # another predicted object (a second cell region or a decoy) right against this one
        "touches_other": bool((ring & (labels > 0)).any()),
    })
    if maxprob is not None:
        out["boundary_conf"] = float(maxprob[band].mean())
        out["uncertain_frac"] = float((maxprob[grown] < 0.9).mean())
    return out


@dataclass
class Gates:
    """Pass/fail limits for one frame's segmentation. Set 2026-10-21 at about the 1st percentile of
    the GOOD masks (Dice >= 0.85) among the annotated test frames, so a good mask is rarely rejected;
    `extract_features.py calibrate` re-measures this for any model."""
    max_trap_dist: float = 45.0         # px from the trap position (good masks: 99th pct 43)
    min_solidity: float = 0.95          # a round cell is near 1; merges fall below (good: 1st pct 0.955)
    area_ratio_range: Tuple[float, float] = (0.85, 1.20)   # vs the median of the neighbouring laser steps
    min_boundary_conf: float = 0.95     # mean top-class probability along the outline (good: 1st pct 0.952)
    allow_border: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def apply_gates(c: Dict[str, float], g: Gates) -> Tuple[bool, List[str]]:
    """(passed, reasons it failed). A NaN measurement never fails a gate (e.g. no probabilities saved)."""
    if not c["found"]:
        return False, ["no cell near the trap"]
    why = []
    if c["trap_dist"] > g.max_trap_dist:
        why.append(f"{c['trap_dist']:.0f} px from the trap")
    if not g.allow_border and c["touches_border"]:
        why.append("cut by the frame edge")
    if c["solidity"] < g.min_solidity:
        why.append(f"solidity {c['solidity']:.2f}")
    lo, hi = g.area_ratio_range
    if not (lo <= c["area_ratio"] <= hi) and not np.isnan(c["area_ratio"]):
        why.append(f"area x{c['area_ratio']:.2f} of typical")
    if c["boundary_conf"] < g.min_boundary_conf:
        why.append(f"boundary confidence {c['boundary_conf']:.2f}")
    return not why, why


# ── what did the camera see ──────────────────────────────────────────────────

def image_quality(raw: np.ndarray, cell: Optional[np.ndarray] = None, pad_px: int = 25) -> Dict[str, float]:
    """Focus, illumination, contrast and noise of one raw grayscale frame.

    focus_frame       variance of the Laplacian over the whole frame (higher = sharper)
    focus_cell        the same in a window around the cell (needs ``cell``)
    illum_mean        mean background brightness (cell excluded when given)
    illum_gradient    brightness spread of the heavily blurred frame (5th-95th pct): uneven lighting
    contrast_rms      RMS contrast of the frame (std / mean)
    contrast_cell     |mean cell - mean surrounding ring| / ring mean (needs ``cell``)
    noise_sigma       noise level from the median absolute deviation of a high-pass residual
    """
    f = raw.astype(np.float32)
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=3)
    bg = np.ones(f.shape, bool) if cell is None else ~(cv2.dilate(cell.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    blur = cv2.GaussianBlur(f, (0, 0), 40)
    resid = f - cv2.GaussianBlur(f, (0, 0), 1.0)
    out = {"focus_frame": float(lap.var()),
           "illum_mean": float(f[bg].mean()),
           "illum_gradient": float(np.percentile(blur, 95) - np.percentile(blur, 5)),
           "contrast_rms": float(f.std() / (f.mean() + 1e-6)),
           "noise_sigma": float(1.4826 * np.median(np.abs(resid - np.median(resid)))),
           "focus_cell": float("nan"), "contrast_cell": float("nan")}
    if cell is not None and cell.any():
        ys, xs = np.nonzero(cell)
        y0, y1 = max(ys.min() - pad_px, 0), min(ys.max() + pad_px + 1, f.shape[0])
        x0, x1 = max(xs.min() - pad_px, 0), min(xs.max() + pad_px + 1, f.shape[1])
        out["focus_cell"] = float(lap[y0:y1, x0:x1].var())
        ring = (cv2.dilate(cell.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0) & ~cell
        rm = f[ring].mean()
        out["contrast_cell"] = float(abs(f[cell].mean() - rm) / (rm + 1e-6))
    return out
