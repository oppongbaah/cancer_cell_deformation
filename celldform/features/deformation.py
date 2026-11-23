"""
Deformation features of the trapped cell: per frame (from one mask) and per cell (from its trajectory over
the laser steps). Added 2026-10-22 to the 15 shape features of ``MorphologyExtractor``.

The dual-beam trap stretches the cell along the beam, which runs along the image x axis (on the hand-drawn
masks, 81 % of the elongated cells have their long axis within 30 deg of x). Lengths are therefore measured
in FIXED directions -- along x and along y -- not along the major/minor axes, which turn with the cell.

Per frame (``frame_features``):
  len_par, len_perp   extent along / across the beam, from second moments: L = 4 sqrt(mu / m00)
                      (the full length of an ellipse with that variance; robust to a few stray pixels)
  orientation         angle of the major axis to the beam, degrees in (-90, 90]
  volume              prolate spheroid about the beam: V = pi/6 * len_par * len_perp^2   (px^3)
  ellip_f2            radial profile r(phi) about the centre, Fourier harmonic 2, relative: 2|c2| / c0
  ellip_phase         direction of that elongation to the beam, degrees
  roughness           outline roughness: sqrt(sum_{n=3..12} (2|c_n|/c0)^2)

Per cell (``cell_summary``), from per-step values (median of the passing frames of each step):
  strain_par/perp     epsilon(s) = L(s) / L0 - 1, L0 = median over steps 0-9
  compliance_par/perp Theil-Sen slope of epsilon against the laser step (strain per step)
  poisson             -(Theil-Sen slope of epsilon_perp against epsilon_par)
  nonlinearity        quadratic term of epsilon_par(s), as strain added over 50 steps: b * 50^2
  onset_step          first step from which epsilon_par stays beyond 3 robust SD of its baseline (3 steps)
  volume_change       median volume over the last ten steps / baseline - 1
  ellip_change        ellip_f2: last ten steps minus baseline
  roughness_median    median roughness over the passing frames
  orient_spread       circular SD of the orientation (deg): how steady the stretch direction is
Within a step (needs several frames per step; ``within_step``):
  fluctuation         median over steps of SD(len_par) / mean(len_par) within the step
  creep               mean within-step profile of len_par (relative), last minus first frame
  creep_tau           retardation time tau (s) of a Kelvin-Voigt fit to that profile, A(1 - e^{-t/tau}) + c,
                      NaN when the fit is not significant. Without a force calibration only tau = eta/E is
                      accessible -- not E or eta themselves.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import cv2
import numpy as np
import pandas as pd

FRAME_FEATURES = ["len_par", "len_perp", "orientation", "volume", "ellip_f2", "ellip_phase", "roughness"]
CELL_FEATURES = ["len_par_base", "len_perp_base", "compliance_par", "compliance_perp", "poisson", "nonlinearity",
                 "onset_step", "volume_change", "ellip_change", "roughness_median", "orient_spread"]
STEP_FEATURES = ["fluctuation", "creep", "creep_tau"]
BASE_STEPS = 10          # baseline = steps 0..9
N_ANGLES = 256           # radial profile sampling


def frame_features(mask: Optional[np.ndarray]) -> Dict[str, float]:
    """The per-frame deformation features of one (selected) region mask."""
    out = {k: np.nan for k in FRAME_FEATURES}
    if mask is None or not mask.any():
        return out
    m = mask.astype(np.uint8)
    M = cv2.moments(m, binaryImage=True)
    if M["m00"] <= 0:
        return out
    mu20, mu02, mu11 = M["mu20"] / M["m00"], M["mu02"] / M["m00"], M["mu11"] / M["m00"]
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    lp, lq = 4 * np.sqrt(mu20), 4 * np.sqrt(mu02)
    out.update(len_par=float(lp), len_perp=float(lq),
               orientation=float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02))),
               volume=float(np.pi / 6 * lp * lq ** 2))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    if len(c) < 12:
        return out
    phi = np.arctan2(c[:, 1] - cy, c[:, 0] - cx)
    r = np.hypot(c[:, 0] - cx, c[:, 1] - cy)
    order = np.argsort(phi)
    phi, r = phi[order], r[order]
    grid = np.linspace(-np.pi, np.pi, N_ANGLES, endpoint=False)
    rr = np.interp(grid, phi, r, period=2 * np.pi)
    cn = np.fft.rfft(rr) / N_ANGLES
    c0 = np.abs(cn[0])
    if c0 <= 0:
        return out
    out.update(ellip_f2=float(2 * np.abs(cn[2]) / c0),
               ellip_phase=float(np.degrees(-np.angle(cn[2]) / 2)),
               roughness=float(np.sqrt(np.sum((2 * np.abs(cn[3:13]) / c0) ** 2))))
    return out


def theil_sen(x: Sequence[float], y: Sequence[float]) -> float:
    """Median of the pairwise slopes -- robust to a few stray points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    keep = dx != 0
    return float(np.median((y[j] - y[i])[keep] / dx[keep])) if keep.any() else np.nan


def _circ_sd_deg(angles_deg: np.ndarray) -> float:
    """Circular SD of an AXIS (period 180 deg), in degrees."""
    a = np.radians(np.asarray(angles_deg, float)[np.isfinite(angles_deg)]) * 2
    if len(a) < 2:
        return np.nan
    R = np.hypot(np.cos(a).mean(), np.sin(a).mean())
    return float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9))) / 2))


def per_step(frames: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """One value per laser step: the median of the step's passing frames."""
    f = frames[frames.passed.astype(bool)]
    return f.groupby("laser_step")[list(cols)].median()


def cell_summary(frames: pd.DataFrame) -> Dict[str, float]:
    """The per-cell deformation features from one cell's frames (columns: laser_step, passed, FRAME_FEATURES)."""
    s = per_step(frames, ["len_par", "len_perp", "volume", "ellip_f2", "roughness"])
    out = {k: np.nan for k in CELL_FEATURES}
    base = s[s.index < BASE_STEPS]
    late = s[s.index >= s.index.max() - (BASE_STEPS - 1)] if len(s) else s
    if len(base) < 3 or len(s) < 10:
        return out
    L0, Q0, V0 = base.len_par.median(), base.len_perp.median(), base.volume.median()
    ep, eq = s.len_par / L0 - 1, s.len_perp / Q0 - 1
    steps = s.index.values.astype(float)
    out.update(len_par_base=float(L0), len_perp_base=float(Q0),
               compliance_par=theil_sen(steps, ep), compliance_perp=theil_sen(steps, eq),
               volume_change=float(late.volume.median() / V0 - 1),
               ellip_change=float(late.ellip_f2.median() - base.ellip_f2.median()),
               roughness_median=float(frames.loc[frames.passed.astype(bool), "roughness"].median()),
               orient_spread=_circ_sd_deg(frames.loc[frames.passed.astype(bool), "orientation"].values))
    sl = theil_sen(ep.values, eq.values)
    out["poisson"] = -sl if np.isfinite(sl) else np.nan
    if len(steps) >= 5:
        b = np.polyfit(steps, ep.values, 2)[0]
        out["nonlinearity"] = float(b * 50 ** 2)
    noise = 1.4826 * np.median(np.abs(ep[base.index] - ep[base.index].median()))
    if noise > 0:
        beyond = (ep - ep[base.index].median()).abs() > 3 * noise
        run = beyond.astype(int).rolling(3).sum() >= 3          # three consecutive steps
        hit = run[run].index
        out["onset_step"] = float(hit[0] - 2) if len(hit) else np.nan
    return out


def within_step(frames: pd.DataFrame, fps: float = 22.0, frames_per_step: int = 200) -> Dict[str, float]:
    """Within-step behaviour from several frames per step (columns: frame_index, laser_step, passed, len_par)."""
    out = {k: np.nan for k in STEP_FEATURES}
    f = frames[frames.passed.astype(bool)].copy()
    if f.empty:
        return out
    f["offset"] = f.frame_index - f.laser_step * frames_per_step
    g = f.groupby("laser_step")
    full = [k for k, x in g if len(x) >= 4]
    if len(full) < 5:
        return out
    f = f[f.laser_step.isin(full)]
    f["rel"] = f.len_par / f.groupby("laser_step").len_par.transform("mean") - 1
    out["fluctuation"] = float(f.groupby("laser_step").len_par.agg(lambda x: x.std() / x.mean()).median())
    prof = f.groupby("offset").rel.mean()
    prof = prof[f.groupby("offset").size() >= 5]
    if len(prof) < 4:
        return out
    out["creep"] = float(prof.iloc[-1] - prof.iloc[0])
    t = prof.index.values / fps
    try:
        from scipy.optimize import curve_fit
        model = lambda t_, A, tau, c: c + A * (1 - np.exp(-t_ / tau))   # noqa: E731
        p, cov = curve_fit(model, t, prof.values, p0=(prof.iloc[-1] - prof.iloc[0], 2.0, prof.iloc[0]),
                           bounds=([-1, 0.05, -1], [1, 60, 1]), maxfev=5000)
        se = np.sqrt(np.diag(cov))
        if np.isfinite(se[0]) and abs(p[0]) > 2 * se[0]:
            out["creep_tau"] = float(p[1])
    except Exception:
        pass
    return out
