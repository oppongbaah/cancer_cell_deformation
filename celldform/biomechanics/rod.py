"""
Rate of deformation (RoD) of one trapped cell: the slope of its strain against the laser step.

    strain(s) = 100 * (x(s) / baseline - 1)        x = the median length of step s's passing frames
    RoD       = d strain / d step                  [% per laser step]; / STEP_SECONDS for % per second

The interval uses heteroscedasticity- and autocorrelation-robust (Newey-West) standard errors: consecutive steps of
one cell are not independent, and a plain OLS interval would be too narrow.

For a cell watched for only a few steps, `shrink` combines its own estimate with what earlier cells say about
typical RoDs (precision-weighted, empirical Bayes): the fewer steps, the more it leans on the prior.

The saved RoD model (rate_of_deformation.py -> checkpoints/biomechanics/<model>/rod_model.json) holds the method's
settings and the priors from all usable cells; `estimate(frames, load_model(path), her2)` gives a new cell's RoD.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE_STEPS = 10                       # baseline = median over steps 0..9 (as celldform.features.deformation)
FRAMES_PER_STEP, FPS = 200, 22.0
STEP_SECONDS = FRAMES_PER_STEP / FPS  # ~9.1 s per laser step


@dataclass
class Rod:
    slope: float                      # % strain per laser step
    se: float                         # robust standard error of the slope
    n_steps: int

    @property
    def ci(self) -> tuple[float, float]:
        return self.slope - 1.96 * self.se, self.slope + 1.96 * self.se

    @property
    def per_second(self) -> float:
        return self.slope / STEP_SECONDS


def step_strain(frames: pd.DataFrame, measure: str = "len_par", base_steps: int = BASE_STEPS) -> pd.Series:
    """One cell's strain (%) per laser step: the median of each step's passing frames against the baseline
    (median over the first `base_steps` steps). Index: laser_step."""
    f = frames[frames.passed.astype(bool)]
    s = f.groupby("laser_step")[measure].median().sort_index()
    base = s[s.index < base_steps].median() if (s.index < base_steps).any() else s.iloc[:3].median()
    return (s / base - 1) * 100


def rate(strain: pd.Series, lags: int = 3) -> Rod:
    """RoD of one cell: OLS slope of strain on laser step, Newey-West (HAC) standard error."""
    import statsmodels.api as sm
    y = strain.dropna()
    if len(y) < 4:
        return Rod(np.nan, np.nan, len(y))
    X = sm.add_constant(y.index.to_numpy(float))
    m = sm.OLS(y.to_numpy(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": min(lags, len(y) // 3)})
    return Rod(float(m.params[1]), float(m.bse[1]), len(y))


def prior(rods: list[Rod]) -> tuple[float, float]:
    """Mean and between-cell SD of the true RoD across earlier cells (method of moments: the spread of their
    estimates minus their own estimation noise)."""
    slopes = np.array([r.slope for r in rods]); se2 = np.array([r.se ** 2 for r in rods])
    tau2 = max(float(np.var(slopes, ddof=1) - np.mean(se2)), 0.0)
    return float(np.mean(slopes)), float(np.sqrt(tau2))


def shrink(own: Rod, prior_mean: float, prior_sd: float) -> Rod:
    """Precision-weighted combination of a cell's own RoD estimate with the prior from earlier cells."""
    if not np.isfinite(own.slope) or not np.isfinite(own.se) or own.se == 0:
        return Rod(prior_mean, prior_sd, own.n_steps)
    if prior_sd == 0:
        return Rod(prior_mean, 0.0, own.n_steps)
    w_own, w_prior = 1 / own.se ** 2, 1 / prior_sd ** 2
    return Rod((w_own * own.slope + w_prior * prior_mean) / (w_own + w_prior), float(np.sqrt(1 / (w_own + w_prior))),
               own.n_steps)


def load_model(path) -> dict:
    """The saved RoD model: settings and priors (pooled and per HER2 class) for each measure."""
    import json
    from pathlib import Path
    return json.loads(Path(path).read_text())


def estimate(frames: pd.DataFrame, model: dict, her2: str | None = None, measure: str | None = None) -> dict:
    """A new cell's RoD from its frames so far (columns laser_step, passed, the measure): its own estimate, and
    shrunk toward the prior of its HER2 class (from a classifier) or, without a class, the pooled prior."""
    m = measure or model["measures"][0]
    st = step_strain(frames, m, base_steps=model["base_steps"])
    own = rate(st, lags=model["hac_lags"])
    pr = model["priors"][m][her2 if her2 in ("high", "low") else "pooled"]
    sh = shrink(own, pr["mean"], pr["sd"])
    step_s = model["step_seconds"]
    return {"measure": m, "n_steps": own.n_steps, "rod": own.slope, "ci": own.ci, "rod_per_s": own.slope / step_s,
            "shrunk": sh.slope, "shrunk_ci": sh.ci, "prior": "class " + her2 if her2 in ("high", "low") else "pooled"}
