"""Rate of deformation (celldform.biomechanics.rod)."""
import numpy as np
import pandas as pd

from celldform.biomechanics.rod import Rod, prior, rate, shrink, step_strain


def _frames(slope_pct: float, steps: int = 50, per_step: int = 6, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = np.repeat(np.arange(steps), per_step)
    length = 36.0 * (1 + slope_pct / 100 * s) * (1 + noise * rng.standard_normal(len(s)))
    return pd.DataFrame({"laser_step": s, "len_par": length, "passed": True})


def test_step_strain_is_zero_at_the_baseline():
    st = step_strain(_frames(0.0))
    assert np.allclose(st.values, 0.0)


def test_rate_recovers_a_linear_slope():
    r = rate(step_strain(_frames(0.2, noise=0.003)))
    assert abs(r.slope - 0.2 / (1 + 0.2 / 100 * 4.5)) < 0.02          # strain is relative to the steps 0-9 median
    lo, hi = r.ci
    assert lo < r.slope < hi and r.n_steps == 50


def test_failed_frames_are_ignored():
    f = _frames(0.1)
    f.loc[f.laser_step == 30, "len_par"] = 1e6
    f.loc[f.laser_step == 30, "passed"] = False
    assert 29 not in step_strain(f).index or step_strain(f).max() < 10


def test_shrink_moves_towards_the_prior_by_precision():
    own = Rod(0.4, 0.2, 5)
    s = shrink(own, 0.0, 0.2)
    assert abs(s.slope - 0.2) < 1e-9 and s.se < own.se
    assert shrink(Rod(0.4, 0.01, 50), 0.0, 0.2).slope > 0.39        # a precise estimate barely moves


def test_prior_subtracts_estimation_noise():
    rods = [Rod(v, 0.0, 50) for v in (-0.2, 0.0, 0.2)]
    m, sd = prior(rods)
    assert abs(m) < 1e-12 and abs(sd - 0.2) < 1e-9
    assert prior([Rod(v, 1.0, 5) for v in (-0.2, 0.0, 0.2)])[1] == 0.0   # spread smaller than the noise
