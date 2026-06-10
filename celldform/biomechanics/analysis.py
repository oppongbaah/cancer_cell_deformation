"""
Biomechanical analysis — deformation rate, optical parameter correlation.

Key computations
----------------

1. Deformation rate (RoD)
   Computed as the finite-difference temporal derivative of a chosen
   deformation metric (default: deformation index D or aspect ratio AR):

       RoD = ΔMetric / Δt   [units: 1/s]

   A linear fit over the full time-series gives the mean deformation rate,
   while a sliding-window version tracks non-linear transient dynamics.

2. Optical parameter correlation
   For each unique experimental condition (laser power P, trap current I),
   the mean RoD and peak deformation are computed.  scipy.stats.pearsonr /
   spearmanr quantify the monotonic relationship between optical parameters
   and deformation outcome, supporting the thesis research question:
   "How does varying optical power and electrical drive current influence
   the deformation dynamics of trapped cancer cells?"

3. Kelvin–Voigt viscoelastic fitting  (placeholder)
   The creep response ε(t) is fitted to the Kelvin–Voigt model:
       σ(t) = E·ε(t) + η·dε/dt   (Eq. 2)
   Young's modulus E and viscosity η are estimated via scipy.optimize.curve_fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ExperimentCondition:
    """Metadata for a single optical trapping experiment run."""

    laser_power_mW: float
    trap_current_mA: float
    label: str = ""  # optional free-text description (e.g., "SKBR3_100mW")


class DeformationAnalyzer:
    """Quantitative biomechanical analysis of cell deformation time-series.

    Parameters
    ----------
    condition:
        The optical parameters used in this experiment.  Can be set later
        via :meth:`set_condition`.
    metric:
        Which feature column to use as the primary deformation signal.
        ``"deformation_index"`` (Eq. 4) or ``"aspect_ratio"`` (Eq. 6).
    window_s:
        Sliding-window width in seconds for local RoD estimation.
        Pass *None* to skip sliding-window computation.
    """

    def __init__(
        self,
        laser_power_mW: float = 0.0,
        trap_current_mA: float = 0.0,
        metric: str = "deformation_index",
        window_s: Optional[float] = None,
    ) -> None:
        self.condition = ExperimentCondition(
            laser_power_mW=laser_power_mW,
            trap_current_mA=trap_current_mA,
        )
        self.metric = metric
        self.window_s = window_s

    def set_condition(self, laser_power_mW: float, trap_current_mA: float) -> None:
        self.condition.laser_power_mW = laser_power_mW
        self.condition.trap_current_mA = trap_current_mA

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deformation_rate(
        self,
        metric_values: np.ndarray,
        timestamps: np.ndarray,
    ) -> float:
        """Estimate mean deformation rate as a linear slope over time.

        Parameters
        ----------
        metric_values:
            1-D array of deformation metric values (e.g. deformation index).
        timestamps:
            1-D array of corresponding timestamps in seconds.

        Returns
        -------
        Mean deformation rate in metric-units / second (slope of linear fit).
        """
        if len(metric_values) < 2:
            return float("nan")

        # Linear fit: metric ~ slope * t + intercept
        slope, _, _, _, _ = stats.linregress(timestamps, metric_values)
        return float(slope)

    def sliding_rate(
        self,
        metric_values: np.ndarray,
        timestamps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute local deformation rate over a sliding window.

        Returns
        -------
        (window_centres, local_rates) — both 1-D arrays of the same length.
        """
        if self.window_s is None:
            raise ValueError("Set window_s in the constructor to use sliding_rate.")

        rates, centres = [], []
        dt = np.median(np.diff(timestamps))        # average sampling interval
        half = self.window_s / 2.0

        for t_c in timestamps:
            mask = (timestamps >= t_c - half) & (timestamps <= t_c + half)
            if mask.sum() < 2:
                continue
            slope, *_ = stats.linregress(timestamps[mask], metric_values[mask])
            rates.append(slope)
            centres.append(t_c)

        return np.array(centres), np.array(rates)

    def analyze_batch(
        self,
        experiments: List[Dict],
    ) -> pd.DataFrame:
        """Aggregate results across multiple experimental conditions.

        Parameters
        ----------
        experiments:
            List of dicts, each with keys:
            ``laser_power_mW``, ``trap_current_mA``,
            ``metric_values`` (array), ``timestamps`` (array).

        Returns
        -------
        DataFrame with columns: laser_power_mW, trap_current_mA,
        mean_rod, peak_deformation, n_frames.
        """
        rows = []
        for exp in experiments:
            mv = np.asarray(exp["metric_values"])
            ts = np.asarray(exp["timestamps"])
            rod = self.deformation_rate(mv, ts)
            rows.append(
                {
                    "laser_power_mW": exp["laser_power_mW"],
                    "trap_current_mA": exp["trap_current_mA"],
                    "mean_rod": rod,
                    "peak_deformation": float(np.nanmax(mv)),
                    "n_frames": len(mv),
                }
            )
        return pd.DataFrame(rows)

    def optical_correlation(
        self,
        summary_df: pd.DataFrame,
        x_col: str = "laser_power_mW",
        y_col: str = "mean_rod",
    ) -> Dict[str, float]:
        """Compute Pearson and Spearman correlation between an optical parameter
        and a deformation outcome across experimental conditions.

        Returns
        -------
        Dict with keys ``pearson_r``, ``pearson_p``, ``spearman_r``, ``spearman_p``.
        """
        x = summary_df[x_col].values
        y = summary_df[y_col].values

        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)

        return {
            "pearson_r": float(pr),
            "pearson_p": float(pp),
            "spearman_r": float(sr),
            "spearman_p": float(sp),
        }

    # ------------------------------------------------------------------
    # Viscoelastic model fitting (Kelvin–Voigt)
    # ------------------------------------------------------------------

    def fit_kelvin_voigt(
        self,
        stress: np.ndarray,
        strain: np.ndarray,
        dt: float,
    ) -> Dict[str, float]:
        """Estimate E and η from the Kelvin–Voigt constitutive relation.

        σ(t) = E·ε(t) + η·dε/dt   (Eq. 2 in proposal)

        Uses finite differences for dε/dt and linear least-squares to solve
        for E and η simultaneously.

        Parameters
        ----------
        stress:
            Applied optical stress time-series.
        strain:
            Measured strain (deformation) time-series.
        dt:
            Sampling interval in seconds.

        Returns
        -------
        Dict with ``E`` (Young's modulus) and ``eta`` (viscosity coefficient).
        """
        d_strain = np.gradient(strain, dt)   # dε/dt via central differences

        # Stack [ε, dε/dt] as the design matrix, solve σ = A @ [E, η]^T.
        A = np.column_stack([strain, d_strain])
        params, *_ = np.linalg.lstsq(A, stress, rcond=None)

        return {"E": float(params[0]), "eta": float(params[1])}
