"""Tests for celldform.biomechanics.DeformationAnalyzer."""

import numpy as np
import pandas as pd
import pytest

from celldform.biomechanics.analysis import DeformationAnalyzer


class TestDeformationAnalyzer:

    def _linear_signal(self, n=50, slope=0.01):
        t = np.linspace(0, n - 1, n) / 10.0   # 0 … 4.9 s at 10 fps
        v = slope * t + 0.05
        return t, v

    def test_deformation_rate_linear_signal(self):
        da = DeformationAnalyzer()
        t, v = self._linear_signal(slope=0.02)
        rod = da.deformation_rate(v, t)
        assert abs(rod - 0.02) < 1e-3

    def test_deformation_rate_constant_signal_is_near_zero(self):
        da = DeformationAnalyzer()
        t = np.linspace(0, 5, 50)
        v = np.ones(50) * 0.3
        rod = da.deformation_rate(v, t)
        assert abs(rod) < 1e-8

    def test_deformation_rate_single_point_returns_nan(self):
        da = DeformationAnalyzer()
        rod = da.deformation_rate(np.array([0.1]), np.array([0.0]))
        assert np.isnan(rod)

    def test_sliding_rate_requires_window_s(self):
        da = DeformationAnalyzer(window_s=None)
        t, v = self._linear_signal()
        with pytest.raises(ValueError):
            da.sliding_rate(v, t)

    def test_sliding_rate_returns_correct_shape(self):
        da = DeformationAnalyzer(window_s=0.5)
        t, v = self._linear_signal(n=50)
        centres, rates = da.sliding_rate(v, t)
        assert centres.shape == rates.shape
        assert len(centres) > 0

    def test_analyze_batch_returns_dataframe(self):
        da = DeformationAnalyzer()
        t, v = self._linear_signal()
        experiments = [
            {"laser_power_mW": 100.0, "trap_current_mA": 200.0, "metric_values": v, "timestamps": t},
            {"laser_power_mW": 150.0, "trap_current_mA": 250.0, "metric_values": v * 1.5, "timestamps": t},
        ]
        df = da.analyze_batch(experiments)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "mean_rod" in df.columns

    def test_optical_correlation_returns_required_keys(self):
        da = DeformationAnalyzer()
        df = pd.DataFrame({
            "laser_power_mW": [100, 120, 140, 160, 180],
            "mean_rod": [0.01, 0.015, 0.018, 0.022, 0.027],
        })
        result = da.optical_correlation(df)
        for key in ("pearson_r", "pearson_p", "spearman_r", "spearman_p"):
            assert key in result

    def test_fit_kelvin_voigt_recovers_parameters(self):
        da = DeformationAnalyzer()
        dt = 0.1
        t = np.arange(0, 5, dt)
        E_true, eta_true = 2.0, 0.5
        strain = np.exp(-t)          # synthetic strain
        d_strain = np.gradient(strain, dt)
        stress = E_true * strain + eta_true * d_strain
        result = da.fit_kelvin_voigt(stress, strain, dt)
        assert abs(result["E"] - E_true) < 0.1
        assert abs(result["eta"] - eta_true) < 0.1
