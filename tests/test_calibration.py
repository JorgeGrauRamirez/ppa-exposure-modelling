"""
Tests for the calibration module.

These cover:
- AR(1) -> OU parameter back-out is internally consistent
- Seasonal OLS recovers planted coefficients
"""

import numpy as np
import pandas as pd
import pytest

from ppa_exposure.calibration import (
    fit_ar1_dynamics,
    fit_seasonal_ols,
    calibrate_schwartz,
)


class TestAR1Dynamics:
    def test_ar1_recovers_known_parameters(self):
        """Generate synthetic OU data and check kappa/sigma recovery."""
        true_kappa = 8.0          # per year
        true_sigma = 0.5          # annualised
        delta_t = 1.0 / 12.0      # monthly
        b_true = np.exp(-true_kappa * delta_t)
        var_eps = true_sigma ** 2 * (1 - np.exp(-2 * true_kappa * delta_t)) / (2 * true_kappa)

        rng = np.random.default_rng(0)
        n = 5000
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = b_true * y[t - 1] + np.sqrt(var_eps) * rng.standard_normal()

        kappa_hat, sigma_hat, _ = fit_ar1_dynamics(y, delta_t)
        assert abs(kappa_hat - true_kappa) / true_kappa < 0.05
        assert abs(sigma_hat - true_sigma) / true_sigma < 0.05

    def test_ar1_rejects_non_stationary(self):
        """AR(1) coefficient outside (0, 1) should raise."""
        # Force a non-stationary sequence
        y = np.cumsum(np.random.default_rng(0).standard_normal(1000))
        with pytest.raises(ValueError):
            fit_ar1_dynamics(y, 1.0 / 12.0)


class TestSeasonalOLS:
    def test_seasonal_ols_recovers_harmonics(self):
        """Plant a known seasonal pattern; check the OLS recovers it."""
        rng = np.random.default_rng(0)
        n_days = 365 * 5
        t = np.arange(n_days, dtype=float)
        omega = 2 * np.pi / 365.25
        theta = 4.5
        A1, B1, A2, B2 = 0.20, 0.10, 0.05, -0.03

        signal = theta + A1 * np.cos(omega * t) + B1 * np.sin(omega * t) \
                       + A2 * np.cos(2 * omega * t) + B2 * np.sin(2 * omega * t)
        noise = 0.1 * rng.standard_normal(n_days)
        y = signal + noise

        theta_hat, A1_hat, B1_hat, A2_hat, B2_hat, r2, *_ = fit_seasonal_ols(y, t)
        assert abs(theta_hat - theta) < 0.05
        assert abs(A1_hat - A1) < 0.02
        assert abs(B1_hat - B1) < 0.02
        assert r2 > 0.5
