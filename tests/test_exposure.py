"""
Tests for the exposure module.

Properties verified:

1. At the par fixed price, MtM evaluated over ALL cashflows at inception
   equals zero (definition of par).

2. The empirical mean MtM across simulated paths at each evaluation time
   t_k matches the deterministic analytical value (Y=0 case), within Monte
   Carlo standard error. This validates the closed-form conditional-forward
   implementation. Note that this analytical mean is NOT zero at t_k > 0 —
   it reflects the deterministic curve roll-off effect.

3. PFE >= EE on the credit side (positive-MtM dominant under backwardated
   curve). This does not hold on the liquidity side for highly-skewed
   distributions, where the quantile and the mean can have opposite ordering.
"""

import numpy as np
import pandas as pd
import pytest

from ppa_exposure.monte_carlo import simulate
from ppa_exposure.exposure import (
    compute_par_fixed_price,
    compute_mtm_matrix,
    compute_exposure_metrics,
)
from ppa_exposure.validation import (
    check_par_condition,
    check_mtm_mean_matches_analytical,
)


@pytest.fixture
def fixture():
    """Reproducible setup with non-trivial backwardated forward curve."""
    valuation_date = pd.Timestamp("2026-06-01")
    delivery_months = pd.date_range("2026-07-01", periods=24, freq="MS")
    forward_curve = np.array([
        100.0, 95.0, 90.0, 95.0, 110.0, 120.0,
        100.0, 90.0, 80.0, 80.0, 75.0, 75.0,
        80.0, 78.0, 76.0, 75.0, 80.0, 85.0,
        78.0, 73.0, 70.0, 72.0, 71.0, 71.0,
    ])

    sim = simulate(
        valuation_date=valuation_date,
        delivery_months=delivery_months,
        forward_curve=forward_curve,
        kappa=8.0,
        sigma=0.5,
        shift_constant_c=100.0,
        n_paths=10000,
        seed=42,
    )
    return {"sim": sim, "delivery_months": delivery_months, "forward_curve": forward_curve}


class TestParCondition:
    """At par fixed price, the contract value at inception is zero."""

    def test_par_mtm_at_inception_is_zero(self, fixture):
        F_fix = compute_par_fixed_price(
            fixture["forward_curve"], fixture["delivery_months"], fixture["sim"].T_grid,
            discount_rate=0.02,
        )
        check = check_par_condition(
            forward_curve=fixture["forward_curve"],
            fixed_price=F_fix,
            volume_mw=100.0,
            delivery_months=fixture["delivery_months"],
            T_grid=fixture["sim"].T_grid,
            discount_rate=0.02,
        )
        assert check["passes"], (
            f"Par condition not met: MtM_0 = {check['mtm_at_inception_eur']:.2f} EUR "
            f"(relative error {check['relative_error']:.2e})"
        )


class TestMtmAnalyticalConsistency:
    """Empirical mean MtM matches the deterministic curve-roll-off value."""

    def test_empirical_mean_matches_analytical(self, fixture):
        F_fix = compute_par_fixed_price(
            fixture["forward_curve"], fixture["delivery_months"], fixture["sim"].T_grid,
        )
        MtM = compute_mtm_matrix(
            fixture["sim"], F_fix, volume_mw=100.0,
            kappa=8.0, sigma=0.5, shift_constant_c=100.0,
        )
        check = check_mtm_mean_matches_analytical(
            MtM=MtM,
            forward_curve=fixture["forward_curve"],
            fixed_price=F_fix,
            volume_mw=100.0,
            delivery_months=fixture["delivery_months"],
            T_grid=fixture["sim"].T_grid,
            discount_rate=0.02,
            se_tolerance=5.0,
        )
        assert check.attrs["passes"], (
            f"Analytical mean check failed: max |z| = {check.attrs['max_abs_z']:.2f}"
        )


class TestExposureShape:
    """Shape sanity checks on the exposure profile."""

    def test_pfe_geq_ee_credit_side(self, fixture):
        """PFE 95% >= EE on the credit side (positive-MtM dominant)."""
        F_fix = compute_par_fixed_price(
            fixture["forward_curve"], fixture["delivery_months"], fixture["sim"].T_grid,
        )
        MtM = compute_mtm_matrix(
            fixture["sim"], F_fix, volume_mw=100.0,
            kappa=8.0, sigma=0.5, shift_constant_c=100.0,
        )
        metrics = compute_exposure_metrics(MtM, fixture["delivery_months"], F_fix, 100.0)
        assert (metrics.PFE_credit >= metrics.EE_credit).all()

    def test_exposure_terminates_at_zero(self, fixture):
        """At the last month, no future cashflows remain -> MtM = 0."""
        F_fix = compute_par_fixed_price(
            fixture["forward_curve"], fixture["delivery_months"], fixture["sim"].T_grid,
        )
        MtM = compute_mtm_matrix(
            fixture["sim"], F_fix, volume_mw=100.0,
            kappa=8.0, sigma=0.5, shift_constant_c=100.0,
        )
        np.testing.assert_array_equal(MtM[:, -1], np.zeros(MtM.shape[0]))


class TestFixedPriceFormula:
    """Par fixed price reproduces the discount + hours-weighted curve average."""

    def test_par_price_in_range(self, fixture):
        F_fix = compute_par_fixed_price(
            fixture["forward_curve"], fixture["delivery_months"], fixture["sim"].T_grid,
        )
        assert fixture["forward_curve"].min() < F_fix < fixture["forward_curve"].max()

    def test_par_price_reasonable_for_constant_curve(self):
        """For a flat curve, par fixed price = constant level."""
        forward_curve = np.full(12, 80.0)
        delivery_months = pd.date_range("2026-07-01", periods=12, freq="MS")
        T_grid = np.arange(12, dtype=float) / 12 + 1.0 / 24
        F_fix = compute_par_fixed_price(forward_curve, delivery_months, T_grid)
        assert abs(F_fix - 80.0) < 1e-6
