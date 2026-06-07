"""
Diagnostic and validation utilities.

Provides explicit sanity checks that should be runnable by audit:
- Forward curve reconciliation (M -> Q -> Y consistency)
- Martingale check on Monte Carlo paths
- MtM mean-zero check at inception (par condition)
- OU stationary variance check on simulated Y paths
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ppa_exposure.monte_carlo import SimulationResult


def check_martingale(sim: SimulationResult, tolerance_z: float = 3.0) -> pd.DataFrame:
    """
    Test E_emp[S_t] == F(0, t) within Monte Carlo standard error.

    Returns a per-month table with z-scores. Passing condition: max |z| < tolerance_z.
    """
    N = sim.n_paths
    emp_mean = sim.S_paths.mean(axis=0)
    emp_std = sim.S_paths.std(axis=0, ddof=1)
    se = emp_std / np.sqrt(N)
    z = (emp_mean - sim.F_market) / se

    df = pd.DataFrame({
        "delivery_month": sim.delivery_months.strftime("%Y-%m"),
        "F_market": sim.F_market,
        "F_empirical": emp_mean,
        "diff": emp_mean - sim.F_market,
        "monte_carlo_se": se,
        "z_score": z,
    })
    df.attrs["max_abs_z"] = float(np.abs(z).max())
    df.attrs["passes"] = bool(np.abs(z).max() < tolerance_z)
    return df


def check_ou_stationary_variance(
    sim: SimulationResult,
    kappa: float,
    sigma: float,
    rel_tolerance: float = 0.1,
) -> pd.DataFrame:
    """
    Verify empirical Var(Y_t) matches theoretical sigma^2 / (2*kappa) * (1 - exp(-2*kappa*t)).

    For times t >> 1/kappa, both should converge to the stationary variance
    sigma^2 / (2*kappa).
    """
    theo_var = (sigma ** 2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * sim.T_grid))
    emp_var = sim.Y_paths.var(axis=0, ddof=1)
    rel_err = (emp_var - theo_var) / np.where(theo_var > 0, theo_var, 1.0)

    df = pd.DataFrame({
        "delivery_month": sim.delivery_months.strftime("%Y-%m"),
        "theoretical_var": theo_var,
        "empirical_var": emp_var,
        "relative_error": rel_err,
    })
    df.attrs["max_abs_rel_err"] = float(np.abs(rel_err).max())
    df.attrs["passes"] = bool(np.abs(rel_err).max() < rel_tolerance)
    return df


def check_par_condition(
    forward_curve: np.ndarray,
    fixed_price: float,
    volume_mw: float,
    delivery_months: pd.DatetimeIndex,
    T_grid: np.ndarray,
    discount_rate: float = 0.02,
    rel_tolerance: float = 1e-6,
) -> dict:
    """
    Verify that the fixed price truly makes the contract at par.

    The par condition is:  MtM_0 = sum_m DF(0, t_m) * (F_fix - F(0, m)) * V * h_m == 0
    where the sum is over ALL months (the full contract at inception).

    Note: At t_k > 0 the EMPIRICAL mean MtM across paths is NOT zero — the
    deterministic curve roll-off pushes expected MtM positive when the curve
    is backwardated (early months are high-price). The model property at par
    is only that the original contract value at inception is zero.
    """
    from ppa_exposure.exposure import _hours_per_month

    hours = _hours_per_month(delivery_months)
    df = np.exp(-discount_rate * T_grid)
    mtm_at_inception = float((df * (fixed_price - forward_curve) * volume_mw * hours).sum())
    notional = float((volume_mw * hours).sum() * fixed_price)
    rel = abs(mtm_at_inception) / notional if notional > 0 else 0.0

    return {
        "mtm_at_inception_eur": mtm_at_inception,
        "notional_eur": notional,
        "relative_error": rel,
        "passes": bool(rel < rel_tolerance),
    }


def check_mtm_mean_matches_analytical(
    MtM: np.ndarray,
    forward_curve: np.ndarray,
    fixed_price: float,
    volume_mw: float,
    delivery_months: pd.DatetimeIndex,
    T_grid: np.ndarray,
    discount_rate: float = 0.02,
    se_tolerance: float = 5.0,
) -> pd.DataFrame:
    """
    Verify that the empirical mean MtM across paths at each evaluation time
    matches the deterministic curve-roll-off MtM (Y=0 analytical value).

    This is a stronger property than the par condition: it validates that the
    analytical conditional-forward formula is correctly implemented. At each
    t_k:

        E_0[MtM_k] = sum_{m > k} DF(t_k, t_m) * (F_fix - F(0, m)) * V * h_m
    """
    from ppa_exposure.exposure import _hours_per_month

    N, M = MtM.shape
    hours = _hours_per_month(delivery_months)

    # Analytical mean per time step (no nested sim needed)
    analytical_mean = np.zeros(M)
    for k in range(M - 1):
        future = slice(k + 1, M)
        dt = T_grid[future] - T_grid[k]
        df = np.exp(-discount_rate * dt)
        analytical_mean[k] = (df * (fixed_price - forward_curve[future]) * volume_mw * hours[future]).sum()

    emp_mean = MtM.mean(axis=0)
    emp_std = MtM.std(axis=0, ddof=1)
    se = emp_std / np.sqrt(N)
    diff = emp_mean - analytical_mean
    z = diff / np.where(se > 0, se, 1.0)

    out = pd.DataFrame({
        "delivery_month": delivery_months.strftime("%Y-%m"),
        "analytical_mean": analytical_mean,
        "empirical_mean": emp_mean,
        "diff": diff,
        "monte_carlo_se": se,
        "z_score": z,
    })
    out.attrs["max_abs_z"] = float(np.abs(z).max())
    out.attrs["passes"] = bool(np.abs(z).max() < se_tolerance)
    return out
