"""
Monte Carlo simulation engine for the Schwartz one-factor model.

Simulates Y_t paths using exact discretization of the OU process:

    Y_{t+dt} = Y_t * exp(-kappa * dt) + sigma * sqrt((1 - exp(-2*kappa*dt))/(2*kappa)) * Z

Spot paths are reconstructed via:

    S_t = exp(alpha(t) + Y_t) - c

By construction, the empirical mean of S_paths matches the market forward
curve at every horizon (subject to Monte Carlo error), giving the model its
market-consistent property.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ppa_exposure.price_model import compute_alpha_t


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation output."""

    S_paths: np.ndarray         # (N, M) — simulated spot prices
    Y_paths: np.ndarray         # (N, M) — simulated state variable
    T_grid: np.ndarray          # (M,) — time grid in years from valuation date
    alpha_t: np.ndarray         # (M,) — deterministic drift
    F_market: np.ndarray        # (M,) — market forward curve used
    delivery_months: pd.DatetimeIndex  # (M,) — calendar months
    n_paths: int
    seed: int


def _build_time_grid(
    valuation_date: pd.Timestamp,
    delivery_months: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Time grid in years from valuation date to the *midpoint* of each
    delivery month — the natural representative time for monthly settlement.
    """
    midpoints = delivery_months + pd.offsets.MonthBegin(0) + pd.Timedelta(days=14)
    return ((midpoints - valuation_date).days / 365.25).values.astype(float)


def simulate(
    valuation_date: str | pd.Timestamp,
    delivery_months: pd.DatetimeIndex,
    forward_curve: np.ndarray,
    kappa: float,
    sigma: float,
    shift_constant_c: float,
    n_paths: int = 10_000,
    use_antithetic: bool = True,
    seed: int = 42,
) -> SimulationResult:
    """
    Run the full Monte Carlo simulation under the calibrated Schwartz model.

    Parameters
    ----------
    valuation_date : str | Timestamp
        The 'today' date for which the forward curve is observed.
    delivery_months : pd.DatetimeIndex
        Month-begin timestamps for each settlement period (length M).
    forward_curve : np.ndarray, shape (M,)
        Market-observed forward prices F(0, t_m) for each delivery month.
    kappa, sigma : float
        Calibrated OU parameters.
    shift_constant_c : float
        Log shift constant.
    n_paths : int
        Number of simulated paths. Must be even if use_antithetic=True.
    use_antithetic : bool
        Apply antithetic variates for variance reduction.
    seed : int

    Returns
    -------
    SimulationResult
    """
    valuation_date = pd.Timestamp(valuation_date)
    T_grid = _build_time_grid(valuation_date, delivery_months)
    M = len(T_grid)

    if use_antithetic and n_paths % 2 != 0:
        raise ValueError("n_paths must be even when use_antithetic=True.")

    # Time steps between consecutive simulation points
    # First step is from t=0 (Y_0 = 0) to t = T_grid[0]; subsequent steps are
    # between successive T_grid points.
    delta_t = np.diff(np.concatenate([[0.0], T_grid]))           # (M,)
    decay = np.exp(-kappa * delta_t)                              # (M,)
    step_std = sigma * np.sqrt((1 - np.exp(-2 * kappa * delta_t)) / (2 * kappa))

    # Random number generation
    rng = np.random.default_rng(seed)
    if use_antithetic:
        half = n_paths // 2
        Z_half = rng.standard_normal(size=(half, M))
        Z = np.concatenate([Z_half, -Z_half], axis=0)
    else:
        Z = rng.standard_normal(size=(n_paths, M))

    # Exact OU discretization
    Y = np.zeros((n_paths, M))
    Y[:, 0] = step_std[0] * Z[:, 0]            # Y_0 = 0 implicit
    for i in range(1, M):
        Y[:, i] = decay[i] * Y[:, i - 1] + step_std[i] * Z[:, i]

    # Market-consistent alpha(t)
    alpha = compute_alpha_t(forward_curve, T_grid, kappa, sigma, shift_constant_c)

    # Reconstruct spot
    S = np.exp(alpha[np.newaxis, :] + Y) - shift_constant_c

    return SimulationResult(
        S_paths=S,
        Y_paths=Y,
        T_grid=T_grid,
        alpha_t=alpha,
        F_market=forward_curve,
        delivery_months=delivery_months,
        n_paths=n_paths,
        seed=seed,
    )


def martingale_check(result: SimulationResult, tolerance_std: float = 3.0) -> pd.DataFrame:
    """
    Verify that empirical mean of S_paths matches the market forward curve
    within Monte Carlo standard error.

    Returns a DataFrame with one row per month showing the deviation in
    multiples of standard error. The check passes when |z| < tolerance_std.
    """
    N = result.n_paths
    emp_mean = result.S_paths.mean(axis=0)
    emp_std = result.S_paths.std(axis=0, ddof=1)
    se = emp_std / np.sqrt(N)
    z = (emp_mean - result.F_market) / se

    return pd.DataFrame({
        "delivery_month": result.delivery_months.strftime("%Y-%m"),
        "F_market": result.F_market,
        "F_empirical": emp_mean,
        "diff": emp_mean - result.F_market,
        "monte_carlo_se": se,
        "z_score": z,
    })
