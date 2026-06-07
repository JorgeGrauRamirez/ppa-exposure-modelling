"""
PPA MtM and exposure metrics.

Mark-to-market is computed analytically using the closed-form conditional
forward — no nested Monte Carlo. For Ørsted as fixed receiver:

    MtM_k^(n) = sum_{m > k} DF(t_k, t_m) * (F_fix - F_{t_k}^(n)(m)) * V * h_m

where F_{t_k}^(n)(m) = E[S_{t_m} | Y_{t_k}^(n)] in closed form.

Exposure metrics are derived from the resulting MtM matrix:
- Credit (Ørsted in-the-money): EE, PFE, EPE, Max PFE
- Liquidity (Ørsted out-of-money): EE, PFE, EPE, Max PFE
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ppa_exposure.monte_carlo import SimulationResult


@dataclass
class ExposureMetrics:
    """Container for exposure calculation outputs."""

    MtM: np.ndarray                # (N, M)
    F_fix: float                   # fixed price at par
    notional_eur: float            # total contract notional

    # Credit (Ørsted in-the-money)
    EE_credit: np.ndarray          # (M,)
    PFE_credit: np.ndarray         # (M,)
    EPE_credit: float              # time-averaged EE
    Max_PFE_credit: float
    Max_PFE_credit_month: pd.Timestamp

    # Liquidity (Ørsted out-of-money)
    EE_liq: np.ndarray             # (M,)
    PFE_liq: np.ndarray            # (M,)
    EPE_liq: float
    Max_PFE_liq: float
    Max_PFE_liq_month: pd.Timestamp

    # Reference
    delivery_months: pd.DatetimeIndex
    pfe_quantile: float
    discount_rate: float


def _hours_per_month(delivery_months: pd.DatetimeIndex) -> np.ndarray:
    """Number of hours in each delivery month."""
    month_ends = delivery_months + pd.offsets.MonthEnd(0)
    days = (month_ends - delivery_months).days + 1
    return days.values * 24.0


def compute_par_fixed_price(
    forward_curve: np.ndarray,
    delivery_months: pd.DatetimeIndex,
    T_grid: np.ndarray,
    discount_rate: float = 0.02,
) -> float:
    """
    Compute the fixed price at par: MtM at inception equals zero.

    F_fix = sum_m DF(0, t_m) * F(0, m) * h_m / sum_m DF(0, t_m) * h_m
    """
    hours = _hours_per_month(delivery_months)
    df = np.exp(-discount_rate * T_grid)
    return float((df * forward_curve * hours).sum() / (df * hours).sum())


def compute_mtm_matrix(
    sim: SimulationResult,
    fixed_price: float,
    volume_mw: float,
    kappa: float,
    sigma: float,
    shift_constant_c: float,
    discount_rate: float = 0.02,
) -> np.ndarray:
    """
    Compute MtM at every (path, evaluation time) analytically.

    The MtM is evaluated at each T_grid[k] looking at remaining cashflows
    m > k. Uses the closed-form conditional forward — no nested simulation.

    Parameters
    ----------
    sim : SimulationResult
        Output of monte_carlo.simulate.
    fixed_price : float
        Contract fixed price in EUR/MWh.
    volume_mw : float
        Baseload capacity in MW.
    kappa, sigma, shift_constant_c : float
        Schwartz model parameters.
    discount_rate : float
        Continuous discount rate.

    Returns
    -------
    MtM : np.ndarray, shape (N, M)
    """
    N, M = sim.Y_paths.shape
    hours = _hours_per_month(sim.delivery_months)
    MtM = np.zeros((N, M))

    # k = M-1: no future cashflows -> MtM stays 0
    for k in range(M - 1):
        future = slice(k + 1, M)
        dt = sim.T_grid[future] - sim.T_grid[k]                            # (M-k-1,)
        decay = np.exp(-kappa * dt)
        cond_var = (sigma ** 2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt))
        jensen = 0.5 * cond_var

        cond_Y_mean = sim.Y_paths[:, k:k+1] * decay[np.newaxis, :]         # (N, M-k-1)
        log_F = sim.alpha_t[future][np.newaxis, :] + cond_Y_mean + jensen[np.newaxis, :]
        F_cond = np.exp(log_F) - shift_constant_c

        df = np.exp(-discount_rate * dt)
        cf_pv = df[np.newaxis, :] * (fixed_price - F_cond) * volume_mw * hours[future][np.newaxis, :]
        MtM[:, k] = cf_pv.sum(axis=1)

    return MtM


def compute_exposure_metrics(
    MtM: np.ndarray,
    delivery_months: pd.DatetimeIndex,
    fixed_price: float,
    volume_mw: float,
    discount_rate: float = 0.02,
    pfe_quantile: float = 0.95,
) -> ExposureMetrics:
    """
    Derive EE, PFE, EPE, and Max PFE for credit and liquidity sides.
    """
    hours = _hours_per_month(delivery_months)
    total_volume = (volume_mw * hours).sum()
    notional = total_volume * fixed_price

    exp_credit = np.maximum(MtM, 0.0)
    exp_liq = np.maximum(-MtM, 0.0)

    EE_credit = exp_credit.mean(axis=0)
    PFE_credit = np.percentile(exp_credit, pfe_quantile * 100, axis=0)
    EE_liq = exp_liq.mean(axis=0)
    PFE_liq = np.percentile(exp_liq, pfe_quantile * 100, axis=0)

    return ExposureMetrics(
        MtM=MtM,
        F_fix=fixed_price,
        notional_eur=notional,
        EE_credit=EE_credit,
        PFE_credit=PFE_credit,
        EPE_credit=float(EE_credit.mean()),
        Max_PFE_credit=float(PFE_credit.max()),
        Max_PFE_credit_month=delivery_months[int(PFE_credit.argmax())],
        EE_liq=EE_liq,
        PFE_liq=PFE_liq,
        EPE_liq=float(EE_liq.mean()),
        Max_PFE_liq=float(PFE_liq.max()),
        Max_PFE_liq_month=delivery_months[int(PFE_liq.argmax())],
        delivery_months=delivery_months,
        pfe_quantile=pfe_quantile,
        discount_rate=discount_rate,
    )
