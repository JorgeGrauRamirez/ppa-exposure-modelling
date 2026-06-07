"""
Schwartz one-factor price model.

The model decomposes the log-shifted spot as:
    ln(S_t + c) = alpha(t) + Y_t,   dY_t = -kappa * Y_t * dt + sigma * dW_t

where:
- (kappa, sigma) are calibrated to historical dynamics (see calibration.py)
- alpha(t) is determined analytically so that E[S_t] = F(0, t) exactly
  matches the market-observed forward curve at every horizon
- c is the log shift constant

This module provides the pure analytical building blocks:
- compute_alpha_t      : market-consistent drift
- ou_conditional_mean  : E[Y_t | Y_s] for s < t
- ou_conditional_var   : Var[Y_t | Y_s] for s < t
- conditional_forward  : E[S_t | Y_s] in closed form
"""

from __future__ import annotations

import numpy as np


def compute_alpha_t(
    forward_curve: np.ndarray,
    t_grid: np.ndarray,
    kappa: float,
    sigma: float,
    shift_constant_c: float,
) -> np.ndarray:
    """
    Compute the deterministic component alpha(t) such that E_0[S_t] = F(0, t).

    Under the model ln(S_t + c) = alpha(t) + Y_t with Y_0 = 0 and
    Y_t ~ Normal(0, sigma^2 / (2*kappa) * (1 - exp(-2*kappa*t))), the
    expected spot is:

        E_0[S_t] = exp(alpha(t) + Var[Y_t] / 2) - c.

    Setting E_0[S_t] = F(0, t):

        alpha(t) = log(F(0, t) + c) - sigma^2 / (4*kappa) * (1 - exp(-2*kappa*t))

    Parameters
    ----------
    forward_curve : np.ndarray, shape (M,)
        Market forward prices F(0, t_m) for the M target months.
    t_grid : np.ndarray, shape (M,)
        Time grid in years from valuation date.
    kappa, sigma, shift_constant_c : float
        Model parameters.
    """
    var_Yt = (sigma ** 2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * t_grid))
    alpha_t = np.log(forward_curve + shift_constant_c) - 0.5 * var_Yt
    return alpha_t


def ou_conditional_mean(
    Y_s: np.ndarray | float,
    kappa: float,
    delta_t: float | np.ndarray,
) -> np.ndarray | float:
    """E[Y_t | Y_s] = Y_s * exp(-kappa * (t - s))."""
    return Y_s * np.exp(-kappa * delta_t)


def ou_conditional_var(
    kappa: float,
    sigma: float,
    delta_t: float | np.ndarray,
) -> float | np.ndarray:
    """Var[Y_t | Y_s] = sigma^2 / (2*kappa) * (1 - exp(-2*kappa*(t-s)))."""
    return (sigma ** 2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * delta_t))


def conditional_forward(
    Y_s: np.ndarray,
    alpha_t: np.ndarray,
    t_grid: np.ndarray,
    s: float,
    kappa: float,
    sigma: float,
    shift_constant_c: float,
) -> np.ndarray:
    """
    Closed-form conditional forward: F_s(t) = E[S_t | Y_s] for s <= t.

    F_s(t) = exp(alpha(t) + Y_s * exp(-kappa*(t-s)) + Var[Y_t|Y_s] / 2) - c.

    Parameters
    ----------
    Y_s : np.ndarray, shape (N,) or (N, 1)
        Simulated state at time s for each of N paths.
    alpha_t : np.ndarray, shape (M_future,)
        Deterministic drift at future times t > s.
    t_grid : np.ndarray, shape (M_future,)
        Future time grid (in years from valuation date).
    s : float
        Conditioning time (in years from valuation date).
    kappa, sigma, shift_constant_c : float

    Returns
    -------
    F_cond : np.ndarray, shape (N, M_future)
    """
    Y_s = np.asarray(Y_s).reshape(-1, 1)
    dt = (t_grid - s).reshape(1, -1)
    decay = np.exp(-kappa * dt)
    cond_var = (sigma ** 2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt))
    log_F = alpha_t.reshape(1, -1) + Y_s * decay + 0.5 * cond_var
    return np.exp(log_F) - shift_constant_c
