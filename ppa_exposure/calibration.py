"""
Schwartz one-factor calibration.

Fits the model
    d(ln(S_t + c)) = kappa * (theta + f(t) - ln(S_t + c)) * dt + sigma * dW_t

where f(t) is a deterministic seasonal function (annual + semi-annual harmonics).

Two-step procedure (standard textbook approach):
    1. OLS on log-shifted prices for theta and the seasonal coefficients.
    2. AR(1) on the residuals (which are mean-zero stationary OU by construction)
       for the dynamics parameters kappa and sigma.

Daily data captures fast intra-week dynamics but produces near-deterministic
MtM at monthly resolution. For PPA exposure at monthly settlement, calibrate
on monthly aggregated data — see `frequency` parameter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CalibrationResult:
    """Container for calibrated parameters and diagnostics."""

    # Core model parameters
    kappa_per_year: float
    sigma_annualized: float
    shift_constant_c: float

    # Deterministic component
    theta: float
    A1: float            # cos(omega * t)
    B1: float            # sin(omega * t)
    A2: float            # cos(2*omega * t)
    B2: float            # sin(2*omega * t)
    omega: float

    # Reference for time axis
    t0_ref_date: str

    # Calibration metadata
    frequency: str       # 'daily' or 'monthly'
    calibration_window_start: str
    calibration_window_end: str
    n_observations: int

    # Diagnostics
    r2_ols_seasonal: float
    half_life_days: float
    seasonal_amplitude_annual: float
    seasonal_amplitude_semi: float
    acf_ols_resid_lag1: float
    acf_ar1_innov_lag1: float
    jb_pvalue: float
    innovation_skewness: float
    innovation_excess_kurtosis: float

    def to_dict(self) -> dict:
        d = asdict(self)
        # Wrap parameters in a 'parameters' sub-dict for backward compatibility
        return {
            "model": "schwartz_one_factor_with_seasonality",
            "calibration_date": self.calibration_window_end,
            "calibration_window": {
                "start": self.calibration_window_start,
                "end": self.calibration_window_end,
                "n_observations": self.n_observations,
                "frequency": self.frequency,
            },
            "shift_constant_c": self.shift_constant_c,
            "transform": "X_t = log(S_t + c)",
            "parameters": {
                "kappa_per_year": self.kappa_per_year,
                "sigma_annualized": self.sigma_annualized,
                "theta": self.theta,
                "A1": self.A1, "B1": self.B1,
                "A2": self.A2, "B2": self.B2,
                "omega": self.omega,
            },
            "reference": {"t0_ref_date": self.t0_ref_date},
            "derived": {
                "half_life_days": self.half_life_days,
                "seasonal_amplitude_annual": self.seasonal_amplitude_annual,
                "seasonal_amplitude_semi_annual": self.seasonal_amplitude_semi,
                "r2_ols_seasonal": self.r2_ols_seasonal,
            },
            "diagnostics": {
                "acf_ols_resid_lag1": self.acf_ols_resid_lag1,
                "acf_ar1_innov_lag1": self.acf_ar1_innov_lag1,
                "jb_pvalue": self.jb_pvalue,
                "innovation_skewness": self.innovation_skewness,
                "innovation_excess_kurtosis": self.innovation_excess_kurtosis,
            },
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def _autocorr(x: np.ndarray, max_lag: int = 30) -> np.ndarray:
    """Sample autocorrelation up to `max_lag` lags."""
    x = x - x.mean()
    var = x.var()
    return np.array([
        (x[: len(x) - k] * x[k:]).mean() / var
        for k in range(max_lag + 1)
    ])


def fit_seasonal_ols(
    log_prices: np.ndarray,
    t_days: np.ndarray,
) -> tuple[float, float, float, float, float, float, np.ndarray, np.ndarray]:
    """
    Fit constant + annual + semi-annual harmonics to log-shifted prices.

    Returns (theta, A1, B1, A2, B2, r2, fitted, residuals).
    """
    omega = 2 * np.pi / 365.25
    X = np.column_stack([
        np.ones(len(t_days)),
        np.cos(omega * t_days),
        np.sin(omega * t_days),
        np.cos(2 * omega * t_days),
        np.sin(2 * omega * t_days),
    ])
    beta, *_ = np.linalg.lstsq(X, log_prices, rcond=None)
    theta, A1, B1, A2, B2 = beta
    fitted = X @ beta
    resid = log_prices - fitted
    ss_tot = ((log_prices - log_prices.mean()) ** 2).sum()
    ss_res = (resid ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return theta, A1, B1, A2, B2, r2, fitted, resid


def fit_ar1_dynamics(
    residuals: np.ndarray,
    delta_t_years: float,
) -> tuple[float, float, np.ndarray]:
    """
    Fit AR(1) without intercept to mean-zero residuals; back out continuous-
    time OU parameters.

    Parameters
    ----------
    residuals : np.ndarray (n,)
        Consecutive observations of the mean-zero OU process. Caller must
        ensure observations are evenly spaced.
    delta_t_years : float
        Time step between successive observations, in years.

    Returns
    -------
    (kappa_per_year, sigma_annualized, innovations)
    """
    # AR(1) regression: resid[t+1] = b * resid[t] + eps
    X = residuals[:-1].reshape(-1, 1)
    y = residuals[1:]
    b_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    b_hat = float(b_hat[0])

    innovations = y - b_hat * X.ravel()
    var_eps = innovations.var(ddof=1)

    if not (0 < b_hat < 1):
        raise ValueError(
            f"AR(1) coefficient b = {b_hat:.4f} outside (0, 1). "
            "Residuals are not consistent with a stationary OU process."
        )

    kappa = -np.log(b_hat) / delta_t_years
    sigma_sq = var_eps * (2 * kappa) / (1 - np.exp(-2 * kappa * delta_t_years))
    sigma = float(np.sqrt(sigma_sq))
    return float(kappa), sigma, innovations


def calibrate_schwartz(
    spot_df: pd.DataFrame,
    frequency: str = "monthly",
    shift_constant_c: float = 100.0,
    date_col: str | None = None,
    price_col: str = "SpotPriceEUR",
) -> CalibrationResult:
    """
    Full two-step calibration of the Schwartz one-factor model.

    Parameters
    ----------
    spot_df : pd.DataFrame
        Time-indexed spot prices. Must contain the date column and `price_col`.
    frequency : str
        'daily' or 'monthly'. Determines delta_t for AR(1) -> continuous conversion.
    shift_constant_c : float
        Constant added to S before log transformation. Must satisfy c > -min(S).
    date_col : str | None
        Name of the date column. If None, inferred from frequency
        ('Date' for daily, 'YearMonth' for monthly).
    price_col : str
        Name of the spot price column in EUR/MWh.

    Returns
    -------
    CalibrationResult
    """
    if date_col is None:
        date_col = "Date" if frequency == "daily" else "YearMonth"

    df = spot_df[[date_col, price_col]].copy().sort_values(date_col).reset_index(drop=True)
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"])

    # Sanity check on shift
    min_price = df["price"].min()
    if min_price + shift_constant_c <= 0:
        raise ValueError(
            f"Shift constant c={shift_constant_c} is insufficient: "
            f"min(S) = {min_price:.2f}. Need c > {-min_price:.2f}."
        )

    # Log-shift
    df["X"] = np.log(df["price"] + shift_constant_c)

    # Time axis in days since first observation
    t0_ref = df["date"].iloc[0]
    df["t_days"] = (df["date"] - t0_ref).dt.days.astype(float)

    # Step 1: OLS seasonal
    theta, A1, B1, A2, B2, r2, fitted, resid = fit_seasonal_ols(
        df["X"].values, df["t_days"].values,
    )

    amp_annual = float(np.sqrt(A1 ** 2 + B1 ** 2))
    amp_semi = float(np.sqrt(A2 ** 2 + B2 ** 2))

    # Step 2: AR(1) on residuals — only consecutive observations
    if frequency == "daily":
        df["dt_days"] = (df["date"] - df["date"].shift(1)).dt.days
        delta_t_years = 1.0 / 365.25
        mask_consecutive = df["dt_days"] == 1
    elif frequency == "monthly":
        df["months_diff"] = (df["date"].dt.year - df["date"].shift(1).dt.year) * 12 + \
                            (df["date"].dt.month - df["date"].shift(1).dt.month)
        delta_t_years = 1.0 / 12.0
        mask_consecutive = df["months_diff"] == 1
    else:
        raise ValueError(f"frequency must be 'daily' or 'monthly', got '{frequency}'")

    df["resid"] = resid

    # Filter to consecutive pairs and extract sequence
    df["resid_lag"] = df["resid"].shift(1)
    pairs = df[mask_consecutive & df["resid_lag"].notna()][["resid_lag", "resid"]].dropna()
    if len(pairs) < 50:
        raise ValueError(f"Too few consecutive pairs for AR(1) fit: {len(pairs)}")

    # Fit AR(1)
    resid_array = np.concatenate([pairs["resid_lag"].values[:1], pairs["resid"].values])
    kappa, sigma, innovations = fit_ar1_dynamics(resid_array, delta_t_years)

    half_life_days = float(np.log(2) / kappa * 365.25)

    # Diagnostics
    acf_resid = _autocorr(df["resid"].dropna().values, max_lag=5)
    acf_innov = _autocorr(innovations, max_lag=5)
    jb_stat, jb_pvalue = stats.jarque_bera(innovations)

    return CalibrationResult(
        kappa_per_year=kappa,
        sigma_annualized=sigma,
        shift_constant_c=shift_constant_c,
        theta=float(theta),
        A1=float(A1), B1=float(B1), A2=float(A2), B2=float(B2),
        omega=2 * np.pi / 365.25,
        t0_ref_date=str(t0_ref.date()),
        frequency=frequency,
        calibration_window_start=str(df["date"].min().date()),
        calibration_window_end=str(df["date"].max().date()),
        n_observations=int(len(df)),
        r2_ols_seasonal=float(r2),
        half_life_days=half_life_days,
        seasonal_amplitude_annual=amp_annual,
        seasonal_amplitude_semi=amp_semi,
        acf_ols_resid_lag1=float(acf_resid[1]),
        acf_ar1_innov_lag1=float(acf_innov[1]),
        jb_pvalue=float(jb_pvalue),
        innovation_skewness=float(stats.skew(innovations)),
        innovation_excess_kurtosis=float(stats.kurtosis(innovations)),
    )


def _cli_calibrate() -> None:
    """Command-line entry: `python -m ppa_exposure.calibration calibrate ...`"""
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate Schwartz one-factor model.")
    parser.add_argument("--spot", required=True, help="Path to spot CSV (daily or monthly).")
    parser.add_argument("--frequency", default="monthly", choices=["daily", "monthly"])
    parser.add_argument("--shift", type=float, default=100.0, help="Log shift constant c.")
    parser.add_argument("--output", default="data/model_params.json")
    args = parser.parse_args()

    if args.frequency == "daily":
        df = pd.read_csv(args.spot, parse_dates=["Date"])
    else:
        df = pd.read_csv(args.spot, parse_dates=["YearMonth"])

    result = calibrate_schwartz(df, frequency=args.frequency, shift_constant_c=args.shift)
    result.to_json(args.output)

    print(f"Calibration complete. Output: {args.output}")
    print(f"  kappa (1/year):   {result.kappa_per_year:.2f}")
    print(f"  sigma:            {result.sigma_annualized:.4f}")
    print(f"  half-life (days): {result.half_life_days:.2f}")
    print(f"  R^2 (seasonal):   {result.r2_ols_seasonal:.4f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        sys.argv = sys.argv[:1] + sys.argv[2:]
        _cli_calibrate()
    else:
        print("Usage: python -m ppa_exposure.calibration calibrate --spot <csv> [options]")
