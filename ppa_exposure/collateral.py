"""
CSA collateral overlay and liquidity metrics.

Given a MtM matrix and a Credit Support Annex threshold, computes the
collateral posted by Ørsted at each (path, time) and derives the metrics a
treasury function uses for liquidity sizing:

- Peak collateral posted per path (distribution -> buffer sizing)
- Max single-period margin call per path (distribution -> cash management)
- Threshold sensitivity (credit vs liquidity trade-off)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LiquidityMetrics:
    """Container for CSA-overlay liquidity outputs."""

    threshold_eur: float
    collateral_posted: np.ndarray         # (N, M)
    margin_calls: np.ndarray              # (N, M) — positive = Ørsted posts more

    EE_collateral: np.ndarray             # (M,)
    PFE_collateral: np.ndarray            # (M,)
    Max_PFE_collateral: float
    Max_PFE_month: pd.Timestamp

    peak_per_path: np.ndarray             # (N,) — max collateral over deal life
    max_call_per_path: np.ndarray         # (N,) — biggest single posting

    peak_pfe95: float
    peak_pfe99: float
    max_call_pfe95: float
    max_call_pfe99: float
    fraction_no_posting: float

    delivery_months: pd.DatetimeIndex
    pfe_quantile: float


def compute_collateral_posted(MtM: np.ndarray, threshold: float) -> np.ndarray:
    """
    Collateral posted by Ørsted at each (path, time).

    Coll posted = max(-MtM - threshold, 0)

    Equivalent to applying a symmetric threshold T to liquidity exposure:
    Ørsted posts only when out-of-money beyond T.
    """
    return np.maximum(-MtM - threshold, 0.0)


def compute_margin_calls(collateral_posted: np.ndarray) -> np.ndarray:
    """
    Period-over-period change in collateral posted.

    Positive values represent additional posting by Ørsted (cash out);
    negative values represent collateral returned (cash in).
    """
    initial = np.zeros((collateral_posted.shape[0], 1))
    series = np.concatenate([initial, collateral_posted], axis=1)
    return np.diff(series, axis=1)


def compute_liquidity_metrics(
    MtM: np.ndarray,
    delivery_months: pd.DatetimeIndex,
    threshold: float,
    pfe_quantile: float = 0.95,
) -> LiquidityMetrics:
    """
    Full liquidity metric computation for one threshold.
    """
    collateral = compute_collateral_posted(MtM, threshold)
    calls = compute_margin_calls(collateral)

    EE = collateral.mean(axis=0)
    PFE = np.percentile(collateral, pfe_quantile * 100, axis=0)

    peak_per_path = collateral.max(axis=1)
    max_call_per_path = np.maximum(calls, 0).max(axis=1)

    return LiquidityMetrics(
        threshold_eur=threshold,
        collateral_posted=collateral,
        margin_calls=calls,
        EE_collateral=EE,
        PFE_collateral=PFE,
        Max_PFE_collateral=float(PFE.max()),
        Max_PFE_month=delivery_months[int(PFE.argmax())],
        peak_per_path=peak_per_path,
        max_call_per_path=max_call_per_path,
        peak_pfe95=float(np.percentile(peak_per_path, 95)),
        peak_pfe99=float(np.percentile(peak_per_path, 99)),
        max_call_pfe95=float(np.percentile(max_call_per_path, 95)),
        max_call_pfe99=float(np.percentile(max_call_per_path, 99)),
        fraction_no_posting=float((peak_per_path == 0).mean()),
        delivery_months=delivery_months,
        pfe_quantile=pfe_quantile,
    )


def threshold_sensitivity(
    MtM: np.ndarray,
    thresholds: np.ndarray,
    delivery_months: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Sweep through a range of thresholds and report the headline metrics.

    Parameters
    ----------
    thresholds : np.ndarray
        Threshold values to evaluate, in EUR.
    """
    rows = []
    for T in thresholds:
        m = compute_liquidity_metrics(MtM, delivery_months, T)
        rows.append({
            "threshold_eur": T,
            "frac_no_posting_pct": m.fraction_no_posting * 100,
            "mean_peak_eur": m.peak_per_path.mean(),
            "peak_pfe95_eur": m.peak_pfe95,
            "peak_pfe99_eur": m.peak_pfe99,
            "max_call_pfe95_eur": m.max_call_pfe95,
            "max_call_pfe99_eur": m.max_call_pfe99,
        })
    return pd.DataFrame(rows)
