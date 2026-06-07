"""
Forward curve construction via seasonal bootstrap.

Given observed M/Q/Y EEX forward contracts and a historical spot series,
constructs a 60-month forward curve F(0, m) that exactly reproduces the
M/Q/Y settlement prices when re-aggregated.

Methodology:
1. Estimate multiplicative monthly seasonal factors s(m) from historical
   spot, using the ratio of monthly-to-annual averages across complete
   calendar years (robust to level shifts like 2022 crisis).
2. Normalise so that the days-weighted mean of s(m) equals 1.
3. Process contracts in order of granularity (Month -> Quarter -> Year).
   For each contract, set the unset months within its period via:
       level_C = (F_C * D_C - S_set) / sum(s(m) * d_m for m in unset)
       F(0, m) = level_C * s(m)
   This guarantees by construction that the days-weighted average of
   monthly forwards over the contract's period equals F_C exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def compute_seasonal_factors(spot_daily: pd.DataFrame) -> pd.Series:
    """
    Estimate multiplicative monthly seasonal factors from historical spot.

    Uses only complete calendar years (with all 12 months present) to avoid
    bias from partial periods. For each year, computes monthly-mean /
    annual-mean ratios. Averages ratios across years for each calendar month.
    Normalises so that days-weighted mean equals 1.

    Parameters
    ----------
    spot_daily : pd.DataFrame
        Must have columns 'Date' and 'SpotPriceEUR'.

    Returns
    -------
    pd.Series indexed 1..12 with multiplicative seasonal factors.
    """
    df = spot_daily.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    months_per_year = df.groupby("Year")["Month"].nunique()
    complete_years = months_per_year[months_per_year == 12].index.tolist()
    df = df[df["Year"].isin(complete_years)]

    monthly_avg = df.groupby(["Year", "Month"])["SpotPriceEUR"].mean().rename("MonthAvg")
    yearly_avg = df.groupby("Year")["SpotPriceEUR"].mean().rename("YearAvg")

    ratios = monthly_avg.reset_index().merge(yearly_avg.reset_index(), on="Year")
    ratios["Ratio"] = ratios["MonthAvg"] / ratios["YearAvg"]

    raw = ratios.groupby("Month")["Ratio"].mean()

    days_in_month = pd.Series(
        [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
        index=range(1, 13),
    )
    normalizer = (raw * days_in_month).sum() / days_in_month.sum()
    factors = (raw / normalizer).rename("s(m)")
    return factors


def _apply_contract(
    curve: pd.DataFrame,
    contract_row: pd.Series,
    source_label: str,
) -> None:
    """
    Assign forwards in `curve` to months not yet covered by a more granular
    contract, preserving the contract's days-weighted mean settlement price.
    In-place modification of `curve`.
    """
    F_C = contract_row["settlement_price_eur_mwh"]
    period_start = contract_row["delivery_start"]
    period_end = contract_row["delivery_end"]

    mask = (curve["month_start"] >= period_start) & (curve["month_end"] <= period_end)
    if not mask.any():
        return

    set_mask = mask & curve["forward"].notna()
    unset_mask = mask & curve["forward"].isna()
    if not unset_mask.any():
        return

    D_C = curve.loc[mask, "days_in_month"].sum()
    S_set = (
        curve.loc[set_mask, "forward"] * curve.loc[set_mask, "days_in_month"]
    ).sum()
    weighted_s_unset = (
        curve.loc[unset_mask, "s_m"] * curve.loc[unset_mask, "days_in_month"]
    ).sum()

    level = (F_C * D_C - S_set) / weighted_s_unset
    curve.loc[unset_mask, "forward"] = level * curve.loc[unset_mask, "s_m"]
    curve.loc[unset_mask, "source"] = source_label


def build_monthly_forward_curve(
    spot_daily: pd.DataFrame,
    forwards: pd.DataFrame,
    target_start: str | pd.Timestamp,
    n_months: int = 60,
) -> pd.DataFrame:
    """
    Build the monthly forward curve via seasonal bootstrap.

    Parameters
    ----------
    spot_daily : pd.DataFrame
        Historical daily spot for seasonal-factor estimation.
    forwards : pd.DataFrame
        Observed M/Q/Y contracts with columns:
        contract_type, delivery_start, delivery_end, settlement_price_eur_mwh.
    target_start : str | Timestamp
        First month of the target curve (e.g. '2026-07-01').
    n_months : int
        Number of months to construct (default 60 for a 5-year PPA).

    Returns
    -------
    pd.DataFrame with columns:
        month_start, month_end, days_in_month, s_m, forward, source.
    """
    factors = compute_seasonal_factors(spot_daily)

    target_months = pd.date_range(target_start, periods=n_months, freq="MS")
    curve = pd.DataFrame({"month_start": target_months})
    curve["month_end"] = curve["month_start"] + pd.offsets.MonthEnd(0)
    curve["days_in_month"] = (curve["month_end"] - curve["month_start"]).dt.days + 1
    curve["s_m"] = curve["month_start"].dt.month.map(factors)
    curve["forward"] = np.nan
    curve["source"] = ""

    # Process in cascade: M -> Q -> Y
    for contract_type in ["M", "Q", "Y"]:
        for _, row in forwards[forwards["contract_type"] == contract_type].iterrows():
            _apply_contract(curve, row, f"{contract_type}:{row['delivery_label']}")

    if curve["forward"].isna().any():
        missing = curve.loc[curve["forward"].isna(), "month_start"].tolist()
        raise RuntimeError(f"Months remain unassigned: {missing}")

    return curve


def reconcile_contracts(
    curve: pd.DataFrame,
    forwards: pd.DataFrame,
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """
    Verify that the monthly curve reproduces each M/Q/Y settlement.

    Returns a DataFrame with one row per contract showing market vs implied
    prices and the difference. Raises if any |diff| exceeds `tolerance`.
    """
    checks: list[dict] = []
    for _, row in forwards.iterrows():
        mask = (
            (curve["month_start"] >= row["delivery_start"]) &
            (curve["month_end"] <= row["delivery_end"])
        )
        if not mask.any():
            continue
        covered = curve[mask]
        implied = (
            (covered["forward"] * covered["days_in_month"]).sum()
            / covered["days_in_month"].sum()
        )
        diff = implied - row["settlement_price_eur_mwh"]
        checks.append({
            "contract_type": row["contract_type"],
            "label": row["delivery_label"],
            "market": row["settlement_price_eur_mwh"],
            "implied": implied,
            "diff": diff,
            "months_covered": int(mask.sum()),
        })

    check_df = pd.DataFrame(checks)
    max_abs = check_df["diff"].abs().max()
    if max_abs > tolerance:
        raise AssertionError(
            f"Reconciliation failed: max |diff| = {max_abs:.6f} > {tolerance}"
        )
    return check_df


def save_monthly_curve(curve: pd.DataFrame, path: str | Path) -> None:
    """Persist the monthly curve to CSV with the canonical schema."""
    out = curve[["month_start", "forward", "source"]].copy()
    out.columns = ["delivery_month", "forward_eur_mwh", "source_contract"]
    out["delivery_month"] = out["delivery_month"].dt.strftime("%Y-%m")
    out.to_csv(path, index=False)
