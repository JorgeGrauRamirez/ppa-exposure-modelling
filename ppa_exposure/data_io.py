"""
Data sourcing and loading utilities.

This module handles:
- Fetching historical day-ahead spot prices from the Energinet Open Data API,
  stitching the legacy `Elspotprices` dataset (until 2025-09-30) with the
  current `DayAheadPrices` dataset (15-min resolution, aggregated to hourly).
- Loading EEX forward contracts and the constructed monthly forward curve.
- Aggregation utilities (hourly -> daily, daily -> monthly).
"""

from __future__ import annotations

import json
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests


API_BASE = "https://api.energidataservice.dk/dataset"
LEGACY_CUTOFF = pd.Timestamp("2025-09-30")

DEFAULT_HEADERS = {
    "User-Agent": "PPA-Exposure-Modelling/0.1",
    "Accept": "application/json",
}


# --------------------------------------------------------------------------- #
# Energinet API client
# --------------------------------------------------------------------------- #

def _get_with_retry(
    url: str,
    params: dict,
    headers: dict = DEFAULT_HEADERS,
    max_retries: int = 5,
    timeout: int = 60,
) -> requests.Response:
    """GET with exponential backoff on 429 / 5xx errors."""
    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        if response.status_code == 429:
            wait = 10 * (2 ** attempt)
            print(f"  Rate limited. Waiting {wait}s before retry {attempt+1}/{max_retries}...")
            time.sleep(wait)
            continue
        if response.status_code >= 500:
            wait = 5 * (2 ** attempt)
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response
    raise RuntimeError(f"Max retries ({max_retries}) exceeded for {url}")


def _fetch_paginated(
    dataset: str,
    start: str,
    end: str,
    price_area: str,
    time_col: str,
    page_size: int = 50_000,
) -> pd.DataFrame:
    """Paginated fetch from a single Energinet dataset."""
    url = f"{API_BASE}/{dataset}"
    all_records: list[dict] = []
    offset = 0

    while True:
        params = {
            "offset": offset,
            "limit": page_size,
            "start": start,
            "end": end,
            "filter": json.dumps({"PriceArea": price_area}, separators=(",", ":")),
            "sort": f"{time_col} ASC",
        }
        response = _get_with_retry(url, params)
        records = response.json().get("records", [])
        if not records:
            break
        all_records.extend(records)
        if len(records) < page_size:
            break
        offset += page_size
        time.sleep(1.0)

    return pd.DataFrame(all_records)


def fetch_spot_prices(
    start_date: str,
    end_date: str,
    price_area: str = "DK1",
) -> pd.DataFrame:
    """
    Fetch hourly day-ahead spot prices, stitching legacy Elspotprices and
    current DayAheadPrices datasets.

    Elspotprices was discontinued at 2025-09-30. Its successor DayAheadPrices
    uses 15-min resolution (EU MTU transition), which we aggregate back to
    hourly means for consistency across the full historical window.

    Parameters
    ----------
    start_date, end_date : str
        Inclusive bounds in 'YYYY-MM-DD' format.
    price_area : str
        Energinet price area code ('DK1', 'DK2', etc.).

    Returns
    -------
    pd.DataFrame with columns: HourUTC, HourDK, PriceArea, SpotPriceEUR, SpotPriceDKK.
    """
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    parts: list[pd.DataFrame] = []

    # Legacy: Elspotprices (hourly)
    if start <= LEGACY_CUTOFF:
        end_legacy = min(end, LEGACY_CUTOFF)
        df = _fetch_paginated(
            "Elspotprices",
            str(start.date()),
            str(end_legacy.date() + pd.Timedelta(days=1)),
            price_area,
            time_col="HourUTC",
        )
        if not df.empty:
            df["HourUTC"] = pd.to_datetime(df["HourUTC"])
            df["HourDK"] = pd.to_datetime(df["HourDK"])
            df = df[["HourUTC", "HourDK", "PriceArea", "SpotPriceEUR", "SpotPriceDKK"]]
            parts.append(df)

    # Current: DayAheadPrices (15-min, aggregated to hourly)
    if end > LEGACY_CUTOFF:
        start_current = max(start, LEGACY_CUTOFF + pd.Timedelta(days=1))
        df = _fetch_paginated(
            "DayAheadPrices",
            str(start_current.date()),
            str(end.date() + pd.Timedelta(days=1)),
            price_area,
            time_col="TimeUTC",
        )
        if not df.empty:
            df["TimeUTC"] = pd.to_datetime(df["TimeUTC"])
            df["TimeDK"] = pd.to_datetime(df["TimeDK"])
            df["HourUTC"] = df["TimeUTC"].dt.floor("h")
            df["HourDK"] = df["TimeDK"].dt.floor("h")
            hourly = (
                df.groupby(["HourUTC", "HourDK", "PriceArea"], as_index=False)
                  .agg(
                      SpotPriceEUR=("DayAheadPriceEUR", "mean"),
                      SpotPriceDKK=("DayAheadPriceDKK", "mean"),
                  )
            )
            parts.append(hourly)

    if not parts:
        raise RuntimeError(f"No data for {price_area} {start_date} -> {end_date}")

    combined = (
        pd.concat(parts, ignore_index=True)
          .drop_duplicates(subset=["HourUTC", "PriceArea"])
          .sort_values("HourUTC")
          .reset_index(drop=True)
    )
    return combined


# --------------------------------------------------------------------------- #
# Aggregation utilities
# --------------------------------------------------------------------------- #

def aggregate_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly day-ahead prices to daily baseload (24-hour mean)."""
    daily = (
        hourly
        .assign(Date=hourly["HourDK"].dt.date)
        .groupby("Date", as_index=False)
        .agg(
            SpotPriceEUR=("SpotPriceEUR", "mean"),
            SpotPriceDKK=("SpotPriceDKK", "mean"),
            HoursAvailable=("HourUTC", "count"),
        )
    )
    daily["Date"] = pd.to_datetime(daily["Date"])
    return daily


def aggregate_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily baseload to monthly mean."""
    monthly = (
        daily
        .assign(YearMonth=daily["Date"].dt.to_period("M"))
        .groupby("YearMonth", as_index=False)
        .agg(
            SpotPriceEUR=("SpotPriceEUR", "mean"),
            DaysInMonth=("Date", "count"),
        )
    )
    monthly["YearMonth"] = monthly["YearMonth"].dt.to_timestamp()
    return monthly


# --------------------------------------------------------------------------- #
# CSV loaders
# --------------------------------------------------------------------------- #

def load_spot_daily(path: str | Path) -> pd.DataFrame:
    """Load a previously-saved daily spot CSV."""
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_spot_monthly(path: str | Path) -> pd.DataFrame:
    """Load a previously-saved monthly spot CSV."""
    df = pd.read_csv(path, parse_dates=["YearMonth"])
    return df.sort_values("YearMonth").reset_index(drop=True)


def load_forwards(path: str | Path) -> pd.DataFrame:
    """Load the EEX forward contracts CSV (M, Q, Y)."""
    df = pd.read_csv(path, parse_dates=["delivery_start", "delivery_end", "trading_day"])
    return df


def load_monthly_curve(path: str | Path) -> pd.DataFrame:
    """Load the constructed monthly forward curve."""
    df = pd.read_csv(path)
    df["delivery_month"] = pd.to_datetime(df["delivery_month"], format="%Y-%m")
    return df.sort_values("delivery_month").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def _cli_fetch_spot() -> None:
    """Command-line entry: `python -m ppa_exposure.data_io fetch-spot ...`"""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch historical spot prices from Energinet.")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--area", default="DK1", help="Price area (default DK1)")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.area.lower()}_{args.start}_{args.end}"

    print(f"Fetching {args.area} spot {args.start} -> {args.end}...")
    hourly = fetch_spot_prices(args.start, args.end, args.area)
    daily = aggregate_to_daily(hourly)
    monthly = aggregate_to_monthly(daily)

    hourly.to_csv(out_dir / f"spot_{tag}_hourly.csv", index=False)
    daily.to_csv(out_dir / f"spot_{tag}_daily.csv", index=False)
    monthly.to_csv(out_dir / f"spot_{tag}_monthly.csv", index=False)
    print(f"Saved to {out_dir}/spot_{tag}_*.csv ({len(hourly):,} hourly records)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fetch-spot":
        sys.argv = sys.argv[:1] + sys.argv[2:]
        _cli_fetch_spot()
    else:
        print("Usage: python -m ppa_exposure.data_io fetch-spot --start YYYY-MM-DD --end YYYY-MM-DD [--area DK1]")
