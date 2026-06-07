"""
Centralised configuration for the PPA exposure modelling framework.

All assumptions, parameters, and toggles flow through this module. Default
values reflect the base-case prototype against DK1 power. Override by
instantiating `Config(...)` with custom values or by loading from JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class DealSpec:
    """Power Purchase Agreement contract specification."""

    volume_mw: float = 100.0
    """Baseload capacity in MW. Volume per month = volume_mw * 24 * days_in_month."""

    start_date: str = "2026-07-01"
    """First delivery month (YYYY-MM-DD, day 1 of month)."""

    end_date: str = "2031-06-30"
    """Last delivery month end."""

    fixed_price_eur_mwh: float | None = None
    """Fixed price in EUR/MWh. If None, computed at par from the forward curve."""

    perspective: str = "seller"
    """'seller' = Ørsted receives fixed, pays floating spot. Inverts MtM sign for 'buyer'."""


@dataclass
class ModelParams:
    """Schwartz one-factor calibrated parameters."""

    kappa_per_year: float = 8.0
    """Mean-reversion speed in 1/year. Indicative monthly-calibrated value."""

    sigma_annualized: float = 0.55
    """Volatility of the log-shifted price, annualised."""

    shift_constant_c: float = 100.0
    """Log shift c such that X_t = ln(S_t + c). Must satisfy c > -min(S_t)."""


@dataclass
class CsaSpec:
    """Credit Support Annex parameters."""

    threshold_eur: float = 5_000_000.0
    """Symmetric threshold for collateral posting."""

    mta_eur: float = 0.0
    """Minimum Transfer Amount. Default 0 (not modelled at monthly resolution)."""

    independent_amount_eur: float = 0.0
    """Initial margin / independent amount. Default 0 (bilateral OTC)."""


@dataclass
class MonteCarloSpec:
    """Monte Carlo simulation settings."""

    n_paths: int = 10_000
    """Number of simulated paths. Half are antithetic if `use_antithetic=True`."""

    use_antithetic: bool = True
    """Antithetic variates for variance reduction. n_paths must be even."""

    seed: int = 42
    """Random seed for reproducibility."""

    discount_rate: float = 0.02
    """Continuous discount rate per year (flat curve assumption)."""

    pfe_quantile: float = 0.95
    """Confidence level for Potential Future Exposure metric (0 < q < 1)."""


@dataclass
class DataPaths:
    """Paths to input and output data files."""

    data_dir: str = "data"
    spot_daily_csv: str = "spot_dk1_daily.csv"
    spot_monthly_csv: str = "spot_dk1_monthly.csv"
    forwards_csv: str = "forwards_dk1_eex.csv"
    monthly_curve_csv: str = "monthly_forward_curve_dk1.csv"
    model_params_json: str = "model_params.json"

    def __post_init__(self) -> None:
        self.data_dir = str(Path(self.data_dir))


@dataclass
class Config:
    """Top-level configuration aggregating all sub-specs."""

    deal: DealSpec = field(default_factory=DealSpec)
    model: ModelParams = field(default_factory=ModelParams)
    csa: CsaSpec = field(default_factory=CsaSpec)
    mc: MonteCarloSpec = field(default_factory=MonteCarloSpec)
    paths: DataPaths = field(default_factory=DataPaths)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        return cls(
            deal=DealSpec(**raw["deal"]),
            model=ModelParams(**raw["model"]),
            csa=CsaSpec(**raw["csa"]),
            mc=MonteCarloSpec(**raw["mc"]),
            paths=DataPaths(**raw["paths"]),
        )


def default_config() -> Config:
    """Return the base-case configuration."""
    return Config()
