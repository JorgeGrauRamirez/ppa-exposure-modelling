"""
PPA Exposure Modelling — market-consistent Monte Carlo framework for
credit and liquidity exposure on long-dated Power Purchase Agreements.
"""

__version__ = "0.1.0"
__author__ = "Jorge Grau Ramírez"

from ppa_exposure.config import Config
from ppa_exposure.calibration import calibrate_schwartz
from ppa_exposure.forward_curve import build_monthly_forward_curve
from ppa_exposure.monte_carlo import simulate
from ppa_exposure.exposure import (
    compute_par_fixed_price,
    compute_mtm_matrix,
    compute_exposure_metrics,
)
from ppa_exposure.collateral import (
    compute_collateral_posted,
    compute_liquidity_metrics,
    threshold_sensitivity,
)

__all__ = [
    "Config",
    "calibrate_schwartz",
    "build_monthly_forward_curve",
    "simulate",
    "compute_par_fixed_price",
    "compute_mtm_matrix",
    "compute_exposure_metrics",
    "compute_collateral_posted",
    "compute_liquidity_metrics",
    "threshold_sensitivity",
]
