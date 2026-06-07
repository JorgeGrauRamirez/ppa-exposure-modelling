# PPA Exposure Modelling

A market-consistent Monte Carlo framework for measuring credit and liquidity exposure on long-dated Power Purchase Agreements (PPAs). Prototype developed against DK1 (Western Denmark) zonal power as a representative case for European renewable-generator PPAs.

## What this does

Given a fixed-for-floating Power Purchase Agreement, this tool answers two questions a Credit and Liquidity Risk team must size:

1. **Credit exposure**: how much could the counterparty owe us, if they default?
2. **Liquidity exposure**: how much collateral could we have to post under a CSA, even while remaining solvent?

Both questions are answered through the same Monte Carlo engine. Outputs include Expected Exposure (EE), Potential Future Exposure (PFE), peak collateral required, and maximum single-period margin call distributions, with full sensitivity to CSA threshold.

## Architecture

```
ppa-exposure-modelling/
├── ppa_exposure/             # Python package — the engine
│   ├── config.py             # Centralised configuration
│   ├── data_io.py            # Data sourcing (Energinet API, EEX forwards)
│   ├── calibration.py        # Schwartz one-factor calibration (OLS + AR1)
│   ├── forward_curve.py      # Seasonal bootstrap M/Q/Y → monthly
│   ├── price_model.py        # Market-consistent OU dynamics
│   ├── monte_carlo.py        # Simulation engine with antithetic variates
│   ├── exposure.py           # Analytical MtM + EE/PFE/EPE metrics
│   ├── collateral.py         # CSA overlay + liquidity metrics
│   └── validation.py         # Martingale check, reconciliation, diagnostics
│
├── notebooks/                # Step-by-step exploration and validation
│
├── app/                      # Streamlit front-end
│   └── streamlit_app.py
│
├── tests/                    # Unit tests
│
├── data/                     # Sourced data (CSVs, JSON, NPZ)
│
└── docs/
    └── technical_document.md # Methodology, assumptions, limitations
```

## Quick start

### Install

```bash
git clone https://github.com/<your-user>/ppa-exposure-modelling.git
cd ppa-exposure-modelling
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Sourcing data

1. **Historical spot DK1** (Energinet, free public API):
   ```bash
   python -m ppa_exposure.data_io fetch-spot --start 2021-01-01 --end 2026-06-04 --area DK1
   ```
   This pulls hourly day-ahead prices via the Energinet Open Data API, stitching the legacy `Elspotprices` dataset (until 2025-09-30) with the current `DayAheadPrices` dataset (15-min resolution, aggregated to hourly). Outputs `data/spot_dk1_2021-01-01_2026-06-04_{hourly,daily,monthly}.csv`.

2. **EEX DK1 forward curve snapshot** (manual, 15 minutes on `eex.com`):
   - Navigate to *Market Data → Power → Futures*, set Area to **DK1**, and for each maturity (Month, Quarter, Year) record the settlement price for each available delivery period.
   - Save as `data/forwards_dk1_eex_<YYYYMMDD>.csv` with columns `contract_type, delivery_label, delivery_start, delivery_end, settlement_price_eur_mwh, trading_day, source, gross_open_interest_lots`.

### Run end to end

```bash
# 1. Build monthly forward curve via seasonal bootstrap
python -m ppa_exposure.forward_curve build-monthly \
    --spot data/spot_dk1_..._daily.csv \
    --forwards data/forwards_dk1_eex_20260604.csv

# 2. Calibrate Schwartz model
python -m ppa_exposure.calibration calibrate \
    --spot data/spot_dk1_..._monthly.csv \
    --shift 100 --output data/model_params.json

# 3. Run Streamlit app
streamlit run app/streamlit_app.py
```

The Streamlit app loads the data above and lets the reviewer change deal terms (volume, fixed price, CSA threshold) and model parameters interactively, with all exposure metrics recomputed on the fly.

### Run notebooks

```bash
jupyter notebook notebooks/
```

The notebooks reproduce the data sourcing, calibration, forward-curve bootstrap, and validation step by step. They import from the `ppa_exposure` package; modifying parameters there propagates everywhere.

## Methodology summary

The model decomposes the (shift-adjusted) log spot as:

```
ln(S_t + c) = alpha(t) + Y_t
dY_t = -kappa * Y_t * dt + sigma * dW_t,   Y_0 = 0
```

where:
- **`kappa, sigma`** are calibrated to **historical** DK1 spot dynamics (Schwartz one-factor with annual and semi-annual seasonal harmonics, AR(1) on residuals).
- **`alpha(t)`** is determined analytically so that `E[S_t] = F(0, t)` matches the **market-observed** forward curve at every monthly horizon.
- **`c`** is a shift constant (default 100 EUR/MWh) accommodating negative day-ahead prices in the log transform.

This split — dynamics from history, levels from market — is standard practice in counterparty credit risk modelling. The market-consistent calibration is validated via a martingale check: empirical mean of N simulated spot paths must reproduce the forward curve within Monte Carlo standard error at all 60 monthly horizons.

MtM at each (path, evaluation time) is computed analytically using the closed-form conditional expectation of the OU process. No nested simulation. Exposure metrics (EE, PFE, EPE) and CSA-adjusted liquidity metrics (peak collateral, max margin call, threshold sensitivity) are then derived.

Full methodology, assumptions, limitations, and suggested extensions in `docs/technical_document.md`.

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Underlying | DK1 zonal price | Physical settlement zone for Danish renewable generation; consistent calibration on Energinet spot + EEX DK1 zonal futures. |
| Calibration frequency | Monthly | Matches PPA settlement frequency; captures the multi-month timescale relevant for monthly exposure. Daily-calibrated kappa is too fast (~1.7-day half-life) and produces near-deterministic MtM. |
| Forward curve | Seasonal bootstrap from M/Q/Y EEX contracts | Preserves market-observed levels; uses historical seasonality only for *shape* within contract periods. Reconciliation guaranteed by construction. |
| Distribution | Gaussian innovations | Documented limitation. Power markets are leptokurtic; PFE is biased conservative-low at the tail. Lucia-Schwartz two-factor or jump-diffusion are documented extensions. |
| Volume | Baseload (100 MW constant) | Most common PPA structure for industrial/utility offtake. As-generated profile (with capture-price effect) is identified as a domain-specific extension. |
| Discount | Flat 2% continuous | Approximate EUR short rate. In production an OIS curve would replace this. |

## Testing

```bash
pytest tests/
```

Core unit tests cover martingale consistency, M/Q/Y reconciliation, and MtM analytical formula.

## Reproducibility

All Monte Carlo runs use a fixed seed (configurable in `ppa_exposure/config.py`). Input data is versioned in `data/` with filenames including the snapshot trading day. Calibrated parameters are persisted in `data/model_params.json` for inspection.

## Author

Jorge Grau Ramírez — MSc Mathematical Modelling and Computation (DTU) / Quantitative Methods in Finance (CBS).

## License

This is a prototype for educational and recruitment purposes. No license granted for production use.
