# Technical Document — PPA Exposure Modelling

**Prototype for credit and liquidity exposure on long-dated Power Purchase Agreements**
*Reference instrument: DK1 zonal power, 5-year fixed-for-floating contract*

---

## 1. Executive summary

This document describes a market-consistent Monte Carlo framework for measuring credit and liquidity exposure on long-dated Power Purchase Agreements. The model decomposes the (shift-adjusted) log spot price into a deterministic drift calibrated to the market forward curve and a stochastic state variable calibrated to historical dynamics. Mark-to-market is computed analytically using the closed-form conditional forward, avoiding nested simulation. Exposure metrics include Expected Exposure (EE), Potential Future Exposure (PFE), Expected Positive Exposure (EPE), and CSA-adjusted liquidity metrics (peak collateral required, max single-period margin call, threshold sensitivity).

The framework is implemented as a Python package (`ppa_exposure`) with a Streamlit front-end and is structured to support production hand-off to an internal development team.

## 2. Contract and reference instrument

### 2.1 Power Purchase Agreement structure

A standard fixed-for-floating PPA. At each monthly settlement, the seller (Ørsted) receives a fixed price `F_fix` per MWh delivered and pays the floating market reference. Net cashflow at month *m*:

```
CF_m = (F_fix - S_m) × V × h_m
```

where `V` is baseload capacity in MW, `h_m` is the number of hours in month *m*, and `S_m` is the realised floating reference.

### 2.2 Base-case contract specification

| Parameter | Value | Rationale |
|---|---|---|
| Reference | DK1 day-ahead baseload | Western Denmark — the physical settlement zone for the majority of Danish renewable generation. |
| Tenor | 5 years (Jul-26 → Jun-31) | Representative long-dated PPA tenor for corporate offtake. |
| Settlement | Monthly | Standard. |
| Volume | 100 MW baseload | Mid-sized PPA. Volume per month = 100 × 24 × days_in_month. |
| Fixed price | At par | F_fix chosen so that MtM at inception = 0. |
| Discount rate | 2% continuous, flat | Approximate EUR short rate. Production implementation would use an OIS curve. |

## 3. Data

### 3.1 Historical spot

Hourly day-ahead prices for the DK1 price zone, 2021-01-01 to 2026-06-04, sourced from the Energinet Open Data Service. Two datasets concatenated:

- `Elspotprices` for 2021-01-01 to 2025-09-30 (hourly resolution).
- `DayAheadPrices` for 2025-10-01 onward (15-min resolution following the EU MTU transition, aggregated to hourly means).

Aggregated to daily and monthly baseload averages for downstream calibration.

### 3.2 Forward curve

18 EEX DK1 power futures contracts as of trading day 2026-06-04:
- 6 monthly contracts (Jul-26 → Dec-26)
- 7 quarterly contracts (Q3-26 → Q1-28)
- 5 yearly contracts (Cal-27 → Cal-31)

Internal consistency validated: the implied monthly curve reproduces each M/Q/Y settlement to within 0.05 EUR/MWh by construction (see §4).

### 3.3 Discounting

Flat 2% continuous discount rate applied throughout. A formal EUR OIS / IRS curve is identified as a production extension.

## 4. Forward curve construction

The 60-month forward curve is constructed via a **seasonal bootstrap**:

### 4.1 Step 1 — Multiplicative seasonal factors

For each complete calendar year *y* in the historical sample, compute the ratio of monthly to annual baseload averages: `R(y, m) = S̄(y, m) / S̄(y)`. The seasonal factor for calendar month *m* is then the cross-year average: `s(m) = mean_y R(y, m)`. Normalised so that the days-weighted average of `s(m)` equals one.

This produces a typical Nordic profile: winter peaks (s(Dec) ≈ 1.4), summer trough (s(Jul) ≈ 0.7).

### 4.2 Step 2 — Cascading disaggregation

Contracts are processed by granularity (Month → Quarter → Year). For each contract with settlement `F_C` covering months *M_C*:

```
level_C = (F_C × D_C − S_set) / Σ_{m ∈ M_C, unset} s(m) × d_m
F(0, m) = level_C × s(m)        ∀ m ∈ M_C \ already_set
```

where `D_C` is total days in the contract period, `d_m` is days in month *m*, and `S_set` is the days-weighted sum of forwards already pinned by more granular contracts.

This procedure guarantees by construction that the days-weighted average of monthly forwards over each contract's period equals its market settlement. Reconciliation is verified post-construction (`forward_curve.reconcile_contracts`).

## 5. Price model

### 5.1 Specification

The (shift-adjusted) log spot follows a mean-reverting Ornstein–Uhlenbeck process with deterministic drift:

```
X_t = ln(S_t + c) = α(t) + Y_t
dY_t = −κ Y_t dt + σ dW_t,    Y_0 = 0
```

- **`α(t)`** is the deterministic market-consistent drift.
- **`Y_t`** is the mean-reverting stochastic state.
- **`κ, σ`** are calibrated to historical dynamics.
- **`c`** is a constant log shift (default 100 EUR/MWh) accommodating negative day-ahead prices. The shift constant must satisfy `c > −min(S_t)` over the calibration window.

### 5.2 Calibration

The standard two-step procedure:

**Step A — OLS on log-shifted prices** to fit `θ` and harmonic seasonal coefficients on a Fourier basis with annual and semi-annual frequencies:

```
X_t = θ + A1 cos(ωt) + B1 sin(ωt) + A2 cos(2ωt) + B2 sin(2ωt) + ε_t,    ω = 2π / 365.25
```

**Step B — AR(1) on residuals** `ε_t`:

```
ε_{t+Δt} = b × ε_t + η_t
```

Continuous-time parameters back out as:

```
κ = −ln(b) / Δt
σ² = Var(η) × 2κ / (1 − exp(−2κΔt))
```

### 5.3 Calibration frequency — a critical choice

Daily calibration on DK1 spot yields **κ ≈ 150 per year (half-life ≈ 1.7 days)**: a single timescale that captures intra-week mean reversion but produces near-deterministic MtM at monthly resolution, because `exp(−κ × 1 month) ≈ 0`. The conditional forward then collapses to the unconditional market forward and the contract loses path-dependence.

**Resolution adopted in this prototype**: re-calibrate the same one-factor model on **monthly aggregated data**. The resulting κ (single-digit per year, half-life of months) captures the multi-month dynamics that matter for a monthly-settled PPA over a 5-year horizon. The intra-week dynamics observable in daily data are not material for monthly settlement and are deliberately not modelled.

**Limitation acknowledged**: a one-factor model can only capture a single timescale. The proper extension is the **Lucia–Schwartz two-factor model**, which separately captures a slow drift component and a fast noise component. See §10.

### 5.4 Market-consistency: closed-form α(t)

Under the model, `E_0[S_t] = exp(α(t) + Var[Y_t]/2) − c`. Setting `E_0[S_t] = F(0, t)` yields the analytical drift:

```
α(t) = ln(F(0, t) + c) − σ² / (4κ) × (1 − exp(−2κt))
```

This ensures that, by construction, the model exactly prices the observed market forward curve at every monthly horizon. Empirical validation (martingale check) on simulated paths confirms this within Monte Carlo standard error.

## 6. Monte Carlo simulation

### 6.1 Exact OU discretization

The OU process admits an exact discrete-time representation:

```
Y_{t+Δt} = Y_t × exp(−κΔt) + σ × sqrt((1 − exp(−2κΔt)) / (2κ)) × Z,    Z ~ N(0, 1)
```

This is preferred over Euler–Maruyama because it is exact for any step size, eliminating discretization bias.

### 6.2 Simulation settings

| Setting | Default | Notes |
|---|---|---|
| Number of paths | 10,000 | Halved for antithetic pairs. |
| Antithetic variates | Yes | Standard variance reduction; ensures mean Z = 0 exactly. |
| Random seed | 42 | Fixed for reproducibility. |
| Time grid | Month midpoints | Each `T_grid[k]` = month-start + 14 days. |

### 6.3 Validation: martingale check

Empirical mean of S_paths is compared against the market forward curve at every monthly horizon. Acceptance criterion: max |z-score| < 3, where `z = (E_emp[S_t] − F(0, t)) / (σ_emp / sqrt(N))`. The base-case calibration passes comfortably (max |z| typically < 2).

A second check verifies that the empirical variance of `Y_t` matches the theoretical OU variance `σ² / (2κ) × (1 − exp(−2κt))` within ~10% relative error.

## 7. Mark-to-market

### 7.1 Analytical formulation

The MtM at evaluation time `t_k` along simulated path *n* equals the present value of remaining cashflows, given the simulated state `Y_{t_k}^{(n)}`:

```
MtM_k^{(n)} = Σ_{m > k} DF(t_k, t_m) × (F_fix − F_{t_k}^{(n)}(m)) × V × h_m
```

The **conditional forward** is closed-form:

```
F_s(t) = E[S_t | Y_s] = exp(α(t) + Y_s × exp(−κ(t−s)) + Var[Y_t|Y_s] / 2) − c
       Var[Y_t|Y_s] = σ² / (2κ) × (1 − exp(−2κ(t−s)))
```

This eliminates nested Monte Carlo, giving MtM exactly under the model in O(N × M²) operations.

### 7.2 Fixed price at par

The par fixed price (MtM at inception = 0) is:

```
F_fix = Σ_m DF(0, t_m) × F(0, m) × h_m / Σ_m DF(0, t_m) × h_m
```

For the base-case run, `F_fix ≈ 79 EUR/MWh`, between the hours-weighted and discount-weighted curve averages.

### 7.3 Validation: MtM zero-mean property

By the tower property of conditional expectations and the market-consistency of α(t), `E_0[MtM_{t_k}] = 0` for all `t_k` at par. Empirical validation on simulated paths confirms this within Monte Carlo standard error.

## 8. Exposure metrics

### 8.1 Credit exposure

From Ørsted's perspective as fixed receiver, credit exposure arises when Ørsted is in-the-money (positive MtM) — the counterparty could default and Ørsted loses the unrealised gain.

```
EE_credit(t)  = E[max(MtM_t, 0)]
PFE_credit(t) = quantile_{95%}[max(MtM_t, 0)]
EPE_credit    = time-average of EE_credit
Max_PFE       = max_t PFE_credit(t)
```

### 8.2 Liquidity exposure (pre-CSA)

The mirror-image metric — when Ørsted is out-of-money, it potentially has to post collateral.

```
EE_liq(t)  = E[max(−MtM_t, 0)]
PFE_liq(t) = quantile_{95%}[max(−MtM_t, 0)]
EPE_liq    = time-average of EE_liq
```

### 8.3 Profile interpretation

For the DK1 base case (high-priced early months, low-priced later years), credit exposure dominates in level — the deterministic curve roll-off as high-price near-term months settle drives expected MtM into positive territory. Liquidity exposure shows a more complex shape with two peaks: an early peak when high-price winter months are still concentrated in the remaining contract, and a later peak when accumulated path variance is at maximum.

## 9. CSA collateral overlay

### 9.1 Mechanism

Under a symmetric Credit Support Annex with threshold `T`, collateral posted by Ørsted at time `t_k` along path *n* is:

```
Coll_k^{(n)} = max(−MtM_k^{(n)} − T, 0)
```

The threshold shifts the liquidity exposure downward by T uniformly. Below `T`, no collateral is posted (the threshold is the "unsecured tolerance").

**MTA (Minimum Transfer Amount)** is set to zero in the prototype. Standard deviation of monthly MtM moves (~10 M EUR) exceeds typical MTA values by an order of magnitude, so MTA does not materially affect monthly-resolution outputs.

### 9.2 Liquidity metrics

| Metric | Definition | Use |
|---|---|---|
| Peak collateral per path | max over time of Coll_k^{(n)} | Per-path maximum amount of cash tied up. Distribution → buffer sizing. |
| Peak collateral PFE 95% | 95th percentile of peak per path | Recommended liquidity buffer for the deal. |
| Max single margin call per path | max over time of (Coll_k − Coll_{k−1}) | Largest single-period cash outflow. Distribution → cash management. |
| Fraction with no posting | proportion of paths with peak = 0 | Probability the deal is fully secured-friendly under base assumptions. |

### 9.3 Threshold sensitivity — the trade-off

Sweeping threshold from 0 to 30 M EUR demonstrates the explicit credit-vs-liquidity trade-off that a Credit & Liquidity Risk team negotiates with counterparties:

- Higher T → more unsecured credit exposure, less liquidity burden.
- Lower T → tight collateral coverage, larger and more frequent posting requirements.

The optimal threshold depends on the counterparty's credit quality (justifies higher T for strong credits) and the marginal cost of liquidity to the issuer.

## 10. Limitations and extensions

### 10.1 Acknowledged simplifications

| Simplification | Impact | Production extension |
|---|---|---|
| Single-factor model | One timescale only; cannot simultaneously capture daily and multi-month dynamics. | **Lucia–Schwartz two-factor model** with separate fast and slow components. |
| Gaussian innovations | Tail too thin for power markets; PFE biased conservative-low. | Empirical or Student-t innovations; or jump-diffusion. |
| Baseload volume profile | Realistic for industrial offtake; misses renewable-shape effects. | As-generated profile with hourly capture-price effect. |
| Flat discount rate | Acceptable for indicative figures; production needs term structure. | EUR OIS / IRS curve at relevant tenors. |
| Monthly settlement granularity | Misses intra-month dynamics (intraday peaks, weekend effects). | Hourly granularity for short-tenor deals (capture-price modelling). |
| MTA = 0 | Negligible at monthly resolution. | Discrete MTA threshold simulation for daily-resolution exposure. |
| No counterparty default modelling | Pre-default exposure only. | Bivariate model with default intensity → CVA / FVA. |
| No wrong-way risk | Counterparty exposure assumed independent of credit quality. | Joint price–credit dynamics. |

### 10.2 Production hand-off considerations

A list of items required to take the prototype to production:

- **Data quality controls**: automated reconciliation of spot data; alerts on missing days; OIS curve sourcing.
- **Model governance**: documented model risk classification, periodic re-calibration cadence, back-testing framework.
- **Performance**: vectorise MtM across path × time tensor for portfolios of >100 contracts.
- **Aggregation**: portfolio-level exposure netting under master agreement.
- **Integration**: feed exposure metrics into existing limit framework; feed liquidity metrics into treasury cashflow projection.
- **User interface**: production UI on existing internal BI stack (Power BI / Tableau / proprietary).

## 11. Reproducibility

All Monte Carlo runs use a fixed seed (default 42). Input data is versioned in `data/` with filenames carrying the snapshot trading day. Calibrated parameters are persisted in `data/model_params.json` with full diagnostic information for inspection.

A test suite (`tests/`) covers:
- AR(1) → OU parameter recovery on synthetic data.
- Seasonal OLS coefficient recovery on planted harmonics.
- Martingale property of the Monte Carlo engine.
- Par condition of the analytical MtM at inception.
- Antithetic variance reduction.

Tests pass via `pytest tests/`.

## 12. References

- Schwartz, E.S. (1997). *The Stochastic Behaviour of Commodity Prices: Implications for Valuation and Hedging.* Journal of Finance, 52, 923–973.
- Lucia, J.J. and Schwartz, E.S. (2002). *Electricity Prices and Power Derivatives: Evidence from the Nordic Power Exchange.* Review of Derivatives Research, 5, 5–50.
- Eydeland, A. and Wolyniec, K. (2003). *Energy and Power Risk Management: New Developments in Modeling, Pricing, and Hedging.* Wiley.
- Bjerksund, P., Rasmussen, H. and Stensland, G. (2010). *Valuation and risk management in the Norwegian electricity market.* Energy, Natural Resources and Environmental Economics.

---

*This document accompanies the open-source `ppa-exposure-modelling` repository. Full code, tests, and an interactive Streamlit front-end are provided in the same package.*
