# Technical Document — PPA Exposure Modelling

**Prototype for credit and liquidity exposure on long-dated Power Purchase Agreements**
*Reference instrument: DK1 zonal power, 5-year fixed-for-floating contract*

---

## Executive summary

This document specifies a market-consistent Monte Carlo framework for measuring credit and liquidity exposure on long-dated Power Purchase Agreements (PPAs). The model is calibrated against the DK1 (Western Denmark) zonal power market and a 5-year fixed-for-floating PPA with 100 MW baseload volume.

**Headline figures (base case)**:

| Metric | Value | % of notional |
|---|---|---|
| Contract notional | **347 M EUR** | 100% |
| Fixed price (at par) | **79.15 EUR/MWh** | — |
| Maximum credit PFE (95%) | **38.4 M EUR** in Dec-27 | 11.1% |
| Maximum liquidity PFE (95%), no CSA | **17.8 M EUR** in Aug-26 | 5.1% |
| Model-implied liquidity buffer (CSA T = 5M) | **27.6 M EUR** (peak per path PFE 95%) | 7.9% |
| Largest plausible single-month margin call (T = 5M) | **19.7 M EUR** (PFE 95%) | 5.7% |
| Probability of zero collateral posting (T = 5M) | **8.5%** of paths | — |

**Key findings**:

1. **Credit exposure dominates liquidity exposure by ~2:1**, opposite to the intuitive log-normal asymmetry argument. This is driven by the *deterministic roll-off* of the backwardated DK1 forward curve, not by stochastic skew — see §8.3.
2. **Liquidity exposure is bimodal**, peaking early (Aug-26, when high-price winter months still remain) and again in 2030 (when accumulated path variance is maximal). The early peak is the structural risk for the Credit & Liquidity Risk team.
3. **CSA threshold sensitivity is roughly linear**: each additional 1 M EUR of threshold removes ~1 M EUR of peak collateral PFE, up to T ≈ 30 M EUR where over 90% of paths require no posting. This quantifies the credit-vs-liquidity trade-off for counterparty negotiation.
4. The model is **market-consistent by construction** — the empirical mean of simulated paths reproduces the EEX forward curve at every monthly horizon (martingale check passes within Monte Carlo standard error).

The framework is implemented as a Python package (`ppa_exposure`) with a Streamlit front-end and is structured for production hand-off.

---

## 1. Contract and reference instrument

### 1.1 Power Purchase Agreement structure

A standard fixed-for-floating PPA. At each monthly settlement, the seller (Ørsted) receives a fixed price `F_fix` per MWh delivered and pays the floating market reference. Net cashflow at month *m*:

```
CF_m = (F_fix − S_m) × V × h_m
```

where `V` is baseload capacity in MW, `h_m` is the number of hours in month *m*, and `S_m` is the realised floating reference for that month.

### 1.2 Base-case contract specification

| Parameter | Value | Rationale |
|---|---|---|
| Reference index | DK1 day-ahead baseload | Western Denmark — physical settlement zone for the majority of Danish renewable generation. Consistent calibration on Energinet spot + EEX DK1 zonal futures. |
| Tenor | 5 years (Jul-26 → Jun-31) | Representative long-dated PPA tenor for corporate offtake. Matches the available EEX yearly forward grid (Cal-27 to Cal-31) without requiring extrapolation. |
| Settlement | Monthly | Standard market convention for utility-scale PPAs. |
| Volume | 100 MW baseload | Mid-sized PPA; representative of corporate offtake (Microsoft/Google deals are 200-500 MW; small corporate deals 10-50 MW). 100 MW × 24 h × ~30 days ≈ 72 GWh per month. |
| Fixed price | At par (79.15 EUR/MWh) | MtM = 0 at inception. Standard market convention for fair-value pricing. |
| Discount rate | 2% continuous, flat | Approximate EUR short rate. Production implementation would use a EUR OIS curve. |
| Perspective | Ørsted as seller / fixed receiver | Seller of physical power, receives fixed. Mirrors Ørsted's actual position as a renewable generator. |

> **All numerical results in this document refer to the base-case run specified above.** The accompanying Streamlit application permits interactive exploration of alternative parameter configurations (volume, threshold, model parameters, simulation settings). For reproducibility, the model artefacts (random seed 42, calibrated parameters persisted in `data/model_params.json`, monthly forward curve in `data/monthly_forward_curve_dk1_20260604.csv`) deterministically recreate the exact figures shown in this document.

---

## 2. Data

### 2.1 Historical spot

Hourly day-ahead prices for the DK1 price zone, **2021-01-01 to 2026-06-04**, sourced from the Energinet Open Data Service. The five-year window deliberately spans the pre-crisis, crisis (2022), and post-crisis regimes, giving a seasonal pattern that averages across price levels and a calibration window that includes meaningful realised volatility. Two datasets are concatenated:

- `Elspotprices` for 2021-01-01 to 2025-09-30 (hourly resolution).
- `DayAheadPrices` for 2025-10-01 onward (15-min resolution following the EU MTU transition, aggregated to hourly means).

Both datasets measure the same underlying day-ahead clearing price; the EU MTU transition is purely a reporting-granularity change. Data is then aggregated to daily and monthly baseload averages for downstream calibration. Stitching, retry logic, and rate-limit handling are implemented in `ppa_exposure/data_io.py`.

### 2.2 Forward curve

18 EEX DK1 power futures contracts collected manually from `eex.com` as of trading day **2026-06-04**:

- 6 monthly contracts (Jul-26 → Dec-26)
- 7 quarterly contracts (Q3-26 → Q1-28)
- 5 yearly contracts (Cal-27 → Cal-31)

A single trading-day snapshot is used (rather than a multi-day average) to maintain temporal consistency with the discount rate and to follow market convention for valuation. Internal consistency was validated: the days-weighted average of overlapping contracts (e.g. Jul+Aug+Sep 2026 vs Q3-26) reconciles to **within 0.05 EUR/MWh tolerance**. Manual collection is documented as a productivization gap; an EEX data subscription would be required for automation.

### 2.3 Discounting

A flat 2% continuous discount rate is applied throughout. This is an approximate EUR short rate as of 2026-Q2 and is acceptable for indicative figures. A formal EUR OIS / IRS curve at relevant tenors is identified as a production extension.

---

## 3. Forward curve construction

The 60-month forward curve is constructed via a **seasonal bootstrap** that decomposes each contract's market price into a "level" (set by the contract) and a multiplicative "shape" (set by historical seasonality).

### 3.1 Step 1 — Multiplicative seasonal factors

For each complete calendar year *y* in the historical sample, compute the ratio of monthly to annual baseload averages: `R(y, m) = S̄(y, m) / S̄(y)`. The seasonal factor for calendar month *m* is then the cross-year average: `s(m) = mean_y R(y, m)`. Only complete calendar years are used to avoid bias from partial periods. Multiplicative (not additive) seasonality is the standard choice for power markets, which scale proportionally with price levels (a doubling of gas prices doubles the absolute seasonal swing). Factors are normalised so that the days-weighted average of `s(m)` equals one, ensuring the seasonal carries no level information.

This produces a typical Nordic profile: winter peaks (s(Dec) ≈ 1.3-1.4), summer trough (s(Jul) ≈ 0.7-0.8).

### 3.2 Step 2 — Cascading disaggregation

Contracts are processed by **increasing granularity (Month → Quarter → Year)**. The Month-first ordering is essential: more granular contracts pin specific months, and the broader contracts (Quarters, Years) then fill the months they cover but no narrower contract has already set. Processing Year-first would overwrite Quarter and Month settlements.

For each contract with settlement `F_C` covering months `M_C` (with already-set months `S_set`):

```
level_C = (F_C × D_C − S_set) / Σ_{m ∈ M_C, unset} s(m) × d_m
F(0, m) = level_C × s(m)        ∀ m ∈ M_C \ already_set
```

where `D_C` is total days in the contract period and `d_m` is days in month *m*. This guarantees by construction that the days-weighted average of monthly forwards over each contract's period equals its market settlement. Reconciliation is verified post-construction (`forward_curve.reconcile_contracts`) and must pass within the 0.05 EUR/MWh tolerance noted above.

---

## 4. Price model

### 4.1 Specification

The shift-adjusted log spot follows a mean-reverting Ornstein-Uhlenbeck process with a deterministic drift:

```
X_t = ln(S_t + c) = α(t) + Y_t
dY_t = −κ Y_t dt + σ dW_t,    Y_0 = 0
```

- **`α(t)`** is the deterministic market-consistent drift.
- **`Y_t`** is the mean-reverting stochastic state.
- **`κ, σ`** are calibrated to historical dynamics.
- **`c`** is a log shift constant accommodating negative day-ahead prices.

**The shift constant `c = 100 EUR/MWh`** is chosen relative to the historical DK1 minimum of approximately −61 EUR/MWh. The constraint `c > −min(S_t)` over the calibration window must hold; 100 provides a safety margin and conceptually corresponds to a practical price floor for the Nordic system under high renewable penetration (deeply negative prices are bounded by curtailment economics).

### 4.2 Two-step calibration

**Step A — OLS on log-shifted prices** fits a constant and harmonic seasonal coefficients on a Fourier basis with annual and semi-annual frequencies:

```
X_t = θ + A1 cos(ωt) + B1 sin(ωt) + A2 cos(2ωt) + B2 sin(2ωt) + ε_t,    ω = 2π / 12
```

Here, `t` is measured in monthly steps, so the annual seasonal frequency is `ω = 2π / 12`.

Only annual and semi-annual harmonics are retained. Higher-order terms (third or fourth harmonic) capture short-period weekly patterns but do not contribute meaningfully to monthly-resolution exposure modelling; their omission also reduces overfitting risk on a finite sample.

**Step B — AR(1) on residuals** `ε_t`:

```
ε_{t+Δt} = b × ε_t + η_t
```

Continuous-time parameters back out as:

```
κ = −ln(b) / Δt
σ² = Var(η) × 2κ / (1 − exp(−2κΔt))
```

### 4.3 Calibration frequency — the critical design choice

Daily calibration on DK1 spot yields **κ ≈ 150 per year (half-life ≈ 1.7 days)**: a single timescale that captures intra-week mean reversion (the weekly peak-trough pattern) but produces near-deterministic MtM at monthly resolution. With `exp(−κ × 1 month) ≈ 4 × 10⁻⁶`, the conditional forward collapses to the unconditional market forward and the contract loses path dependence entirely.

**The model adopts monthly calibration** on aggregated baseload data. The resulting κ (single-digit per year, half-life of months) captures the multi-month dynamics that matter for a monthly-settled PPA over a 5-year horizon. The intra-week dynamics observable in daily data are not material for monthly settlement and are deliberately not modelled. The single-sentence defence: *calibration frequency must match settlement frequency*.

**Sample-size caveat**: monthly aggregation reduces the calibration window from ~1,950 daily observations to ~65 monthly observations, excluding the incomplete final month. This is sufficient for OLS seasonal estimation and borderline-adequate for AR(1) — large enough for point estimates, but tail diagnostics (Jarque-Bera, ACF at higher lags) carry more uncertainty than they would on a longer sample. A formal extension would use a longer historical window (pre-2021 data is available from Energinet but spans a regime with materially different supply mix).

**One-factor limitation acknowledged**: a single-factor model captures one timescale only. The proper extension is the **Lucia-Schwartz two-factor model**, which simultaneously captures a slow drift component (multi-month) and a fast noise component (daily). This is the canonical next step.

### 4.4 Market-consistency: closed-form α(t)

Under the model, `E_0[S_t] = exp(α(t) + Var[Y_t]/2) − c`. Setting `E_0[S_t] = F(0, t)` yields the analytical drift:

```
α(t) = ln(F(0, t) + c) − σ² / (4κ) × (1 − exp(−2κt))
```

This ensures that, by construction, the model exactly prices the observed market forward curve at every monthly horizon. The methodology is the standard "dynamics from history, levels from market" decomposition used in counterparty credit risk modelling.

---

## 5. Monte Carlo simulation

### 5.1 Exact OU discretization

The OU process admits an exact discrete-time representation:

```
Y_{t+Δt} = Y_t × exp(−κΔt) + σ × sqrt((1 − exp(−2κΔt)) / (2κ)) × Z,    Z ~ N(0, 1)
```

This is preferred over Euler-Maruyama because it is exact for any step size, eliminating discretization bias regardless of the time grid.

### 5.2 Simulation settings

| Setting | Default | Notes |
|---|---|---|
| Number of paths | 10,000 | Halved for antithetic pairs (5,000 independent draws). Trades off precision against runtime; sub-second execution. |
| Antithetic variates | Yes | Standard variance reduction; mean Z = 0 holds exactly per pair. |
| Random seed | 42 | Fixed for reproducibility — same inputs produce bit-identical outputs. |
| Time grid | Month midpoints (day 15 of each month) | A monthly settlement averages 24×~30 hourly prices; the midpoint is the natural representative single-time stand-in for the monthly average price. The alternative (month-end) would systematically misrepresent the settlement timing. |

### 5.3 Validation: martingale check

The empirical mean of `S_paths` is compared against the market forward curve at every monthly horizon. The acceptance criterion is `max |z| < 3`, where `z = (E_emp[S_t] − F(0, t)) / (σ_emp / sqrt(N))`. The base-case run passes with `max |z| < 2` across all 60 months.

A secondary check verifies that the empirical variance of `Y_t` matches the theoretical OU variance `σ²/(2κ) × (1 − exp(−2κt))` within ~10% relative error.

---

## 6. Mark-to-market

### 6.1 Analytical formulation

The MtM at evaluation time `t_k` along simulated path *n* equals the present value of remaining cashflows, given the simulated state `Y_{t_k}^(n)`:

```
MtM_k^(n) = Σ_{m > k} DF(t_k, t_m) × (F_fix − F_{t_k}^(n)(m)) × V × h_m
```

The **conditional forward** is closed-form:

```
F_s(t) = E[S_t | Y_s] = exp(α(t) + Y_s × exp(−κ(t−s)) + Var[Y_t|Y_s] / 2) − c
       Var[Y_t|Y_s] = σ² / (2κ) × (1 − exp(−2κ(t−s)))
```

This eliminates nested Monte Carlo. MtM at every (path, time) is computed exactly under the model in O(N × M²) operations.

**Timing convention**: `MtM_k` is interpreted as the value of the remaining contract *just after* the settlement of month *k*. The MtM matrix at evaluation index *k* therefore excludes month *k*'s cashflow (just paid) and sums over months k+1, ..., M. By the time the last cashflow has settled (k = M − 1), no future cashflows remain and `MtM_{M-1} = 0` for every path by construction.

### 6.2 Fixed price at par

The par fixed price (MtM at inception = 0) is:

```
F_fix = Σ_m DF(0, t_m) × F(0, m) × h_m / Σ_m DF(0, t_m) × h_m
```

For the base-case run, **F_fix = 79.15 EUR/MWh** — between the simple arithmetic mean of the forward curve (~80 EUR/MWh) and the hours- plus discount-weighted average. The total notional is **347 M EUR** (100 MW × ~43,800 deal-life hours × 79.15 EUR/MWh).

### 6.3 Validation: par condition

By construction, the contract value at inception equals zero exactly:

```
MtM_0 = Σ_{m=0}^{M-1} DF(0, t_m) × (F_fix − F(0, m)) × V × h_m = 0
```

This is verified analytically in the test suite (`tests/test_exposure.py::TestParCondition`) to within floating-point precision. Note: at `t_k > 0`, the *expected* MtM across paths is *not* zero — see §7.3 for the curve roll-off effect.

---

## 7. Exposure metrics

### 7.1 Credit exposure (Ørsted in-the-money)

From Ørsted's perspective as fixed receiver, credit exposure arises when MtM > 0 — the counterparty could default and Ørsted loses the unrealised gain.

```
EE_credit(t)  = E[max(MtM_t, 0)]
PFE_credit(t) = quantile_{95%}[max(MtM_t, 0)]
EPE_credit    = time-average of EE_credit
Max_PFE       = max_t PFE_credit(t)
```

The 95% confidence level is industry-standard for PFE reporting. The 99% level is computed in parallel as a tail-scenario metric.

**Base-case results**:

| Metric | Value | Notes |
|---|---|---|
| EPE | 12.4 M EUR | Time-averaged Expected Exposure |
| Max PFE (95%) | 38.4 M EUR | Peak in Dec-27 |
| Max PFE (95%) as % of notional | 11.1% | — |

### 7.2 Liquidity exposure (Ørsted out-of-money)

The mirror-image metric. Symmetric definitions apply with `max(−MtM_t, 0)`.

**Base-case results (gross, no CSA)**:

| Metric | Value | Notes |
|---|---|---|
| EPE | 1.4 M EUR | — |
| Max PFE (95%) | 17.8 M EUR | Peak in Aug-26 |
| Max PFE (95%) as % of notional | 5.1% | — |

### 7.3 Asymmetry — the curve roll-off insight

A naïve application of the log-normal price distribution would predict that liquidity exposure exceeds credit exposure: the heavy right tail of spot (occasional 300+ EUR/MWh spikes) translates into a heavy left tail of MtM for the fixed-receiver, suggesting larger downside (negative MtM, liquidity drain) than upside (positive MtM, credit exposure).

**The base-case results show the opposite**: credit PFE (38.4 M) is approximately 2.2× larger than liquidity PFE (17.8 M). The explanation is structural and worth making explicit:

The DK1 forward curve is **backwardated**: high prices in 2026-2027 (Nov-26 at 115 EUR/MWh, Dec-26 at 120) declining to low prices in 2030-2031 (~65-75 EUR/MWh). The par fixed price of 79 EUR/MWh sits between these regimes.

This creates an asymmetric contract structure:
- **First half (high-price months)**: Ørsted is structurally out-of-money — expected cashflows are negative (Ørsted pays the difference between spot and fixed).
- **Second half (low-price months)**: Ørsted is structurally in-the-money — expected cashflows are positive.

These sum to zero at inception (definition of par). But as time passes, the early *negative* cashflows settle first. The remaining contract — i.e. the MtM — *increases* mechanically over the first two years, peaking around Dec-27 when most of the bad months have rolled off but the deep backwardation tail is still ahead.

This **deterministic roll-off effect** drives expected MtM solidly positive: ~19 M EUR mean at Dec-27 with a standard deviation of ~12 M EUR. The stochastic dispersion is symmetric in the log-normal sense but is *centred on a positive mean*, so most paths cross into credit territory. Liquidity exposure exists only on the negative tail of the dispersion — which is real (Max PFE 17.8 M) but smaller in level than the centred-positive credit side.

**Interpretation for the team**: the asymmetry direction is curve-specific. If the DK1 forward curve flipped to contango (low early, high late), the dominant exposure would reverse. The model framework is general; the direction of the conclusion is an artefact of this specific snapshot.

### 7.4 Liquidity profile shape — the bimodal pattern

The gross liquidity exposure profile is bimodal, peaking in Aug-26 (17.8 M PFE) and again in mid-2030 (~16 M PFE). This is not a numerical artefact:

- **Early peak (Aug-26)**: when high-price winter 2026-27 months still remain in the contract, a positive shock to `Y_t` propagates multiplicatively through all those high-price F_cond(m) values simultaneously. The exposure is concentrated in time.
- **Mid-period trough (2028)**: most high-price months have settled; remaining months are moderate. Less exposure concentration.
- **Late peak (2030)**: accumulated path variance is at its stationary maximum, and the deep-backwardation late-year months still remaining are sensitive to a positive shock that pushes F_cond above F_fix.

For a Liquidity Risk Manager, the early peak is the more material concern — it concentrates the largest liquidity demand in the first months of the deal, when relationships and operational processes with the counterparty are still being established.

---

## 8. CSA collateral overlay

### 8.1 Mechanism

Under a symmetric Credit Support Annex with threshold `T`, collateral posted by Ørsted at time `t_k` along path *n* is:

```
Coll_k^(n) = max(−MtM_k^(n) − T, 0)
```

The threshold uniformly shifts liquidity exposure downward by T. Below T, no collateral is posted (the threshold is the "unsecured tolerance" — the level at which the parties accept counterparty exposure without collateralisation).

### 8.2 Simplifications and their justification

| Simplification | Rationale |
|---|---|
| **Symmetric CSA** (same T both directions) | Standard market practice between corporates of comparable rating. Asymmetric CSAs (lower T for the weaker party) are documented as an extension. |
| **MTA = 0** | The standard deviation of monthly MtM moves is ~10 M EUR; typical MTA values are 100k-500k EUR. MTA is therefore not the binding constraint at monthly resolution and is set to zero. Daily-resolution exposure modelling would need to implement MTA explicitly. |
| **Independent Amount = 0** | Bilateral OTC contract assumption (not cleared). Initial margin would be relevant for cleared PPAs but is not standard for direct utility-to-corporate trades. |
| **Monthly margin call frequency** | Aligned with the simulation grid for consistency. Real CSAs typically operate on daily mark-to-market with daily margin calls; this is a deliberate simplification of the prototype. A daily-resolution model would expose larger single-period margin calls but smoothed monthly aggregates. |

### 8.3 Liquidity metrics

The CSA overlay converts MtM dynamics into the metrics that a treasury function uses for buffer sizing and cash-management planning:

| Metric | Definition | Use |
|---|---|---|
| Peak collateral per path | max over time of `Coll_k^(n)` | Per-path maximum cash tied up. Distribution → buffer sizing. |
| Peak collateral PFE 95% | 95th percentile of peak per path | Model-implied liquidity buffer for the deal. |
| Max single margin call per path | max over time of (`Coll_k − Coll_{k−1}`) | Largest single-period cash outflow event. Distribution → cash management. |
| Fraction with no posting | proportion of paths with peak = 0 | Probability the deal is entirely secured under base assumptions. |

**Base-case results (T = 5 M EUR)**:

| Metric | Value | Notes |
|---|---|---|
| Peak collateral PFE 95% | **27.6 M EUR** | Model-implied liquidity buffer (7.9% of notional). |
| Peak collateral PFE 99% | **36.0 M EUR** | Tail liquidity scenario. |
| Max single-period margin call PFE 95% | **19.7 M EUR** | Largest plausible single-month cash outflow. |
| Max single-period margin call PFE 99% | **25.3 M EUR** | Tail scenario for cash management. |
| Fraction of paths with no posting | **8.5%** | Only ~1 in 12 paths avoids posting collateral entirely. |

The very low "no-posting" fraction (8.5%) is a meaningful finding: under base assumptions this PPA *structurally* requires collateral management capacity, not as a rare contingency but as a near-certainty over a 5-year horizon.

### 8.4 Threshold sensitivity — the credit/liquidity trade-off

The most operationally relevant output of the model. Sweeping threshold from 0 to 30 M EUR:

| T (M EUR) | % paths no-post | Mean peak (M) | PFE 95% peak (M) | PFE 99% peak (M) | PFE 95% max call (M) |
|---|---|---|---|---|---|
| 0.0 | 1.2% | 16.2 | 32.5 | 40.9 | 22.0 |
| 2.5 | 3.7% | 13.8 | 30.0 | 38.4 | 20.8 |
| **5.0** | **8.5%** | **11.4** | **27.6** | **35.9** | **19.7** |
| 7.5 | 15.6% | 9.2 | 25.0 | 33.4 | 18.4 |
| 10.0 | 25.9% | 7.2 | 22.5 | 30.9 | 16.9 |
| 15.0 | 48.5% | 4.1 | 17.5 | 25.9 | 13.9 |
| 20.0 | 69.5% | 2.1 | 12.5 | 20.9 | 11.0 |
| 30.0 | 92.6% | 0.6 | 2.5 | 10.9 | — |

The relationship between threshold and peak PFE 95% is **almost exactly linear**: each additional 1 M EUR of threshold reduces peak PFE 95% by approximately 1 M EUR. This is expected: the threshold uniformly shifts the liquidity exposure curve downward and the 95th percentile shifts in lock-step until very high thresholds where the distribution becomes degenerate.

**The implication for counterparty negotiation**: the trade-off is quantitative, not qualitative. For a counterparty rated A or above, raising T from 5 M to 15 M reduces liquidity buffer by 10 M EUR at the cost of 10 M EUR additional unsecured credit exposure. The optimal choice is determined by the marginal cost of liquidity at the issuer (favouring higher T when liquidity is constrained) versus the credit quality of the counterparty (favouring lower T for weaker credits).

---

## 9. Limitations and extensions

### 9.1 Acknowledged simplifications

| Simplification | Impact | Production extension |
|---|---|---|
| Single-factor model | One timescale only; cannot simultaneously capture daily and multi-month dynamics. | Lucia-Schwartz two-factor model with separate fast and slow components. |
| Gaussian innovations | Tail too thin for power markets; high-quantile PFE could be understated. | Empirical or Student-t innovations; jump-diffusion. |
| Baseload volume profile | Realistic for industrial offtake; misses renewable-shape effects. | As-generated profile with hourly capture-price effect. |
| Flat 2% discount rate | Acceptable for indicative figures. | EUR OIS / IRS curve at relevant tenors. |
| Monthly settlement granularity | Misses intra-month dynamics (intraday peaks, weekend effects). | Hourly granularity for short-tenor deals (capture-price modelling). |
| Monthly margin calls in CSA overlay | Smooths daily realised cash flows. | Daily-resolution margin call simulation. |
| MTA = 0 | Negligible at monthly resolution; potentially material daily. | Discrete MTA threshold simulation for daily-resolution exposure. |
| No counterparty default modelling | Pre-default exposure only. | Bivariate model with default intensity → CVA / FVA. |
| No wrong-way risk | Counterparty exposure assumed independent of credit quality. | Joint price-credit dynamics, particularly relevant for power-sector counterparties. |
| Calibration window ~65 monthly obs | Borderline for AR(1) tail diagnostics. | Extend window to pre-2021 (regime caveats apply); or use a Bayesian shrinkage prior. |
| Trading-day snapshot | Single-day forward curve. | Multi-day average or stochastic curve model. |
| Manual EEX data collection | Cannot be automated as part of the pipeline. | EEX market data subscription with API access. |

### 9.2 Production hand-off considerations

A short list of items required to take the prototype to production:

- **Data quality controls**: automated reconciliation of spot data; alerts on missing days; OIS curve sourcing; EEX API integration.
- **Model governance**: documented model-risk classification, periodic re-calibration cadence (typically quarterly), back-testing framework against realised P&L.
- **Performance**: vectorise MtM computation across path × time × deal tensor for portfolios of >100 contracts.
- **Aggregation**: portfolio-level exposure netting under master netting agreements; wrong-way-risk-adjusted aggregation across the trading book.
- **Integration**: feed exposure metrics into existing limit framework (credit lines per counterparty); feed liquidity metrics into treasury cash-flow projection.
- **User interface**: production UI on existing internal BI stack (Power BI / Tableau / proprietary).

---

## 10. Reproducibility

All Monte Carlo runs use a fixed seed (default 42). Input data is versioned in `data/` with filenames carrying the snapshot trading day. Calibrated parameters are persisted in `data/model_params.json` with full diagnostic information for inspection.

A test suite (`tests/`) covers:
- AR(1) → OU parameter recovery on synthetic data.
- Seasonal OLS coefficient recovery on planted harmonics.
- Martingale property of the Monte Carlo engine.
- Par condition of the analytical MtM at inception.
- Empirical mean of MtM matches the deterministic curve-roll-off value within Monte Carlo standard error.
- Antithetic variance reduction.
- Reproducibility (same seed → bit-identical output).

All 13 tests pass via `pytest tests/`.

---

## 11. References

- Schwartz, E.S. (1997). *The Stochastic Behaviour of Commodity Prices: Implications for Valuation and Hedging.* Journal of Finance, 52, 923-973.
- Lucia, J.J. and Schwartz, E.S. (2002). *Electricity Prices and Power Derivatives: Evidence from the Nordic Power Exchange.* Review of Derivatives Research, 5, 5-50.
- Eydeland, A. and Wolyniec, K. (2003). *Energy and Power Risk Management: New Developments in Modeling, Pricing, and Hedging.* Wiley.
- Bjerksund, P., Rasmussen, H. and Stensland, G. (2010). *Valuation and risk management in the Norwegian electricity market.* Energy, Natural Resources and Environmental Economics.

---

*Source code, tests, interactive Streamlit application, and step-by-step development notebooks are available in the `ppa-exposure-modelling` repository.*
