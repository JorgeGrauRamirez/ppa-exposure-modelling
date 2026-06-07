"""
Streamlit front-end for the PPA Exposure Modelling prototype.

Run with:
    streamlit run app/streamlit_app.py

The app loads pre-built data artifacts (forward curve, calibration) from
`data/`, then runs the Monte Carlo + exposure pipeline on the fly given
user-configurable deal terms, CSA parameters, and simulation settings.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when running `streamlit run app/streamlit_app.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ppa_exposure import (
    compute_collateral_posted,
    compute_exposure_metrics,
    compute_liquidity_metrics,
    compute_mtm_matrix,
    compute_par_fixed_price,
    simulate,
    threshold_sensitivity,
)
from ppa_exposure.validation import check_martingale

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="PPA Exposure Modelling",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = ROOT / "data"


# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------

@st.cache_data
def load_forward_curve() -> pd.DataFrame:
    """Load the monthly forward curve."""
    candidates = sorted(DATA_DIR.glob("monthly_forward_curve_*.csv"))
    if not candidates:
        return pd.DataFrame()
    df = pd.read_csv(candidates[-1])
    df["delivery_month"] = pd.to_datetime(df["delivery_month"], format="%Y-%m")
    return df


@st.cache_data
def load_model_params() -> dict | None:
    """Load the calibrated model parameters."""
    candidates = sorted(DATA_DIR.glob("model_params*.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text())


# ----------------------------------------------------------------------------
# Sidebar — user controls
# ----------------------------------------------------------------------------

st.sidebar.title("Configuration")

with st.sidebar.expander("Deal terms", expanded=True):
    volume_mw = st.number_input("Baseload volume (MW)", min_value=1.0, max_value=1000.0, value=100.0, step=10.0)
    discount_rate = st.slider("Discount rate (%)", 0.0, 5.0, 2.0, 0.25) / 100
    fixed_price_mode = st.radio(
        "Fixed price",
        ["At par (computed)", "Manual"],
        help="Par fixed price = MtM zero at inception.",
    )
    manual_fixed = None
    if fixed_price_mode == "Manual":
        manual_fixed = st.number_input("Fixed price (EUR/MWh)", value=80.0, step=1.0)

with st.sidebar.expander("CSA / collateral", expanded=True):
    threshold_m = st.slider("Threshold (M EUR, symmetric)", 0.0, 30.0, 5.0, 0.5)
    threshold = threshold_m * 1e6

with st.sidebar.expander("Monte Carlo", expanded=False):
    n_paths = st.select_slider("Number of paths", options=[1000, 2000, 5000, 10000, 20000], value=10000)
    use_antithetic = st.checkbox("Antithetic variates", value=True)
    seed = st.number_input("Random seed", value=42, step=1)
    pfe_quantile = st.slider("PFE quantile (%)", 90, 99, 95) / 100

with st.sidebar.expander("Model parameters", expanded=False):
    params = load_model_params()
    if params:
        default_kappa = params["parameters"]["kappa_per_year"]
        default_sigma = params["parameters"]["sigma_annualized"]
        default_c = params["shift_constant_c"]
    else:
        default_kappa, default_sigma, default_c = 8.0, 0.55, 100.0

    override_params = st.checkbox("Override calibrated parameters", value=False)
    if override_params:
        kappa = st.number_input("kappa (1/year)", min_value=0.1, max_value=200.0, value=float(default_kappa), step=0.5)
        sigma = st.number_input("sigma (annualised)", min_value=0.01, max_value=3.0, value=float(default_sigma), step=0.05)
    else:
        kappa, sigma = default_kappa, default_sigma
    shift_c = float(default_c)


# ----------------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------------

st.title("PPA Exposure Modelling Prototype")
st.caption(
    "Market-consistent Monte Carlo framework for credit and liquidity exposure "
    "on long-dated Power Purchase Agreements. Reference instrument: DK1 zonal power."
)

# Load forward curve
fc_df = load_forward_curve()
if fc_df.empty:
    st.error(
        "No monthly forward curve found in `data/`. "
        "Run `notebooks/03_seasonal_bootstrap.ipynb` to build it before launching the app."
    )
    st.stop()

forward_curve = fc_df["forward_eur_mwh"].values
delivery_months = pd.DatetimeIndex(fc_df["delivery_month"].values)
valuation_date = (delivery_months[0] - pd.DateOffset(months=1)).replace(day=1)

# ----------------------------------------------------------------------------
# Run simulation
# ----------------------------------------------------------------------------

with st.spinner("Running Monte Carlo..."):
    sim = simulate(
        valuation_date=valuation_date,
        delivery_months=delivery_months,
        forward_curve=forward_curve,
        kappa=kappa,
        sigma=sigma,
        shift_constant_c=shift_c,
        n_paths=int(n_paths),
        use_antithetic=use_antithetic,
        seed=int(seed),
    )

    F_fix = (
        manual_fixed if manual_fixed is not None
        else compute_par_fixed_price(forward_curve, delivery_months, sim.T_grid, discount_rate)
    )

    MtM = compute_mtm_matrix(
        sim, fixed_price=F_fix, volume_mw=volume_mw,
        kappa=kappa, sigma=sigma, shift_constant_c=shift_c,
        discount_rate=discount_rate,
    )

    exposure = compute_exposure_metrics(
        MtM, delivery_months, F_fix, volume_mw,
        discount_rate=discount_rate, pfe_quantile=pfe_quantile,
    )

    liquidity = compute_liquidity_metrics(
        MtM, delivery_months, threshold, pfe_quantile=pfe_quantile,
    )


# ----------------------------------------------------------------------------
# Top-of-page summary
# ----------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Notional", f"{exposure.notional_eur/1e6:,.0f} M EUR")
col2.metric("Fixed price (par)", f"{F_fix:,.2f} EUR/MWh")
col3.metric("Max PFE — Credit", f"{exposure.Max_PFE_credit/1e6:.1f} M EUR",
            help=f"At {exposure.Max_PFE_credit_month.strftime('%b-%Y')}")
col4.metric("Peak collateral PFE 95%", f"{liquidity.peak_pfe95/1e6:.1f} M EUR",
            help=f"Threshold T = {threshold_m:.1f} M EUR")


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------

tab_price, tab_exposure, tab_liquidity, tab_sensitivity, tab_assumptions = st.tabs([
    "Price model", "Exposure profile", "Liquidity overlay", "Sensitivity", "Assumptions",
])


# ----------------- Tab 1: Price model ----------------------------------------

with tab_price:
    st.subheader("Forward curve and simulated paths")

    fig, ax = plt.subplots(figsize=(12, 4))
    # Sample paths
    sample = np.random.default_rng(0).choice(sim.n_paths, size=min(200, sim.n_paths), replace=False)
    for i in sample:
        ax.plot(delivery_months, sim.S_paths[i], linewidth=0.3, alpha=0.2, color="steelblue")
    ax.plot(delivery_months, forward_curve, linewidth=2.5, color="red", label="Market forward F(0, t)")
    ax.plot(delivery_months, sim.S_paths.mean(axis=0), "--", linewidth=2, color="black",
            label="Empirical mean (martingale check)")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Simulated spot paths under the market-consistent Schwartz model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Martingale check (audit-friendly)"):
        mart = check_martingale(sim)
        st.write(f"Max |z-score|: **{mart.attrs['max_abs_z']:.2f}** "
                 f"(passes within 3σ: {'✅' if mart.attrs['passes'] else '❌'})")
        st.dataframe(mart.style.format({
            "F_market": "{:.2f}", "F_empirical": "{:.2f}",
            "diff": "{:+.3f}", "monte_carlo_se": "{:.3f}", "z_score": "{:+.2f}",
        }), height=200)


# ----------------- Tab 2: Exposure profile -----------------------------------

with tab_exposure:
    st.subheader("Credit and liquidity exposure profiles")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, ee, pfe, label, color, max_pfe, max_month in [
        (axes[0], exposure.EE_credit, exposure.PFE_credit, "Credit (Ørsted in-the-money)", "steelblue",
         exposure.Max_PFE_credit, exposure.Max_PFE_credit_month),
        (axes[1], exposure.EE_liq, exposure.PFE_liq, "Liquidity (Ørsted out-of-money)", "crimson",
         exposure.Max_PFE_liq, exposure.Max_PFE_liq_month),
    ]:
        ax.fill_between(delivery_months, 0, pfe / 1e6, alpha=0.25, color=color, label=f"PFE {int(pfe_quantile*100)}%")
        ax.plot(delivery_months, ee / 1e6, linewidth=2, color=color, label="EE")
        ax.plot(delivery_months, pfe / 1e6, linewidth=1, color=color)
        ax.axhline(max_pfe / 1e6, linestyle="--", color="black", alpha=0.5,
                   label=f"Max = {max_pfe/1e6:.1f} M @ {max_month.strftime('%b-%y')}")
        ax.set_title(label)
        ax.set_ylabel("M EUR")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    col1.markdown(f"""
    **Credit metrics**
    - EPE: **{exposure.EPE_credit/1e6:.2f} M EUR**
    - Max PFE: **{exposure.Max_PFE_credit/1e6:.2f} M EUR** ({exposure.Max_PFE_credit_month.strftime('%b-%Y')})
    - As % of notional: **{exposure.Max_PFE_credit/exposure.notional_eur*100:.1f}%**
    """)
    col2.markdown(f"""
    **Liquidity metrics (gross, no CSA)**
    - EPE: **{exposure.EPE_liq/1e6:.2f} M EUR**
    - Max PFE: **{exposure.Max_PFE_liq/1e6:.2f} M EUR** ({exposure.Max_PFE_liq_month.strftime('%b-%Y')})
    - As % of notional: **{exposure.Max_PFE_liq/exposure.notional_eur*100:.1f}%**
    """)


# ----------------- Tab 3: Liquidity overlay ----------------------------------

with tab_liquidity:
    st.subheader(f"Collateral posted by Ørsted — CSA threshold T = {threshold_m:.1f} M EUR")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(delivery_months, 0, liquidity.PFE_collateral / 1e6, alpha=0.25, color="crimson",
                    label=f"PFE {int(pfe_quantile*100)}%")
    ax.plot(delivery_months, liquidity.EE_collateral / 1e6, linewidth=2, color="crimson", label="Expected")
    ax.plot(delivery_months, liquidity.PFE_collateral / 1e6, linewidth=1, color="crimson")
    ax.set_title("Collateral posted profile")
    ax.set_ylabel("M EUR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Peak collateral posted per path**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(liquidity.peak_per_path / 1e6, bins=60, color="crimson", edgecolor="white", alpha=0.7)
        for pct in [50, 95, 99]:
            val = np.percentile(liquidity.peak_per_path, pct) / 1e6
            ax.axvline(val, linestyle="--", linewidth=1)
            ax.text(val, ax.get_ylim()[1] * 0.95, f"p{pct}={val:.1f}", rotation=90, va="top", fontsize=8)
        ax.set_xlabel("M EUR")
        ax.set_ylabel("Paths")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("**Max single-period margin call per path**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(liquidity.max_call_per_path / 1e6, bins=60, color="darkorange", edgecolor="white", alpha=0.7)
        for pct in [50, 95, 99]:
            val = np.percentile(liquidity.max_call_per_path, pct) / 1e6
            ax.axvline(val, linestyle="--", linewidth=1)
            ax.text(val, ax.get_ylim()[1] * 0.95, f"p{pct}={val:.1f}", rotation=90, va="top", fontsize=8)
        ax.set_xlabel("M EUR")
        ax.set_ylabel("Paths")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(f"""
    | Metric | Value |
    |---|---|
    | Peak collateral PFE 95% | **{liquidity.peak_pfe95/1e6:.2f} M EUR** |
    | Peak collateral PFE 99% | **{liquidity.peak_pfe99/1e6:.2f} M EUR** |
    | Max single margin call PFE 95% | **{liquidity.max_call_pfe95/1e6:.2f} M EUR** |
    | Max single margin call PFE 99% | **{liquidity.max_call_pfe99/1e6:.2f} M EUR** |
    | Fraction of paths with no posting | **{liquidity.fraction_no_posting*100:.1f}%** |
    """)


# ----------------- Tab 4: Sensitivity ----------------------------------------

with tab_sensitivity:
    st.subheader("Threshold sensitivity — the credit vs liquidity trade-off")

    thresholds = np.array([0, 2.5, 5, 7.5, 10, 15, 20, 30]) * 1e6
    sens = threshold_sensitivity(MtM, thresholds, delivery_months)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(sens["threshold_eur"] / 1e6, sens["peak_pfe95_eur"] / 1e6, "o-", linewidth=2, color="crimson", label="PFE 95%")
    axes[0].plot(sens["threshold_eur"] / 1e6, sens["peak_pfe99_eur"] / 1e6, "s-", linewidth=2, color="purple", label="PFE 99%")
    axes[0].plot(sens["threshold_eur"] / 1e6, sens["mean_peak_eur"] / 1e6, "^-", linewidth=2, color="orange", label="Mean")
    axes[0].set_xlabel("Threshold (M EUR)")
    axes[0].set_ylabel("Peak collateral posted (M EUR)")
    axes[0].set_title("Peak collateral required vs threshold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sens["threshold_eur"] / 1e6, sens["frac_no_posting_pct"], "o-", linewidth=2, color="seagreen")
    axes[1].set_xlabel("Threshold (M EUR)")
    axes[1].set_ylabel("% paths with no posting")
    axes[1].set_title("Probability of zero collateral over deal life")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    display = sens.copy()
    for col in ["threshold_eur", "mean_peak_eur", "peak_pfe95_eur", "peak_pfe99_eur",
                "max_call_pfe95_eur", "max_call_pfe99_eur"]:
        display[col] = display[col] / 1e6
    display.columns = ["T (M)", "% no-post", "Mean peak (M)", "PFE95 peak (M)", "PFE99 peak (M)",
                       "PFE95 call (M)", "PFE99 call (M)"]
    st.dataframe(display.style.format({c: "{:.2f}" for c in display.columns if c != "T (M)"}), hide_index=True)


# ----------------- Tab 5: Assumptions ----------------------------------------

with tab_assumptions:
    st.subheader("Current run configuration")

    st.json({
        "deal": {
            "volume_mw": volume_mw,
            "fixed_price_eur_mwh": F_fix,
            "fixed_price_mode": fixed_price_mode,
            "discount_rate": discount_rate,
            "valuation_date": str(valuation_date.date()),
            "delivery_window": f"{delivery_months[0].strftime('%b-%Y')} -> {delivery_months[-1].strftime('%b-%Y')}",
            "n_months": len(delivery_months),
        },
        "model": {
            "kappa_per_year": float(kappa),
            "sigma_annualized": float(sigma),
            "shift_constant_c": float(shift_c),
            "half_life_days": float(np.log(2) / kappa * 365.25),
            "calibration_source": "data/model_params.json" if params else "manual override",
        },
        "csa": {"threshold_eur": threshold, "mta_eur": 0.0, "independent_amount": 0.0},
        "monte_carlo": {
            "n_paths": int(n_paths), "use_antithetic": use_antithetic,
            "seed": int(seed), "pfe_quantile": pfe_quantile,
        },
    })

    st.markdown("""
    ### Key methodology choices
    - **Dynamics from history, levels from market**. `kappa` and `sigma` calibrated to historical DK1 spot;
      `alpha(t)` set analytically so that `E[S_t] = F(0, t)`.
    - **Monthly calibration frequency**. Daily-calibrated kappa is too fast (half-life ~ 1.7 days) for monthly
      PPA exposure; monthly calibration captures the relevant timescale.
    - **MtM analytical**. Conditional forwards in closed form — no nested simulation.
    - **Gaussian innovations**. Documented limitation: power markets are leptokurtic; PFE biased conservative-low.
    - **MTA simplified to zero**. Standard deviation of monthly MtM moves (~10M EUR) >>500k MTA;
      no material impact at monthly resolution.

    See `docs/technical_document.md` for the full methodology, derivations, and limitations.
    """)
