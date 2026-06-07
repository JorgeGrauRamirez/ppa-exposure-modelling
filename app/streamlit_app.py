"""
Streamlit front-end for the PPA Exposure Modelling prototype.

Run with:
    streamlit run app/streamlit_app.py

Tabs are designed so a reviewer with no prior context can navigate them
sequentially: each opens with a short narrative, presents annotated charts,
and ends with the headline business takeaway.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the package importable when running from app/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ppa_exposure import (
    compute_exposure_metrics,
    compute_liquidity_metrics,
    compute_mtm_matrix,
    compute_par_fixed_price,
    simulate,
    threshold_sensitivity,
)
from ppa_exposure.validation import check_martingale

DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Page configuration and theming
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PPA Exposure Modelling",
    page_icon="../logo/Orsted_logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color palette
CREDIT_COLOR = "#4FC3F7"        # cool blue
LIQUIDITY_COLOR = "#EF5350"     # warm red
ACCENT = "#FFA726"              # amber for highlights
NEUTRAL = "#9E9E9E"             # gray
TEXT_DIM = "#BDBDBD"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    margin=dict(l=50, r=20, t=50, b=40),
    font=dict(family="Inter, sans-serif", size=12),
    hoverlabel=dict(font_size=12),
)

# Subtle CSS tweaks
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    [data-testid="stMetricDelta"] { font-size: 0.85rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background: rgba(255,255,255,0.03);
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(239, 83, 80, 0.15);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_forward_curve() -> pd.DataFrame:
    candidates = sorted(DATA_DIR.glob("monthly_forward_curve_*.csv"))
    if not candidates:
        return pd.DataFrame()
    df = pd.read_csv(candidates[-1])
    df["delivery_month"] = pd.to_datetime(df["delivery_month"], format="%Y-%m")
    return df


@st.cache_data
def load_model_params() -> dict | None:
    candidates = sorted(DATA_DIR.glob("model_params*.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text())


# ---------------------------------------------------------------------------
# Sidebar — user controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.caption("Adjust deal terms and model settings. All metrics recompute live.")

    with st.expander("📄 Deal terms", expanded=True):
        volume_mw = st.number_input("Baseload volume (MW)", 1.0, 1000.0, 100.0, 10.0,
                                    help="Constant capacity sold/bought 24/7.")
        discount_rate = st.slider("Discount rate (%)", 0.0, 5.0, 2.0, 0.25,
                                  help="Flat continuous discount applied to all cashflows.") / 100
        fixed_mode = st.radio("Fixed price", ["At par (computed)", "Manual override"],
                              help="At par = MtM is zero at signing. Standard market convention.")
        manual_fixed = st.number_input("Manual fixed price (EUR/MWh)", value=80.0, step=1.0) \
            if fixed_mode == "Manual override" else None

    with st.expander("🔒 CSA / collateral", expanded=True):
        threshold_m = st.slider("Threshold (M EUR, symmetric)", 0.0, 30.0, 5.0, 0.5,
                                help="No collateral posted until exposure exceeds this amount.")
        threshold = threshold_m * 1e6

    with st.expander("🎲 Monte Carlo settings", expanded=False):
        n_paths = st.select_slider("Paths", [1000, 2000, 5000, 10000, 20000], 10000)
        use_antithetic = st.checkbox("Antithetic variates", True)
        seed = st.number_input("Random seed", value=42, step=1)
        pfe_quantile = st.slider("PFE confidence (%)", 90, 99, 95) / 100

    with st.expander("🧪 Model parameters", expanded=False):
        params = load_model_params()
        if params:
            default_kappa = params["parameters"]["kappa_per_year"]
            default_sigma = params["parameters"]["sigma_annualized"]
            default_c = params["shift_constant_c"]
            st.caption(f"Calibrated from `{params.get('calibration_window', {}).get('frequency', '?')}` data")
        else:
            default_kappa, default_sigma, default_c = 8.0, 0.55, 100.0
            st.warning("No calibration file found — using defaults.")

        override = st.checkbox("Override calibrated parameters", False)
        if override:
            kappa = st.number_input("κ (1/year)", 0.1, 200.0, float(default_kappa), 0.5)
            sigma = st.number_input("σ (annualized)", 0.01, 3.0, float(default_sigma), 0.05)
        else:
            kappa, sigma = float(default_kappa), float(default_sigma)
        shift_c = float(default_c)
        st.caption(f"Half-life: **{np.log(2)/kappa*365.25:.0f} days** "
                   f"({'fast' if kappa > 50 else 'slow' if kappa < 5 else 'moderate'} reversion)")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# ⚡ PPA Exposure Modelling")
st.markdown(
    f"<p style='color:{TEXT_DIM}; font-size:1.05rem; margin-top:-0.5rem;'>"
    "Credit and liquidity exposure on a long-dated Power Purchase Agreement — "
    "DK1 zonal power, market-consistent Monte Carlo."
    "</p>",
    unsafe_allow_html=True,
)

# Load curve
fc_df = load_forward_curve()
if fc_df.empty:
    st.error("⚠️ No monthly forward curve found in `data/`. "
             "Run `notebooks/03_seasonal_bootstrap.ipynb` before launching the app.")
    st.stop()

forward_curve = fc_df["forward_eur_mwh"].values
delivery_months = pd.DatetimeIndex(fc_df["delivery_month"].values)
valuation_date = (delivery_months[0] - pd.DateOffset(months=1)).replace(day=1)


# ---------------------------------------------------------------------------
# Run the pipeline (cached at this seed/path count level isn't easy due to
# multiple parameters; we recompute on each interaction — fast enough)
# ---------------------------------------------------------------------------

with st.spinner("Running Monte Carlo..."):
    sim = simulate(valuation_date, delivery_months, forward_curve,
                   kappa=kappa, sigma=sigma, shift_constant_c=shift_c,
                   n_paths=int(n_paths), use_antithetic=use_antithetic, seed=int(seed))

    F_fix = (manual_fixed if manual_fixed is not None else
             compute_par_fixed_price(forward_curve, delivery_months, sim.T_grid, discount_rate))

    MtM = compute_mtm_matrix(sim, F_fix, volume_mw, kappa, sigma, shift_c, discount_rate)
    exposure = compute_exposure_metrics(MtM, delivery_months, F_fix, volume_mw,
                                        discount_rate=discount_rate, pfe_quantile=pfe_quantile)
    liquidity = compute_liquidity_metrics(MtM, delivery_months, threshold, pfe_quantile=pfe_quantile)


# ---------------------------------------------------------------------------
# Headline summary cards
# ---------------------------------------------------------------------------

st.markdown("##### Headline metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Notional", f"{exposure.notional_eur/1e6:,.0f} M EUR",
          help=f"{volume_mw:.0f} MW × {(exposure.notional_eur/(volume_mw*sum(((delivery_months + pd.offsets.MonthEnd(0) - delivery_months).days + 1)*24))*100):.2f} avg EUR/MWh × hours")
c2.metric("Fixed price", f"{F_fix:,.2f} EUR/MWh",
          help=f"Hours- and discount-weighted average of forward curve ({fixed_mode}).")
c3.metric("Max PFE — Credit",
          f"{exposure.Max_PFE_credit/1e6:.1f} M EUR",
          delta=f"{exposure.Max_PFE_credit/exposure.notional_eur*100:.1f}% of notional",
          delta_color="off",
          help=f"Peak counterparty exposure in {exposure.Max_PFE_credit_month.strftime('%b %Y')}.")
c4.metric("Peak collateral PFE 95%",
          f"{liquidity.peak_pfe95/1e6:.1f} M EUR",
          delta=f"{liquidity.peak_pfe95/exposure.notional_eur*100:.1f}% of notional",
          delta_color="off",
          help=f"Recommended liquidity buffer at T = {threshold_m:.1f} M EUR threshold.")

st.markdown("---")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_price, tab_exp, tab_liq, tab_sens, tab_assum = st.tabs([
    "📈 Price model", "📊 Exposure profile", "💧 Liquidity overlay",
    "⚖️ Sensitivity", "📋 Assumptions"
])


# ============================== TAB 1: PRICE MODEL ============================

with tab_price:
    st.markdown("### Forward curve and simulated price paths")
    st.markdown(
        f"<p style='color:{TEXT_DIM};'>"
        "The DK1 forward curve sets the <b>expected</b> spot at every monthly delivery. "
        "Monte Carlo simulates 10,000 alternative realisations of how the actual spot might evolve, "
        "calibrated to historical DK1 dynamics. The model is <b>market-consistent</b>: the empirical "
        "mean across paths reproduces the market curve at every horizon (martingale check below)."
        "</p>",
        unsafe_allow_html=True,
    )

    # Compute percentile bands for fan chart
    pct5 = np.percentile(sim.S_paths, 5, axis=0)
    pct25 = np.percentile(sim.S_paths, 25, axis=0)
    pct75 = np.percentile(sim.S_paths, 75, axis=0)
    pct95 = np.percentile(sim.S_paths, 95, axis=0)
    mean_emp = sim.S_paths.mean(axis=0)

    fig = go.Figure()

    # 5-95 band
    fig.add_trace(go.Scatter(x=delivery_months, y=pct95, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=delivery_months, y=pct5, mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(79, 195, 247, 0.12)",
                             name="5–95% band", hovertemplate="p5–p95<extra></extra>"))

    # 25-75 band
    fig.add_trace(go.Scatter(x=delivery_months, y=pct75, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=delivery_months, y=pct25, mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(79, 195, 247, 0.25)",
                             name="25–75% band", hovertemplate="p25–p75<extra></extra>"))

    # Market forward
    fig.add_trace(go.Scatter(x=delivery_months, y=forward_curve, mode="lines+markers",
                             line=dict(color=ACCENT, width=2.5), marker=dict(size=4),
                             name="Market forward F(0, t)",
                             hovertemplate="Market: %{y:.1f} EUR/MWh<extra></extra>"))

    # Empirical mean (validates martingale)
    fig.add_trace(go.Scatter(x=delivery_months, y=mean_emp, mode="lines",
                             line=dict(color="white", width=1, dash="dot"),
                             name="Simulated mean",
                             hovertemplate="Sim mean: %{y:.1f} EUR/MWh<extra></extra>"))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=480,
        yaxis_title="EUR/MWh",
        xaxis_title=None,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=1.08, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Side observations
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""
    **Curve range**
    {forward_curve.min():.1f} – {forward_curve.max():.1f} EUR/MWh
    """)
    col2.markdown(f"""
    **Shape**
    {'Backwardated (early > late)' if forward_curve[:6].mean() > forward_curve[-6:].mean() else 'Contango (early < late)'}
    """)
    col3.markdown(f"""
    **Implied volatility (σ)**
    {sigma:.2f} annualized, half-life {np.log(2)/kappa*365.25:.0f} days
    """)

    # Martingale check expandable
    with st.expander("🔍 Martingale check (audit-friendly)", expanded=False):
        mart = check_martingale(sim)
        max_z = mart.attrs["max_abs_z"]
        if max_z < 3.0:
            st.success(f"✅ Passes within 3σ tolerance — max |z| = {max_z:.2f}")
        else:
            st.warning(f"⚠️ Max |z| = {max_z:.2f} exceeds 3σ. May indicate insufficient paths.")

        st.caption("The empirical mean of simulated spot must equal the market forward at every horizon. "
                   "Z-score = (empirical − market) / Monte Carlo standard error.")

        st.dataframe(
            mart.style.format({
                "F_market": "{:.2f}", "F_empirical": "{:.2f}",
                "diff": "{:+.3f}", "monte_carlo_se": "{:.3f}", "z_score": "{:+.2f}",
            }).background_gradient(subset=["z_score"], cmap="RdYlGn_r", vmin=-3, vmax=3),
            height=280, use_container_width=True,
        )


# ============================== TAB 2: EXPOSURE PROFILE =======================

with tab_exp:
    st.markdown("### Credit and liquidity exposure profiles")

    col_intro, col_summary = st.columns([2, 1])
    with col_intro:
        st.markdown(
            f"<p style='color:{TEXT_DIM};'>"
            "From <b>Ørsted's perspective as fixed receiver</b>:<br>"
            "<span style='color:" + CREDIT_COLOR + ";'>● <b>Credit</b></span> exposure arises when "
            "Ørsted is in-the-money — if the counterparty defaults, Ørsted loses the unrealised gain.<br>"
            "<span style='color:" + LIQUIDITY_COLOR + ";'>● <b>Liquidity</b></span> exposure arises when "
            "Ørsted is out-of-money — under a CSA, Ørsted would have to post collateral."
            "</p>",
            unsafe_allow_html=True,
        )
    with col_summary:
        ratio = exposure.Max_PFE_credit / max(exposure.Max_PFE_liq, 1)
        if ratio > 1.5:
            st.info(f"📌 **Credit dominates** ({ratio:.1f}× larger). The backwardated curve "
                    f"creates a structural positive drift in MtM as early high-price months settle.")
        elif ratio < 0.67:
            st.info(f"📌 **Liquidity dominates** ({1/ratio:.1f}× larger).")
        else:
            st.info(f"📌 **Roughly symmetric** — credit and liquidity exposures are comparable.")

    # Build double-panel exposure chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Credit (Ørsted in-the-money)",
                                        "Liquidity (Ørsted out-of-money)"),
                        vertical_spacing=0.12)

    # Credit panel
    fig.add_trace(go.Scatter(x=delivery_months, y=exposure.PFE_credit/1e6,
                             mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=delivery_months, y=np.zeros_like(exposure.PFE_credit),
                             mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(79, 195, 247, 0.20)",
                             name=f"PFE {int(pfe_quantile*100)}%",
                             hovertemplate="PFE: %{y:.2f} M EUR<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=delivery_months, y=exposure.EE_credit/1e6,
                             mode="lines", line=dict(color=CREDIT_COLOR, width=2.5),
                             name="EE (expected)",
                             hovertemplate="EE: %{y:.2f} M EUR<extra></extra>"),
                  row=1, col=1)

    # Annotation at credit peak
    fig.add_annotation(
        x=exposure.Max_PFE_credit_month, y=exposure.Max_PFE_credit/1e6,
        text=f"Max PFE: {exposure.Max_PFE_credit/1e6:.1f} M",
        showarrow=True, arrowhead=2, ax=0, ay=-35,
        bgcolor="rgba(79,195,247,0.85)", font=dict(color="black", size=11),
        row=1, col=1,
    )

    # Liquidity panel
    fig.add_trace(go.Scatter(x=delivery_months, y=exposure.PFE_liq/1e6,
                             mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=delivery_months, y=np.zeros_like(exposure.PFE_liq),
                             mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(239, 83, 80, 0.20)",
                             name=f"PFE {int(pfe_quantile*100)}% (liq)",
                             showlegend=False,
                             hovertemplate="PFE: %{y:.2f} M EUR<extra></extra>"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=delivery_months, y=exposure.EE_liq/1e6,
                             mode="lines", line=dict(color=LIQUIDITY_COLOR, width=2.5),
                             name="EE (liquidity)",
                             hovertemplate="EE: %{y:.2f} M EUR<extra></extra>"),
                  row=2, col=1)

    fig.add_annotation(
        x=exposure.Max_PFE_liq_month, y=exposure.Max_PFE_liq/1e6,
        text=f"Max PFE: {exposure.Max_PFE_liq/1e6:.1f} M",
        showarrow=True, arrowhead=2, ax=0, ay=-35,
        bgcolor="rgba(239,83,80,0.85)", font=dict(color="black", size=11),
        row=2, col=1,
    )

    fig.update_layout(**PLOTLY_LAYOUT, height=560, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=1.05, x=0))
    fig.update_yaxes(title_text="M EUR", row=1, col=1)
    fig.update_yaxes(title_text="M EUR", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.markdown("##### Key metrics")
    summary = pd.DataFrame({
        "Metric": ["Expected exposure (EPE)", "Maximum PFE", "Peak month", "% of notional"],
        "Credit": [
            f"{exposure.EPE_credit/1e6:.2f} M EUR",
            f"{exposure.Max_PFE_credit/1e6:.2f} M EUR",
            exposure.Max_PFE_credit_month.strftime("%b %Y"),
            f"{exposure.Max_PFE_credit/exposure.notional_eur*100:.1f}%",
        ],
        "Liquidity (no CSA)": [
            f"{exposure.EPE_liq/1e6:.2f} M EUR",
            f"{exposure.Max_PFE_liq/1e6:.2f} M EUR",
            exposure.Max_PFE_liq_month.strftime("%b %Y"),
            f"{exposure.Max_PFE_liq/exposure.notional_eur*100:.1f}%",
        ],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)


# ============================== TAB 3: LIQUIDITY OVERLAY ======================

with tab_liq:
    st.markdown(f"### Collateral posted by Ørsted — CSA threshold T = {threshold_m:.1f} M EUR")
    st.markdown(
        f"<p style='color:{TEXT_DIM};'>"
        "Under the CSA, Ørsted posts collateral only when out-of-money <i>beyond</i> the threshold T. "
        "This tab converts the liquidity exposure into <b>cash-out metrics</b> a treasury function "
        "actually uses to size buffers: how much collateral is posted at any point in time, and how big "
        "the single-period shocks can be."
        "</p>",
        unsafe_allow_html=True,
    )

    # Recommendation callout
    rec_buffer = liquidity.peak_pfe95 / 1e6
    rec_shock = liquidity.max_call_pfe95 / 1e6
    no_post_pct = liquidity.fraction_no_posting * 100

    col_rec1, col_rec2, col_rec3 = st.columns(3)
    col_rec1.success(f"💰 **Recommended buffer**\n\n**{rec_buffer:.1f} M EUR**\n\n"
                     f"95th percentile of peak collateral across paths")
    col_rec2.warning(f"⚡ **Max single shock**\n\n**{rec_shock:.1f} M EUR**\n\n"
                     f"Largest plausible month-over-month outflow (p95)")
    col_rec3.info(f"📉 **No-posting probability**\n\n**{no_post_pct:.1f}%**\n\n"
                  f"Paths that never post collateral over the deal life")

    # Profile chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=delivery_months, y=liquidity.PFE_collateral/1e6,
                             mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=delivery_months, y=np.zeros_like(liquidity.PFE_collateral),
                             mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(239, 83, 80, 0.20)",
                             name=f"PFE {int(pfe_quantile*100)}%",
                             hovertemplate="PFE: %{y:.2f} M EUR<extra></extra>"))
    fig.add_trace(go.Scatter(x=delivery_months, y=liquidity.EE_collateral/1e6,
                             mode="lines", line=dict(color=LIQUIDITY_COLOR, width=2.5),
                             name="Expected",
                             hovertemplate="Expected: %{y:.2f} M EUR<extra></extra>"))

    fig.update_layout(**PLOTLY_LAYOUT, height=380,
                      title=dict(text="Collateral posted profile", font=dict(size=14)),
                      yaxis_title="M EUR posted",
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=1.1, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # Distribution histograms
    st.markdown("##### Distributions across all simulated paths")
    col_h1, col_h2 = st.columns(2)

    def percentile_lines(arr, percentiles):
        return [(p, float(np.percentile(arr, p))) for p in percentiles]

    with col_h1:
        peak = liquidity.peak_per_path / 1e6
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(x=peak, nbinsx=60, marker=dict(color=LIQUIDITY_COLOR, opacity=0.7),
                                     hovertemplate="Peak ≈ %{x:.1f} M EUR<br>Paths: %{y}<extra></extra>"))
        ymax = np.histogram(peak, bins=60)[0].max()
        for pct, val in percentile_lines(peak, [50, 95, 99]):
            fig_h.add_vline(x=val, line_dash="dash", line_color="white", opacity=0.5)
            fig_h.add_annotation(x=val, y=ymax*0.95, text=f"p{pct}={val:.1f}",
                                 showarrow=False, font=dict(color="white", size=10),
                                 xshift=10, textangle=-90)
        fig_h.update_layout(**PLOTLY_LAYOUT, height=320,
                            title=dict(text="Peak collateral per path", font=dict(size=13)),
                            xaxis_title="M EUR", yaxis_title="Number of paths",
                            showlegend=False)
        st.plotly_chart(fig_h, use_container_width=True)
        st.caption("How much collateral Ørsted ends up posting at the worst moment of the deal life, per simulated path.")

    with col_h2:
        calls = liquidity.max_call_per_path / 1e6
        fig_c = go.Figure()
        fig_c.add_trace(go.Histogram(x=calls, nbinsx=60, marker=dict(color=ACCENT, opacity=0.7),
                                     hovertemplate="Max call ≈ %{x:.1f} M EUR<br>Paths: %{y}<extra></extra>"))
        ymax = np.histogram(calls, bins=60)[0].max()
        for pct, val in percentile_lines(calls, [50, 95, 99]):
            fig_c.add_vline(x=val, line_dash="dash", line_color="white", opacity=0.5)
            fig_c.add_annotation(x=val, y=ymax*0.95, text=f"p{pct}={val:.1f}",
                                 showarrow=False, font=dict(color="white", size=10),
                                 xshift=10, textangle=-90)
        fig_c.update_layout(**PLOTLY_LAYOUT, height=320,
                            title=dict(text="Max single-period margin call per path", font=dict(size=13)),
                            xaxis_title="M EUR", yaxis_title="Number of paths",
                            showlegend=False)
        st.plotly_chart(fig_c, use_container_width=True)
        st.caption("The largest cash outflow over a single month, per simulated path. Critical for cash management.")


# ============================== TAB 4: SENSITIVITY ============================

with tab_sens:
    st.markdown("### Threshold sensitivity — the credit/liquidity trade-off")
    st.markdown(
        f"<p style='color:{TEXT_DIM};'>"
        "This is the trade-off the Credit & Liquidity Risk team explicitly negotiates with each counterparty: "
        "<b>higher threshold = less collateral burden but more unsecured credit exposure</b>. "
        "Stronger counterparties justify higher thresholds; weaker ones the opposite. "
        "This chart quantifies the trade-off on the actual DK1 curve."
        "</p>",
        unsafe_allow_html=True,
    )

    thresholds = np.array([0, 2.5, 5, 7.5, 10, 15, 20, 30]) * 1e6
    sens = threshold_sensitivity(MtM, thresholds, delivery_months)

    # Two-panel sensitivity chart
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Peak collateral required vs threshold",
                                        "Probability of zero collateral over deal life"),
                        horizontal_spacing=0.12)

    # Left: peak collateral curves
    fig.add_trace(go.Scatter(x=sens["threshold_eur"]/1e6, y=sens["peak_pfe99_eur"]/1e6,
                             mode="lines+markers", line=dict(color="#9C27B0", width=2.5),
                             marker=dict(size=7), name="PFE 99%",
                             hovertemplate="T=%{x:.1f}M → PFE99=%{y:.1f}M<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sens["threshold_eur"]/1e6, y=sens["peak_pfe95_eur"]/1e6,
                             mode="lines+markers", line=dict(color=LIQUIDITY_COLOR, width=2.5),
                             marker=dict(size=7), name="PFE 95%",
                             hovertemplate="T=%{x:.1f}M → PFE95=%{y:.1f}M<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sens["threshold_eur"]/1e6, y=sens["mean_peak_eur"]/1e6,
                             mode="lines+markers", line=dict(color=ACCENT, width=2.5),
                             marker=dict(size=7), name="Mean",
                             hovertemplate="T=%{x:.1f}M → Mean=%{y:.1f}M<extra></extra>"),
                  row=1, col=1)

    # Highlight current threshold
    fig.add_vline(x=threshold_m, line_dash="dash", line_color="white", opacity=0.5,
                  annotation_text=f"Current T = {threshold_m:.1f}M",
                  annotation_position="top", row=1, col=1)

    # Right: zero-posting probability
    fig.add_trace(go.Scatter(x=sens["threshold_eur"]/1e6, y=sens["frac_no_posting_pct"],
                             mode="lines+markers", line=dict(color="#66BB6A", width=2.5),
                             marker=dict(size=8), name="P(no posting)",
                             showlegend=False,
                             hovertemplate="T=%{x:.1f}M → %{y:.1f}% paths<extra></extra>"),
                  row=1, col=2)
    fig.add_vline(x=threshold_m, line_dash="dash", line_color="white", opacity=0.5, row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT, height=420,
                      legend=dict(orientation="h", yanchor="top", y=1.12, x=0))
    fig.update_xaxes(title_text="Threshold (M EUR)", row=1, col=1)
    fig.update_xaxes(title_text="Threshold (M EUR)", row=1, col=2)
    fig.update_yaxes(title_text="Peak collateral (M EUR)", row=1, col=1)
    fig.update_yaxes(title_text="% paths", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)

    # Compare two thresholds
    st.markdown("##### Compare two threshold scenarios")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        t_low = st.selectbox("Conservative (low T)", thresholds/1e6, index=1)
    with col_c2:
        t_high = st.selectbox("Permissive (high T)", thresholds/1e6, index=5)

    row_low = sens[sens["threshold_eur"] == t_low * 1e6].iloc[0]
    row_high = sens[sens["threshold_eur"] == t_high * 1e6].iloc[0]

    delta_pfe = (row_low["peak_pfe95_eur"] - row_high["peak_pfe95_eur"]) / 1e6
    delta_unsec = (t_high - t_low)  # increase in unsecured credit

    st.markdown(f"""
    Moving from **T = {t_low:.1f}M** to **T = {t_high:.1f}M**:
    - Liquidity buffer (PFE 95%) **decreases by {delta_pfe:+.1f} M EUR**
    - Unsecured credit exposure **increases by {delta_unsec:+.1f} M EUR**
    - Probability of never posting goes from **{row_low['frac_no_posting_pct']:.0f}% → {row_high['frac_no_posting_pct']:.0f}%**

    This is the precise quantification a credit team uses when negotiating the CSA.
    """)

    # Sensitivity table
    with st.expander("Full sensitivity table"):
        display = sens.copy()
        for col in ["threshold_eur", "mean_peak_eur", "peak_pfe95_eur", "peak_pfe99_eur",
                    "max_call_pfe95_eur", "max_call_pfe99_eur"]:
            display[col] = display[col] / 1e6
        display.columns = ["T (M)", "% no-post", "Mean peak (M)", "PFE95 peak (M)",
                           "PFE99 peak (M)", "PFE95 call (M)", "PFE99 call (M)"]
        st.dataframe(display.style.format({c: "{:.2f}" for c in display.columns if c != "T (M)"}),
                     hide_index=True, use_container_width=True)


# ============================== TAB 5: ASSUMPTIONS ============================

with tab_assum:
    st.markdown("### Current run configuration")
    st.markdown(
        f"<p style='color:{TEXT_DIM};'>"
        "A snapshot of every assumption driving the numbers above. "
        "Full methodology, derivations, and limitations are documented in <code>docs/technical_document.md</code>."
        "</p>",
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Contract")
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Volume | **{volume_mw:.0f} MW** baseload |
        | Tenor | **{(delivery_months[-1] - delivery_months[0]).days / 365.25:.1f} years** ({delivery_months[0].strftime('%b %Y')} → {delivery_months[-1].strftime('%b %Y')}) |
        | Settlement | Monthly |
        | Fixed price | **{F_fix:.2f} EUR/MWh** ({fixed_mode}) |
        | Discount rate | {discount_rate*100:.2f}% continuous |
        | Perspective | Ørsted as seller |
        | Notional | **{exposure.notional_eur/1e6:.1f} M EUR** |
        """)

        st.markdown("#### CSA")
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Threshold | **{threshold_m:.1f} M EUR** (symmetric) |
        | MTA | 0 (immaterial at monthly resolution) |
        | Independent amount | 0 (bilateral OTC) |
        """)

    with col_r:
        st.markdown("#### Price model")
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | κ (mean reversion) | **{kappa:.2f} per year** |
        | σ (volatility, annualised) | **{sigma:.4f}** |
        | Log shift c | {shift_c:.0f} EUR/MWh |
        | Half-life | {np.log(2)/kappa*365.25:.0f} days |
        | Calibration source | {'override' if override else 'calibrated'} |
        """)

        st.markdown("#### Monte Carlo")
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Paths | **{int(n_paths):,}** |
        | Antithetic | {'Yes' if use_antithetic else 'No'} |
        | Seed | {int(seed)} |
        | PFE quantile | {int(pfe_quantile*100)}% |
        """)

    st.markdown("---")

    st.markdown("#### Methodology highlights")

    with st.expander("Market consistency"):
        st.markdown(r"""
        The model decomposes the log-shifted spot as $\ln(S_t + c) = \alpha(t) + Y_t$ where:
        - $Y_t$ is a mean-reverting OU process calibrated to **historical dynamics** (κ, σ)
        - $\alpha(t)$ is the deterministic drift, set analytically so that $E_0[S_t] = F(0,t)$ — the simulated mean matches the **market** forward curve at every horizon.

        This is the standard "dynamics from history, levels from market" decomposition used in counterparty credit risk.
        """)

    with st.expander("MtM analytical, not nested Monte Carlo"):
        st.markdown(r"""
        At each (path, evaluation time), the MtM is computed in closed form using the conditional expectation
        of the OU process:

        $$F_{t_k}^{(n)}(m) = E[S_{t_m} \mid Y_{t_k}^{(n)}] = \exp\left(\alpha(t_m) + Y_{t_k} \cdot e^{-\kappa(t_m - t_k)} + \frac{\sigma^2}{4\kappa}(1 - e^{-2\kappa(t_m-t_k)})\right) - c$$

        This avoids nested simulation and gives exact MtM under the model.
        """)

    with st.expander("Calibration frequency choice"):
        st.markdown("""
        Daily calibration produces κ ≈ 150/year (half-life ≈ 1.7 days), which collapses the model to
        a near-deterministic curve roll-off at monthly resolution. **Monthly calibration** captures
        the timescale relevant for monthly-settled PPA exposure. Lucia–Schwartz two-factor is the
        canonical extension to capture both timescales simultaneously.
        """)

    with st.expander("Acknowledged limitations"):
        st.markdown("""
        - Gaussian innovations underestimate tail risk (PFE biased conservative-low)
        - Single-factor model captures one timescale only
        - Baseload volume profile (no renewable shape effects)
        - Flat discount rate (no EUR OIS curve)
        - No counterparty default modelling (pre-default exposure only)
        """)

    st.markdown("---")
    st.caption("Full methodology in `docs/technical_document.md`. Source code: see the GitHub repository.")
