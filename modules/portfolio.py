"""Module 1 — Portfolio Evaluation of European options on stocks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from common import (K_, N_, S_, T_, bs_greeks, bs_price, d1_, d2_, download_table, help_button,
                    intrinsic_value, page_header, pdf_button, pdf_report, r_, sig_, spot_input,
                    verdict)

COLS = ["Instrument", "Position", "Quantity", "Strike K", "Maturity T (years)", "Volatility σ",
        "Contract multiplier"]

DEFAULT_LEGS = pd.DataFrame([
    {"Instrument": "Call", "Position": "Long", "Quantity": 10, "Strike K": 100.0,
     "Maturity T (years)": 0.5, "Volatility σ": 0.25, "Contract multiplier": 1},
    {"Instrument": "Call", "Position": "Short", "Quantity": 10, "Strike K": 110.0,
     "Maturity T (years)": 0.5, "Volatility σ": 0.25, "Contract multiplier": 1},
    {"Instrument": "Put", "Position": "Long", "Quantity": 5, "Strike K": 95.0,
     "Maturity T (years)": 0.25, "Volatility σ": 0.28, "Contract multiplier": 1},
])


# ----------------------------------------------------------------------------
# Help
# ----------------------------------------------------------------------------

def _help():
    n_, m_, P_, c_ = sp.symbols("n m P c")
    eps = sp.Symbol(r"\varepsilon")
    call = S_ * N_(d1_) - K_ * sp.exp(-r_ * T_) * N_(d2_)
    put = K_ * sp.exp(-r_ * T_) * N_(-d2_) - S_ * N_(-d1_)
    help_button(
        "pf",
        "Portfolio Evaluation",
        theory=(
            "A portfolio (position) is a collection of *legs*: European calls, European puts and shares of the "
            "underlying stock, each held **long** (bought, +1) or **short** (sold/written, −1). Each option leg is "
            "valued with the **Black-Scholes** model; the stock leg is worth the spot price. The **market value** "
            "of the position is the signed sum of the leg values, while its **intrinsic value** is what the "
            "options would be worth if exercised immediately, max(S−K,0) for a call and max(K−S,0) for a put. "
            "The difference is the **time value** (optionality left before expiry). Because pricing is linear, "
            "position Greeks are the signed, quantity-weighted sums of the per-option Greeks, which describe "
            "how the position value reacts to the spot (Δ, Γ), volatility (vega) and the passage of time (θ)."
        ),
        assumptions=[
            "European exercise only (no early exercise); underlying pays no dividends.",
            "Spot follows a geometric Brownian motion with constant volatility σ per leg.",
            "Constant, continuously-compounded risk-free rate r; frictionless market (no costs, no taxes, "
            "short selling allowed, continuous trading).",
            "Leg value = sign × quantity × contract multiplier × unit price (sign = +1 long, −1 short).",
            "Spot is fetched online (Yahoo Finance last close) or entered manually.",
        ],
        formulas=[
            ("d1 and d2", sp.Eq(sp.Symbol("d_1"), d1_)),
            ("", sp.Eq(sp.Symbol("d_2"), d2_)),
            ("Black-Scholes call", sp.Eq(sp.Symbol("c"), call)),
            ("Black-Scholes put", sp.Eq(sp.Symbol("p"), put)),
            ("Intrinsic value (call / put)",
             r"IV_{call}=\max(S-K,0)\qquad IV_{put}=\max(K-S,0)"),
            ("Position (leg) value, ε = +1 long, −1 short", sp.Eq(sp.Symbol("V_i"), eps * n_ * m_ * P_)),
            ("Portfolio market value", r"V=\sum_i \varepsilon_i\, n_i\, m_i\, P_i \qquad TV = V - \sum_i "
                                       r"\varepsilon_i n_i m_i IV_i"),
            ("Delta (call / put)", r"\Delta_c=" + sp.latex(N_(d1_)) + r",\qquad \Delta_p = N(d_1)-1"),
            ("Gamma", sp.Eq(sp.Symbol(r"\Gamma"),
                            sp.exp(-d1_ ** 2 / 2) / sp.sqrt(2 * sp.pi) / (S_ * sig_ * sp.sqrt(T_)))),
            ("Vega", sp.Eq(sp.Symbol(r"\nu"), S_ * sp.sqrt(T_) * sp.exp(-sp.Symbol("d_1") ** 2 / 2)
                           / sp.sqrt(2 * sp.pi))),
            ("Theta (call)", r"\Theta_c=-\frac{S\,\varphi(d_1)\,\sigma}{2\sqrt{T}}-rKe^{-rT}N(d_2)"),
            ("Position Greeks", r"G_{pos}=\sum_i \varepsilon_i\, n_i\, m_i\, G_i \quad(\Delta_{stock}=1)"),
        ],
        example=(
            "S = 100, r = 5 %, σ = 25 %, T = 0.5. **Long 10 calls K = 100** and **short 10 calls K = 110** "
            "(a bull call spread), multiplier 1.\n\n"
            "1. d1 = [ln(100/100) + (0.05 + 0.03125)·0.5] / (0.25·√0.5) = 0.2298, d2 = 0.0530 → "
            "c(100) = 100·N(0.2298) − 100e^{−0.025}·N(0.0530) ≈ **8.26**.\n"
            "2. Same for K = 110: d1 = −0.3093, d2 = −0.4861 → c(110) ≈ **4.23**.\n"
            "3. Market value = +10·8.26 − 10·4.23 ≈ **40.3** (net premium paid).\n"
            "4. Intrinsic value = 10·max(100−100,0) − 10·max(100−110,0) = **0** → time value ≈ 40.3.\n"
            "5. Max payoff at expiry = 10·(110−100) = 100, so max profit ≈ 100 − 40.3 = 59.7.\n\n"
            "Load these legs in the table (they are the first two default rows) to verify."
        ),
    )


# ----------------------------------------------------------------------------
# Computation
# ----------------------------------------------------------------------------

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in COLS:
        if c not in df:
            df[c] = np.nan
    df = df[COLS]
    df["Instrument"] = df["Instrument"].fillna("Call")
    df["Position"] = df["Position"].fillna("Long")
    num_defaults = {"Quantity": 1, "Strike K": 100.0, "Maturity T (years)": 0.5, "Volatility σ": 0.2,
                    "Contract multiplier": 1}
    for c, d in num_defaults.items():
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(d)
    df["Strike K"] = df["Strike K"].clip(lower=0.01)
    df["Maturity T (years)"] = df["Maturity T (years)"].clip(lower=0.0)
    df["Volatility σ"] = df["Volatility σ"].clip(lower=1e-4)
    df = df[df["Quantity"] != 0]
    return df.reset_index(drop=True)


def _unit_value(inst, S, K, T, r, sig):
    if inst == "Stock":
        return S
    return bs_price(S, K, T, r, sig, inst.lower())


def _portfolio_value(legs: pd.DataFrame, S, r, dt=0.0):
    """Portfolio value at spot S after dt years have elapsed (vectorised over S if array)."""
    S = np.atleast_1d(np.asarray(S, dtype=float))
    out = np.zeros_like(S)
    for _, L in legs.iterrows():
        w = (1 if L["Position"] == "Long" else -1) * L["Quantity"] * L["Contract multiplier"]
        T = max(L["Maturity T (years)"] - dt, 0.0)
        out += w * np.array([_unit_value(L["Instrument"], s, L["Strike K"], T, r, L["Volatility σ"]) for s in S])
    return out


def _payoff(legs: pd.DataFrame, S):
    S = np.asarray(S, dtype=float)
    out = np.zeros_like(S)
    for _, L in legs.iterrows():
        w = (1 if L["Position"] == "Long" else -1) * L["Quantity"] * L["Contract multiplier"]
        if L["Instrument"] == "Stock":
            out += w * S
        elif L["Instrument"] == "Call":
            out += w * np.maximum(S - L["Strike K"], 0)
        else:
            out += w * np.maximum(L["Strike K"] - S, 0)
    return out


def _evaluate(legs: pd.DataFrame, S, r) -> pd.DataFrame:
    rows = []
    for i, L in legs.iterrows():
        inst, pos = L["Instrument"], L["Position"]
        sign = 1 if pos == "Long" else -1
        qty, mult = L["Quantity"], L["Contract multiplier"]
        K, T, sig = L["Strike K"], L["Maturity T (years)"], L["Volatility σ"]
        w = sign * qty * mult
        if inst == "Stock":
            price, iv = S, S
            g = {"delta": 1.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        else:
            price = bs_price(S, K, T, r, sig, inst.lower())
            iv = intrinsic_value(S, K, inst.lower())
            g = bs_greeks(S, K, T, r, sig, inst.lower())
        rows.append({
            "Leg": i + 1,
            "Description": f"{pos} {qty:g} × {inst}" + ("" if inst == "Stock" else f" K={K:g} T={T:g}"),
            "Unit price": price,
            "Unit intrinsic": iv,
            "Unit time value": price - iv,
            "Cash flow (premium)": -w * price,
            "Market value": w * price,
            "Intrinsic value": w * iv,
            "Time value": w * (price - iv),
            "Δ": w * g["delta"],
            "Γ": w * g["gamma"],
            "Vega (1%)": w * g["vega"] / 100,
            "θ/day": w * g["theta"] / 365,
            "ρ (1%)": w * g["rho"] / 100,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------

def _spot_grid(legs, S):
    strikes = legs.loc[legs["Instrument"] != "Stock", "Strike K"].tolist()
    lo = min([S] + strikes) * 0.6
    hi = max([S] + strikes) * 1.4
    return np.linspace(max(lo, 0.01), hi, 161)


def _payoff_fig(legs, S, r, V0):
    grid = _spot_grid(legs, S)
    pay = _payoff(legs, grid)
    now = _portfolio_value(legs, grid, r)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=pay, name="Value at expiry (payoff)", line=dict(width=2.5, color="#4f7cff")))
    fig.add_trace(go.Scatter(x=grid, y=now, name="Current BS value", line=dict(width=2, dash="dash",
                                                                               color="#f28e2b")))
    fig.add_trace(go.Scatter(x=grid, y=pay - V0, name="P&L at expiry (net of cost)",
                             line=dict(width=1.5, dash="dot", color="#59a14f")))
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.add_vline(x=S, line_color="gray", line_dash="dot", annotation_text=f"S = {S:,.2f}")
    fig.update_layout(height=430, xaxis_title="Spot price at evaluation", yaxis_title="Position value",
                      hovermode="x unified", legend=dict(orientation="h", y=1.1), margin=dict(t=40, b=40))
    return fig


def _heatmap_fig(legs, S, r, V0, horizon_days):
    spots = np.linspace(S * 0.7, S * 1.3, 31)
    days = np.linspace(0, horizon_days, min(horizon_days, 30) + 1).round().astype(int)
    days = np.unique(days)
    Z = np.array([_portfolio_value(legs, spots, r, dt=d / 365) - V0 for d in days])
    lim = float(np.nanmax(np.abs(Z))) or 1.0
    fig = go.Figure(go.Heatmap(x=np.round(spots, 2), y=days, z=Z, colorscale=[[0, "#D64545"], [0.5, "#F3F5FA"], [1, "#0E9F6E"]], zmid=0, zmin=-lim, zmax=lim,
                               colorbar=dict(title="P&L"),
                               hovertemplate="S=%{x}<br>day %{y}<br>P&L=%{z:,.2f}<extra></extra>"))
    fig.update_layout(height=430, xaxis_title="Spot price", yaxis_title="Days elapsed", margin=dict(t=30, b=40))
    return fig


def _mpl_payoff(legs, S, r, V0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = _spot_grid(legs, S)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(grid, _payoff(legs, grid), lw=2.2, color="#4f7cff", label="Value at expiry (payoff)")
    ax.plot(grid, _portfolio_value(legs, grid, r), lw=1.8, ls="--", color="#f28e2b", label="Current BS value")
    ax.plot(grid, _payoff(legs, grid) - V0, lw=1.4, ls=":", color="#59a14f", label="P&L at expiry")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(S, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Spot price")
    ax.set_ylabel("Position value")
    ax.set_title("Portfolio payoff vs current value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------

def render():
    page_header("Portfolio Evaluation",
                "Market value, intrinsic value and Greeks of a portfolio of European options on a stock.",
                "portfolio")
    _help()

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("<div class='panel-label'>Inputs</div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Underlying asset**")
            S = spot_input("pf_spot", default=100.0)
            st.markdown("**Market parameters**")
            r = st.number_input("Risk-free rate r (cont. comp., decimal)", value=0.05, step=0.005,
                                format="%.4f", key="pf_r")

    st.markdown("<div class='panel-label'>Position legs</div>", unsafe_allow_html=True)
    st.caption("Add or remove rows freely. For stock legs, strike, maturity and volatility are ignored. "
               "Quantity = number of contracts (or shares); multiplier = shares per contract (e.g. 100).")
    edited = st.data_editor(
        DEFAULT_LEGS, num_rows="dynamic", width="stretch", key="pf_legs",
        column_config={
            "Instrument": st.column_config.SelectboxColumn(options=["Call", "Put", "Stock"], required=True),
            "Position": st.column_config.SelectboxColumn(options=["Long", "Short"], required=True),
            "Quantity": st.column_config.NumberColumn(min_value=0, step=1, format="%d"),
            "Strike K": st.column_config.NumberColumn(min_value=0.01, step=1.0, format="%.2f"),
            "Maturity T (years)": st.column_config.NumberColumn(min_value=0.0, step=0.05, format="%.3f"),
            "Volatility σ": st.column_config.NumberColumn(min_value=0.0001, step=0.01, format="%.3f"),
            "Contract multiplier": st.column_config.NumberColumn(min_value=1, step=1, format="%d", default=1),
        },
    )
    legs = _clean(pd.DataFrame(edited))
    if legs.empty:
        with right:
            st.info("Add at least one leg with a non-zero quantity.")
        return

    res = _evaluate(legs, S, r)
    V0 = float(res["Market value"].sum())
    IV = float(res["Intrinsic value"].sum())
    TV = float(res["Time value"].sum())
    paid = float(res.loc[res["Cash flow (premium)"] < 0, "Cash flow (premium)"].sum()) * -1
    received = float(res.loc[res["Cash flow (premium)"] > 0, "Cash flow (premium)"].sum())

    with right:
        st.markdown("<div class='panel-label'>Results</div>", unsafe_allow_html=True)
        if V0 >= 0:
            verdict("neutral", "Net debit position", f"{V0:,.2f}", "Cost to set up today",
                    "Long value exceeds short value.")
        else:
            verdict("loss", "Net credit position", f"{-V0:,.2f}", "Collected today",
                    "You carry liabilities on the short legs.")
        m = st.columns(3)
        m[0].metric("Market value", f"{V0:,.2f}")
        m[1].metric("Intrinsic value", f"{IV:,.2f}")
        m[2].metric("Time value", f"{TV:,.2f}")
        m = st.columns(4)
        m[0].metric("Net Δ", f"{res['Δ'].sum():,.3f}", help="Change in value for +1 in spot")
        m[1].metric("Net Γ", f"{res['Γ'].sum():,.4f}")
        m[2].metric("Net vega (per 1% σ)", f"{res['Vega (1%)'].sum():,.3f}")
        m[3].metric("Net θ / day", f"{res['θ/day'].sum():,.3f}")
        m = st.columns(3)
        m[0].metric("Premium paid (long legs)", f"{paid:,.2f}")
        m[1].metric("Premium received (short legs)", f"{received:,.2f}")
        m[2].metric("Net cost to enter", f"{paid - received:,.2f}",
                    help="Positive = net debit (you pay); negative = net credit (you receive).")
        st.plotly_chart(_payoff_fig(legs, S, r, V0), width="stretch", key="pf_payoff_chart")
        opt = legs[legs["Instrument"] != "Stock"]
        if opt["Maturity T (years)"].nunique() > 1:
            st.caption("Legs have different maturities: the expiry curve uses each leg's own payoff.")

    st.markdown("<div class='panel-label'>Per-leg valuation</div>", unsafe_allow_html=True)
    st.dataframe(res.style.format({c: "{:,.4f}" for c in res.columns if c not in ("Leg", "Description")}),
                 width="stretch", hide_index=True)
    download_table(res, "portfolio_valuation", key="pf_dl")

    st.markdown("<div class='panel-label'>P&L heatmap (spot × days)</div>", unsafe_allow_html=True)
    max_T = float(legs["Maturity T (years)"].max())
    max_days = max(int(round(max_T * 365)), 1)
    horizon = st.slider("Horizon (days elapsed)", 1, max_days, min(max_days, max(1, max_days // 2)),
                        key="pf_horizon")
    st.plotly_chart(_heatmap_fig(legs, S, r, V0, horizon), width="stretch", key="pf_heat_chart")
    st.caption("P&L relative to today's market value, repricing every leg with Black-Scholes at the "
               "remaining maturity (σ and r held constant).")

    st.markdown("<div class='panel-label'>Report</div>", unsafe_allow_html=True)
    if st.toggle("Prepare PDF report", key="pf_pdf_toggle"):
        import matplotlib.pyplot as plt
        fig = _mpl_payoff(legs, S, r, V0)
        summary = (
            f"Spot S = {S:,.4f}; risk-free rate r = {r:.4%}; {len(legs)} legs.\n"
            f"Market value = {V0:,.4f}; intrinsic value = {IV:,.4f}; time value = {TV:,.4f}.\n"
            f"Premium paid = {paid:,.4f}; premium received = {received:,.4f}; net cost = {paid - received:,.4f}.\n"
            f"Net Delta = {res['Δ'].sum():,.4f}; Gamma = {res['Γ'].sum():,.5f}; "
            f"Vega (1%) = {res['Vega (1%)'].sum():,.4f}; Theta/day = {res['θ/day'].sum():,.4f}."
        )
        method = ("Each option leg is priced with Black-Scholes (European, no dividends). Leg value = sign x "
                  "quantity x multiplier x unit price, sign = +1 long / -1 short. Intrinsic value = max(S-K,0) "
                  "for calls, max(K-S,0) for puts; time value = market value - intrinsic value. Position Greeks "
                  "are signed, weighted sums of per-option Greeks.")
        pdf_tbl = res[["Leg", "Unit price", "Market value", "Intrinsic value", "Time value", "Δ", "Γ",
                       "Vega (1%)", "θ/day"]]
        data = pdf_report("Portfolio Evaluation Report", [("Summary", summary), ("Method", method)],
                          tables=[("Input legs", legs), ("Per-leg valuation", pdf_tbl)], figures=[fig])
        plt.close(fig)
        pdf_button("Download PDF report", data, "portfolio_report.pdf", key="pf_pdf")
