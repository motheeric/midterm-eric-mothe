"""Module 3 - Delta and Delta-Gamma hedging calculator.

Daily re-hedging of an option position along a simulated (GBM) or historical
spot path, with cash account, P&L of hedged vs unhedged book, charts,
adjustment reports and a printable PDF.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from common import (S_, K_, T_, r_, sig_, N_, d1_, bs_price, bs_greeks, spot_input, fetch_history,
                    help_button, download_table, pdf_report, pdf_button, step_list, page_header, verdict)

DT = 1 / 252
COLORS = {"Unhedged": "#8A93A8", "Delta": "#6A4CFF", "Delta-Gamma": "#D08A00"}


# ----------------------------------------------------------------------------
# Core engine
# ----------------------------------------------------------------------------

def simulate_hedge(path, K, T, r, sigma, opt, side, n_opt, method, hedge=None, tc_bps=0.0):
    """Re-hedge daily along `path` (path[0] = today's spot).

    method: "Delta" or "Delta-Gamma". hedge = dict(opt, K, T) for the gamma hedge option.
    The hedge instruments are financed through a cash account accruing r, so the
    initial hedged book value equals the initial position value.
    Returns a DataFrame (one row per day) of the daily adjustment report.
    """
    sgn = 1 if side == "Long" else -1
    rows = []
    shares = n_h = 0.0
    cash = 0.0
    v0 = None
    for d, S in enumerate(path):
        tau = max(T - d * DT, 0.0)
        g = bs_greeks(S, K, tau, r, sigma, opt)
        V = bs_price(S, K, tau, r, sigma, opt)
        pos_val = sgn * n_opt * V
        delta_p, gamma_p = sgn * n_opt * g["delta"], sgn * n_opt * g["gamma"]
        Vh = dh = gh = 0.0
        if method == "Delta-Gamma":
            tau_h = max(hedge["T"] - d * DT, 0.0)
            gH = bs_greeks(S, hedge["K"], tau_h, r, sigma, hedge["opt"])
            Vh, dh, gh = bs_price(S, hedge["K"], tau_h, r, sigma, hedge["opt"]), gH["delta"], gH["gamma"]
        if d > 0:
            cash *= math.exp(r * DT)
        # target hedge
        if method == "Delta-Gamma":
            new_nh = -gamma_p / gh if gh > 1e-10 else 0.0
        else:
            new_nh = 0.0
        new_shares = -(delta_p + new_nh * dh)
        tr_sh, tr_h = new_shares - shares, new_nh - n_h
        cost = tc_bps / 1e4 * (abs(tr_sh) * S + abs(tr_h) * Vh)
        cash -= tr_sh * S + tr_h * Vh + cost
        shares, n_h = new_shares, new_nh
        total = pos_val + shares * S + n_h * Vh + cash
        if v0 is None:
            v0, pos0 = total, pos_val
        rows.append({
            "Day": d, "Spot S": S, "T left (y)": tau,
            "Option Δ": g["delta"], "Option Γ": g["gamma"],
            "Position Δ": delta_p, "Position Γ": gamma_p,
            "Hedge opts held": n_h, "Hedge opt trade": tr_h,
            "Shares held": shares, "Share trade": tr_sh,
            "Cash": cash, "Hedged value": total,
            "Hedged P&L": total - v0, "Unhedged P&L": pos_val - pos0,
        })
    df = pd.DataFrame(rows)
    df["Hedged daily P&L"] = df["Hedged P&L"].diff().fillna(0.0)
    df["Unhedged daily P&L"] = df["Unhedged P&L"].diff().fillna(0.0)
    return df


def gbm_paths(S0, mu, sigma, days, n_paths, seed, shock_day=None, shock_pct=0.0):
    rng = np.random.default_rng(int(seed))
    z = rng.standard_normal((n_paths, days))
    rets = (mu - 0.5 * sigma ** 2) * DT + sigma * math.sqrt(DT) * z
    if shock_day and 1 <= shock_day <= days and shock_pct:
        rets[:, shock_day - 1] += math.log(1 + shock_pct / 100)
    paths = S0 * np.exp(np.cumsum(rets, axis=1))
    return np.hstack([np.full((n_paths, 1), S0), paths])


def parse_prices(text):
    vals = []
    for tok in text.replace(";", ",").replace("\n", ",").replace(" ", ",").split(","):
        tok = tok.strip()
        if tok:
            try:
                v = float(tok)
                if v > 0:
                    vals.append(v)
            except ValueError:
                pass
    return vals


# ----------------------------------------------------------------------------
# Help
# ----------------------------------------------------------------------------

def _help():
    nh, Gp, Gh, Dp, Dh, ns = sp.symbols(r"n_h \Gamma_p \Gamma_h \Delta_p \Delta_h n_S")
    delta_call = N_(d1_)
    gamma = sp.exp(-d1_ ** 2 / 2) / (sp.sqrt(2 * sp.pi) * S_ * sig_ * sp.sqrt(T_))
    help_button(
        "hdg", "Delta & Delta-Gamma Hedging",
        theory=(
            "A dealer holding options is exposed to moves in the underlying. **Delta hedging** holds "
            "−Δ shares per unit of position Δ so that the book is insensitive to *small* spot moves. "
            "Because Δ changes with S (convexity, measured by Γ) and with time, the hedge must be "
            "**rebalanced** (here: daily). Between rebalancings a delta-hedged short option loses "
            "≈ ½Γ(ΔS)² on large moves and earns theta - the classic gamma/theta trade-off.\n\n"
            "**Delta-gamma hedging** first adds a second option (the hedge option) in quantity n_h to "
            "cancel the book's Γ, then uses shares to cancel the remaining Δ. The book is then protected "
            "against both first- and second-order spot moves, and hedging errors are much smaller.\n\n"
            "Each day the module recomputes Black-Scholes Greeks with maturity reduced by 1/252, trades "
            "the difference between target and current holdings, and finances the trades through a "
            "cash account that accrues interest at r."
        ),
        assumptions=[
            "Black-Scholes world: lognormal (GBM) spot, constant r and σ, no dividends.",
            "European options; Greeks and prices from Black-Scholes; same σ for all options.",
            "Rebalancing once per trading day (Δt = 1/252); fractional shares/options allowed.",
            "Hedge trades are self-financed by borrowing/lending cash at r; optional proportional costs.",
            "The gamma hedge option must have Γ_h > 0 (typically a longer-dated option, T2 > T).",
        ],
        formulas=[
            ("Black-Scholes d1", sp.Eq(sp.Symbol("d_1"), d1_)),
            ("Delta of a call (put: N(d1) − 1)", sp.Eq(sp.Symbol(r"\Delta_{call}"), delta_call)),
            ("Gamma (call = put)", sp.Eq(sp.Symbol(r"\Gamma"), gamma)),
            ("Delta-neutral condition", sp.Eq(Dp + ns, 0)),
            ("Delta hedge: shares to hold", sp.Eq(ns, -Dp)),
            ("Delta-gamma neutral: hedge options", sp.Eq(nh, -Gp / Gh)),
            ("Delta-gamma neutral: shares", sp.Eq(ns, -(Dp + nh * Dh))),
            ("Hedging P&L over one day (delta-hedged)",
             r"\delta\Pi \approx \tfrac12\,\Gamma_p\,(\delta S)^2 + \Theta_p\,\delta t"),
        ],
        example=(
            "Short 100 calls, S = 100, K = 100, T = 0.5, r = 5%, σ = 20%. Call Δ = 0.598, Γ = 0.0274 → "
            "position Δ_p = −59.8, Γ_p = −2.74.\n\n"
            "- **Delta hedge:** buy 59.8 shares.\n"
            "- **Delta-gamma:** hedge with a call K2 = 105, T2 = 1 (Δ_h = 0.542, Γ_h = 0.01984): "
            "n_h = −(−2.736)/0.01984 ≈ 137.9 calls bought, then shares = −(−59.77 + 137.9 × 0.542) ≈ −15.0 "
            "(sell 15.0 shares).\n"
            "- Next day S moves → recompute Δ, Γ with T − 1/252 and trade the differences."
        ),
    )


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------

def _plotly_spot(path_df, fan=None):
    fig = go.Figure()
    if fan is not None:
        days = np.arange(fan.shape[1])
        p5, p25, p50, p75, p95 = np.percentile(fan, [5, 25, 50, 75, 95], axis=0)
        fig.add_trace(go.Scatter(x=days, y=p95, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=days, y=p5, fill="tonexty", fillcolor="rgba(106,76,255,0.12)",
                                 line=dict(width=0), name="5–95%"))
        fig.add_trace(go.Scatter(x=days, y=p75, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=days, y=p25, fill="tonexty", fillcolor="rgba(106,76,255,0.25)",
                                 line=dict(width=0), name="25–75%"))
        fig.add_trace(go.Scatter(x=days, y=p50, line=dict(color="#6A4CFF", dash="dot"), name="Median"))
    fig.add_trace(go.Scatter(x=path_df["Day"], y=path_df["Spot S"], mode="lines+markers",
                             line=dict(color="#14213D", width=2.5), name="Path used"))
    fig.update_layout(title="Spot path", xaxis_title="Day", yaxis_title="S", height=360,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


def _plotly_lines(results, col, title, ytitle, unhedged=False):
    fig = go.Figure()
    first = next(iter(results.values()))
    if unhedged:
        fig.add_trace(go.Scatter(x=first["Day"], y=first["Unhedged P&L"], mode="lines+markers",
                                 name="Unhedged", line=dict(color=COLORS["Unhedged"], dash="dash")))
    for m, df in results.items():
        fig.add_trace(go.Scatter(x=df["Day"], y=df[col], mode="lines+markers", name=m,
                                 line=dict(color=COLORS[m], width=2.5)))
    fig.update_layout(title=title, xaxis_title="Day", yaxis_title=ytitle, height=360,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


def _mpl_figures(results, fan_final=None):
    figs = []
    first = next(iter(results.values()))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(first["Day"], first["Spot S"], "o-", color="#14213D")
    ax[0].set_title("Spot path"); ax[0].set_xlabel("Day"); ax[0].grid(alpha=.3)
    for m, df in results.items():
        ax[1].plot(df["Day"], df["Shares held"], "o-", color=COLORS[m], label=f"{m}: shares")
        if m == "Delta-Gamma":
            ax[1].plot(df["Day"], df["Hedge opts held"], "s--", color=COLORS[m], alpha=.6,
                       label="Delta-Gamma: hedge options")
    ax[1].set_title("Hedge holdings"); ax[1].set_xlabel("Day"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    fig.tight_layout(); figs.append(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(first["Day"], first["Unhedged P&L"], "o--", color=COLORS["Unhedged"], label="Unhedged")
    for m, df in results.items():
        ax.plot(df["Day"], df["Hedged P&L"], "o-", color=COLORS[m], label=f"{m} hedged")
    ax.axhline(0, color="k", lw=.6); ax.set_title("Cumulative P&L"); ax.set_xlabel("Day")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); figs.append(fig)

    if fan_final is not None:
        fig, ax = plt.subplots(figsize=(10, 3.6))
        for name, vals in fan_final.items():
            ax.hist(vals, bins=40, alpha=.5, color=COLORS[name], label=name)
        ax.set_title("Distribution of final P&L across simulated paths"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); figs.append(fig)
    return figs


REPORT_COLS = ["Day", "Spot S", "Option Δ", "Option Γ", "Position Δ", "Position Γ", "Shares held", "Share trade",
               "Hedge opts held", "Hedge opt trade", "Cash", "Hedged daily P&L", "Hedged P&L", "Unhedged P&L"]


def _trade_text(x, unit):
    if abs(x) < 1e-9:
        return "hold"
    return f"{'buy' if x > 0 else 'sell'} {abs(x):,.2f} {unit}"


# ----------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------

def _label(text):
    st.markdown(f"<div class='panel-label'>{text}</div>", unsafe_allow_html=True)


def _inputs():
    """Left column: every input. Returns a dict, or None if the price list is invalid."""
    _label("Position")
    c = st.columns(2)
    opt = c[0].selectbox("Option type", ["call", "put"], key="hdg_opt")
    side = c[1].selectbox("Side", ["Short", "Long"], key="hdg_side")
    c = st.columns(2)
    n_opt = c[0].number_input("Number of options", min_value=1.0, value=100.0, step=10.0, key="hdg_n")
    K = c[1].number_input("Strike K", min_value=0.01, value=100.0, step=1.0, key="hdg_K")
    c = st.columns(3)
    T = c[0].number_input("Maturity T (y)", min_value=0.01, value=0.5, step=0.05, key="hdg_T")
    r = c[1].number_input("Rate r (%)", value=5.0, step=0.25, key="hdg_r") / 100
    sigma = c[2].number_input("Vol σ (%)", min_value=0.1, value=20.0, step=1.0, key="hdg_sig") / 100

    _label("Hedge method")
    method_sel = st.selectbox("Hedging method", ["Delta", "Delta-Gamma", "Compare both"], key="hdg_method")
    methods = ["Delta", "Delta-Gamma"] if method_sel == "Compare both" else [method_sel]
    hedge = None
    if "Delta-Gamma" in methods:
        _label("Second option (gamma hedge)")
        c = st.columns(2)
        h_opt = c[0].selectbox("Hedge option type", ["call", "put"], key="hdg_h_opt")
        K2 = c[1].number_input("Strike K2", min_value=0.01, value=105.0, step=1.0, key="hdg_K2")
        c = st.columns(2)
        T2 = c[0].number_input("Maturity T2 (y)", min_value=0.01, value=1.0, step=0.05, key="hdg_T2")
        tc = c[1].number_input("Cost (bps)", min_value=0.0, value=0.0, step=1.0, key="hdg_tc_dg")
        hedge = {"opt": h_opt, "K": K2, "T": T2}
        if T2 <= T:
            st.caption("Tip: a hedge option with T2 > T keeps Γ_h > 0 over the whole horizon.")
    else:
        tc = st.number_input("Transaction cost (bps)", min_value=0.0, value=0.0, step=1.0, key="hdg_tc_d")

    _label("Spot path")
    source = st.radio("Path source", ["Simulator (forecast)", "Historical"], horizontal=True, key="hdg_src")
    fan = None
    if source.startswith("Simulator"):
        S0 = spot_input("hdg_spot", 100.0)
        c = st.columns(2)
        days = c[0].slider("Days", 1, 5, 5, key="hdg_days")
        seed = c[1].number_input("Seed", min_value=0, value=42, step=1, key="hdg_seed")
        c = st.columns(2)
        mu = c[0].number_input("Drift μ (%/yr)", value=8.0, step=1.0, key="hdg_mu") / 100
        sim_sig = c[1].number_input("Simulation σ (%/yr)", min_value=0.1, value=float(sigma * 100), step=1.0,
                                    key="hdg_simsig") / 100
        n_paths = st.number_input("Fan-chart paths", min_value=1, max_value=2000, value=300, step=50,
                                  key="hdg_npaths")
        use_shock = st.checkbox("Add a shock", key="hdg_shock_on")
        c = st.columns(2)
        shock_day = c[0].number_input("Shock day", 1, days, min(2, days), key="hdg_shock_day",
                                      disabled=not use_shock)
        shock_pct = c[1].number_input("Shock (%)", value=-5.0, step=1.0, key="hdg_shock_pct",
                                      disabled=not use_shock)
        sd, sp_ = (int(shock_day), shock_pct) if use_shock else (None, 0.0)
        fan = gbm_paths(S0, mu, sim_sig, days, int(n_paths), seed, sd, sp_)
        path = fan[0]
        path_label = (f"GBM forecast, S0 = {S0:.2f}, μ = {mu:.1%}, σ_sim = {sim_sig:.1%}, {days} day(s), "
                      f"seed {int(seed)}" + (f", shock {shock_pct:+.1f}% on day {sd}" if sd else ""))
    else:
        c = st.columns([2, 1])
        ticker = c[0].text_input("Ticker (Yahoo Finance)", value="AAPL", key="hdg_hist_ticker").strip().upper()
        n_days = c[1].number_input("Last N days", min_value=1, max_value=60, value=5, step=1, key="hdg_hist_n")
        hist = fetch_history(ticker, "6mo") if ticker else None
        if hist is not None and len(hist) >= 2:
            h = hist.tail(int(n_days) + 1)
            path = h.to_numpy(dtype=float)
            path_label = f"Historical closes of {ticker}, {h.index[0].date()} → {h.index[-1].date()}"
            st.caption(path_label)
        else:
            st.warning("Online history unavailable. Paste closing prices (oldest first) instead.")
            txt = st.text_area("Closing prices", "100, 101.2, 99.8, 98.5, 100.4, 102.1", key="hdg_hist_txt")
            vals = parse_prices(txt)
            if len(vals) < 2:
                st.error("Enter at least two positive prices.")
                return None
            path = np.array(vals, dtype=float)
            path_label = f"User-supplied price list ({len(vals) - 1} day(s))"
        if len(path) - 1 > T * 252:
            st.warning("Path is longer than the option's life; the option expires before the end.")
    return dict(opt=opt, side=side, n_opt=n_opt, K=K, T=T, r=r, sigma=sigma, methods=methods, hedge=hedge,
                tc=tc, fan=fan, path=path, path_label=path_label)


def render():
    page_header("Delta & Delta-Gamma Hedging",
                "Re-hedge an option position every day along a simulated or historical spot path "
                "and see how much of the P&L swing the hedge removes.", "hedging")
    _help()

    left, right = st.columns([1, 2], gap="large")
    with left:
        inp = _inputs()
    if inp is None:
        return
    opt, side, n_opt, K, T, r, sigma = (inp[k] for k in ("opt", "side", "n_opt", "K", "T", "r", "sigma"))
    methods, hedge, tc, fan, path, path_label = (inp[k] for k in
                                                 ("methods", "hedge", "tc", "fan", "path", "path_label"))

    # ---------------- Compute ----------------
    results = {m: simulate_hedge(path, K, T, r, sigma, opt, side, n_opt, m, hedge, tc) for m in methods}
    first = results[methods[0]]
    S_init = path[0]
    if abs(K / S_init - 1) > 0.3 or (isinstance(hedge, dict) and abs(hedge['K'] / S_init - 1) > 0.3):
        st.warning(f"The strike (K = {K:,.2f}) is far from today's spot ({S_init:,.2f}), so the option is deep in or out "
                   f"of the money and the hedge barely changes. For a meaningful test, set K (and K2) close to the spot.")
    last = {m: df.iloc[-1] for m, df in results.items()}
    un = first["Unhedged P&L"].iloc[-1]
    n_days = len(path) - 1

    def _reduction(m):
        h = last[m]["Hedged P&L"]
        return (1 - abs(h) / abs(un)) if abs(un) > 1e-9 else float("nan")

    best = max(methods, key=lambda m: -abs(last[m]["Hedged P&L"]))
    red = _reduction(best)

    fan_final, fan_tbl = None, None
    if fan is not None and len(fan) > 1:
        fan_final = {"Unhedged": []}
        for m in methods:
            fan_final[m] = []
        for p in fan:
            for j, m in enumerate(methods):
                df = simulate_hedge(p, K, T, r, sigma, opt, side, n_opt, m, hedge, tc)
                if j == 0:
                    fan_final["Unhedged"].append(df["Unhedged P&L"].iloc[-1])
                fan_final[m].append(df["Hedged P&L"].iloc[-1])
        fan_final = {k: np.array(v) for k, v in fan_final.items()}
        fan_tbl = pd.DataFrame({
            "Strategy": list(fan_final),
            "Mean P&L": [v.mean() for v in fan_final.values()],
            "Std dev": [v.std() for v in fan_final.values()],
            "5% quantile (VaR)": [np.percentile(v, 5) for v in fan_final.values()],
            "95% quantile": [np.percentile(v, 95) for v in fan_final.values()],
            "Worst": [v.min() for v in fan_final.values()],
        })

    # ---------------- Key results (right column) ----------------
    with right:
        if math.isnan(red):
            kind, head = "neutral", f"{best} hedging: the unhedged position barely moved"
        elif red >= 0:
            kind = "gain" if red >= 0.5 else "neutral"
            head = f"{best} hedging cut the P&L swing by {red:.0%}"
        else:
            kind, head = "loss", f"{best} hedging made the P&L swing {-red:.0%} larger"
        detail = f"Hedged P&L after {n_days} day(s): {last[best]['Hedged P&L']:,.2f}. Unhedged P&L: {un:,.2f}. Final spot {path[-1]:,.2f} ({path[-1] / S_init - 1:+.2%})."
        if len(methods) > 1:
            other = [m for m in methods if m != best][0]
            detail += f" {other} hedged P&L: {last[other]['Hedged P&L']:,.2f}."
        verdict(kind, head, (f"{abs(red):.0%}" if not math.isnan(red) else "n/a"),
                ("smaller" if (math.isnan(red) or red >= 0) else "larger") + " P&L swing than the unhedged position", detail)

        cols = st.columns(2 + len(methods))
        cols[0].metric("Final spot", f"{path[-1]:,.2f}", f"{path[-1] / S_init - 1:+.2%}")
        cols[1].metric("Unhedged P&L", f"{un:,.2f}")
        for i, m in enumerate(methods):
            rm = _reduction(m)
            cols[2 + i].metric(f"{m} hedged P&L", f"{last[m]['Hedged P&L']:,.2f}",
                               f"{rm:.0%} smaller |P&L|" if not math.isnan(rm) else None, delta_color="off")
        st.plotly_chart(_plotly_lines(results, "Hedged P&L", "Cumulative P&L: hedged vs unhedged", "P&L", True),
                        width="stretch", key="hdg_ch_pnl")

    # ---------------- Day-0 hedge ----------------
    st.markdown("### Initial hedge (day 0)")
    cols = st.columns(2 + len(methods))
    cols[0].metric("Position Δ", f"{first['Position Δ'].iloc[0]:,.3f}")
    cols[1].metric("Position Γ", f"{first['Position Γ'].iloc[0]:,.4f}")
    for i, m in enumerate(methods):
        d0 = results[m].iloc[0]
        txt = _trade_text(d0["Share trade"], "shares")
        if m == "Delta-Gamma":
            txt = _trade_text(d0["Hedge opt trade"], f"{hedge['opt']}s K2") + " · " + txt
        cols[2 + i].metric(f"{m} hedge", txt)
    steps = []
    for m in methods:
        d0 = results[m].iloc[0]
        if m == "Delta":
            steps.append(f"<b>Delta:</b> n_S = −Δ_p = {d0['Shares held']:,.3f} shares.")
        else:
            gh = bs_greeks(S_init, hedge["K"], hedge["T"], r, sigma, hedge["opt"])
            steps.append(f"<b>Delta-Gamma:</b> Γ_h = {gh['gamma']:.5f}, Δ_h = {gh['delta']:.4f} → "
                         f"n_h = −Γ_p/Γ_h = {d0['Hedge opts held']:,.3f}; then n_S = −(Δ_p + n_h·Δ_h) = "
                         f"{d0['Shares held']:,.3f} shares.")
    steps.append("Each following day: recompute Greeks with T − 1/252 at the new spot, trade the difference, "
                 "accrue interest on cash.")
    step_list(steps)

    # ---------------- Other charts ----------------
    st.markdown("### Spot path and hedge holdings")
    c1, c2 = st.columns(2)
    c1.plotly_chart(_plotly_spot(first, fan if fan is not None and len(fan) > 1 else None),
                    width="stretch", key="hdg_ch_spot")
    figh = go.Figure()
    for m, df in results.items():
        figh.add_trace(go.Scatter(x=df["Day"], y=df["Shares held"], mode="lines+markers",
                                  name=f"{m}: shares", line=dict(color=COLORS[m], width=2.5)))
        if m == "Delta-Gamma":
            figh.add_trace(go.Scatter(x=df["Day"], y=df["Hedge opts held"], mode="lines+markers",
                                      name="Delta-Gamma: hedge options", line=dict(color=COLORS[m], dash="dot")))
    figh.update_layout(title="Hedge ratio / holdings", xaxis_title="Day", yaxis_title="Units", height=360,
                       margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
    c2.plotly_chart(figh, width="stretch", key="hdg_ch_ratio")
    figd = go.Figure()
    figd.add_trace(go.Bar(x=first["Day"], y=first["Unhedged daily P&L"], name="Unhedged",
                          marker_color=COLORS["Unhedged"]))
    for m, df in results.items():
        figd.add_trace(go.Bar(x=df["Day"], y=df["Hedged daily P&L"], name=m, marker_color=COLORS[m]))
    figd.update_layout(title="Daily P&L", barmode="group", xaxis_title="Day", height=360,
                       margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(figd, width="stretch", key="hdg_ch_daily")

    # ---------------- Fan-chart P&L distribution ----------------
    if fan_final is not None:
        st.markdown("### Projected P&L across simulated paths")
        c1, c2 = st.columns([3, 2])
        figf = go.Figure()
        for k, v in fan_final.items():
            figf.add_trace(go.Histogram(x=v, name=k, opacity=0.55, marker_color=COLORS[k], nbinsx=40))
        figf.update_layout(barmode="overlay", title=f"Final P&L distribution ({len(fan)} paths)",
                           xaxis_title="P&L", height=340, margin=dict(l=10, r=10, t=40, b=10),
                           legend=dict(orientation="h", y=-0.2))
        c1.plotly_chart(figf, width="stretch", key="hdg_ch_fan")
        c2.dataframe(fan_tbl.style.format({c: "{:,.2f}" for c in fan_tbl.columns[1:]}), hide_index=True,
                     width="stretch")

    # ---------------- Daily adjustment reports ----------------
    st.markdown("### Daily adjustment report")
    st.caption(path_label)
    tabs = st.tabs(methods)
    reports = {}
    for tab, m in zip(tabs, methods):
        with tab:
            df = results[m]
            cols = [c for c in REPORT_COLS if m == "Delta-Gamma" or "Hedge opt" not in c]
            rep = df[cols].copy()
            rep.insert(1, "Action", [
                (_trade_text(a, "hedge opts") + " · " if m == "Delta-Gamma" else "") + _trade_text(b, "shares")
                for a, b in zip(df["Hedge opt trade"], df["Share trade"])])
            reports[m] = rep
            fmt = {c: "{:,.4f}" for c in rep.columns if c not in ("Day", "Action")}
            st.dataframe(rep.style.format(fmt), hide_index=True, width="stretch")
            download_table(rep, f"hedging_report_{m.replace('-', '_').lower()}", key=f"hdg_dl_{m}")

    # ---------------- PDF ----------------
    with st.expander("Printable PDF report"):
        sections = [
            ("Position", f"{side} {n_opt:,.0f} European {opt}s, K = {K}, T = {T} y, r = {r:.2%}, "
                         f"sigma = {sigma:.2%}. Hedging method(s): {', '.join(methods)}."
                         + (f" Gamma hedge: {hedge['opt']} K2 = {hedge['K']}, T2 = {hedge['T']} y." if hedge else "")
                         + (f" Transaction costs: {tc} bps." if tc else "")),
            ("Spot path", path_label + ". Path: " + ", ".join(f"{x:.2f}" for x in path[:40])),
            ("Summary", "\n".join(
                [f"Unhedged final P&L: {un:,.2f}"]
                + [f"{m} hedged final P&L: {last[m]['Hedged P&L']:,.2f} (final cash {last[m]['Cash']:,.2f}, "
                   f"shares {last[m]['Shares held']:,.3f})" for m in methods])),
        ]
        pdf_cols = ["Day", "Spot S", "Option Δ", "Option Γ", "Shares held", "Share trade",
                    "Hedge opts held", "Cash", "Hedged P&L", "Unhedged P&L"]
        tables = [(f"Daily adjustment report - {m}",
                   results[m][[c for c in pdf_cols if m == "Delta-Gamma" or c != "Hedge opts held"]])
                  for m in methods]
        if fan_tbl is not None:
            tables.append(("Projected final P&L across simulated paths", fan_tbl))
        if st.button("Build PDF", key="hdg_pdf_build"):
            figs = _mpl_figures(results, fan_final)
            data = pdf_report("Delta / Delta-Gamma Hedging Report", sections, tables, figs)
            for f in figs:
                plt.close(f)
            pdf_button("Download PDF", data, "hedging_report.pdf", key="hdg_pdf_dl")
