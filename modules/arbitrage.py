"""Arbitrage Calculator.

Mode 1 - Put-call parity: compares market call/put with their synthetic
replicas (stock + bond + other option) and prints the exact arbitrage trades.
Mode 2 - Single option vs Black-Scholes fair value (user sigma), with implied
volatility of the market price and the delta-hedged trade.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

from common import (K_, S_, T_, bs_greeks, bs_price, download_table, help_button, implied_vol, page_header,
                    pdf_button, pdf_report, r_, sig_, spot_input, step_list, verdict)

C_, P_, B_, ST_ = sp.symbols("C P B S_T", positive=True)


# ----------------------------------------------------------------------------
# Help
# ----------------------------------------------------------------------------

def _help():
    pvk = K_ * sp.exp(-r_ * T_)
    help_button(
        "arb",
        "Arbitrage Calculator",
        theory=(
            "Two portfolios with identical payoffs at maturity must have the same price today, otherwise "
            "one can **buy the cheap one and sell the expensive one** for a riskless profit. For European "
            "options on a non-dividend stock, a call plus a zero-coupon bond paying K is worth exactly the same "
            "as a put plus one share at T (both pay max(S_T, K)). This is **put-call parity**. "
            "Rearranging it lets us build any option *synthetically* from the other option, the stock and a "
            "bond (borrowing or lending K·e^(−rT)). If the market call differs from the synthetic call by more "
            "than the transaction costs, we sell the expensive leg and buy the cheap one: the payoffs cancel "
            "in every state at T, and the price difference is locked in at t = 0.\n\n"
            "*Mode 2* compares one option with its Black-Scholes value for a volatility you believe in. This is "
            "a **model-based** (statistical) arbitrage: the profit is only realised if the stock really has that "
            "volatility and the position is delta-hedged continuously."
        ),
        assumptions=[
            "No transaction costs or taxes (optionally a fixed cost per trade can be entered).",
            "European options (exercise only at T), same strike K and maturity T for the call and the put.",
            "The underlying pays no dividends during the life of the options.",
            "Unlimited borrowing and lending at the same continuously-compounded risk-free rate r.",
            "Short selling is allowed, with full use of the proceeds.",
            "Markets are frictionless and traders take advantage of arbitrage opportunities as they appear.",
        ],
        formulas=[
            ("Put-call parity", sp.Eq(C_ + pvk, P_ + S_)),
            ("Bond (present value of the strike)", sp.Eq(B_, pvk)),
            ("Synthetic call = put + stock - bond", sp.Eq(sp.Symbol("C_{syn}"), P_ + S_ - pvk)),
            ("Synthetic put = call - stock + bond", sp.Eq(sp.Symbol("P_{syn}"), C_ - S_ + pvk)),
            ("Parity gap (arbitrage profit per share at t = 0)", sp.Eq(sp.Symbol(r"\Delta"), C_ - P_ - S_ + pvk)),
            ("Profit at maturity (future value)", sp.Eq(sp.Symbol(r"\Pi_T"), sp.Abs(sp.Symbol(r"\Delta")) * sp.exp(r_ * T_))),
            ("Conversion payoff at T (short call, long put, long stock, repay K)",
             sp.Eq(sp.Symbol(r"V_T"), -sp.Max(ST_ - K_, 0) + sp.Max(K_ - ST_, 0) + ST_ - K_)),
            ("Mode 2 - Black-Scholes call", r"C_{BS}=S\,N(d_1)-K·e^(−rT)N(d_2),\quad d_{1,2}=\frac{\ln(S/K)+(r\pm\sigma^2/2)T}{\sigma\sqrt{T}}"),
        ],
        example=(
            "S = 100, K = 100, T = 1, r = 5%  ->  PV(K) = 100·e^(−0.05) = 95.12.\n\n"
            "Market: C = 12.00, P = 5.00. Synthetic call = P + S - PV(K) = 5 + 100 - 95.12 = **9.88**.\n\n"
            "The call is overpriced by 12.00 - 9.88 = **2.12**. Strategy (*conversion*): sell the call (+12.00), "
            "buy the put (-5.00), buy one share (-100.00), borrow 95.12 (+95.12). Net cash today = **+2.12**.\n\n"
            "At T: if S_T = 80 the put pays 20, the share is worth 80, total 100 repays the loan; if S_T = 120 the "
            "share is called away for 100, which repays the loan. The position is worth 0 in every state, so the "
            "2.12 (worth 2.12·e^(0.05) = 2.23 at T) is a riskless profit."
        ),
    )


# ----------------------------------------------------------------------------
# Mode 1: put-call parity arbitrage
# ----------------------------------------------------------------------------

def _fmt(x):
    return f"{x:+,.2f}"


def _parity_inputs():
    c1, c2 = st.columns(2)
    C = c1.number_input("Market call price C", min_value=0.0, value=12.0, step=0.1, key="arb_call")
    P = c2.number_input("Market put price P", min_value=0.0, value=5.0, step=0.1, key="arb_put")
    return C, P


def _parity_mode(right, S, K, T, r, tc, n, mult, C, P):

    pvk = K * math.exp(-r * T)
    syn_call = P + S - pvk
    syn_put = C - S + pvk
    gap = C - P - (S - pvk)                    # per share
    shares = n * mult
    cost_total = 4 * tc                        # 4 trades (call, put, stock, bond)
    cost_ps = cost_total / shares if shares else 0.0

    # lower bounds sanity check
    warn = []
    if C < max(S - pvk, 0) - 1e-12:
        warn.append(f"call below its lower bound max(S - PV(K), 0) = {max(S - pvk, 0):.4f}")
    if P < max(pvk - S, 0) - 1e-12:
        warn.append(f"put below its lower bound max(PV(K) - S, 0) = {max(pvk - S, 0):.4f}")

    if abs(gap) <= cost_ps or abs(gap) < 1e-9:
        direction = None
    elif gap > 0:
        direction = "conversion"
    else:
        direction = "reversal"

    s = 1 if direction != "reversal" else -1   # +1 conversion (also used for display when no arb)
    legs = [
        ("Call", "Sell" if s > 0 else "Buy", f"{n} contract(s) (x{mult}) @ {C:.2f}", s * C),
        ("Put", "Buy" if s > 0 else "Sell", f"{n} contract(s) (x{mult}) @ {P:.2f}", -s * P),
        ("Stock", "Buy" if s > 0 else "Short-sell", f"{shares:,.0f} shares @ {S:.2f}", -s * S),
        ("Bond (PV K)", "Borrow" if s > 0 else "Lend", f"{pvk:.4f} per share for T = {T:g} y at r = {r:.2%}",
         s * pvk),
    ]
    rows = [{"Leg": l, "Action": a, "Quantity": q, "Cash flow t=0 / share": cf,
             "Cash flow t=0 total": cf * shares} for l, a, q, cf in legs]
    if tc > 0:
        rows.append({"Leg": "Transaction costs", "Action": "Pay", "Quantity": "4 trades",
                     "Cash flow t=0 / share": -cost_ps, "Cash flow t=0 total": -cost_total})
    net_ps = abs(gap) - cost_ps
    net_tot = net_ps * shares
    rows.append({"Leg": "NET", "Action": "", "Quantity": "", "Cash flow t=0 / share": net_ps if direction else s * gap - cost_ps,
                 "Cash flow t=0 total": net_tot if direction else (s * gap - cost_ps) * shares})
    df_cf = pd.DataFrame(rows)

    fv = net_ps * math.exp(r * T)
    steps = []
    if direction:
        if direction == "conversion":
            steps = [
                f"<b>Sell {n} call contract(s)</b> (K = {K:g}, T = {T:g} y) at {C:.2f} -> receive "
                f"<b>{C * shares:,.2f}</b>.",
                f"<b>Buy {n} put contract(s)</b> (same K, T) at {P:.2f} -> pay <b>{P * shares:,.2f}</b>.",
                f"<b>Buy {shares:,.0f} shares</b> at {S:.2f} -> pay <b>{S * shares:,.2f}</b>.",
                f"<b>Borrow {pvk * shares:,.2f}</b> (= PV(K) x {shares:,.0f}) at r = {r:.2%} until T; you will repay "
                f"<b>{K * shares:,.2f}</b> at maturity.",
                f"Net cash today: <b>{net_tot:+,.2f}</b> ({net_ps:+.4f} per share"
                + (f", after {cost_total:,.2f} of costs" if tc > 0 else "") + "). Invest it at r.",
                f"At T: if S_T > K the call is exercised against you - deliver the shares, receive {K * shares:,.2f}, "
                f"repay the loan. If S_T <= K exercise the put - sell the shares for {K * shares:,.2f}, repay the loan. "
                f"The hedge is worth 0 in every state; locked profit at T = <b>{fv * shares:,.2f}</b>.",
            ]
        else:
            steps = [
                f"<b>Buy {n} call contract(s)</b> (K = {K:g}, T = {T:g} y) at {C:.2f} -> pay <b>{C * shares:,.2f}</b>.",
                f"<b>Sell {n} put contract(s)</b> (same K, T) at {P:.2f} -> receive <b>{P * shares:,.2f}</b>.",
                f"<b>Short-sell {shares:,.0f} shares</b> at {S:.2f} -> receive <b>{S * shares:,.2f}</b>.",
                f"<b>Lend {pvk * shares:,.2f}</b> (buy a zero-coupon bond = PV(K) x {shares:,.0f}) at r = {r:.2%}; "
                f"it pays <b>{K * shares:,.2f}</b> at T.",
                f"Net cash today: <b>{net_tot:+,.2f}</b> ({net_ps:+.4f} per share"
                + (f", after {cost_total:,.2f} of costs" if tc > 0 else "") + "). Invest it at r.",
                f"At T: collect {K * shares:,.2f} from the bond. If S_T > K exercise the call (pay K, receive the shares) "
                f"and close the short. If S_T <= K the put is exercised against you (pay K, receive the shares) and you "
                f"close the short. Hedge worth 0 in every state; locked profit at T = <b>{fv * shares:,.2f}</b>.",
            ]

    with right:
        if direction is None:
            verdict("neutral", "No arbitrage: put-call parity holds within costs", f"{abs(gap):,.4f}",
                    "parity gap per share",
                    f"The gap does not exceed transaction costs of {cost_ps:.4f} per share, so there is no "
                    "riskless trade to make.")
        elif direction == "conversion":
            verdict("gain", "The call is overpriced: sell it and buy the synthetic call", f"{net_tot:,.2f}",
                    "profit locked in today",
                    f"The market call is {gap:.4f} per share above put + stock - PV(K). A conversion captures "
                    f"the gap, worth {fv * shares:,.2f} at maturity.")
        else:
            verdict("gain", "The call is underpriced: buy it and sell the synthetic call", f"{net_tot:,.2f}",
                    "profit locked in today",
                    f"The market call is {-gap:.4f} per share below put + stock - PV(K). A reversal captures "
                    f"the gap, worth {fv * shares:,.2f} at maturity.")
        if warn:
            st.info("Note: " + "; ".join(warn) + ". This is itself an arbitrage signal.")
        m = st.columns(2)
        m[0].metric("PV(K) = K·e^(−rT)", f"{pvk:,.4f}")
        m[1].metric("Parity gap C - P - (S - PV(K))", f"{gap:+,.4f}")
        m = st.columns(2)
        m[0].metric("Synthetic call P + S - PV(K)", f"{syn_call:,.4f}", f"market - synthetic {C - syn_call:+.4f}",
                    delta_color="off")
        m[1].metric("Synthetic put C - S + PV(K)", f"{syn_put:,.4f}", f"market - synthetic {P - syn_put:+.4f}",
                    delta_color="off")
        if direction:
            st.markdown("<div class='panel-label'>Step-by-step strategy</div>", unsafe_allow_html=True)
            step_list(steps)
            k = st.columns(3)
            k[0].metric("Profit locked at t = 0", f"{net_tot:,.2f}", f"{net_ps:.4f} / share", delta_color="off")
            k[1].metric("Future value at T", f"{fv * shares:,.2f}", f"x e^(rT) = {math.exp(r * T):.4f}",
                        delta_color="off")
            k[2].metric("Return on capital", "riskless", "zero net investment", delta_color="off")

    st.subheader("Cash flows at t = 0")
    st.dataframe(df_cf.style.format({"Cash flow t=0 / share": "{:+,.4f}", "Cash flow t=0 total": "{:+,.2f}"}),
                 hide_index=True, width="stretch")

    # payoff at T proof
    st.subheader("Payoff at maturity (proof of zero risk)")
    scen = [("S_T < K", 0.8 * K), ("S_T = K", K), ("S_T > K", 1.2 * K)]
    prow = []
    for name, st_ in scen:
        call = s * -max(st_ - K, 0)
        put = s * max(K - st_, 0)
        stock = s * st_
        bond = -s * K
        prow.append({"Scenario": f"{name} ({st_:,.2f})", "Call": call, "Put": put, "Stock": stock,
                     "Bond": bond, "Hedge total": call + put + stock + bond,
                     "Invested profit": fv if direction else 0.0})
    df_pay = pd.DataFrame(prow)
    st.dataframe(df_pay.style.format({c: "{:+,.2f}" for c in df_pay.columns if c != "Scenario"}),
                 hide_index=True, width="stretch")
    st.caption("Per share, from the arbitrageur's point of view. The hedge always nets to zero, so the only "
               "remaining cash flow is the invested t = 0 profit.")

    grid = np.linspace(0.01, 2 * K, 201)
    leg_call = s * -np.maximum(grid - K, 0)
    leg_put = s * np.maximum(K - grid, 0)
    leg_stock = s * grid
    leg_bond = -s * K * np.ones_like(grid)
    total = leg_call + leg_put + leg_stock + leg_bond + (fv if direction else 0.0)
    fig = go.Figure()
    for y, nm, col in [(leg_call, "Call leg", "#e4572e"), (leg_put, "Put leg", "#17bebb"),
                       (leg_stock, "Stock leg", "#a0a0a0"), (leg_bond, "Bond leg", "#ffc914")]:
        fig.add_trace(go.Scatter(x=grid, y=y, name=nm, line=dict(dash="dot", color=col, width=1.5)))
    fig.add_trace(go.Scatter(x=grid, y=total, name="Arbitrage position (incl. profit FV)",
                             line=dict(color="#4f7cff", width=4)))
    fig.add_vline(x=K, line_dash="dash", line_color="gray", annotation_text="K")
    fig.update_layout(height=430, xaxis_title="Stock price at maturity S_T", yaxis_title="Payoff per share at T",
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=30))
    st.plotly_chart(fig, width="stretch", key="arb_plot_parity")

    # exports
    st.subheader("Export")
    download_table(df_cf, "arbitrage_cashflows", "arb_dl_cf")
    download_table(df_pay, "arbitrage_payoff_T", "arb_dl_pay")
    verdict_txt = ("No arbitrage: parity holds within transaction costs." if not direction else
               f"{direction.title()} arbitrage. Locked profit {net_tot:,.2f} today, {fv * shares:,.2f} at T.")
    if st.button("Build PDF report", key="arb_pdf_build"):
        figs = []
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(figsize=(7, 3.8))
            for y, nm in [(leg_call, "Call"), (leg_put, "Put"), (leg_stock, "Stock"), (leg_bond, "Bond")]:
                ax.plot(grid, y, ":", label=nm)
            ax.plot(grid, total, lw=3, label="Arbitrage position")
            ax.axvline(K, color="gray", ls="--")
            ax.set_xlabel("S_T"); ax.set_ylabel("Payoff per share"); ax.legend(); ax.grid(alpha=.3)
            figs.append(f)
        except Exception:
            pass
        sections = [
            ("Inputs", f"S = {S:.4f}, K = {K:.4f}, T = {T:g} y, r = {r:.4%}, C = {C:.4f}, P = {P:.4f}, "
                       f"cost/trade = {tc:.2f}, contracts = {n} x {mult}."),
            ("Parity analysis", f"PV(K) = {pvk:.4f}; synthetic call = {syn_call:.4f}; synthetic put = {syn_put:.4f}; "
                                f"parity gap = {gap:+.4f} per share."),
            ("Verdict", verdict_txt),
        ]
        if direction:
            import re
            sections.append(("Strategy", "\n".join(f"{i}. {re.sub('<[^>]+>', '', t)}"
                                                   for i, t in enumerate(steps, 1))))
        st.session_state["arb_pdf_bytes"] = pdf_report("Arbitrage Calculator - Put-call parity", sections,
                                                       [("Cash flows at t=0", df_cf), ("Payoff at T", df_pay)], figs)
    if st.session_state.get("arb_pdf_bytes"):
        pdf_button("Download PDF report", st.session_state["arb_pdf_bytes"], "arbitrage_report.pdf", "arb_pdf_dl")


# ----------------------------------------------------------------------------
# Mode 2: single option vs Black-Scholes
# ----------------------------------------------------------------------------

def _bs_inputs():
    opt = st.selectbox("Option", ["call", "put"], key="arb_bs_type")
    c2, c3 = st.columns(2)
    mkt = c2.number_input("Market option price", min_value=0.0, value=12.0, step=0.1, key="arb_bs_mkt")
    sig = c3.number_input("Your volatility sigma (%)", min_value=0.1, value=20.0, step=0.5, key="arb_bs_sig") / 100
    return opt, mkt, sig


def _bs_metrics(fair, mkt, diff, iv, sig, delta):
    m = st.columns(2)
    m[0].metric("Black-Scholes fair value", f"{fair:,.4f}")
    m[1].metric("Market price", f"{mkt:,.4f}", f"{diff:+.4f} vs fair", delta_color="off")
    m = st.columns(2)
    m[0].metric("Implied vol of market price", "n/a" if math.isnan(iv) else f"{iv:.2%}",
                None if math.isnan(iv) else f"{(iv - sig) * 100:+.2f} vol pts vs yours", delta_color="off")
    m[1].metric("Delta (at your sigma)", f"{delta:+.4f}")
    if math.isnan(iv):
        st.info("No implied volatility exists for this price (it violates the no-arbitrage bounds). "
                "This is a pure arbitrage, not just a model view.")


def _bs_mode(right, S, K, T, r, tc, n, mult, opt, mkt, sig):

    fair = bs_price(S, K, T, r, sig, opt)
    iv = implied_vol(mkt, S, K, T, r, opt)
    g = bs_greeks(S, K, T, r, sig, opt)
    delta = g["delta"]
    shares = n * mult
    cost_ps = 3 * tc / shares if shares else 0.0
    diff = mkt - fair

    if abs(diff) <= cost_ps or abs(diff) < 1e-9:
        with right:
            verdict("neutral", "No mispricing beyond transaction costs", f"{diff:+,.4f}", "market minus fair value",
                    "The market price is consistent with your volatility, so there is no trade to make.")
            _bs_metrics(fair, mkt, diff, iv, sig, delta)
        return
    over = diff > 0

    # hedge: short option -> hold +delta shares; long option -> hold -delta shares
    hold = delta if over else -delta
    bond = hold * S + (-fair if over else fair)   # +borrow / -lend: replicates the option at fair value
    steps = [
        f"<b>{'Sell' if over else 'Buy'} {n} {opt} contract(s)</b> at {mkt:.2f} -> "
        f"{'receive' if over else 'pay'} <b>{mkt * shares:,.2f}</b>.",
        f"<b>{'Buy' if hold > 0 else 'Short-sell'} {abs(hold) * shares:,.1f} shares</b> "
        f"(|Delta| = {abs(delta):.4f} per option) at {S:.2f} -> "
        f"{'pay' if hold > 0 else 'receive'} <b>{abs(hold) * S * shares:,.2f}</b>. Position is now delta-neutral.",
        f"<b>{'Borrow' if bond > 0 else 'Lend'} {abs(bond) * shares:,.2f}</b> at r = {r:.2%} (the bond leg of the Black-Scholes replica). "
        f"Net cash today = mispricing = <b>{abs(diff) * shares:,.2f}</b>.",
        "Re-hedge regularly: re-compute Delta with your sigma and adjust the shares, financing through the bond.",
        f"At T the replicating portfolio's value matches the option payoff. Expected profit at t = 0 = "
        f"<b>{(abs(diff) - cost_ps) * shares:,.2f}</b> ({abs(diff) - cost_ps:.4f} / share), i.e. "
        f"<b>{(abs(diff) - cost_ps) * shares * math.exp(r * T):,.2f}</b> at T.",
    ]
    with right:
        verdict("gain", f"The {opt} is {'overpriced: sell it' if over else 'underpriced: buy it'} and delta-hedge",
                f"{(abs(diff) - cost_ps) * shares:,.2f}", "expected profit today",
                f"Mispriced by {abs(diff):.4f} per share (implied vol {iv:.2%} {'>' if over else '<'} your "
                f"{sig:.2%}). Model-based: the profit is expected, not guaranteed. It requires your sigma to be "
                "the true realised volatility and continuous delta re-hedging.")
        _bs_metrics(fair, mkt, diff, iv, sig, delta)
        st.markdown("<div class='panel-label'>Hedged trade</div>", unsafe_allow_html=True)
        step_list(steps)
    df = pd.DataFrame([
        {"Leg": f"{opt.title()} option", "Action": "Sell" if over else "Buy", "Units / share": -1 if over else 1,
         "Cash flow t=0 / share": mkt if over else -mkt},
        {"Leg": "Stock (delta hedge)", "Action": "Buy" if hold > 0 else "Short", "Units / share": hold,
         "Cash flow t=0 / share": -hold * S},
        {"Leg": "Bond", "Action": "Borrow" if bond > 0 else "Lend", "Units / share": bond,
         "Cash flow t=0 / share": bond},
        {"Leg": "Transaction costs", "Action": "Pay", "Units / share": None, "Cash flow t=0 / share": -cost_ps},
        {"Leg": "NET (expected edge)", "Action": "", "Units / share": None,
         "Cash flow t=0 / share": abs(diff) - cost_ps},
    ])
    df["Total"] = df["Cash flow t=0 / share"] * shares
    st.dataframe(df, hide_index=True, width="stretch")

    # P&L of the hedged book right after inception, revalued at your sigma
    grid = np.linspace(0.6 * S, 1.4 * S, 121)
    sign = -1 if over else 1
    pnl = [sign * (bs_price(x, K, T, r, sig, opt) - mkt) + hold * (x - S) for x in grid]
    unh = [sign * (bs_price(x, K, T, r, sig, opt) - mkt) for x in grid]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=unh, name="Option alone", line=dict(dash="dot", color="#e4572e")))
    fig.add_trace(go.Scatter(x=grid, y=pnl, name="Delta-hedged position", line=dict(color="#4f7cff", width=4)))
    fig.add_hline(y=abs(diff), line_dash="dash", line_color="green", annotation_text="edge")
    fig.add_vline(x=S, line_dash="dash", line_color="gray", annotation_text="S")
    fig.update_layout(height=400, xaxis_title="Stock price immediately after the trade",
                      yaxis_title="P&L per share (revalued at your sigma)", legend=dict(orientation="h", y=-0.2),
                      margin=dict(t=30))
    st.plotly_chart(fig, width="stretch", key="arb_plot_bs")
    st.caption("Near the current spot the hedged position is flat at the mispricing (the edge); the curvature "
               "is gamma, which is why the hedge must be rebalanced.")
    download_table(df, "bs_mispricing_trade", "arb_dl_bs")
    if st.button("Build PDF report", key="arb_bs_pdf_build"):
        import re
        st.session_state["arb_bs_pdf_bytes"] = pdf_report(
            "Arbitrage Calculator - Black-Scholes mispricing",
            [("Inputs", f"S = {S:.4f}, K = {K:.4f}, T = {T:g}, r = {r:.4%}, {opt} market = {mkt:.4f}, sigma = {sig:.2%}"),
             ("Result", f"BS fair value {fair:.4f}, implied vol {iv:.2%}, mispricing {diff:+.4f} per share, "
                        f"delta {delta:+.4f}."),
             ("Hedged trade", "\n".join(f"{i}. {re.sub('<[^>]+>', '', t)}" for i, t in enumerate(steps, 1)))],
            [("Trade legs", df)])
    if st.session_state.get("arb_bs_pdf_bytes"):
        pdf_button("Download PDF report", st.session_state["arb_bs_pdf_bytes"], "bs_mispricing_report.pdf",
                   "arb_bs_pdf_dl")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def render():
    page_header("Arbitrage Calculator",
                "Check whether option prices break put-call parity and get the exact trades that lock in the gap.",
                "arbitrage")
    _help()

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("<div class='panel-label'>Inputs</div>", unsafe_allow_html=True)
        mode = st.radio("Mode", ["1 - Put-call parity (call vs put, model-free)",
                                 "2 - Single option vs Black-Scholes value"], key="arb_mode")
        S = spot_input("arb_spot", 100.0)
        K = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0, key="arb_K")
        c = st.columns(2)
        T = c[0].number_input("Maturity T (years)", min_value=0.001, value=1.0, step=0.25, key="arb_T")
        r = c[1].number_input("Risk-free rate r (%, cont.)", value=5.0, step=0.25, key="arb_r") / 100
        extra = _parity_inputs() if mode.startswith("1") else _bs_inputs()
        c = st.columns(2)
        n = int(c[0].number_input("Number of contracts", min_value=1, value=1, step=1, key="arb_n"))
        mult = int(c[1].number_input("Contract multiplier", min_value=1, value=100, step=1, key="arb_mult"))
        tc = st.number_input("Transaction cost per trade ($)", min_value=0.0, value=0.0, step=1.0, key="arb_tc")

    if mode.startswith("1"):
        _parity_mode(right, S, K, T, r, tc, n, mult, *extra)
    else:
        _bs_mode(right, S, K, T, r, tc, n, mult, *extra)
