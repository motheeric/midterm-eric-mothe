"""Module 4 - SABR Calculator, calibration, smile/surface, pricing & arbitrage, Monte Carlo."""
from __future__ import annotations

import io
import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from scipy.optimize import minimize
from scipy.stats import norm

from common import (bs_greeks, bs_price, download_table, help_button, implied_vol, page_header, pdf_button,
                    pdf_report, spot_input, step_list, verdict)

SAMPLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data",
                           "sabr_sample.csv")

# Defaults (also the parameter set used to generate the bundled sample)
DEF = dict(alpha=0.25, beta=1.0, rho=-0.30, nu=0.60)

# Full-width keyword compatible with old (use_container_width) and new (width="stretch") Streamlit
_V = tuple(int(x) for x in st.__version__.split(".")[:2])
_W = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}


# ----------------------------------------------------------------------------
# Hagan (2002) lognormal implied volatility
# ----------------------------------------------------------------------------

def hagan_vol(F, K, T, alpha, beta, rho, nu):
    """Hagan et al. (2002) Black implied vol. Vectorised over F, K, T.
    Handles ATM (z/x(z) -> 1) with a Taylor series for small |z|."""
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    F, K, T = np.broadcast_arrays(F, K, T)
    omb = 1.0 - beta
    logFK = np.log(F / K)
    fkb = (F * K) ** (omb / 2.0)
    z = (nu / alpha) * fkb * logFK
    # z / x(z), stable for small z
    zx = np.empty_like(z)
    small = np.abs(z) < 1e-6
    zs = z[small]
    zx[small] = 1.0 - 0.5 * rho * zs + (2.0 - 3.0 * rho ** 2) * zs ** 2 / 12.0
    zl = z[~small]
    xz = np.log((np.sqrt(1.0 - 2.0 * rho * zl + zl ** 2) + zl - rho) / (1.0 - rho))
    zx[~small] = zl / xz
    denom = fkb * (1.0 + omb ** 2 / 24.0 * logFK ** 2 + omb ** 4 / 1920.0 * logFK ** 4)
    corr = 1.0 + (omb ** 2 / 24.0 * alpha ** 2 / fkb ** 2
                  + rho * beta * nu * alpha / (4.0 * fkb)
                  + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2) * T
    out = alpha / denom * zx * corr
    return out if out.ndim else float(out)


def black76(F, K, T, r, sigma, option="call"):
    """Black-76 on the forward, discounted. Equals BS with S = F e^{-rT}."""
    if T <= 0 or sigma <= 0:
        return math.exp(-r * T) * max((F - K) if option == "call" else (K - F), 0.0)
    sd = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sd ** 2) / sd
    d2 = d1 - sd
    df = math.exp(-r * T)
    if option == "call":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def calibrate(K, vols, F, T, beta, x0=None):
    """Least squares in vol space. Returns dict(alpha, rho, nu, rmse, success)."""
    K, vols, F, T = (np.asarray(a, dtype=float) for a in (K, vols, F, T))
    bounds = [(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)]

    def obj(p):
        try:
            m = hagan_vol(F, K, T, p[0], beta, p[1], p[2])
            if not np.all(np.isfinite(m)):
                return 1e6
            return float(np.sum((m - vols) ** 2))
        except Exception:
            return 1e6

    # ATM-ish vol to seed alpha (alpha ~ sigma_ATM * F^(1-beta))
    i_atm = int(np.argmin(np.abs(np.log(K / F))))
    a0 = float(vols[i_atm] * F[i_atm] ** (1 - beta))
    starts = [x0] if x0 is not None else []
    starts += [[a0, r0, n0] for r0 in (-0.5, 0.0, 0.5) for n0 in (0.3, 1.0)]
    best = None
    for s in starts:
        res = minimize(obj, np.clip(s, [b[0] for b in bounds], [b[1] for b in bounds]),
                       method="L-BFGS-B", bounds=bounds, options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    a, rh, n = best.x
    return dict(alpha=float(a), rho=float(rh), nu=float(n), rmse=float(math.sqrt(best.fun / len(K))),
                success=bool(best.success))


# ----------------------------------------------------------------------------
# Monte Carlo for the SABR SDEs (forward measure)
# ----------------------------------------------------------------------------

SCHEMES = ["Euler-Maruyama (absorbing at 0)", "Log transform (σ) + full truncation (F)", "Log-Euler for F (β = 1)"]


def simulate_sabr(F0, alpha, beta, rho, nu, T, n_paths, n_steps, seed, antithetic, scheme, keep=25):
    """Vectorised simulation. Returns terminal F, terminal sigma, sample paths (keep × n_steps+1)."""
    rng = np.random.default_rng(int(seed))
    dt = T / n_steps
    sq = math.sqrt(dt)
    half = (n_paths + 1) // 2 if antithetic else n_paths
    F = np.full(n_paths, float(F0))
    sig = np.full(n_paths, float(alpha))
    keep = min(keep, n_paths)
    paths = np.empty((keep, n_steps + 1))
    paths[:, 0] = F0
    rc = math.sqrt(max(1.0 - rho ** 2, 0.0))
    for i in range(n_steps):
        z1 = rng.standard_normal(half)
        z2 = rng.standard_normal(half)
        if antithetic:
            z1 = np.concatenate([z1, -z1])[:n_paths]
            z2 = np.concatenate([z2, -z2])[:n_paths]
        w1 = z1
        w2 = rho * z1 + rc * z2
        if scheme.startswith("Euler"):
            Fp = np.maximum(F, 0.0)
            F_new = F + sig * Fp ** beta * sq * w1
            sig = np.maximum(sig + nu * sig * sq * w2, 0.0)
            F = np.maximum(F_new, 0.0)  # absorption at zero
        elif scheme.startswith("Log transform"):
            Fp = np.maximum(F, 0.0)
            F = F + sig * Fp ** beta * sq * w1           # full truncation: F^+ in the diffusion
            sig = sig * np.exp(-0.5 * nu ** 2 * dt + nu * sq * w2)  # exact lognormal step
        else:  # log-Euler for F, valid for beta = 1
            F = F * np.exp(-0.5 * sig ** 2 * dt + sig * sq * w1)
            sig = sig * np.exp(-0.5 * nu ** 2 * dt + nu * sq * w2)
        paths[:, i + 1] = np.maximum(F[:keep], 0.0)
    return np.maximum(F, 0.0), sig, paths


# ----------------------------------------------------------------------------
# SymPy formulas for Help
# ----------------------------------------------------------------------------

def _formulas():
    F, K, T, al, be, rh, nu, z, x, sig, t, r, dt = sp.symbols(
        "F K T alpha beta rho nu z x sigma t r Delta_t", real=True)
    W1, W2 = sp.symbols("W_1 W_2")
    fkb = (F * K) ** ((1 - be) / 2)
    L = sp.log(F / K)
    z_e = nu / al * fkb * L
    x_e = sp.log((sp.sqrt(1 - 2 * rh * z + z ** 2) + z - rh) / (1 - rh))
    hag = (al / (fkb * (1 + (1 - be) ** 2 / 24 * L ** 2 + (1 - be) ** 4 / 1920 * L ** 4)) * z / x
           * (1 + ((1 - be) ** 2 / 24 * al ** 2 / fkb ** 2 + rh * be * nu * al / (4 * fkb)
                   + (2 - 3 * rh ** 2) / 24 * nu ** 2) * T))
    atm = al / F ** (1 - be) * (1 + ((1 - be) ** 2 / 24 * al ** 2 / F ** (2 - 2 * be)
                                     + rh * be * nu * al / (4 * F ** (1 - be)) + (2 - 3 * rh ** 2) / 24 * nu ** 2) * T)
    sb = sp.Symbol(r"\sigma_B")
    return [
        ("SABR SDEs (forward measure)", r"dF_t=\sigma_t F_t^{\beta}\,dW^1_t,\qquad d\sigma_t=\nu\,\sigma_t\,dW^2_t,"
                                        r"\qquad dW^1_t\,dW^2_t=\rho\,dt,\qquad \sigma_0=\alpha"),
        ("Forward of a non-dividend stock", sp.Eq(F, sp.Symbol("S") * sp.exp(r * T))),
        ("Hagan's implied Black volatility", sp.Eq(sb, hag)),
        ("with z and x(z)", r"z=" + sp.latex(z_e) + r",\qquad x(z)=" + sp.latex(x_e)),
        ("ATM limit (K = F, z/x(z) → 1)", sp.Eq(sp.Symbol(r"\sigma_{ATM}"), atm)),
        ("Euler-Maruyama", r"F_{n+1}=\max\!\left(F_n+\sigma_n F_n^{\beta}\sqrt{\Delta t}\,Z^1_n,0\right),\quad "
                           r"\sigma_{n+1}=\sigma_n+\nu\sigma_n\sqrt{\Delta t}\,Z^2_n"),
        ("Log transform (σ) + full truncation (F)",
         r"\sigma_{n+1}=\sigma_n\exp\!\left(-\tfrac12\nu^2\Delta t+\nu\sqrt{\Delta t}\,Z^2_n\right),\quad "
         r"F_{n+1}=F_n+\sigma_n\,(F_n^{+})^{\beta}\sqrt{\Delta t}\,Z^1_n"),
        ("Log-Euler for F (β = 1)", r"F_{n+1}=F_n\exp\!\left(-\tfrac12\sigma_n^2\Delta t+\sigma_n\sqrt{\Delta t}\,Z^1_n\right),"
                                    r"\quad Z^2=\rho Z^1+\sqrt{1-\rho^2}\,\varepsilon"),
        ("Present value of the average payoff", r"\hat C=e^{-rT}\frac1N\sum_{i=1}^{N}\left(F_T^{(i)}-K\right)^+,"
                                                r"\qquad 95\%\ \mathrm{CI}:\ \hat C\pm1.96\,\frac{s}{\sqrt N}"),
    ]


HELP_THEORY = (
    "The **SABR** model (Stochastic Alpha Beta Rho, Hagan et al. 2002) lets the volatility of the forward follow "
    "its own lognormal process, correlated with the forward. **α** sets the overall vol level, **β** the backbone "
    "(β = 1 lognormal, standard for equities; β = 0 normal), **ρ** the skew (ρ < 0 gives the equity skew) and **ν** "
    "(vol-of-vol) the curvature of the smile. Hagan's asymptotic expansion gives a closed-form Black implied "
    "volatility for any strike, which is then plugged into Black-76 / Black-Scholes to price options.\n\n"
    "**Calibration** fixes β and finds (α, ρ, ν) minimising the squared error between Hagan vols and market vols. "
    "**Monte Carlo** simulates the SDEs directly under the T-forward measure, so the option price is the discounted "
    "average payoff; it is used to check Hagan's approximation (which degrades for long maturities / large ν)."
)
HELP_ASSUMPTIONS = [
    "European options on a non-dividend-paying stock; F = S·e^(rT) with constant r.",
    "Under the T-forward measure the forward is a martingale and F_T = S_T at expiry.",
    "β is chosen by the user (β = 1 for equities) and kept fixed during calibration.",
    "Hagan's formula is an asymptotic approximation (accurate for short T and moderate ν·√T).",
    "Mispricing strategies assume no transaction costs and frictionless delta-hedging.",
]
HELP_EXAMPLE = (
    "S = 100, r = 5 %, T = 1 → F = 100·e^(0.05) = 105.13. With β = 1, α = 0.25, ρ = −0.3, ν = 0.6, the ATM vol is "
    "0.25·(1 + (−0.3·0.6·0.25/4 + (2 − 3·0.09)·0.36/24)·1) = 0.25·(1 − 0.01125 + 0.02595) = **0.2537**. "
    "The ATM call (K = 105.13) then costs ≈ e^(−0.05)·105.13·(2N(0.1268)−1) ≈ **10.09**. A market quote of 11.00 is "
    "overpriced by 0.91: sell the call, buy Δ ≈ 0.55 shares and rebalance — expected edge ≈ 0.91 per option. "
    "Monte Carlo (20k paths, 100 steps) gives ≈ 10.09 ± 0.20, consistent with Hagan."
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _params():
    s = st.session_state
    return (float(s.get("sabr_alpha", DEF["alpha"])), float(s.get("sabr_beta", DEF["beta"])),
            float(s.get("sabr_rho", DEF["rho"])), float(s.get("sabr_nu", DEF["nu"])))


def _apply_calibrated():
    c = st.session_state.get("sabr_calib")
    if c:
        st.session_state["sabr_alpha"] = round(c["alpha"], 6)
        st.session_state["sabr_rho"] = round(float(np.clip(c["rho"], -0.999, 0.999)), 6)
        st.session_state["sabr_nu"] = round(c["nu"], 6)
        st.session_state["sabr_beta"] = float(c["beta"])
        st.session_state["sabr_applied_msg"] = (
            f"Calibrated parameters applied to all tabs: α = {c['alpha']:.4f}, β = {float(c['beta']):g}, "
            f"ρ = {c['rho']:.4f}, ν = {c['nu']:.4f}.")


def _read_upload(up):
    name = up.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(up)
    return pd.read_csv(up)


def _normalise(df):
    """Map user columns to Strike / MarketVol (+ optional Forward, T). Vols given in % are converted."""
    cols = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    pick = lambda *names: next((cols[n] for n in names if n in cols), None)
    k = pick("strike", "k")
    v = pick("marketvol", "vol", "impliedvol", "iv", "volatility", "sigma")
    if k is None or v is None:
        raise ValueError("File needs a 'Strike' and a 'MarketVol' column.")
    out = pd.DataFrame({"Strike": pd.to_numeric(df[k], errors="coerce"),
                        "MarketVol": pd.to_numeric(df[v], errors="coerce")})
    f = pick("forward", "f")
    t = pick("t", "maturity", "expiry", "tenor")
    if f is not None:
        out["Forward"] = pd.to_numeric(df[f], errors="coerce")
    if t is not None:
        out["T"] = pd.to_numeric(df[t], errors="coerce")
    out = out.dropna(subset=["Strike", "MarketVol"])
    out = out[(out.Strike > 0) & (out.MarketVol > 0)]
    if out.MarketVol.median() > 3:  # given in percent
        out["MarketVol"] = out.MarketVol / 100
    return out.reset_index(drop=True)


def _mpl_smile(K, vols_model, K_mkt=None, vols_mkt=None, title="SABR smile"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(K, np.asarray(vols_model) * 100, lw=2, label="SABR (Hagan)")
    if K_mkt is not None:
        ax.scatter(K_mkt, np.asarray(vols_mkt) * 100, color="crimson", zorder=3, label="Market")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied vol (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def _layout(fig, title, xt, yt, h=420):
    fig.update_layout(title=title, xaxis_title=xt, yaxis_title=yt, height=h, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2))
    return fig


# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------

def _tab_calculator(alpha, beta, rho, nu):
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.markdown("<div class='panel-label'>Option inputs</div>", unsafe_allow_html=True)
        S = spot_input("sabr_calc_spot", 100.0)
        r = st.number_input("Risk-free rate r (cont.)", value=0.05, step=0.005, format="%.4f", key="sabr_calc_r")
        T = st.number_input("Maturity T (years)", min_value=0.01, value=1.0, step=0.25, key="sabr_calc_T")
        F = S * math.exp(r * T)
        use_f = st.checkbox("Enter forward F manually", key="sabr_calc_fman")
        if use_f:
            F = st.number_input("Forward F", min_value=0.01, value=float(round(F, 4)), key="sabr_calc_F")
        K = st.number_input("Strike K", min_value=0.01, value=float(round(F, 2)), step=1.0, key="sabr_calc_K")
        opt = st.radio("Option", ["call", "put"], horizontal=True, key="sabr_calc_opt")
    vol = hagan_vol(F, K, T, alpha, beta, rho, nu)
    atm = hagan_vol(F, F, T, alpha, beta, rho, nu)
    S_eff = F * math.exp(-r * T)
    price = bs_price(S_eff, K, T, r, vol, opt)
    g = bs_greeks(S_eff, K, T, r, vol, opt)
    with c2:
        st.markdown("<div class='panel-label'>Hagan implied volatility</div>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Forward F = S·e^(rT)", f"{F:,.4f}")
        m2.metric("Hagan σ_B(K)", f"{vol * 100:.3f} %")
        m1.metric("ATM σ_B(F)", f"{atm * 100:.3f} %")
        m2.metric(f"{opt.capitalize()} price (Black-76)", f"{price:,.4f}")
        m1.metric("Delta", f"{g['delta']:.4f}")
        m2.metric("Vega (per 1 %)", f"{g['vega'] / 100:.4f}")
        st.caption(f"ln(F/K) = {math.log(F / K):+.4f}, z = {nu / alpha * (F * K) ** ((1 - beta) / 2) * math.log(F / K):+.4f}")
    df = pd.DataFrame([{"S": S, "r": r, "T": T, "Forward F": F, "Strike K": K, "Option": opt, "alpha": alpha,
                        "beta": beta, "rho": rho, "nu": nu, "Hagan vol": vol, "ATM vol": atm, "Price": price,
                        "Delta": g["delta"]}])
    download_table(df, "sabr_hagan_calc", "sabr_calc_dl")
    return df


def _tab_calibration(beta_ui):
    st.markdown("<div class='panel-label'>Calibrate α, ρ and ν to market implied vols (β fixed)</div>",
                unsafe_allow_html=True)
    src = st.radio("Data source", ["Bundled sample", "Upload CSV / Excel"], horizontal=True, key="sabr_cal_src")
    raw = None
    if src.startswith("Upload"):
        up = st.file_uploader("File with columns Strike, MarketVol (optional Forward, T)",
                              type=["csv", "xlsx", "xls"], key="sabr_cal_file")
        if up is not None:
            try:
                raw = _read_upload(up)
            except Exception as e:
                st.error(f"Could not read file: {e}")
        else:
            st.info("Upload a file, or switch to the bundled sample.")
    else:
        try:
            raw = pd.read_csv(SAMPLE_PATH)
        except Exception as e:
            st.error(f"Sample file missing: {e}")
    if raw is None:
        return None
    try:
        data = _normalise(raw)
    except ValueError as e:
        st.error(str(e))
        return None
    if len(data) < 3:
        st.error("Need at least 3 valid strikes.")
        return None

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("<div class='panel-label'>Market inputs</div>", unsafe_allow_html=True)
    S = left.number_input("Spot S", min_value=0.01, value=100.0, key="sabr_cal_S")
    r = left.number_input("Rate r", value=0.05, step=0.005, format="%.4f", key="sabr_cal_r")
    T = left.number_input("Maturity T", min_value=0.01, value=1.0, key="sabr_cal_T")
    beta = left.number_input("β (fixed)", min_value=0.0, max_value=1.0, value=float(beta_ui), step=0.1,
                           key="sabr_cal_beta", help="β = 1 for equities")
    if "T" not in data:
        data["T"] = T
    if "Forward" not in data:
        data["Forward"] = S * np.exp(r * data["T"])
    else:
        left.caption("Forward taken from file.")

    res = calibrate(data.Strike.values, data.MarketVol.values, data.Forward.values, data["T"].values, beta)
    res["beta"] = beta
    st.session_state["sabr_calib"] = res
    data["ModelVol"] = hagan_vol(data.Forward.values, data.Strike.values, data["T"].values,
                                 res["alpha"], beta, res["rho"], res["nu"])
    data["Error (bp)"] = (data.ModelVol - data.MarketVol) * 1e4

    with left:
        st.button("Use calibrated parameters in all tabs", key="sabr_cal_apply", on_click=_apply_calibrated,
                  type="primary")
        if st.session_state.get("sabr_applied_msg"):
            st.success(st.session_state.pop("sabr_applied_msg"))

    kk = np.linspace(data.Strike.min() * 0.9, data.Strike.max() * 1.1, 200)
    F0, T0 = float(data.Forward.median()), float(data["T"].median())
    fig = go.Figure()
    fig.add_scatter(x=kk, y=hagan_vol(F0, kk, T0, res["alpha"], beta, res["rho"], res["nu"]) * 100,
                    name="SABR fit", mode="lines")
    fig.add_scatter(x=data.Strike, y=data.MarketVol * 100, name="Market", mode="markers", marker=dict(size=9))
    fig.add_vline(x=F0, line_dash="dot", annotation_text="F")
    with right:
        st.markdown("<div class='panel-label'>Calibrated parameters</div>", unsafe_allow_html=True)
        m = st.columns(5)
        m[0].metric("α", f"{res['alpha']:.5f}")
        m[1].metric("β (fixed)", f"{beta:.2f}")
        m[2].metric("ρ", f"{res['rho']:.5f}")
        m[3].metric("ν", f"{res['nu']:.5f}")
        m[4].metric("RMSE", f"{res['rmse'] * 1e4:.2f} bp")
        if not res["success"]:
            st.warning("Optimizer reported non-convergence; results are the best found.")
        st.plotly_chart(_layout(fig, "Market vs calibrated SABR vols", "Strike K", "Implied vol (%)"),
                        **_W, key="sabr_cal_chart")

    data["MarketPrice"] = [black76(f, k, t, r, v) for f, k, t, v in
                           zip(data.Forward, data.Strike, data["T"], data.MarketVol)]
    data["ModelPrice"] = [black76(f, k, t, r, v) for f, k, t, v in
                          zip(data.Forward, data.Strike, data["T"], data.ModelVol)]
    st.markdown("<div class='panel-label'>Market vs model, strike by strike</div>", unsafe_allow_html=True)
    st.dataframe(data.style.format(precision=4), **_W, hide_index=True)
    download_table(data, "sabr_calibration", "sabr_cal_dl")
    params = pd.DataFrame([{"alpha": res["alpha"], "beta": beta, "rho": res["rho"], "nu": res["nu"],
                            "RMSE (vol)": res["rmse"], "n strikes": len(data)}])
    st.session_state["sabr_cal_fig"] = (kk, F0, T0, data.Strike.values, data.MarketVol.values, res)
    return data, params


def _tab_smile(alpha, beta, rho, nu):
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("<div class='panel-label'>Smile inputs</div>", unsafe_allow_html=True)
    r = left.number_input("Rate r", value=0.05, step=0.005, format="%.4f", key="sabr_sm_r")
    T = left.number_input("Maturity T", min_value=0.01, value=1.0, key="sabr_sm_T")
    spots_txt = left.text_input("Spots S (comma-separated)", "90, 100, 110", key="sabr_sm_spots")
    try:
        spots = [float(x) for x in spots_txt.split(",") if x.strip()]
        spots = [s for s in spots if s > 0] or [100.0]
    except ValueError:
        left.warning("Could not parse spots; using 100.")
        spots = [100.0]
    k1, k2 = left.columns(2)
    kmin = k1.number_input("K min", min_value=0.01, value=60.0, key="sabr_sm_kmin")
    kmax = k2.number_input("K max", min_value=0.02, value=150.0, key="sabr_sm_kmax")
    if kmax <= kmin:
        left.error("K max must exceed K min.")
        return None
    kk = np.linspace(kmin, kmax, 120)
    fig = go.Figure()
    rows = []
    grid_k = np.linspace(kmin, kmax, 10)
    for S in spots:
        F = S * math.exp(r * T)
        fig.add_scatter(x=kk, y=hagan_vol(F, kk, T, alpha, beta, rho, nu) * 100, mode="lines",
                        name=f"S = {S:g} (F = {F:.2f})")
        for K in grid_k:
            v = hagan_vol(F, K, T, alpha, beta, rho, nu)
            rows.append({"Spot S": S, "Forward F": F, "Strike K": K, "SABR vol": v,
                         "Call (Black-76)": black76(F, K, T, r, v, "call"),
                         "Put (Black-76)": black76(F, K, T, r, v, "put")})
    with right:
        st.markdown("<div class='panel-label'>Smile across spots (F = S·e^(rT))</div>", unsafe_allow_html=True)
        st.plotly_chart(_layout(fig, f"SABR smile, T = {T:g}", "Strike K", "Implied vol (%)"),
                        **_W, key="sabr_sm_chart")
    tab = pd.DataFrame(rows)
    st.markdown("<div class='panel-label'>Prices and forwards on a strike grid</div>", unsafe_allow_html=True)
    st.dataframe(tab.style.format(precision=4), **_W, hide_index=True, height=260)
    download_table(tab, "sabr_smile_table", "sabr_sm_dl")

    st.markdown("<div class='panel-label'>Volatility surface over strike and maturity</div>",
                unsafe_allow_html=True)
    S0 = st.number_input("Spot for surface", min_value=0.01, value=float(spots[len(spots) // 2]), key="sabr_sf_S")
    TT = np.linspace(0.1, 3.0, 30)
    KK = np.linspace(kmin, kmax, 40)
    Kg, Tg = np.meshgrid(KK, TT)
    Z = hagan_vol(S0 * np.exp(r * Tg), Kg, Tg, alpha, beta, rho, nu) * 100
    sf = go.Figure(go.Surface(x=Kg, y=Tg, z=Z, colorscale=[[0, "#14213D"], [0.5, "#6A4CFF"], [1, "#F2B544"]], colorbar=dict(title="vol %")))
    sf.update_layout(scene=dict(xaxis_title="Strike K", yaxis_title="Maturity T", zaxis_title="Vol (%)"),
                     height=520, margin=dict(l=0, r=0, t=30, b=0), title=f"SABR surface (S = {S0:g})")
    st.plotly_chart(sf, **_W, key="sabr_sf_chart")
    st.session_state["sabr_sm_fig"] = (kk, spots, r, T)
    return tab


def _tab_pricing(alpha, beta, rho, nu):
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.markdown("<div class='panel-label'>Option and market quote</div>", unsafe_allow_html=True)
        S = spot_input("sabr_px_spot", 100.0)
        r = st.number_input("Rate r", value=0.05, step=0.005, format="%.4f", key="sabr_px_r")
        T = st.number_input("Maturity T", min_value=0.01, value=1.0, key="sabr_px_T")
        K = st.number_input("Strike K", min_value=0.01, value=105.0, key="sabr_px_K")
        opt = st.radio("Option", ["call", "put"], horizontal=True, key="sabr_px_opt")
        mkt = st.number_input("Market price (observed)", min_value=0.0, value=11.00, step=0.1, key="sabr_px_mkt")
        n = st.number_input("Number of options (contracts × multiplier)", min_value=1, value=100, step=100,
                            key="sabr_px_n")
        tol = st.number_input("Tolerance (no-trade band)", min_value=0.0, value=0.05, step=0.01, key="sabr_px_tol")
    F = S * math.exp(r * T)
    vol = hagan_vol(F, K, T, alpha, beta, rho, nu)
    model = bs_price(S, K, T, r, vol, opt)
    b76 = black76(F, K, T, r, vol, opt)
    iv_mkt = implied_vol(mkt, S, K, T, r, opt)
    g = bs_greeks(S, K, T, r, vol, opt)
    diff = mkt - model

    delta = g["delta"]
    if abs(diff) <= tol:
        steps = ["No arbitrage opportunity beyond the tolerance band; stay flat."]
        verdict_txt = "Fair"
    elif diff > 0:
        verdict_txt = "Overpriced"
        hedge = f"buy {delta * n:,.2f} shares" if delta > 0 else f"short {-delta * n:,.2f} shares"
        steps = [f"Sell {n:,} {opt}s K = {K:g}, T = {T:g} at the market price {mkt:.4f} → receive {mkt * n:,.2f}.",
                 f"Delta-hedge: option Δ = {delta:.4f} per option, so {hedge} at S = {S:g}.",
                 "Invest/borrow the net cash at r; rebalance the stock position as S moves (Δ from SABR vol).",
                 f"At expiry the hedged short replicates at the model value {model:.4f} → expected edge "
                 f"≈ ({mkt:.4f} − {model:.4f}) × {n:,} = **{diff * n:,.2f}** (PV today)."]
    else:
        verdict_txt = "Underpriced"
        hedge = f"short {delta * n:,.2f} shares" if delta > 0 else f"buy {-delta * n:,.2f} shares"
        steps = [f"Buy {n:,} {opt}s K = {K:g}, T = {T:g} at the market price {mkt:.4f} → pay {mkt * n:,.2f}.",
                 f"Delta-hedge: option Δ = {delta:.4f} per option, so {hedge} at S = {S:g}.",
                 "Invest/borrow the net cash at r; rebalance the stock position as S moves (Δ from SABR vol).",
                 f"The hedged long earns the model value {model:.4f} → expected edge "
                 f"≈ ({model:.4f} − {mkt:.4f}) × {n:,} = **{-diff * n:,.2f}** (PV today)."]

    with c2:
        if verdict_txt == "Fair":
            verdict("neutral", "The market option is fairly priced: no trade", f"{diff:+,.4f}",
                    "market minus model, per option",
                    f"The gap is inside the no-trade band of ±{tol:g}.")
        elif verdict_txt == "Overpriced":
            verdict("gain", "The market option is overpriced: sell it and delta-hedge", f"{diff:,.4f}",
                    "edge per option",
                    f"Market vol is above the SABR vol. Expected edge on {n:,} options: {diff * n:,.2f} (PV today).")
        else:
            verdict("gain", "The market option is underpriced: buy it and delta-hedge", f"{-diff:,.4f}",
                    "edge per option",
                    f"Market vol is below the SABR vol. Expected edge on {n:,} options: {-diff * n:,.2f} (PV today).")
        m = st.columns(3)
        m[0].metric("Forward F", f"{F:,.4f}")
        m[1].metric("SABR vol", f"{vol * 100:.3f} %")
        m[2].metric("Model price (BS = Black-76)", f"{model:,.4f}")
        m = st.columns(3)
        m[0].metric("Market implied vol", "n/a" if math.isnan(iv_mkt) else f"{iv_mkt * 100:.3f} %")
        m[1].metric("Market − model", f"{diff:+.4f}")
        m[2].metric("Delta", f"{delta:.4f}")
        st.markdown("<div class='panel-label'>Strategy</div>", unsafe_allow_html=True)
        step_list(steps)
        st.caption("Edge is model-dependent (statistical arbitrage): it is realised only if SABR is the right model "
                   "and the hedge is rebalanced frequently.")
    tab = pd.DataFrame([{"Spot S": S, "Forward F": F, "Strike K": K, "T": T, "r": r, "Option": opt,
                         "SABR vol": vol, "BS price (SABR vol)": model, "Black-76 price": b76,
                         "Market price": mkt, "Market IV": iv_mkt, "Diff": diff, "Delta": delta,
                         "Verdict": verdict_txt, "Expected edge": abs(diff) * n if verdict_txt != "Fair" else 0.0}])
    download_table(tab, "sabr_pricing", "sabr_px_dl")
    return tab, steps


def _tab_mc(alpha, beta, rho, nu):
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.markdown("<div class='panel-label'>Simulation inputs</div>", unsafe_allow_html=True)
        S = st.number_input("Spot S", min_value=0.01, value=100.0, key="sabr_mc_S")
        r = st.number_input("Rate r", value=0.05, step=0.005, format="%.4f", key="sabr_mc_r")
        T = st.number_input("Maturity T", min_value=0.01, value=1.0, key="sabr_mc_T")
        K = st.number_input("Strike K", min_value=0.01, value=105.0, key="sabr_mc_K")
        opt = st.radio("Option", ["call", "put"], horizontal=True, key="sabr_mc_opt")
        scheme = st.selectbox("Scheme", SCHEMES, index=1, key="sabr_mc_scheme")
        n_paths = st.number_input("Number of simulations", min_value=100, max_value=200_000, value=20_000,
                                  step=1000, key="sabr_mc_n")
        n_steps = st.number_input("Time steps", min_value=1, max_value=1000, value=100, step=10, key="sabr_mc_steps")
        seed = st.number_input("Seed", min_value=0, value=42, step=1, key="sabr_mc_seed")
        anti = st.toggle("Antithetic variates", value=True, key="sabr_mc_anti")
        if scheme.startswith("Log-Euler") and abs(beta - 1) > 1e-9:
            st.warning("Log-Euler for F is only exact in form for β = 1; switch scheme or set β = 1.")
        if n_paths * n_steps > 5e7:
            st.warning("Large run: this may take a while.")
        run = st.button("Run simulation", key="sabr_mc_run", type="primary")

    if run:
        F0 = S * math.exp(r * T)
        with st.spinner("Simulating..."):
            FT, sigT, paths = simulate_sabr(F0, alpha, beta, rho, nu, T, int(n_paths), int(n_steps), int(seed),
                                            anti, scheme)
        disc = math.exp(-r * T)
        pay = np.maximum(FT - K, 0) if opt == "call" else np.maximum(K - FT, 0)
        pv = disc * pay
        if anti:  # pair antithetic samples for a correct standard error
            h = len(pv) // 2
            pairs = 0.5 * (pv[:h] + pv[h:2 * h])
            se = pairs.std(ddof=1) / math.sqrt(len(pairs))
        else:
            se = pv.std(ddof=1) / math.sqrt(len(pv))
        price = float(pv.mean())
        hv = hagan_vol(F0, K, T, alpha, beta, rho, nu)
        hprice = black76(F0, K, T, r, hv, opt)
        st.session_state["sabr_mc_res"] = dict(F0=F0, FT=FT, paths=paths, pv=pv, price=price, se=se, hv=hv,
                                                hprice=hprice, S=S, K=K, T=T, r=r, opt=opt, scheme=scheme,
                                                n=int(n_paths), steps=int(n_steps), anti=anti,
                                                params=(alpha, beta, rho, nu))
    res = st.session_state.get("sabr_mc_res")
    if res is None:
        with c2:
            st.info("Set the inputs and press Run simulation.")
        return None
    FT, pv, disc = res["FT"], res["pv"], math.exp(-res["r"] * res["T"])
    lo, hi = res["price"] - 1.96 * res["se"], res["price"] + 1.96 * res["se"]
    inside = lo <= res["hprice"] <= hi

    # Convergence
    n = len(pv)
    pts = np.unique(np.geomspace(100, n, 40).astype(int))
    if res["anti"]:
        pts = np.unique(np.maximum(2, pts - pts % 2))
        pv_ord = np.empty(n)
        h = n // 2
        pv_ord[0:2 * h:2] = pv[:h]
        pv_ord[1:2 * h:2] = pv[h:2 * h]
        if n % 2:
            pv_ord[-1] = pv[-1]
    else:
        pv_ord = pv
    cm = np.cumsum(pv_ord)[pts - 1] / pts
    cs = np.array([pv_ord[:p].std(ddof=1) / math.sqrt(p) for p in pts])
    fig = go.Figure()
    fig.add_scatter(x=pts, y=cm + 1.96 * cs, mode="lines", line=dict(width=0), showlegend=False)
    fig.add_scatter(x=pts, y=cm - 1.96 * cs, mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(106,76,255,0.18)", name="95 % band")
    fig.add_scatter(x=pts, y=cm, mode="lines+markers", name="MC price", line=dict(color="#6A4CFF", width=2.5), marker=dict(size=5))
    fig.add_hline(y=res["hprice"], line_dash="dash", line_color="#14213D", annotation_text="Hagan")
    fig.update_xaxes(type="log")
    hist = go.Figure(go.Histogram(x=FT, nbinsx=80, marker_color="#6A4CFF"))
    hist.add_vline(x=res["K"], line_dash="dash", line_color="#14213D", annotation_text="K")
    tgrid = np.linspace(0, res["T"], res["paths"].shape[1])
    pf = go.Figure()
    for p in res["paths"]:
        pf.add_scatter(x=tgrid, y=p, mode="lines", line=dict(width=1), showlegend=False)

    with c2:
        verdict("violet", f"Monte Carlo {res['opt']} price: discounted average payoff", f"{res['price']:,.4f}",
                f"over {res['n']:,} paths, {res['steps']} steps",
                f"95 % CI [{lo:.4f}, {hi:.4f}] {'contains' if inside else 'does not contain'} the Hagan price "
                f"{res['hprice']:.4f} (difference {res['price'] - res['hprice']:+.4f}). "
                + ("" if inside else "Hagan's expansion and the time discretisation both add bias; try more steps."))
        m = st.columns(3)
        m[0].metric("Mean terminal F_T = S_T", f"{FT.mean():,.4f}", f"F0 = {res['F0']:.4f}", delta_color="off")
        m[1].metric("PV of mean terminal price, e^(−rT)·E[S_T]", f"{disc * FT.mean():,.4f}", f"S0 = {res['S']:g}",
                    delta_color="off")
        m[2].metric("Hagan-vol price", f"{res['hprice']:,.4f}", f"σ_B = {res['hv'] * 100:.2f} %", delta_color="off")
        st.plotly_chart(_layout(fig, "Convergence", "Number of paths", "Price"), **_W,
                        key="sabr_mc_conv")

    d1, d2 = st.columns(2)
    d1.plotly_chart(_layout(hist, "Terminal price distribution F_T = S_T", "S_T", "Count"),
                    **_W, key="sabr_mc_hist")
    d2.plotly_chart(_layout(pf, f"Sample paths of F_t ({res['scheme']})", "t (years)", "F_t"),
                    **_W, key="sabr_mc_paths")
    a, b, rh, nu_ = res["params"]
    tab = pd.DataFrame([{"Scheme": res["scheme"], "Paths": res["n"], "Steps": res["steps"],
                         "Antithetic": res["anti"], "S0": res["S"], "F0": res["F0"], "K": res["K"], "T": res["T"],
                         "r": res["r"], "Option": res["opt"], "alpha": a, "beta": b, "rho": rh, "nu": nu_,
                         "Mean F_T": float(FT.mean()), "MC price": res["price"], "Std error": res["se"],
                         "CI low": lo, "CI high": hi, "Hagan vol": res["hv"], "Hagan price": res["hprice"]}])
    download_table(tab, "sabr_monte_carlo", "sabr_mc_dl")
    return tab


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def render():
    page_header("SABR & Monte Carlo",
                "Fit the SABR smile to market vols, price options with it, spot mispricing and check it by simulation.",
                "sabr")
    help_button("sabr", "SABR model", HELP_THEORY, HELP_ASSUMPTIONS, _formulas(), HELP_EXAMPLE)

    with st.container(border=True):
        st.markdown("**SABR parameters** (shared by all tabs; use the Calibration tab to fill them from data)")
        for k, v in DEF.items():
            st.session_state.setdefault(f"sabr_{k}", v)
        c = st.columns(4)
        c[0].number_input("α (initial vol)", min_value=1e-4, step=0.01, format="%.5f",
                          key="sabr_alpha")
        c[1].number_input("β (backbone; 1 = equities)", min_value=0.0, max_value=1.0, step=0.1,
                          key="sabr_beta")
        c[2].number_input("ρ (correlation)", min_value=-0.999, max_value=0.999, step=0.05,
                          format="%.5f", key="sabr_rho")
        c[3].number_input("ν (vol of vol)", min_value=1e-4, step=0.05, format="%.5f",
                          key="sabr_nu")
    alpha, beta, rho, nu = _params()

    tabs = st.tabs(["Hagan calculator", "Calibration", "Smile and surface", "Pricing and arbitrage",
                    "Monte Carlo"])
    with tabs[0]:
        calc_df = _tab_calculator(alpha, beta, rho, nu)
    with tabs[1]:
        cal = _tab_calibration(beta)
    with tabs[2]:
        smile_df = _tab_smile(alpha, beta, rho, nu)
    with tabs[3]:
        px_df, steps = _tab_pricing(alpha, beta, rho, nu)
    with tabs[4]:
        mc_df = _tab_mc(alpha, beta, rho, nu)

    st.divider()
    st.markdown("#### Printable report")
    if st.toggle("Build PDF report", key="sabr_pdf_on"):
        try:
            import matplotlib
            matplotlib.use("Agg")
            sections = [("SABR parameters in use", f"alpha = {alpha:.5f}, beta = {beta:.2f}, rho = {rho:.5f}, "
                                                   f"nu = {nu:.5f}"),
                        ("Arbitrage strategy", "\n".join(f"{i}. {s.replace('**', '')}"
                                                         for i, s in enumerate(steps, 1)))]
            tables = [("Hagan calculator", calc_df.T.reset_index().rename(columns={"index": "Field", 0: "Value"})),
                      ("Pricing vs market (BS / Black-76 with SABR vol)",
                       px_df.T.reset_index().rename(columns={"index": "Field", 0: "Value"}))]
            figs = []
            if cal is not None:
                data, params = cal
                tables.insert(0, ("Calibrated parameters", params))
                tables.append(("Calibration: market vs model", data[["Strike", "Forward", "T", "MarketVol",
                                                                    "ModelVol", "MarketPrice", "ModelPrice"]]))
                kk, F0, T0, Km, Vm, res = st.session_state["sabr_cal_fig"]
                figs.append(_mpl_smile(kk, hagan_vol(F0, kk, T0, res["alpha"], res["beta"], res["rho"], res["nu"]),
                                       Km, Vm, "Calibration fit"))
            if smile_df is not None:
                tables.append(("Smile table: prices & forwards", smile_df))
                kk, spots, r, T = st.session_state["sabr_sm_fig"]
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, 3.8))
                for S in spots:
                    F = S * math.exp(r * T)
                    ax.plot(kk, hagan_vol(F, kk, T, alpha, beta, rho, nu) * 100, label=f"S={S:g}, F={F:.2f}")
                ax.set_xlabel("Strike K"); ax.set_ylabel("Implied vol (%)"); ax.set_title("SABR smile")
                ax.grid(alpha=0.3); ax.legend()
                figs.append(fig)
            if mc_df is not None:
                tables.append(("Monte Carlo", mc_df.T.reset_index().rename(columns={"index": "Field", 0: "Value"})))
            pdf = pdf_report("SABR Calculator & Monte Carlo", sections, tables, figs)
            pdf_button("Download PDF report", pdf, "sabr_report.pdf", key="sabr_pdf_dl")
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
