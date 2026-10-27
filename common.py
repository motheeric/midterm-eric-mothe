"""Shared helpers for the Options Toolkit app.

Every module imports from here so that pricing, Greeks, spot fetching,
Help rendering and report export behave identically across the app.
"""
from __future__ import annotations

import io
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
from scipy.stats import norm

# ----------------------------------------------------------------------------
# Black-Scholes (non-dividend-paying stock, optional continuous yield q)
# ----------------------------------------------------------------------------

def _d1_d2(S, K, T, r, sigma, q=0.0):
    T = max(float(T), 1e-12)
    sigma = max(float(sigma), 1e-12)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option="call", q=0.0):
    """Black-Scholes price of a European call/put."""
    if T <= 0:
        return intrinsic_value(S, K, option)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, option="call", q=0.0):
    """Greeks per ONE option. vega/rho per 1.00 change (divide by 100 for 1%),
    theta per year (divide by 365 for per day)."""
    if T <= 0:
        itm = (S > K) if option == "call" else (S < K)
        sign = 1 if option == "call" else -1
        return {"delta": sign * float(itm), "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf = norm.pdf(d1)
    gamma = math.exp(-q * T) * pdf / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * pdf * math.sqrt(T)
    if option == "call":
        delta = math.exp(-q * T) * norm.cdf(d1)
        theta = (-S * math.exp(-q * T) * pdf * sigma / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * norm.cdf(d2) + q * S * math.exp(-q * T) * norm.cdf(d1))
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -math.exp(-q * T) * norm.cdf(-d1)
        theta = (-S * math.exp(-q * T) * pdf * sigma / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm.cdf(-d2) - q * S * math.exp(-q * T) * norm.cdf(-d1))
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def intrinsic_value(S, K, option="call"):
    return max(S - K, 0.0) if option == "call" else max(K - S, 0.0)


def implied_vol(price, S, K, T, r, option="call", q=0.0, lo=1e-4, hi=5.0):
    """Implied volatility by bisection. Returns nan if no solution."""
    f = lambda s: bs_price(S, K, T, r, s, option, q) - price
    try:
        if f(lo) * f(hi) > 0:
            return float("nan")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-8:
                break
        return 0.5 * (lo + hi)
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------------
# Market data: online spot with manual fallback
# ----------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_spot(ticker: str):
    """Last price from Yahoo Finance. Returns (price, message)."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d")
        if hist is None or hist.empty:
            return None, f"No data returned for '{ticker}'."
        return float(hist["Close"].iloc[-1]), f"Last close {hist.index[-1].date()} (Yahoo Finance)"
    except Exception as e:  # network blocked, bad ticker, ...
        return None, f"Online fetch failed: {e.__class__.__name__}"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker: str, period: str = "6mo"):
    """Daily close history from Yahoo Finance, or None."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or hist.empty:
            return None
        return hist["Close"].rename("Close")
    except Exception:
        return None


def spot_input(key: str, default: float = 100.0, label: str = "Spot price S"):
    """Widget: choose 'Online (ticker)' or 'Manual'. Always returns a float S
    (falls back to manual value if the online fetch fails)."""
    mode = st.radio("Spot source", ["Manual", "Online (ticker)"], horizontal=True, key=f"{key}_mode")
    if mode == "Online (ticker)":
        ticker = st.text_input("Ticker (Yahoo Finance)", value="AAPL", key=f"{key}_ticker").strip().upper()
        price, msg = fetch_spot(ticker) if ticker else (None, "Enter a ticker.")
        if price is not None:
            st.metric(f"{ticker} spot", f"{price:,.2f}")
            st.caption(msg)
            return float(price)
        st.warning(f"{msg} Using manual input instead.")
    return float(st.number_input(label, min_value=0.01, value=float(default), step=1.0, key=f"{key}_manual"))


# ----------------------------------------------------------------------------
# Help panel (theory + assumptions + SymPy-rendered formulas)
# ----------------------------------------------------------------------------

def help_button(key: str, title: str, theory: str, assumptions: list[str],
                formulas: list[tuple[str, sp.Basic | str]], example: str | None = None):
    """Render a 'Help' toggle. `formulas` is a list of (label, sympy expr/Eq or
    raw LaTeX string). SymPy objects are converted with sympy.latex()."""
    if st.toggle("Help", key=f"help_{key}"):
        with st.container(border=True):
            st.subheader(f"Help — {title}")
            st.markdown("**Theoretical background**")
            st.markdown(theory)
            st.markdown("**Assumptions**")
            st.markdown("\n".join(f"- {a}" for a in assumptions))
            st.markdown("**Formulas** *(rendered with SymPy)*")
            for label, expr in formulas:
                st.markdown(f"*{label}*")
                st.latex(expr if isinstance(expr, str) else sp.latex(expr))
            if example:
                st.markdown("**Worked example**")
                st.markdown(example)


# Common SymPy symbols for formulas
S_, K_, T_, r_, sig_, t_, q_ = sp.symbols("S K T r sigma t q", positive=True)
N_ = sp.Function("N")
d1_ = (sp.log(S_ / K_) + (r_ + sig_ ** 2 / 2) * T_) / (sig_ * sp.sqrt(T_))
d2_ = d1_ - sig_ * sp.sqrt(T_)


# ----------------------------------------------------------------------------
# UI helpers: metric cards, downloads, PDF report
# ----------------------------------------------------------------------------

# Design tokens ("volatility desk"): cool paper, navy ink, one violet signature for curves.
PAPER, INK, GRAPHITE, RULE, VIOLET, GAIN, LOSS, AMBER = (
    "#F3F5FA", "#14213D", "#5B6478", "#D5DBE7", "#6A4CFF", "#0E9F6E", "#D64545", "#D08A00")


def _register_plotly_template():
    import plotly.graph_objects as go
    import plotly.io as pio
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13),
        title=dict(font=dict(family="Bricolage Grotesque, sans-serif", size=18, color=INK), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[VIOLET, INK, AMBER, GAIN, LOSS, GRAPHITE],
        xaxis=dict(gridcolor=RULE, zerolinecolor=GRAPHITE, linecolor=RULE, ticks="outside", tickcolor=RULE),
        yaxis=dict(gridcolor=RULE, zerolinecolor=GRAPHITE, linecolor=RULE, ticks="outside", tickcolor=RULE),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.02, yanchor="bottom", x=0),
        hoverlabel=dict(bgcolor=INK, font_color="white", font_family="IBM Plex Sans"),
        margin=dict(l=50, r=20, t=60, b=45),
    )
    pio.templates["vol_desk"] = tpl
    pio.templates.default = "plotly_white+vol_desk"


def _style_matplotlib():
    import matplotlib as mpl
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=[VIOLET, INK, AMBER, GAIN, LOSS, GRAPHITE]),
        "axes.edgecolor": RULE, "axes.labelcolor": INK, "xtick.color": GRAPHITE, "ytick.color": GRAPHITE,
        "axes.grid": True, "grid.color": RULE, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlecolor": INK,
    })


_register_plotly_template()
_style_matplotlib()


def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        html, body, [class*="css"], .stMarkdown, p, li, label, input, textarea, button, select,
        div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {{
            font-family: 'IBM Plex Sans', sans-serif;
        }}
        .stApp {{ background: {PAPER}; }}
        .block-container {{ padding-top: 2.2rem; max-width: 1240px; }}

        /* Headings: Bricolage, tight and confident */
        h1, h2, h3, .stApp h1, .stApp h2, .stApp h3 {{
            font-family: 'Bricolage Grotesque', sans-serif !important; color: {INK}; letter-spacing: -0.02em;
        }}
        .stApp h1 {{ font-weight: 800; font-size: 2.6rem; line-height: 1.05; }}
        .stApp h2 {{ font-weight: 700; font-size: 1.6rem; }}
        .stApp h3 {{ font-weight: 700; font-size: 1.2rem; }}

        /* Sidebar = the desk: deep ink panel */
        section[data-testid="stSidebar"] {{ background: {INK}; }}
        section[data-testid="stSidebar"] * {{ color: #E7EBF4 !important; }}
        section[data-testid="stSidebar"] h1 {{ font-size: 1.55rem; color: #FFFFFF !important; }}
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{ color: #A9B3C9 !important; }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            padding: 7px 10px; border-radius: 6px; margin-bottom: 2px; transition: background .15s;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: rgba(255,255,255,0.06); }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
            background: rgba(106,76,255,0.28); box-shadow: inset 3px 0 0 {VIOLET};
        }}
        section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.12); }}

        /* Metrics: no boxed card, a violet rule and a big tabular number */
        div[data-testid="stMetric"] {{
            background: transparent; border: none; border-left: 3px solid {VIOLET};
            border-radius: 0; padding: 2px 0 2px 14px;
        }}
        div[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; font-weight: 600; color: {INK};
            font-size: clamp(1.15rem, 1.7vw, 2rem) !important; }}
        div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {{
            overflow: visible !important; text-overflow: clip !important; white-space: normal !important;
            overflow-wrap: anywhere; }}
        div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {{
            overflow: visible !important; text-overflow: clip !important; white-space: normal !important; }}
        div[data-testid="stMetricLabel"] p {{ color: {GRAPHITE}; font-size: 0.86rem; }}

        /* Bordered containers: thin rule, small radius, no shadow */
        div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {RULE} !important; border-radius: 8px; background: #FFFFFF80; }}

        /* Tabs */
        button[data-baseweb="tab"] p {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 600; font-size: 0.98rem; }}

        /* Buttons */
        .stButton > button, .stDownloadButton > button {{ border-radius: 6px; font-weight: 600; }}
        .stButton > button[kind="primary"] {{ background: {VIOLET}; border-color: {VIOLET}; }}

        /* Strategy steps (a real sequence, so they are numbered) */
        .step-box {{ border-left: 3px solid {VIOLET}; padding: 8px 14px; margin: 6px 0;
            background: rgba(106,76,255,0.06); border-radius: 0 6px 6px 0; }}

        /* Home hero */
        .hero {{ position: relative; padding: 8px 0 6px 0; margin-bottom: 8px; }}
        .hero h1.hero-title {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 4.2rem;
            line-height: 0.95; letter-spacing: -0.04em; color: {INK}; margin: 0; }}
        .hero p.hero-sub {{ font-size: 1.12rem; color: {GRAPHITE}; max-width: 60ch; margin: 14px 0 0 0; }}
        .hero svg {{ width: 100%; height: auto; display: block; margin-top: 4px; }}
        .smile-path {{ stroke-dasharray: 1600; stroke-dashoffset: 1600; animation: draw 2.2s ease-out .2s forwards; }}
        @keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}
        @media (prefers-reduced-motion: reduce) {{ .smile-path {{ animation: none; stroke-dashoffset: 0; }} }}
        .mod-row {{ display: grid; grid-template-columns: 64px 1fr; gap: 6px 18px; padding: 18px 0;
            border-top: 1px solid {RULE}; }}
        .mod-num {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 2.6rem;
            line-height: 1; color: {VIOLET}; }}
        .mod-name {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.3rem; color: {INK}; }}
        .mod-desc {{ color: {GRAPHITE}; max-width: 70ch; margin-top: 4px; }}
        @media (max-width: 640px) {{ .hero h1.hero-title {{ font-size: 2.7rem; }} .mod-row {{ grid-template-columns: 44px 1fr; }}
            .mod-num {{ font-size: 1.9rem; }} }}
        /* Module header with its drawing */
        .mod-head {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 24px;
            border-bottom: 1px solid {RULE}; padding-bottom: 14px; margin-bottom: 18px; }}
        .mod-title {{ font-family: 'Bricolage Grotesque', sans-serif !important; font-weight: 800 !important;
            font-size: 2.5rem !important; line-height: 1.02 !important; letter-spacing: -0.03em; margin: 0 !important;
            padding: 0 !important; color: {INK}; }}
        .mod-sub {{ color: {GRAPHITE}; margin: 8px 0 0 0; max-width: 62ch; }}
        .mod-art svg {{ display: block; }}
        @media (max-width: 760px) {{ .mod-art {{ display: none; }} .mod-title {{ font-size: 1.9rem !important; }} }}

        /* Verdict panel */
        .verdict {{ border-left: 6px solid; background: #FFFFFF; padding: 16px 20px; margin: 10px 0 16px 0;
            border-radius: 0 8px 8px 0; }}
        .vd-head {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.25rem; color: {INK}; }}
        .vd-fig {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 3rem; line-height: 1.05;
            margin-top: 6px; font-variant-numeric: tabular-nums; }}
        .vd-figlab {{ color: {GRAPHITE}; font-size: 0.9rem; }}
        .vd-detail {{ color: {INK}; margin-top: 8px; max-width: 75ch; }}

        /* Control panel (left column of each module) */
        .panel-label {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; color: {INK};
            font-size: 1.05rem; margin: 4px 0 6px 0; }}

        /* Footer */
        .app-foot {{ margin-top: 48px; padding-top: 14px; border-top: 1px solid {RULE}; color: {GRAPHITE};
            font-size: 0.86rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def step_list(steps: list[str]):
    """Numbered strategy steps, rendered as boxes."""
    for i, s in enumerate(steps, 1):
        st.markdown(f"<div class='step-box'><b>Step {i}.</b> {s}</div>", unsafe_allow_html=True)


def download_table(df: pd.DataFrame, name: str, key: str):
    """CSV + Excel download buttons for a table."""
    c1, c2 = st.columns(2)
    c1.download_button("Download CSV", df.to_csv(index=False).encode(), f"{name}.csv", "text/csv", key=f"{key}_csv")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name=name[:30])
    c2.download_button("Download Excel", buf.getvalue(), f"{name}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{key}_xlsx")


def _pdf_safe(text: str) -> str:
    return (str(text).replace("σ", "sigma").replace("θ", "theta").replace("Θ", "Theta").replace("γ", "gamma").replace("μ", "mu").replace("π", "pi").replace("≈", "~").replace("→", "->").replace("−", "-").replace("×", "x").replace("·", "*").replace("Δ", "Delta").replace("Γ", "Gamma").replace("ν", "nu")
            .replace("α", "alpha").replace("β", "beta").replace("ρ", "rho").replace("—", "-").replace("–", "-")
            .replace("’", "'").replace("“", '"').replace("”", '"').encode("latin-1", "replace").decode("latin-1"))


def pdf_report(title: str, sections: list[tuple[str, str]], tables: list[tuple[str, pd.DataFrame]] | None = None,
               figures: list | None = None) -> bytes:
    """Build a printable PDF. sections = [(heading, text)], tables = [(caption, df)],
    figures = list of matplotlib Figures (optional)."""
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _pdf_safe(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Generated {datetime.now():%Y-%m-%d %H:%M} - Options Toolkit", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for head, text in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, _pdf_safe(head), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, _pdf_safe(text))
        pdf.ln(1)
    for cap, df in (tables or []):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, _pdf_safe(cap), new_x="LMARGIN", new_y="NEXT")
        d = df.copy()
        if len(d) > 60:
            d = d.head(60)
        cols = [str(c) for c in d.columns]
        w = (pdf.w - 2 * pdf.l_margin) / max(len(cols), 1)
        pdf.set_font("Helvetica", "B", 7)
        for c in cols:
            pdf.cell(w, 5, _pdf_safe(c)[:18], border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 7)
        for _, row in d.iterrows():
            for v in row:
                txt = f"{v:,.4f}" if isinstance(v, (float, np.floating)) else str(v)
                pdf.cell(w, 5, _pdf_safe(txt)[:18], border=1)
            pdf.ln()
        pdf.ln(2)
    for fig in (figures or []):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)
        pdf.add_page()
        pdf.image(buf, w=pdf.w - 2 * pdf.l_margin)
    return bytes(pdf.output())


def pdf_button(label: str, data: bytes, filename: str, key: str):
    st.download_button(label, data, filename, "application/pdf", key=key)


# ----------------------------------------------------------------------------
# Module identity: header with a small computed drawing, verdict panels
# ----------------------------------------------------------------------------

def _svg_poly(xs, ys, w=240, h=96, pad=8, stroke=VIOLET, width=3, extra=""):
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = lambda x: pad + (x - x0) / ((x1 - x0) or 1) * (w - 2 * pad)
    sy = lambda y: pad + (y1 - y) / ((y1 - y0) or 1) * (h - 2 * pad)
    d = "M" + " L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    base = f"<line x1='{pad}' y1='{h - pad}' x2='{w - pad}' y2='{h - pad}' stroke='{RULE}' stroke-width='1'/>"
    return (f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img' aria-hidden='true'>{base}{extra}"
            f"<path d='{d}' fill='none' stroke='{stroke}' stroke-width='{width}' stroke-linecap='round' "
            f"stroke-linejoin='round'/></svg>")


def art(kind: str) -> str:
    """Small computed drawing that identifies each module."""
    if kind == "portfolio":  # value today vs payoff at expiry of a call: the hockey stick
        xs = [60 + i for i in range(81)]
        pay = [max(x - 100, 0) for x in xs]
        val = [bs_price(x, 100, 0.5, 0.03, 0.25, "call") for x in xs]
        w, h, pad = 240, 96, 8
        x0, x1, y1 = xs[0], xs[-1], max(val)
        sx = lambda x: pad + (x - x0) / (x1 - x0) * (w - 2 * pad)
        sy = lambda y: pad + (y1 - y) / y1 * (h - 2 * pad)
        dp = "M" + " L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, pay))
        return _svg_poly(xs, val, extra=f"<path d='{dp}' fill='none' stroke='{INK}' stroke-width='1.5' "
                                         f"stroke-dasharray='4 4'/>")
    if kind == "arbitrage":  # call - put = S - PV(K): two parallel lines, the parity gap
        xs = [60 + i for i in range(81)]
        lhs = [bs_price(x, 100, 1, 0.04, 0.25, "call") - bs_price(x, 100, 1, 0.04, 0.25, "put") for x in xs]
        w, h, pad = 240, 96, 8
        lo, hi = min(lhs) - 6, max(lhs) + 6
        sx = lambda x: pad + (x - xs[0]) / (xs[-1] - xs[0]) * (w - 2 * pad)
        sy = lambda y: pad + (hi - y) / (hi - lo) * (h - 2 * pad)
        d2 = "M" + " L".join(f"{sx(x):.1f},{sy(y + 6):.1f}" for x, y in zip(xs, lhs))
        d1 = "M" + " L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, lhs))
        return (f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' aria-hidden='true'>"
                f"<path d='{d1}' fill='none' stroke='{INK}' stroke-width='2'/>"
                f"<path d='{d2}' fill='none' stroke='{VIOLET}' stroke-width='3' stroke-linecap='round'/>"
                f"<line x1='{sx(120):.1f}' y1='{sy(lhs[60]):.1f}' x2='{sx(120):.1f}' y2='{sy(lhs[60] + 6):.1f}' "
                f"stroke='{AMBER}' stroke-width='2.5'/></svg>")
    if kind == "hedging":  # a 5-day spot path with daily rebalancing marks
        rng = np.random.default_rng(7)
        s = [100.0]
        for _ in range(40):
            s.append(s[-1] * math.exp(0.012 * rng.standard_normal()))
        xs = list(range(len(s)))
        marks = "".join(
            f"<circle cx='{8 + i / 40 * 224:.1f}' cy='{8 + (max(s) - s[i]) / (max(s) - min(s)) * 80:.1f}' r='4' "
            f"fill='{INK}'/>" for i in range(0, 41, 8))
        return _svg_poly(xs, s, width=2.5).replace("</svg>", marks + "</svg>")
    if kind == "sabr":  # the SABR smile
        F, a, rho, nu = 100.0, 0.22, -0.15, 1.1
        ks = [60 + i for i in range(91)]

        def v(K):
            z = nu / a * math.log(F / K)
            if abs(z) < 1e-7:
                return a
            xz = math.log((math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
            return a * z / xz
        return _svg_poly(ks, [v(k) for k in ks])
    return ""


def page_header(title: str, subtitle: str, kind: str):
    """Module title with its computed drawing on the right."""
    st.markdown(
        f"<div class='mod-head'><div><h1 class='mod-title'>{title}</h1><p class='mod-sub'>{subtitle}</p></div>"
        f"<div class='mod-art'>{art(kind)}</div></div>",
        unsafe_allow_html=True,
    )


def verdict(kind: str, headline: str, figure: str | None = None, figure_label: str | None = None,
            detail: str | None = None):
    """Result panel. kind: 'gain' (arbitrage/edge found), 'neutral' (no opportunity), 'loss' (warning)."""
    color = {"gain": GAIN, "neutral": GRAPHITE, "loss": LOSS}.get(kind, VIOLET)
    fig = (f"<div class='vd-fig' style='color:{color}'>{figure}</div>"
           f"<div class='vd-figlab'>{figure_label or ''}</div>") if figure else ""
    st.markdown(
        f"<div class='verdict' style='border-color:{color}'><div class='vd-head'>{headline}</div>{fig}"
        f"{f'<div class=vd-detail>{detail}</div>' if detail else ''}</div>",
        unsafe_allow_html=True,
    )
