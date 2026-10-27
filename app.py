"""Options Toolkit — Streamlit app.

Run locally:  pip install -r requirements.txt  then  streamlit run app.py
"""
import math

import streamlit as st

from common import inject_css, INK, GRAPHITE, RULE, VIOLET

st.set_page_config(page_title="Options Toolkit", page_icon="📈", layout="wide")
inject_css()

MODULES = {
    "Home": None,
    "1 · Portfolio Evaluation": "modules.portfolio",
    "2 · Arbitrage Calculator": "modules.arbitrage",
    "3 · Delta & Delta-Gamma Hedging": "modules.hedging",
    "4 · SABR & Monte Carlo": "modules.sabr",
}
NAMES = list(MODULES)

if "nav" not in st.session_state:
    st.session_state["nav"] = NAMES[0]


def _go(name):
    st.session_state["nav"] = name


with st.sidebar:
    st.title("Options Toolkit")
    st.caption("Advanced Investment Management, Pace University")
    choice = st.radio("Module", NAMES, key="nav", label_visibility="collapsed")
    st.divider()
    st.caption("Each module has a Help switch with the theory, the assumptions and the formulas.")
    st.caption("Spot prices come live from Yahoo Finance, or from your own input when offline.")


def _smile_svg() -> str:
    """Hero art: a real SABR smile (Hagan, beta = 1), computed, not drawn by hand."""
    F, a, rho, nu, T = 100.0, 0.22, -0.15, 1.1, 1.0

    def vol(K):
        z = nu / a * math.log(F / K)
        xz = math.log((math.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho)) if abs(z) > 1e-7 else 1.0
        ratio = z / xz if abs(z) > 1e-7 else 1.0
        return a * ratio * (1 + (rho * nu * a / 4 + (2 - 3 * rho * rho) * nu * nu / 24) * T)

    W, H, pad = 1200, 230, 26
    Ks = [55 + i * 0.5 for i in range(201)]  # strikes 55 -> 155
    vs = [vol(k) for k in Ks]
    vmin, vmax = min(vs), max(vs)
    X = lambda k: pad + (k - Ks[0]) / (Ks[-1] - Ks[0]) * (W - 2 * pad)
    Y = lambda v: pad + (vmax - v) / (vmax - vmin) * (H - 2 * pad)
    d = "M" + " L".join(f"{X(k):.1f},{Y(v):.1f}" for k, v in zip(Ks, vs))
    ticks = "".join(
        f"<line x1='{X(k):.1f}' y1='{H - pad + 6}' x2='{X(k):.1f}' y2='{H - pad + 12}' stroke='{GRAPHITE}' stroke-width='1'/>"
        f"<text x='{X(k):.1f}' y='{H - 2}' fill='{GRAPHITE}' font-size='12' text-anchor='middle' "
        f"font-family='IBM Plex Sans'>{k}</text>" for k in range(60, 160, 10))
    atm_x, atm_y = X(F), Y(vol(F))
    return f"""
    <svg viewBox='0 0 {W} {H + 4}' role='img' aria-label='SABR implied volatility smile across strikes'>
      <line x1='{pad}' y1='{H - pad + 6}' x2='{W - pad}' y2='{H - pad + 6}' stroke='{RULE}' stroke-width='1'/>
      {ticks}
      <line x1='{atm_x:.1f}' y1='{pad - 6}' x2='{atm_x:.1f}' y2='{H - pad + 6}' stroke='{RULE}' stroke-dasharray='4 5'/>
      <path class='smile-path' d='{d}' fill='none' stroke='{VIOLET}' stroke-width='4' stroke-linecap='round'/>
      <circle cx='{atm_x:.1f}' cy='{atm_y:.1f}' r='6' fill='{INK}'/>
      <text x='{atm_x + 14:.1f}' y='{atm_y - 14:.1f}' fill='{INK}' font-size='13' font-family='IBM Plex Sans'>
        at the money, {vol(F) * 100:.1f}% vol</text>
    </svg>"""


HOME_ROWS = [
    ("1", NAMES[1], "Portfolio Evaluation",
     "Price a mix of calls, puts and stock, long or short. See market value, intrinsic value, "
     "the Greeks of the whole position, and how its P&L moves with the spot and with time."),
    ("2", NAMES[2], "Arbitrage Calculator",
     "Check put-call parity against market prices. When an option is mispriced, the app builds the synthetic "
     "option from stock and bonds and lists every trade that locks in the profit."),
    ("3", NAMES[3], "Delta & Delta-Gamma Hedging",
     "Re-hedge a position day by day on a simulated or historical spot path, up to 5 days, and compare hedged "
     "and unhedged P&L in a printable adjustment report."),
    ("4", NAMES[4], "SABR & Monte Carlo",
     "Compute Hagan's implied volatility, calibrate the model to your data, draw the smile and the surface, "
     "price with SABR, and simulate the SABR dynamics with Euler-Maruyama or full truncation."),
]

if MODULES[choice] is None:
    st.markdown(
        f"""
        <div class='hero'>
          <h1 class='hero-title'>Options Toolkit</h1>
          <p class='hero-sub'>Value, arbitrage, hedge and model European options on stocks, with live market data
          and the maths behind every number.</p>
          {_smile_svg()}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("The curve above is a real SABR smile computed with Hagan's formula (α = 0.22, β = 1, ρ = −0.15, ν = 1.1).")
    for num, key, name, desc in HOME_ROWS:
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.markdown(
            f"<div class='mod-row'><div class='mod-num'>{num}</div>"
            f"<div><div class='mod-name'>{name}</div><div class='mod-desc'>{desc}</div></div></div>",
            unsafe_allow_html=True,
        )
        c2.button("Open", key=f"open_{num}", on_click=_go, args=(key,), use_container_width=True)
else:
    import importlib

    mod = importlib.import_module(MODULES[choice])
    mod.render()

st.markdown(
    "<div class='app-foot'>Options Toolkit, built by Eric Mothe for Advanced Investment Management "
    "(FIN 653), Pace University, Fall 2026. Prices are for education only: Black-Scholes and SABR models, "
    "no transaction costs unless stated.</div>",
    unsafe_allow_html=True,
)
