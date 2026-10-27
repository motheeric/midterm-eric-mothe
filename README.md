# Options Toolkit

A Streamlit app with four calculators for European options on non-dividend-paying stocks:

1. **Portfolio Evaluation**: market value, intrinsic value and Greeks of a multi-leg position, with payoff and P&L charts.
2. **Arbitrage Calculator**: put-call parity check, synthetic options built from stock and bonds, and a step-by-step arbitrage strategy.
3. **Delta and Delta-Gamma Hedging**: daily re-hedging, a spot price simulator over up to 5 days, P&L projection and a daily adjustment report.
4. **SABR Calculator and Monte Carlo**: Hagan implied volatility, calibration from CSV/Excel (beta = 1), volatility smile and surface, SABR-based pricing and arbitrage, and Euler-Maruyama / log-transform with full-truncation simulation.

Each module has a **Help** toggle with the theory, the assumptions and formulas rendered with SymPy. Tables can be downloaded as CSV or Excel, and each module exports a printable PDF report.

## Run it locally

You need Python 3.9 or later.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

To use it on a phone on the same Wi-Fi network, open the "Network URL" that Streamlit prints in the terminal.

## Spot prices

Spot prices are fetched from Yahoo Finance (`yfinance`) when you choose "Online (ticker)". If the fetch fails (no internet, wrong ticker), the app falls back to manual input.

## Sample data

`sample_data/sabr_sample.csv` and `sample_data/sabr_sample.xlsx` can be uploaded in the SABR calibration tab.

## Files

```
app.py              navigation and home page
common.py           Black-Scholes, Greeks, spot fetching, Help panel, downloads, PDF reports
modules/
  portfolio.py      1. Portfolio Evaluation
  arbitrage.py      2. Arbitrage Calculator
  hedging.py        3. Delta and Delta-Gamma Hedging
  sabr.py           4. SABR Calculator and Monte Carlo
sample_data/        sample calibration data
requirements.txt
```
