# 📈 Stock Price Prediction with LSTM

![tests](https://github.com/Purnakash/stock-price-prediction/actions/workflows/tests.yml/badge.svg)

An LSTM that predicts next-day closing prices for **any stock**, with live
data from Yahoo Finance, an honest comparison against a naive baseline, and an
interactive dashboard.

**▶ Live demo:** *(add your Streamlit Community Cloud link here after deploying, see [Deploy](#deploy))*

Rebuilt and upgraded from our B.Tech project *Stock Price Prediction using
Machine Learning* (SRM Institute of Science and Technology). The original code,
recovered from the project report, is kept in [`original/`](original/).

> **Not financial advice.** This is a machine-learning learning project. Daily
> stock prices are close to a random walk, and no model here should be used to trade.

---

## Features

* **Any ticker, live data:** US stocks (`AAPL`, `NVDA`, …) and Indian stocks
  (`TCS.NS`, `RELIANCE.NS`, …) via `yfinance`, or upload your own CSV
  (NSE/BSE exports work as-is).
* **LSTM model** (Keras 3): the original two-layer architecture, or an improved
  version with dropout and early stopping.
* **Predicts daily returns** by default, so the model can follow a stock to new
  highs; the original price-level approach is available for comparison.
* **Honest evaluation:** RMSE, MAE, MAPE and up/down direction on a held-out
  test period, always compared with the **naive baseline** ("tomorrow = today").
  The app tells you plainly whether the LSTM beats it.
* **Forecast** of the next 1–10 trading days.
* **Interactive dashboard** (Streamlit + Plotly): prediction chart with range
  selector, metrics, forecast table, CSV download.
* **Compare stocks:** high/low prices, trading volume and relative performance
  for several tickers (from the original project's dashboard).
* **Tested:** 15 automated tests, run by GitHub Actions on every push.

---

## What changed from the original project

| | Original (2022) | This version |
|---|---|---|
| Data | One fixed CSV (NSE Tata Global, 2013–2018) | Any ticker, live, or any CSV |
| Scaling | Scaler fitted on **all** data, so test data leaked into training | Scaler fitted on the training period only |
| Training | 1 epoch, batch size 1 | Up to 50 epochs, early stopping on a validation split |
| Evaluation | "80–95 % accuracy", not meaningful for price regression | RMSE / MAE / MAPE / direction vs a naive baseline |
| Target | Price level (cannot exceed the training maximum) | Daily log returns by default, price level optional |
| Forecast | None | Recursive multi-day forecast |
| Dashboard | Dash, fixed stocks (Tesla, Apple, Facebook, Microsoft) | Streamlit, any stocks, deployable for free |
| Code | One notebook | Package with modules, CLI, tests and CI |

---

## How it works

```
prices (Yahoo Finance or CSV)
   │  clean: parse dates, sort, map NSE column names, drop bad rows
   ▼
split by time ── first 80 % train │ last 20 % test (never seen in training)
   │
   ▼
daily log returns, standardised (fitted on train only) → windows of the last N days (default 60)
   │              (or: prices scaled to [0, 1], the original approach)
   ▼
LSTM(50) → LSTM(50) → Dense(1)     (+ dropout in the improved version)
   │
   ▼
predicted return → price = today's close × e^return
   │
   ▼
one-step-ahead predictions on the test days ── compared with naive "tomorrow = today"
   │
   ▼
recursive forecast of the next days (each prediction feeds the next)
```

**Why predict returns?** A model trained on prices scaled to 0–1 cannot
output a price above the highest one it saw in training. When a stock keeps
making new highs during the test period, a price-level LSTM therefore lags
below the real price. In one run on AAPL (test year Oct 2025 – Oct 2026) the
price-level model's RMSE was 2.4× the naive baseline's for exactly this
reason. Returns stay in a similar range at any price level, so the model is
not boxed in.

**Why the naive baseline matters.** For daily prices, "tomorrow's close =
today's close" is very hard to beat, because most of each day's change is
unpredictable. A model with a low error can still be no better than that
baseline. If the LSTM's RMSE is not clearly below the naive RMSE, it has not
learned anything useful, and the app says so.

---

## Results

AAPL, 5 years of daily prices, test period 23 Oct 2025 – 22 Oct 2026 (251
trading days, never seen in training), window 60, default settings:

| Model | RMSE | MAE | MAPE | Direction right | vs naive RMSE |
|---|---|---|---|---|---|
| LSTM predicting **returns** (default) | **4.43** | 3.13 | 1.08 % | 53.4 % | −0.3 % |
| LSTM predicting **price level** (original approach) | 10.83 | 8.91 | 3.01 % | 51.4 % | +143.8 % |
| Naive baseline ("tomorrow = today") | 4.44 | 3.15 | 1.09 % | – | – |

**What this shows**

* The price-level LSTM is **2.4× worse than doing nothing**: AAPL traded above
  its training range during the test year, and a model trained on prices
  scaled to 0–1 cannot predict beyond that range, so it lagged throughout.
  The original project could not reveal this, because its scaler had seen the
  test data and it had no baseline.
* Predicting **returns** removes that limit and brings the error down to the
  naive level. Its 0.3 % edge and 53.4 % direction accuracy are, however,
  **not statistically meaningful** (53.4 % over 251 days is p ≈ 0.3 against a
  coin flip). The model effectively learned "tomorrow ≈ today plus a slight
  drift", which is what an honest model should learn from daily prices alone.
* Conclusion: with closing prices as the only input, an LSTM does not
  meaningfully beat the naive forecast. Beating it would need other
  information (news, earnings, market-wide factors) and much more careful
  validation.

Results vary with the stock, the period and the random seed; rerun with
`python -m spp.train --ticker <SYMBOL>` and `--target price` to compare.

---

## Quick start

```powershell
git clone https://github.com/Purnakash/stock-price-prediction.git
cd stock-price-prediction
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

TensorFlow 2.20+ supports Python 3.10–3.13 (use `py -3.13`; if TensorFlow fails to install, use 3.12).

**Dashboard:**

```bash
streamlit run app.py
```

Your browser opens http://localhost:8501. Enter a ticker, click
**Train & predict**, and the model trains in about 10–60 seconds.

**Command line:**

```bash
python -m spp.train --ticker AAPL
python -m spp.train --ticker TATACONSUM.NS --period 10y --epochs 30
python -m spp.train --ticker AAPL --target price          # price-level model, to compare
python -m spp.train --csv data/NSE-TATA.csv --target price --report   # original setup on the original data
```

This prints the test metrics against the naive baseline and a 5-day forecast,
and saves the model, the scaler and the metrics in `models/`.

**Tests:**

```bash
pytest
```

---

## Project layout

```
stock-price-prediction/
├── app.py                 Streamlit dashboard
├── src/spp/
│   ├── data.py            Yahoo Finance / CSV loading and cleaning
│   ├── features.py        time split, returns or prices, leak-free scaling, windows
│   ├── model.py           LSTM (original + improved), training
│   ├── metrics.py         RMSE, MAE, MAPE, direction, naive baseline
│   ├── forecast.py        test-period evaluation, recursive forecast
│   └── train.py           command-line training
├── original/              B.Tech code recovered from the report (for reference)
├── tests/                 pipeline tests (no network or TensorFlow needed)
└── data/                  optional CSV files (not committed)
```

---

## Deploy

Free hosting on **Streamlit Community Cloud**:

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Choose this repo, branch `main`, main file `app.py`. Under *Advanced
   settings*, pick Python **3.12** (the most widely supported version for TensorFlow on the cloud).
4. Deploy. You get a public link like `https://<name>.streamlit.app`; put it at
   the top of this README.

The free tier has about 1 GB of memory. Training uses less than that with the
default settings; keep the epochs and the history moderate.

---

## Limitations

* Uses closing prices only: no news, earnings, fundamentals or market-wide data.
* LSTMs cannot predict prices outside the range seen in training, so in strong
  trends predictions lag behind the real price.
* Forecast errors compound with every step, so days 2–10 are a rough trend at best.
* Results change between runs and between periods; one good test period proves little.

## Team

Original B.Tech project: **Purnakash G**, **Prem Kumar B**,
**Shaik Mohammad Sohail**. Rebuild, fixes and dashboard: Purnakash G.

## References

* S. Hochreiter, J. Schmidhuber, "Long short-term memory", *Neural Computation* 9(8), 1997.
* R. J. Hyndman, G. Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.),
  chapter on simple forecasting methods and the naive benchmark.
