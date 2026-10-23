"""Stock price prediction dashboard (Streamlit).

Run locally:   streamlit run app.py
On Streamlit Community Cloud, point the app at this file.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from spp.data import DataError, load_csv, load_yahoo  # noqa: E402
from spp.features import prepare  # noqa: E402
from spp.forecast import evaluate, forecast  # noqa: E402

# Hide pandas 3.x deprecation notices raised inside yfinance (harmless). yfinance is
# only imported when prices are downloaded, so this still takes effect in time.
warnings.filterwarnings("ignore", message=r".*inplace keyword.*is deprecated.*")

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

PERIODS = {"2 years": "2y", "5 years": "5y", "10 years": "10y", "Max": "max"}
POPULAR = ["AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "TCS.NS", "RELIANCE.NS", "INFY.NS", "TATACONSUM.NS"]


@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(ticker: str, period: str) -> pd.DataFrame:
    return load_yahoo(ticker, period)


@st.cache_resource(show_spinner=False, max_entries=8)
def get_model(key: tuple, _close: pd.Series, window: int, epochs: int, report: bool, target: str):
    """Train once per (data, settings) combination; Streamlit keeps it in memory."""
    from spp.model import predictor, train

    prep = prepare(_close, window=window, target=target)
    model, hist = train(prep, epochs=epochs, report=report)
    return prep, predictor(model), hist


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    source = st.radio("Data source", ["Yahoo Finance", "Upload CSV"], horizontal=True)
    if source == "Yahoo Finance":
        ticker = st.text_input("Ticker symbol", "AAPL", help="Indian stocks: add .NS (NSE) or .BO (BSE)")
        st.caption("Try: " + ", ".join(POPULAR[:7]))
        period = PERIODS[st.select_slider("History", list(PERIODS), value="5 years")]
        upload = None
    else:
        upload = st.file_uploader("CSV with Date and Close columns", type="csv")
        ticker, period = (Path(upload.name).stem if upload else ""), None
    st.divider()
    target_label = st.radio(
        "Model predicts",
        ["Daily returns (recommended)", "Price level (original project)"],
        help="Returns stay in a similar range when a stock hits new highs; price-level models "
        "cannot predict above the prices they were trained on and lag in uptrends.",
    )
    target = "returns" if target_label.startswith("Daily") else "price"
    window = st.slider("Look-back window (days)", 20, 120, 60, step=10)
    epochs = st.slider("Max training epochs", 5, 50, 20, step=5)
    report_arch = st.toggle("Original architecture (no dropout)", value=False)
    days = st.slider("Forecast days", 1, 10, 5)
    run = st.button("🚀 Train & predict", type="primary")

st.title("📈 Stock Price Prediction with LSTM")
st.caption(
    "Educational project, **not financial advice**. Past prices do not reliably predict future prices."
)

tab_pred, tab_cmp, tab_about = st.tabs(["🔮 Prediction", "📊 Compare stocks", "ℹ️ How it works"])

# ── prediction tab ───────────────────────────────────────────────────────────
with tab_pred:
    if not run and "result" not in st.session_state:
        st.info("Choose a stock in the sidebar and click **Train & predict**. Training takes about 10–60 s.")
    if run:
        try:
            with st.spinner(f"Loading prices for {ticker} ..."):
                if upload is not None:
                    df = load_csv(upload)
                elif source == "Upload CSV":
                    raise DataError("Upload a CSV file first.")
                else:
                    df = get_prices(ticker, period)
            close = df["Close"]
            key = (ticker, period, len(close), str(close.index[-1]), window, epochs, report_arch, target)
            with st.spinner("Training the LSTM ..."):
                prep, predict, hist = get_model(key, close, window, epochs, report_arch, target)
            table, metrics = evaluate(prep, predict, close)
            st.session_state["result"] = dict(
                ticker=ticker, df=df, prep=prep, table=table, metrics=metrics, target=target,
                fc=forecast(prep, predict, close, days), epochs=len(hist["loss"]),
            )  # fmt: skip
        except (DataError, ValueError) as exc:
            st.error(str(exc))
            st.session_state.pop("result", None)

    r = st.session_state.get("result")
    if r:
        m, naive, table = r["metrics"]["model"], r["metrics"]["naive"], r["table"]
        last = r["df"]["Close"].iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last close", f"{last:,.2f}")
        c2.metric("LSTM error (RMSE)", f"{m['RMSE']:,.2f}",
                  f"{r['metrics']['rmse_vs_naive_%']:+.1f} % vs naive", delta_color="inverse")  # fmt: skip
        c3.metric("Average error (MAPE)", f"{m['MAPE %']:.2f} %")
        c4.metric("Up/down direction right", f"{m['Direction %']:.0f} %")
        if r["metrics"]["beats_naive"]:
            st.success("The LSTM beats the naive baseline (\"tomorrow = today\") on the test period.")
        else:
            st.warning(
                "The LSTM does **not** beat the naive baseline (\"tomorrow = today\") here. "
                "This is common for daily prices, which behave close to a random walk."
            )

        hist_close = r["df"]["Close"].loc[: table.index[0]]
        fig = go.Figure()
        fig.add_scatter(x=hist_close.index, y=hist_close, name="History", line=dict(color="#94a3b8", width=1))
        fig.add_scatter(x=table.index, y=table["Actual"], name="Actual (test)", line=dict(color="#2563eb"))
        fig.add_scatter(x=table.index, y=table["Predicted"], name="LSTM prediction", line=dict(color="#ea580c"))
        fig.add_scatter(x=r["fc"].index, y=r["fc"], name=f"Forecast ({len(r['fc'])} days)",
                        line=dict(color="#16a34a", dash="dot"), mode="lines+markers")  # fmt: skip
        fig.update_layout(height=520, hovermode="x unified", margin=dict(t=30, l=10, r=10, b=10),
                          legend=dict(orientation="h", y=1.02), yaxis_title="Price",
                          xaxis=dict(rangeslider=dict(visible=True), rangeselector=dict(buttons=[
                              dict(count=6, label="6M", step="month", stepmode="backward"),
                              dict(count=1, label="1Y", step="year", stepmode="backward"),
                              dict(step="all")])))  # fmt: skip
        st.plotly_chart(fig)

        left, right = st.columns([2, 1])
        with left:
            st.subheader("Test-period metrics")
            st.dataframe(pd.DataFrame({"LSTM": m, "Naive baseline": naive}).T.round(2))
            st.caption(f"Test period: {table.index[0]:%Y-%m-%d} to {table.index[-1]:%Y-%m-%d} "
                       f"({len(table)} days), predicting {r['target']}, trained for {r['epochs']} epochs.")  # fmt: skip
        with right:
            st.subheader("Forecast")
            fc_table = r["fc"].round(2).rename("Price").to_frame()
            fc_table.index = fc_table.index.strftime("%a %d %b %Y")
            st.dataframe(fc_table)
            st.caption("Each day is predicted from the previous predictions, so errors grow quickly.")
        st.download_button("⬇️ Download predictions (CSV)", table.to_csv().encode(),
                           file_name=f"{r['ticker']}_predictions.csv", mime="text/csv")  # fmt: skip

# ── compare tab (from the original project's dashboard) ─────────────────────
with tab_cmp:
    picks = st.multiselect("Stocks", POPULAR, default=["TSLA", "AAPL", "META", "MSFT"])
    cmp_period = PERIODS[st.select_slider("Period", list(PERIODS), value="2 years", key="cmp_period")]
    if picks:
        frames = {}
        for t in picks:
            try:
                frames[t] = get_prices(t, cmp_period)
            except DataError as exc:
                st.warning(str(exc))
        if frames:
            hl = go.Figure()
            for t, d in frames.items():
                hl.add_scatter(x=d.index, y=d["High"], name=f"High {t}", mode="lines", opacity=0.8)
                hl.add_scatter(x=d.index, y=d["Low"], name=f"Low {t}", mode="lines", opacity=0.5)
            hl.update_layout(title="High and low prices", height=450, hovermode="x unified",
                             xaxis=dict(rangeslider=dict(visible=True)), yaxis_title="Price")  # fmt: skip
            st.plotly_chart(hl)
            vol = go.Figure()
            for t, d in frames.items():
                vol.add_scatter(x=d.index, y=d["Volume"], name=t, mode="lines", opacity=0.8)
            vol.update_layout(title="Trading volume", height=380, hovermode="x unified", yaxis_title="Shares")
            st.plotly_chart(vol)
            norm = pd.DataFrame({t: d["Close"] / d["Close"].iloc[0] * 100 for t, d in frames.items()})
            st.subheader("Performance (start = 100)")
            st.line_chart(norm)

# ── about tab ────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
**Model.** Two stacked LSTM layers (50 units each) and a dense output, trained
with Adam on mean squared error. Each prediction uses the previous *N* days
(the look-back window).

**What it predicts.** By default the model predicts tomorrow's **return**
(the log of tomorrow's close divided by today's), and the price is
today's close × e^return. The original project predicted the **price level**
directly, scaled to 0–1; such a model cannot output prices above the highest
price it saw in training, so it lags whenever a stock reaches new highs. Both
are available in the sidebar so you can compare them.

**Evaluation.** The last 20 % of days are held out as a test period the model
never sees in training. Each test day is predicted from the real prices before
it (one step ahead) and compared with the **naive baseline**, which simply
predicts that tomorrow's close equals today's.

**Improvements over the original B.Tech project**
* The scaler is fitted on training data only. The original fitted it on all
  data, which leaks test information into training.
* Real error metrics (RMSE, MAE, MAPE, direction) and a baseline, instead of
  an "accuracy" percentage that has no meaning for price regression.
* Several epochs with early stopping (the original trained one epoch).
* Option to predict returns instead of price levels, so new highs can be predicted.
* Live data for any ticker via Yahoo Finance, plus CSV upload.

**Limits.** Prices depend on news, earnings and the wider market, none of which
the model sees. LSTMs also cannot predict prices outside the range they were
trained on, so in strong trends they lag behind. Use this to learn about time
series models, not to trade.

Rebuilt from our B.Tech project *Stock Price Prediction using Machine Learning*
(SRM IST): Purnakash G, Prem Kumar B, Shaik Mohammad Sohail.
"""
    )
