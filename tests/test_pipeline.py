"""Data, windowing, metrics and forecasting tests: no TensorFlow or network needed."""

import importlib.util
import io

import numpy as np
import pandas as pd
import pytest

from spp.data import DataError, clean, load_csv
from spp.features import make_windows, prepare
from spp.forecast import evaluate, forecast
from spp.metrics import directional_accuracy, mae, mape, naive_forecast, report, rmse

HAS_KERAS = importlib.util.find_spec("keras") is not None


def random_walk(n=800, seed=0, start="2018-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx, name="Close")


def nse_csv(n=300):
    """Same layout as the NSE-TATAGLOBAL file of the original project: newest row first."""
    s = random_walk(n)
    df = pd.DataFrame({"Date": s.index.strftime("%Y-%m-%d"), "Open": s, "High": s * 1.01, "Low": s * 0.99,
                       "Last": s, "Close": s, "Total Trade Quantity": 1000, "Turnover (Lacs)": 1.0})  # fmt: skip
    return df.iloc[::-1].to_csv(index=False)


# ── data ─────────────────────────────────────────────────────────────────────


def test_clean_nse_layout_sorts_and_maps_columns():
    df = load_csv(io.StringIO(nse_csv()))
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index.is_monotonic_increasing
    assert (df["Volume"] == 1000).all()  # "Total Trade Quantity" -> Volume


def test_clean_drops_bad_rows_and_duplicates():
    raw = pd.DataFrame({"date": ["2020-01-02", "not a date", "2020-01-03", "2020-01-03", "2020-01-06"],
                        "close": ["10", "11", "x", "12", "13"]})  # fmt: skip
    df = clean(raw)
    assert list(df["Close"]) == [10.0, 12.0, 13.0]
    assert (df["High"] == df["Close"]).all()  # missing OHLC filled from Close


def test_clean_errors_are_readable(tmp_path):
    with pytest.raises(DataError, match="Close"):
        clean(pd.DataFrame({"Date": ["2020-01-01"], "Price": [1]}))
    with pytest.raises(DataError, match="not found"):
        load_csv(tmp_path / "missing.csv")


# ── windows and split ────────────────────────────────────────────────────────


def test_make_windows():
    x, y = make_windows(np.arange(10), 3)
    assert x.shape == (7, 3, 1) and y.shape == (7,)
    assert list(x[0, :, 0]) == [0, 1, 2] and y[0] == 3
    assert list(x[-1, :, 0]) == [6, 7, 8] and y[-1] == 9
    with pytest.raises(ValueError):
        make_windows(np.arange(3), 3)


def test_prepare_price_mode_has_no_leakage_and_correct_alignment():
    close = random_walk(500)
    prep = prepare(close, window=30, test_frac=0.2, target="price")
    n_test = 100
    assert len(prep.test_dates) == n_test == len(prep.y_test) == len(prep.x_test)
    # scaler sees training prices only: their range maps exactly onto [0, 1]
    train = close.iloc[:400]
    assert prep.scaler.data_min_[0] == pytest.approx(train.min())
    assert prep.scaler.data_max_[0] == pytest.approx(train.max())
    # targets are the real prices of the test days
    assert np.allclose(prep.inverse(prep.y_test), close.loc[prep.test_dates].to_numpy(), rtol=1e-5)
    # the last value of each test window is the previous day's close
    assert np.allclose(prep.inverse(prep.x_test[:, -1, 0]), close.shift(1).loc[prep.test_dates], rtol=1e-5)


def test_prepare_returns_mode_alignment_and_scaling():
    close = random_walk(500)
    prep = prepare(close, window=30, test_frac=0.2)  # returns is the default
    assert prep.target == "returns"
    assert len(prep.test_dates) == 100 == len(prep.x_test)
    ret = np.log(close / close.shift(1))
    # scaler fitted on training returns only (day 0 has none)
    assert prep.scaler.mean_[0] == pytest.approx(ret.iloc[1:400].mean())
    # target = that day's return; last input = previous day's return
    assert np.allclose(prep.inverse(prep.y_test), ret.loc[prep.test_dates], atol=1e-6)
    assert np.allclose(prep.inverse(prep.x_test[:, -1, 0]), ret.shift(1).loc[prep.test_dates], atol=1e-6)
    assert len(prep.x_train) == len(prep.train_dates)


def test_prepare_rejects_bad_input():
    with pytest.raises(ValueError, match="Not enough history"):
        prepare(random_walk(80), window=60)
    with pytest.raises(ValueError, match="target"):
        prepare(random_walk(300), target="volume")
    bad = random_walk(300)
    bad.iloc[5] = -1
    with pytest.raises(ValueError, match="positive"):
        prepare(bad)


# ── metrics ──────────────────────────────────────────────────────────────────


def test_metrics_values():
    y, p = np.array([100.0, 110.0]), np.array([90.0, 110.0])
    assert rmse(y, p) == pytest.approx(np.sqrt(50))
    assert mae(y, p) == 5.0
    assert mape(y, p) == pytest.approx(5.0)
    assert directional_accuracy([101, 99], [102, 98], [100, 100]) == 100.0
    assert directional_accuracy([101, 99], [99, 101], [100, 100]) == 0.0
    assert list(naive_forecast([1, 2])) == [1.0, 2.0]


def test_report_flags_whether_model_beats_naive():
    y, prev = np.array([10.0, 11, 12, 13]), np.array([9.0, 10, 11, 12])
    assert report(y, y, prev)["beats_naive"] is True  # perfect model
    assert report(y, prev - 5, prev)["beats_naive"] is False  # worse than naive


# ── evaluate / forecast with stand-in models ─────────────────────────────────


def naive_predictor(prep):
    """A stand-in model whose forecast is exactly "tomorrow = today" in either mode."""
    if prep.target == "price":
        return lambda x: x[:, -1, 0]  # repeat the last price
    zero = float(prep.scaler.transform([[0.0]])[0, 0])
    return lambda x: np.full(len(x), zero)  # predict a zero return


@pytest.mark.parametrize("target", ["price", "returns"])
def test_evaluate_with_persistence_model_equals_naive(target):
    close = random_walk(400)
    prep = prepare(close, window=20, target=target)
    table, m = evaluate(prep, naive_predictor(prep), close)
    assert list(table.columns) == ["Actual", "Predicted", "Naive"]
    assert np.allclose(table["Predicted"], table["Naive"], rtol=1e-5)
    assert m["model"]["RMSE"] == pytest.approx(m["naive"]["RMSE"], rel=1e-4)


def test_returns_mode_can_predict_above_the_training_range():
    # steady uptrend: test prices are all above the training maximum
    idx = pd.bdate_range("2020-01-01", periods=400)
    close = pd.Series(100 * 1.002 ** np.arange(400), index=idx)
    prep = prepare(close, window=20, target="returns")
    up = float(prep.scaler.transform([[np.log(1.002)]])[0, 0])
    table, m = evaluate(prep, lambda x: np.full(len(x), up), close)
    assert table["Predicted"].max() > close.iloc[:320].max()
    assert m["model"]["RMSE"] < 1e-6  # the constant +0.2 %/day model is exact here
    assert m["beats_naive"]


@pytest.mark.parametrize("target", ["price", "returns"])
def test_forecast_dates_and_recursion(target):
    close = random_walk(300)
    prep = prepare(close, window=20, target=target)
    fc = forecast(prep, naive_predictor(prep), close, days=5)
    assert len(fc) == 5
    assert fc.index[0] > close.index[-1] and all(d.weekday() < 5 for d in fc.index)
    assert np.allclose(fc, close.iloc[-1], rtol=1e-5)  # persistence stays flat

    calls = []

    def drift(x):  # +0.01 per step in scaled space, records what it was fed
        calls.append(x[0, -1, 0])
        return x[:, -1, 0] + 0.01

    forecast(prep, drift, close, days=3)
    assert np.allclose(np.diff(calls), 0.01, atol=1e-6)  # each step sees the previous prediction
    with pytest.raises(ValueError):
        forecast(prep, naive_predictor(prep), close, days=0)


@pytest.mark.skipif(not HAS_KERAS, reason="Keras not installed")
def test_lstm_trains_and_predicts():
    from spp.model import build_lstm, predictor, train

    assert [type(layer).__name__ for layer in build_lstm(30, report=True).layers] == ["LSTM", "LSTM", "Dense"]
    close = random_walk(300)
    prep = prepare(close, window=30)
    model, hist = train(prep, epochs=2)  # returns mode (default)
    assert len(hist["loss"]) >= 1
    table, _ = evaluate(prep, predictor(model), close)
    assert table["Predicted"].notna().all()
