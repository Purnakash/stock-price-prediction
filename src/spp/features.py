"""Turn a price series into supervised windows for the LSTM.

Two targets are supported:

* ``target="returns"`` (default): the model sees the last ``window`` daily log
  returns, ln(close_t / close_t-1), standardised, and predicts tomorrow's
  return. The price is then ``today's close * exp(predicted return)``.
  Returns stay in a similar range even when a stock reaches new highs, so the
  model is not stuck inside the price range it was trained on.
* ``target="price"`` (the original project): the model sees the last
  ``window`` closing prices scaled to [0, 1] and predicts tomorrow's price
  directly. It cannot predict prices above the training maximum, so it lags
  in strong uptrends.

The split is by time (no shuffling): the model is trained on the past and
tested on the most recent ``test_frac`` of days. Scalers are fitted on the
training period only; fitting on all data (as the original project did) leaks
test information into training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

TARGETS = ("returns", "price")


def make_windows(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """1-D array -> X (n, window, 1), y (n,) with y[i] = values[i + window]."""
    v = np.asarray(values, dtype=np.float32).ravel()
    if window < 1:
        raise ValueError("window must be >= 1")
    n = len(v) - window
    if n <= 0:
        raise ValueError(f"Need more than {window} values to make windows, got {len(v)}.")
    idx = np.arange(window)[None, :] + np.arange(n)[:, None]
    return v[idx][..., None], v[window:]


def log_returns(close: pd.Series) -> np.ndarray:
    """ln(close_t / close_t-1); the first value is NaN."""
    return np.log(close.to_numpy(float)) - np.log(close.shift(1).to_numpy(float))


@dataclass
class Prepared:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler | StandardScaler
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    window: int
    target: str = "price"

    def inverse(self, scaled: np.ndarray) -> np.ndarray:
        """Scaled model values back to the target's units (prices or log returns)."""
        return self.scaler.inverse_transform(np.asarray(scaled, dtype=np.float64).reshape(-1, 1)).ravel()

    def model_series(self, close: pd.Series) -> np.ndarray:
        """The scaled series the model reads (prices or returns), one value per day."""
        raw = close.to_numpy(float) if self.target == "price" else log_returns(close)
        return self.scaler.transform(raw.reshape(-1, 1)).ravel()

    def to_prices(self, scaled_pred: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
        """Model outputs -> predicted prices, given each day's previous close."""
        values = self.inverse(scaled_pred)
        if self.target == "price":
            return values
        return np.asarray(prev_close, dtype=np.float64) * np.exp(values)


def prepare(close: pd.Series, window: int = 60, test_frac: float = 0.2, target: str = "returns") -> Prepared:
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be between 0 and 1")
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    close = close.dropna().astype(float)
    if (close <= 0).any():
        raise ValueError("Prices must be positive.")
    n_test = max(1, int(round(len(close) * test_frac)))
    n_train = len(close) - n_test
    if n_train <= window + 10:
        raise ValueError(
            f"Not enough history: {len(close)} days with window={window}. "
            "Use a longer period or a smaller window."
        )

    if target == "price":
        raw = close.to_numpy(float)
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(raw[:n_train].reshape(-1, 1))
        first = 0  # first usable day
    else:
        raw = log_returns(close)
        scaler = StandardScaler().fit(raw[1:n_train].reshape(-1, 1))  # day 0 has no return
        first = 1
    scaled = scaler.transform(np.nan_to_num(raw).reshape(-1, 1)).ravel()  # train only: no leakage

    x_train, y_train = make_windows(scaled[first:n_train], window)
    # test windows look back into the training period for context
    x_test, y_test = make_windows(scaled[n_train - window :], window)
    return Prepared(
        x_train, y_train, x_test, y_test, scaler,
        train_dates=close.index[first + window : n_train],
        test_dates=close.index[n_train:],
        window=window,
        target=target,
    )  # fmt: skip
