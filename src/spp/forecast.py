"""Evaluate on the test period and forecast future days.

Both functions take a ``predict`` callable (scaled windows -> scaled values),
so they work with the Keras model and can be tested without TensorFlow, and
with either target (prices or returns).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .features import Prepared
from .metrics import report

Predict = Callable[[np.ndarray], np.ndarray]


def evaluate(prep: Prepared, predict: Predict, close: pd.Series) -> tuple[pd.DataFrame, dict]:
    """One-step-ahead predictions on the test days, plus metrics vs the naive baseline."""
    prev = close.shift(1).loc[prep.test_dates].to_numpy(float)
    pred = prep.to_prices(predict(prep.x_test), prev)
    actual = close.loc[prep.test_dates].to_numpy(float)
    table = pd.DataFrame({"Actual": actual, "Predicted": pred, "Naive": prev}, index=prep.test_dates)
    return table, report(actual, pred, prev)


def forecast(prep: Prepared, predict: Predict, close: pd.Series, days: int = 5) -> pd.Series:
    """Recursive forecast: predict tomorrow, append it, predict the day after, ...

    Errors compound with every step, so treat anything beyond a few days as a
    rough trend, not a price target.
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    window = prep.model_series(close)[-prep.window :]
    price = float(close.iloc[-1])
    out = []
    for _ in range(days):
        nxt = float(np.asarray(predict(window[None, :, None])).ravel()[0])
        price = float(prep.to_prices(np.array([nxt]), np.array([price]))[0])
        out.append(price)
        window = np.append(window[1:], nxt)
    dates = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=days)
    return pd.Series(out, index=dates, name="Forecast")
