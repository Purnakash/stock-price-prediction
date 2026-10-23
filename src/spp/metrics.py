"""Error metrics and the naive baseline every stock model must beat.

The naive forecast says "tomorrow's close = today's close". For daily prices
it is surprisingly hard to beat, so a model is only useful if its error is
clearly lower than the naive one. (The original project reported "80-95 %
accuracy", which is not a meaningful measure for price regression.)
"""

from __future__ import annotations

import numpy as np


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def mape(y: np.ndarray, p: np.ndarray) -> float:
    """Mean absolute percentage error, in percent."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100) if mask.any() else float("nan")


def directional_accuracy(y: np.ndarray, p: np.ndarray, prev: np.ndarray) -> float:
    """Share of days where the predicted move (up/down vs the previous close) was right, in percent."""
    y, p, prev = (np.asarray(a, float) for a in (y, p, prev))
    true_dir, pred_dir = np.sign(y - prev), np.sign(p - prev)
    keep = true_dir != 0
    return float(np.mean(true_dir[keep] == pred_dir[keep]) * 100) if keep.any() else float("nan")


def naive_forecast(prev_close: np.ndarray) -> np.ndarray:
    """Persistence baseline: prediction for day t = actual close of day t-1."""
    return np.asarray(prev_close, float)


def report(y: np.ndarray, pred: np.ndarray, prev: np.ndarray) -> dict:
    """Model vs naive baseline on the same days."""
    naive = naive_forecast(prev)
    out = {
        "model": {"RMSE": rmse(y, pred), "MAE": mae(y, pred), "MAPE %": mape(y, pred),
                  "Direction %": directional_accuracy(y, pred, prev)},
        "naive": {"RMSE": rmse(y, naive), "MAE": mae(y, naive), "MAPE %": mape(y, naive),
                  "Direction %": float("nan")},
    }  # fmt: skip
    out["beats_naive"] = out["model"]["RMSE"] < out["naive"]["RMSE"]
    out["rmse_vs_naive_%"] = (out["model"]["RMSE"] / out["naive"]["RMSE"] - 1) * 100 if out["naive"]["RMSE"] else 0.0
    return out
