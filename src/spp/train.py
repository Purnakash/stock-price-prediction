"""Train and evaluate the LSTM from the command line.

    python -m spp.train --ticker AAPL
    python -m spp.train --ticker TATACONSUM.NS --period 10y --epochs 30
    python -m spp.train --ticker AAPL --target price         # predict price levels (original approach)
    python -m spp.train --csv data/NSE-TATA.csv --target price --report   # original setup

Prints test metrics vs the naive baseline and a 5-day forecast, and saves
models/<name>.keras, models/<name>_scaler.joblib and models/<name>_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticker", help="Yahoo Finance symbol, e.g. AAPL, TCS.NS")
    src.add_argument("--csv", type=Path, help="CSV with Date and Close columns (e.g. NSE export)")
    p.add_argument("--period", default="5y", help="history to download (default 5y)")
    p.add_argument("--window", type=int, default=60, help="days of history per sample (default 60)")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--target", choices=["returns", "price"], default="returns",
                   help="predict daily returns (default) or price levels (original project)")  # fmt: skip
    p.add_argument("--report", action="store_true", help="use the original architecture (no dropout)")
    p.add_argument("--forecast-days", type=int, default=5)
    p.add_argument("--out", type=Path, default=MODELS_DIR)
    a = p.parse_args(argv)

    import joblib

    from .data import DataError, load_csv, load_yahoo
    from .features import prepare
    from .forecast import evaluate, forecast
    from .model import predictor, train

    try:
        df = load_yahoo(a.ticker, a.period) if a.ticker else load_csv(a.csv)
    except DataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", a.ticker or a.csv.stem)
    close = df["Close"]
    print(f"{name}: {len(df)} days, {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}, predicting {a.target}")

    prep = prepare(close, window=a.window, test_frac=a.test_frac, target=a.target)
    model, hist = train(prep, epochs=a.epochs, report=a.report, verbose=2)
    predict = predictor(model)
    table, metrics = evaluate(prep, predict, close)
    fc = forecast(prep, predict, close, days=a.forecast_days)

    print("\nTest period:", f"{prep.test_dates[0]:%Y-%m-%d} to {prep.test_dates[-1]:%Y-%m-%d}",
          f"({len(prep.test_dates)} days)")  # fmt: skip
    print(f"{'':12}{'RMSE':>10}{'MAE':>10}{'MAPE %':>10}{'Dir %':>8}")
    for k in ("model", "naive"):
        m = metrics[k]
        print(f"{'LSTM' if k == 'model' else 'Naive':12}{m['RMSE']:10.2f}{m['MAE']:10.2f}"
              f"{m['MAPE %']:10.2f}{m['Direction %']:8.1f}")  # fmt: skip
    verdict = "beats" if metrics["beats_naive"] else "does NOT beat"
    print(f"LSTM {verdict} the naive baseline ({metrics['rmse_vs_naive_%']:+.1f} % RMSE).")
    print("\nForecast:\n" + fc.round(2).to_string())

    a.out.mkdir(parents=True, exist_ok=True)
    model.save(a.out / f"{name}.keras")
    joblib.dump(prep.scaler, a.out / f"{name}_scaler.joblib")
    (a.out / f"{name}_metrics.json").write_text(json.dumps(
        {"name": name, "target": a.target, "window": a.window, "epochs_run": len(hist["loss"]),
         "report_architecture": a.report,
         "test_start": str(prep.test_dates[0].date()), "test_end": str(prep.test_dates[-1].date()),
         **metrics}, indent=2, default=float))  # fmt: skip
    print(f"\nSaved to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
