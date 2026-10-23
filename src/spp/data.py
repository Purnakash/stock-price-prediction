"""Load daily stock prices from Yahoo Finance or from a CSV file.

Both return the same shape: a DataFrame indexed by date (ascending) with
columns Open, High, Low, Close, Volume (floats), no missing closes.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Column names used by NSE / BSE exports, e.g. the NSE-TATAGLOBAL file used in
# the original project: Date, Open, High, Low, Last, Close,
# Total Trade Quantity, Turnover (Lacs)
_ALIASES = {
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj Close",
    "volume": "Volume",
    "total trade quantity": "Volume",
    "shares traded": "Volume",
}


class DataError(ValueError):
    """Raised with a readable message when prices cannot be loaded."""


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, parse dates, sort ascending, drop bad rows."""
    df = df.rename(columns={c: _ALIASES.get(str(c).strip().lower(), str(c).strip()) for c in df.columns})
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df.index.name = "Date"
    if "Close" not in df.columns:
        raise DataError(f"No 'Close' column found; columns are: {list(df.columns)}")
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = df["Close"] if col != "Volume" else 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[COLUMNS].dropna(subset=["Close"])
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if len(df) == 0:
        raise DataError("No usable rows after cleaning.")
    return df


def load_csv(source) -> pd.DataFrame:
    """CSV from a path or an open file / upload (anything pandas.read_csv accepts)."""
    if isinstance(source, (str, Path)):
        source = Path(source)
        if not source.is_file():
            raise DataError(f"CSV file not found: {source}")
    try:
        raw = pd.read_csv(source)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataError(f"Could not read the CSV file: {exc}") from exc
    return clean(raw)


def load_yahoo(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Daily prices from Yahoo Finance via yfinance (e.g. 'AAPL', 'TATACONSUM.NS')."""
    import yfinance as yf

    ticker = ticker.strip().upper()
    if not ticker:
        raise DataError("Enter a ticker symbol, e.g. AAPL or TCS.NS.")
    try:
        with warnings.catch_warnings():
            # yfinance still uses pandas options that newer pandas deprecates;
            # harmless for us, so keep the output clean.
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception as exc:  # network errors, Yahoo rate limits, ...
        raise DataError(f"Could not download '{ticker}' from Yahoo Finance: {exc}") from exc
    if raw is None or len(raw) == 0:
        raise DataError(
            f"No data for '{ticker}'. Check the symbol; Indian stocks need a suffix, "
            "e.g. TCS.NS (NSE) or TCS.BO (BSE)."
        )
    if isinstance(raw.columns, pd.MultiIndex):  # yfinance >= 0.2.40 returns (field, ticker)
        raw.columns = raw.columns.get_level_values(0)
    return clean(raw.reset_index())
