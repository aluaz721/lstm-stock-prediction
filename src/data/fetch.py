"""
Fetches daily OHLCV data for the stock universe via yfinance.

Note: this sandbox's network access is restricted to package registries and
can't reach Yahoo Finance, so this module is verified by API surface and
return-shape contract (see tests/test_fetch_contract.py), not a live call.
Run it for real on a machine with normal internet access.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "AMD", "INTC", "CRM",
]


def fetch_ticker_history(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """
    Fetch daily OHLCV history for one ticker.

    Returns a DataFrame indexed by date with columns:
    Open, High, Low, Close, Volume (at minimum).
    """
    df = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} in range {start}..{end}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["ticker"] = ticker
    return df


def fetch_universe(
    tickers: list[str] = TICKERS,
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch and concatenate OHLCV history for the full ticker universe.
    Returns one long DataFrame with a 'ticker' column, indexed by date.
    """
    frames = []
    for ticker in tickers:
        try:
            frames.append(fetch_ticker_history(ticker, start, end))
        except ValueError as e:
            print(f"Skipping {ticker}: {e}")
    if not frames:
        raise RuntimeError("No data fetched for any ticker in the universe")
    return pd.concat(frames).sort_index()
