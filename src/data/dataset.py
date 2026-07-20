"""
Turns the engineered feature universe into (X, y) sequences ready for
model training.

Design decisions (updated after the raw-price-level target was diagnosed
as the root cause of a train/test distribution mismatch -- see
features.py's docstring for the empirical evidence):

1. Chronological split uses ONE global cutoff date across all tickers, not
   a per-ticker row-fraction split. If ticker A has a longer history than
   ticker B, splitting by row-fraction would put different calendar dates
   in "train" for A vs B, which both leaks and confuses "chronological."

2. INPUT FEATURES are scaled per ticker (StandardScaler, fit only on that
   ticker's training rows) -- same reasoning as before: a shared pooled
   model needs comparable scales across tickers, and fitting only on train
   avoids leaking future information into normalization.

3. The TARGET (next_log_return) is NOT scaled. Log returns are already
   close to stationary across time (verified empirically), unlike price
   levels, which is specifically why price levels needed scaling in the
   first place. Scaling an already-stationary quantity adds a layer of
   indirection for no real benefit, and skipping it means predictions
   convert back to real prices directly (last_close * exp(predicted
   return)) without an inverse-transform step.

4. Sequences are pooled across all 10 tickers into one training set for
   ONE shared model per architecture.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.features import FEATURE_COLUMNS

TARGET_COLUMN = "next_log_return"
PRICE_COLUMN = "Close"


@dataclass
class SplitData:
    train: pd.DataFrame
    test: pd.DataFrame
    split_date: pd.Timestamp


def chronological_split(universe_df: pd.DataFrame, train_frac: float = 0.65) -> SplitData:
    """
    Picks a single cutoff date at the train_frac quantile of all unique
    trading dates present in the universe, then splits every ticker on
    that same date.
    """
    unique_dates = pd.Series(sorted(universe_df.index.unique()))
    split_idx = int(len(unique_dates) * train_frac)
    split_date = unique_dates.iloc[split_idx]

    train = universe_df[universe_df.index <= split_date]
    test = universe_df[universe_df.index > split_date]
    return SplitData(train=train, test=test, split_date=split_date)


def fit_scalers_per_ticker(train_df: pd.DataFrame) -> dict[str, StandardScaler]:
    """Fits one StandardScaler per ticker on FEATURE_COLUMNS, train rows only."""
    scalers = {}
    for ticker, group in train_df.groupby("ticker", sort=False):
        scaler = StandardScaler()
        scaler.fit(group[FEATURE_COLUMNS].values)
        scalers[ticker] = scaler
    return scalers


def transform_with_scalers(
    df: pd.DataFrame, scalers: dict[str, StandardScaler]
) -> pd.DataFrame:
    """
    Applies each ticker's fitted scaler to FEATURE_COLUMNS only.
    TARGET_COLUMN and PRICE_COLUMN are carried through UNSCALED -- the
    target because it's deliberately not scaled (see module docstring),
    and PRICE_COLUMN because build_sequences needs the real Close value
    to compute last_close for converting predicted returns back to prices.
    """
    pieces = []
    for ticker, group in df.groupby("ticker", sort=False):
        if ticker not in scalers:
            raise KeyError(
                f"No fitted scaler for ticker {ticker!r} -- "
                "was it present in the training split?"
            )
        scaled = scalers[ticker].transform(group[FEATURE_COLUMNS].values)
        scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLUMNS, index=group.index)
        scaled_df["ticker"] = ticker
        scaled_df[TARGET_COLUMN] = group[TARGET_COLUMN].values
        scaled_df[PRICE_COLUMN] = group[PRICE_COLUMN].values
        pieces.append(scaled_df)
    return pd.concat(pieces).sort_index()


def build_sequences(
    scaled_df: pd.DataFrame, seq_len: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds sliding-window sequences per ticker, then pools across tickers.

    Indexing note (easy to get off-by-one on, so spelled out explicitly):
    for a window covering rows [start, start+seq_len), the LAST row in
    the window is at index start+seq_len-1 -- call this day D. We want to
    predict the return realized from day D to day D+1, which is exactly
    next_log_return stored AT ROW D (next_log_return[t] is defined as the
    return from day t to day t+1 -- see features.py's add_return_target).
    So the target is scaled_df[TARGET_COLUMN] at index start+seq_len-1,
    NOT start+seq_len (that indexing would have been correct for the old
    "predict tomorrow's LEVEL using data through today" target, but isn't
    for "predict the RETURN that occurs after today"). Verified against a
    hand-worked example in this module's tests.

    Returns:
        X: shape (n_samples, seq_len, n_features) -- scaled input windows
        y: shape (n_samples,) -- next_log_return, UNSCALED
        tickers: shape (n_samples,)
        dates: shape (n_samples,) -- the date each target return resolves
           on (i.e. the date being predicted), for time-axis plotting
        last_close: shape (n_samples,) -- the real Close price on the
           window's last day, needed to convert a predicted return back
           into a predicted price: predicted_price = last_close *
           exp(predicted_return)
    """
    X_list, y_list, ticker_list, date_list, last_close_list = [], [], [], [], []

    for ticker, group in scaled_df.groupby("ticker", sort=False):
        group = group.sort_index()
        feature_values = group[FEATURE_COLUMNS].values
        target_values = group[TARGET_COLUMN].values
        close_values = group[PRICE_COLUMN].values
        dates = group.index.values

        n_rows = len(feature_values)
        # Need seq_len rows for the window, plus row start+seq_len to exist
        # for the target date -- same loop bound as before.
        for start in range(0, n_rows - seq_len):
            window = feature_values[start : start + seq_len]
            last_window_idx = start + seq_len - 1
            target = target_values[last_window_idx]
            target_date = dates[start + seq_len]
            last_close = close_values[last_window_idx]

            X_list.append(window)
            y_list.append(target)
            ticker_list.append(ticker)
            date_list.append(target_date)
            last_close_list.append(last_close)

    if not X_list:
        raise ValueError(
            f"No sequences could be built with seq_len={seq_len} -- "
            "is there enough history per ticker?"
        )

    return (
        np.stack(X_list),
        np.array(y_list, dtype=np.float32),
        np.array(ticker_list),
        np.array(date_list),
        np.array(last_close_list, dtype=np.float64),
    )
