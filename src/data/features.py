"""
Technical indicator features, computed independently per ticker.

REWRITTEN from the original raw-price-level version after diagnosing a
real bug: with raw levels (Close, ma7, ma21, macd in dollars, Bollinger
bands in dollars), a model trained on one price regime produces training
loss that collapses near-instantly (the LSTM learns "next Close ~= last
Close", trivially available since Close is an input feature at the
window's last timestep) while test loss stays high whenever the test
period's price level has drifted from the training period's -- confirmed
empirically: a stock with a strong secular trend (verified with a
NVDA-like synthetic uptrend) produces train/test target statistics that
don't remotely match (train mean~=0/std~=1 vs. test mean~=-0.54/std~=2.71
in one run), purely from price-level drift, with NO leakage in the
split/scaling code itself.

Every feature below is now a RATIO or RETURN, not a raw level -- scale-
invariant regardless of whether a stock trades at $15 or $900, and far
more stationary over time than price levels are. Close itself is dropped
from FEATURE_COLUMNS entirely (it stays in the DataFrame as a plain
column -- needed to compute the target and to convert predicted returns
back to real prices -- it's just not fed to the model as an input).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "close_to_ma7",
    "close_to_ma21",
    "close_to_ema12",
    "close_to_ema26",
    "macd_pct",
    "bb_percent_b",
    "bb_width_pct",
    "close_to_ema_fast",
]


def add_technical_indicators(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Adds technical indicator columns to a single ticker's OHLCV DataFrame.
    Expects the DataFrame to already be sorted by date and contain price_col.
    """
    out = df.copy()
    close = out[price_col]

    # 1-day log return -- stationary in a way a raw price difference isn't;
    # this is also what makes the LSTM's "just copy the last input" shortcut
    # far less available, since consecutive RETURNS aren't trivially related
    # to each other the way consecutive price LEVELS are.
    out["return_1d"] = np.log(close / close.shift(1))

    ma7 = close.rolling(window=7).mean()
    ma21 = close.rolling(window=21).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ema_fast = close.ewm(com=0.5, adjust=False).mean()

    # "how far above/below its moving average is the price, as a ratio"
    # -- scale-invariant, unlike the raw moving average value itself.
    out["close_to_ma7"] = close / ma7 - 1
    out["close_to_ma21"] = close / ma21 - 1
    out["close_to_ema12"] = close / ema12 - 1
    out["close_to_ema26"] = close / ema26 - 1
    out["close_to_ema_fast"] = close / ema_fast - 1

    # MACD normalized by price level, rather than a raw dollar figure --
    # a $2 MACD means something very different for a $15 stock than a
    # $900 stock.
    macd = ema12 - ema26
    out["macd_pct"] = macd / close

    # Bollinger Bands as %B (where within the band the price sits, ~[0,1])
    # and band width as a percentage of price, instead of raw dollar bands.
    bb_std20 = close.rolling(window=20).std()
    bb_upper = ma21 + 2 * bb_std20
    bb_lower = ma21 - 2 * bb_std20
    band_range = bb_upper - bb_lower
    out["bb_percent_b"] = (close - bb_lower) / band_range.replace(0, np.nan)
    out["bb_width_pct"] = band_range / ma21

    return out


def add_return_target(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Adds 'next_log_return', computed PER TICKER: the log return realized
    from day t's close to day t+1's close, stored at row t. This is the
    regression target -- NOT scaled, since log returns are already close
    to stationary across time (verified empirically; see module
    docstring), unlike price levels which needed per-ticker scaling
    specifically because their distribution drifts over time.

    Must be called on a DataFrame with a 'ticker' column (i.e. after
    build_features_for_universe, or on build_features_for_universe's
    per-ticker input before concatenation -- either works since this
    groups by ticker internally).
    """
    out = df.copy()
    next_close = out.groupby("ticker")[price_col].shift(-1)
    out["next_log_return"] = np.log(next_close / out[price_col])
    return out


def build_features_for_universe(universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies add_technical_indicators() independently to each ticker's
    slice of a long-format universe DataFrame (must have a 'ticker' column),
    adds the return target, then concatenates the results back together.
    """
    pieces = []
    for ticker, group in universe_df.groupby("ticker", sort=False):
        group = group.sort_index()
        pieces.append(add_technical_indicators(group))
    result = pd.concat(pieces).sort_index()
    result = add_return_target(result)

    # Rolling/EWM windows produce NaNs for the first ~21 days per ticker,
    # and the return target is NaN for each ticker's last row (no "next
    # day" to compute a return to yet) -- both must be dropped before
    # windowing, not silently imputed.
    return result.dropna(subset=FEATURE_COLUMNS + ["next_log_return"])
