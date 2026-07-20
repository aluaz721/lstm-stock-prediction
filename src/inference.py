"""
Inference utilities: single next-day prediction, and multi-day recursive
forecasting for the notebook's "what will this stock be worth in X days"
chart.

REWRITTEN alongside features.py/dataset.py's target-formulation fix:
predict_fn now returns a predicted LOG RETURN (unscaled), not a scaled
Close level -- converted back to a real price via
`last_close * exp(predicted_return)` rather than a scaler inverse-
transform. inverse_transform_close is gone entirely; there's nothing to
inverse-transform anymore since the target was never scaled in the first
place (see dataset.py's docstring for why).

Design note on recursive forecasting: to predict day t+2 we need engineered
features (MA7, MACD, Bollinger Bands, ...) for day t+1, which themselves
depend on OHLCV we don't actually have yet -- day t+1 is itself a
prediction. We handle this the standard way for recursive forecasting:
after predicting day t+1's Close (via the return conversion above), we
synthesize a full OHLCV row for it (Open=High=Low=Close=predicted value,
Volume=trailing mean) and append it to the history, then recompute
technical indicators over the extended series so the window for predicting
t+2 is internally consistent. This is a real limitation, not a hidden one:
uncertainty compounds with each step, and the synthesized OHLC collapses
to a single point rather than a true range. State that plainly wherever
this forecast is displayed.

`forecast_horizon` takes a `predict_fn` callable rather than a torch model
directly, so the recursive/indicator-regeneration logic (the part most
likely to have an off-by-one or index-alignment bug) can be tested without
torch -- see tests/test_inference.py, which injects a stub predict_fn and
runs for real in this environment.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.features import add_technical_indicators, FEATURE_COLUMNS

PredictFn = Callable[[np.ndarray], float]  # (seq_len, n_features) scaled window -> predicted log return


def predict_next_close(
    predict_fn: PredictFn,
    ticker_history_df: pd.DataFrame,
    scaler: StandardScaler,
    seq_len: int,
) -> float:
    """
    Single next-day prediction: builds the most recent valid window from
    ticker_history_df (raw OHLCV, unscaled), scales the INPUT features,
    predicts a log return, and converts it to a real price using the
    window's actual last Close (last_close * exp(predicted_return)).
    """
    featured = add_technical_indicators(ticker_history_df).dropna(subset=FEATURE_COLUMNS)
    if len(featured) < seq_len:
        raise ValueError(
            f"Need at least {seq_len} rows of valid features, got {len(featured)}. "
            "Is there enough history for the rolling-window warmup?"
        )
    last_close = float(featured["Close"].iloc[-1])
    window = featured.iloc[-seq_len:][FEATURE_COLUMNS].values
    scaled_window = scaler.transform(window)
    predicted_return = predict_fn(scaled_window)
    return last_close * float(np.exp(predicted_return))


def make_torch_predict_fn(model, device: str = "cpu") -> PredictFn:
    """
    Wraps a trained torch model into the PredictFn signature forecast_horizon
    expects. Kept separate from forecast_horizon itself so the recursive
    logic stays testable without torch (see module docstring).
    """
    import torch  # deferred import: keeps this module importable without torch

    def predict_fn(scaled_window: np.ndarray) -> float:
        model.eval()
        with torch.no_grad():
            x = torch.tensor(scaled_window[None, :, :], dtype=torch.float32, device=device)
            return model(x).item()

    return predict_fn


def make_pyfunc_predict_fn(pyfunc_model) -> PredictFn:
    """
    Same idea as make_torch_predict_fn, but for a model loaded from the
    MLflow registry via mlflow.pyfunc.load_model -- what the FastAPI
    serving layer actually uses, since it loads whatever's aliased
    'production' rather than holding a live torch.nn.Module directly.
    """

    def predict_fn(scaled_window: np.ndarray) -> float:
        result = pyfunc_model.predict(scaled_window[None, :, :])
        return float(np.asarray(result).reshape(-1)[0])

    return predict_fn


def next_business_day(last_date: pd.Timestamp) -> pd.Timestamp:
    return pd.bdate_range(start=last_date, periods=2)[1]


def forecast_horizon(
    predict_fn: PredictFn,
    ticker_history_df: pd.DataFrame,
    scaler: StandardScaler,
    seq_len: int,
    horizon_days: int,
) -> pd.DataFrame:
    """
    Recursively forecasts horizon_days of future Close prices. See module
    docstring for the OHLC-synthesis limitation this relies on, and the
    return-to-price conversion this now uses instead of a scaler
    inverse-transform.

    Returns a DataFrame indexed by predicted future business-day dates,
    with 'predicted_close' and 'predicted_return' columns.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    history = ticker_history_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    predictions = []

    for _ in range(horizon_days):
        featured = add_technical_indicators(history).dropna(subset=FEATURE_COLUMNS)
        if len(featured) < seq_len:
            raise ValueError(
                f"Need at least {seq_len} rows of valid features to forecast, "
                f"got {len(featured)}."
            )
        last_close = float(featured["Close"].iloc[-1])
        window = featured.iloc[-seq_len:][FEATURE_COLUMNS].values
        scaled_window = scaler.transform(window)
        predicted_return = float(predict_fn(scaled_window))
        next_close = last_close * float(np.exp(predicted_return))

        next_date = next_business_day(history.index[-1])
        trailing_volume = history["Volume"].tail(21).mean()
        new_row = pd.DataFrame(
            {
                "Open": [next_close],
                "High": [next_close],
                "Low": [next_close],
                "Close": [next_close],
                "Volume": [trailing_volume],
            },
            index=[next_date],
        )
        history = pd.concat([history, new_row])
        predictions.append((next_date, next_close, predicted_return))

    return pd.DataFrame(
        predictions, columns=["date", "predicted_close", "predicted_return"]
    ).set_index("date")
