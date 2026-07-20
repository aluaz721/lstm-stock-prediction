import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.features import FEATURE_COLUMNS, add_technical_indicators
from src.inference import (
    predict_next_close,
    forecast_horizon,
    next_business_day,
)


def make_ticker_history(n_days=60, start_price=100.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    price = start_price
    closes = [price]
    for _ in range(n_days - 1):
        price *= 1 + rng.normal(0, 0.01)
        closes.append(price)
    return pd.DataFrame(
        {
            "Open": [c * 0.999 for c in closes],
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": rng.integers(1_000_000, 2_000_000, n_days),
        },
        index=dates,
    )


def fit_scaler_on(history_df):
    featured = add_technical_indicators(history_df).dropna(subset=FEATURE_COLUMNS)
    scaler = StandardScaler()
    scaler.fit(featured[FEATURE_COLUMNS].values)
    return scaler, featured


def test_next_business_day_skips_weekend():
    friday = pd.Timestamp("2024-01-05")  # a Friday
    assert friday.day_name() == "Friday"
    result = next_business_day(friday)
    assert result.day_name() == "Monday"
    assert result > friday


def test_predict_next_close_zero_return_gives_last_close():
    """A predicted return of exactly 0 should give predicted price == last known close."""
    history = make_ticker_history()
    scaler, featured = fit_scaler_on(history)

    def stub_predict(scaled_window):
        return 0.0

    pred = predict_next_close(stub_predict, history, scaler, seq_len=10)
    expected = float(featured["Close"].iloc[-1])
    assert abs(pred - expected) < 1e-6


def test_predict_next_close_positive_return_increases_price():
    history = make_ticker_history()
    scaler, featured = fit_scaler_on(history)
    last_close = float(featured["Close"].iloc[-1])

    def stub_predict_up(scaled_window):
        return 0.05  # +5% log return

    pred = predict_next_close(stub_predict_up, history, scaler, seq_len=10)
    expected = last_close * np.exp(0.05)
    assert abs(pred - expected) < 1e-6
    assert pred > last_close


def test_predict_next_close_raises_on_insufficient_history():
    history = make_ticker_history(n_days=15)  # not enough for warmup + seq_len

    def stub_predict(scaled_window):
        return 0.0

    scaler = StandardScaler()
    scaler.mean_ = np.zeros(len(FEATURE_COLUMNS))
    scaler.scale_ = np.ones(len(FEATURE_COLUMNS))
    with pytest.raises(ValueError):
        predict_next_close(stub_predict, history, scaler, seq_len=10)


def test_forecast_horizon_returns_correct_length_and_dates():
    history = make_ticker_history()
    scaler, _ = fit_scaler_on(history)

    def stub_predict(scaled_window):
        return 0.0

    horizon = 5
    forecast = forecast_horizon(stub_predict, history, scaler, seq_len=10, horizon_days=horizon)

    assert len(forecast) == horizon
    assert list(forecast.columns) == ["predicted_close", "predicted_return"]
    assert forecast.index[0] > history.index[-1]
    assert forecast.index.is_monotonic_increasing


def test_forecast_horizon_compounds_correctly_with_constant_return():
    """
    With a constant predicted log return r each step, prices should
    compound geometrically: day i's price = last_real_close * exp(r*(i+1)).
    This is different from the old price-level target's self-consistency
    check (which expected a CONSTANT price) -- returns compound, so a
    constant return produces geometric growth, not a flat line. Exercises
    the same recursive append-and-recompute-indicators loop either way.
    """
    history = make_ticker_history()
    scaler, featured = fit_scaler_on(history)
    last_real_close = float(featured["Close"].iloc[-1])

    r = 0.01  # constant +1% predicted log return every step

    def stub_predict(scaled_window):
        return r

    forecast = forecast_horizon(stub_predict, history, scaler, seq_len=10, horizon_days=5)

    expected_prices = [last_real_close * np.exp(r * (i + 1)) for i in range(5)]
    np.testing.assert_allclose(forecast["predicted_close"].values, expected_prices, rtol=1e-4)
    np.testing.assert_allclose(forecast["predicted_return"].values, [r] * 5, rtol=1e-6)


def test_forecast_horizon_rejects_zero_or_negative_horizon():
    history = make_ticker_history()
    scaler, _ = fit_scaler_on(history)

    def stub_predict(scaled_window):
        return 0.0

    with pytest.raises(ValueError):
        forecast_horizon(stub_predict, history, scaler, seq_len=10, horizon_days=0)


def test_forecast_horizon_raises_on_insufficient_history():
    history = make_ticker_history(n_days=15)
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(len(FEATURE_COLUMNS))
    scaler.scale_ = np.ones(len(FEATURE_COLUMNS))

    def stub_predict(scaled_window):
        return 0.0

    with pytest.raises(ValueError):
        forecast_horizon(stub_predict, history, scaler, seq_len=10, horizon_days=3)
