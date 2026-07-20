import numpy as np
import pandas as pd
import pytest

from src.data.features import build_features_for_universe, add_return_target, FEATURE_COLUMNS
from src.data.dataset import (
    chronological_split,
    fit_scalers_per_ticker,
    transform_with_scalers,
    build_sequences,
    TARGET_COLUMN,
    PRICE_COLUMN,
)


def make_synthetic_universe(tickers, n_days=300, seed=42):
    rng = np.random.default_rng(seed)
    base_prices = {t: p for t, p in zip(tickers, [190.0, 130.0, 100.0, 250.0][: len(tickers)])}
    frames = []
    for ticker in tickers:
        dates = pd.bdate_range("2023-01-01", periods=n_days)
        price = base_prices[ticker]
        closes = [price]
        for _ in range(n_days - 1):
            price *= 1 + rng.normal(0, 0.02)
            closes.append(price)
        df = pd.DataFrame(
            {
                "Open": closes,
                "High": [c * 1.01 for c in closes],
                "Low": [c * 0.99 for c in closes],
                "Close": closes,
                "Volume": rng.integers(1_000_000, 5_000_000, n_days),
            },
            index=dates,
        )
        df["ticker"] = ticker
        frames.append(df)
    return pd.concat(frames).sort_index()


@pytest.fixture
def synthetic_universe():
    return make_synthetic_universe(["AAPL", "NVDA", "AMD"])


@pytest.fixture
def featured(synthetic_universe):
    return build_features_for_universe(synthetic_universe)


def test_features_have_no_nans(featured):
    assert not featured[FEATURE_COLUMNS].isna().any().any()
    assert not featured[TARGET_COLUMN].isna().any()


def test_close_is_not_a_model_input_feature():
    """Regression guard: Close must never be back in FEATURE_COLUMNS -- see features.py docstring."""
    assert "Close" not in FEATURE_COLUMNS


def test_return_target_matches_hand_computed_values():
    """
    add_return_target's next_log_return at row t must equal
    log(Close[t+1] / Close[t]) -- verified against a hand-computed series,
    not just "runs without error."
    """
    closes = [100.0, 102.0, 101.0, 105.0, 103.0]
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    df = pd.DataFrame({"Close": closes, "ticker": "TEST"}, index=dates)

    result = add_return_target(df)
    expected = [
        np.log(102.0 / 100.0),
        np.log(101.0 / 102.0),
        np.log(105.0 / 101.0),
        np.log(103.0 / 105.0),
        np.nan,  # last row has no "next day"
    ]
    np.testing.assert_allclose(result["next_log_return"].values[:-1], expected[:-1], rtol=1e-10)
    assert np.isnan(result["next_log_return"].iloc[-1])


def test_features_drop_warmup_and_last_row(synthetic_universe, featured):
    # rolling/EWM windows need ~21 days warmup per ticker, AND the last
    # row per ticker has no next_log_return (no "next day" to compute it)
    n_tickers = synthetic_universe["ticker"].nunique()
    assert len(featured) < len(synthetic_universe) - n_tickers  # at least the last-row drop
    assert len(featured) > len(synthetic_universe) - n_tickers * 25


def test_chronological_split_no_leakage(featured):
    split = chronological_split(featured, train_frac=0.65)
    assert split.train.index.max() <= split.split_date
    assert split.test.index.min() > split.split_date
    assert len(split.train) + len(split.test) == len(featured)


def test_scalers_fit_per_ticker_only_on_train_features(featured):
    """Scaling applies to FEATURE_COLUMNS -- the target must NOT be scaled (see dataset.py docstring)."""
    split = chronological_split(featured, train_frac=0.65)
    scalers = fit_scalers_per_ticker(split.train)
    assert set(scalers.keys()) == set(featured["ticker"].unique())

    train_scaled = transform_with_scalers(split.train, scalers)
    for ticker in scalers:
        sub = train_scaled[train_scaled["ticker"] == ticker]
        for col in FEATURE_COLUMNS:
            vals = sub[col].values
            assert abs(vals.mean()) < 1e-6
            assert abs(vals.std() - 1.0) < 1e-6

        # target must be UNCHANGED by scaling -- compare against the
        # pre-scaling values for the same ticker/dates
        original = split.train[split.train["ticker"] == ticker][TARGET_COLUMN]
        np.testing.assert_allclose(sub[TARGET_COLUMN].values, original.values, rtol=1e-10)


def test_transform_preserves_price_column_unscaled(featured):
    split = chronological_split(featured, train_frac=0.65)
    scalers = fit_scalers_per_ticker(split.train)
    train_scaled = transform_with_scalers(split.train, scalers)

    # Compare per-ticker: sort_index() alone isn't a stable cross-frame
    # comparison key when multiple tickers share the same date -- ties
    # break differently depending on each frame's pre-sort row order.
    for ticker in split.train["ticker"].unique():
        original = split.train[split.train["ticker"] == ticker].sort_index()[PRICE_COLUMN]
        scaled = train_scaled[train_scaled["ticker"] == ticker].sort_index()[PRICE_COLUMN]
        np.testing.assert_allclose(scaled.values, original.values, rtol=1e-10)


def test_transform_rejects_unknown_ticker(featured):
    split = chronological_split(featured, train_frac=0.65)
    scalers = fit_scalers_per_ticker(split.train)
    bogus = split.test.copy()
    bogus["ticker"] = "NOT_A_REAL_TICKER"
    with pytest.raises(KeyError):
        transform_with_scalers(bogus, scalers)


def test_build_sequences_shapes(featured):
    split = chronological_split(featured, train_frac=0.65)
    scalers = fit_scalers_per_ticker(split.train)
    train_scaled = transform_with_scalers(split.train, scalers)

    seq_len = 10
    X, y, tickers, dates, last_close = build_sequences(train_scaled, seq_len=seq_len)

    assert X.ndim == 3
    assert X.shape[1] == seq_len
    assert X.shape[2] == len(FEATURE_COLUMNS)
    assert y.shape[0] == X.shape[0]
    assert tickers.shape[0] == X.shape[0]
    assert dates.shape[0] == X.shape[0]
    assert last_close.shape[0] == X.shape[0]
    assert set(tickers) == set(split.train["ticker"].unique())
    assert np.all(last_close > 0)  # real prices, sanity check


def test_build_sequences_indexing_matches_hand_worked_example():
    """
    The exact scenario the module docstring works through by hand: window
    ending at day D should be paired with the return from D to D+1, using
    the CLOSE at day D as last_close -- not the return "one row past the
    window" the way the old price-level target used.
    """
    closes = [100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 110.0]
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    df = pd.DataFrame({"Close": closes, "ticker": "TEST"}, index=dates)
    df = add_return_target(df)
    df = df.dropna(subset=["next_log_return"])

    # bypass scaling entirely -- use a trivial single feature column so we
    # can isolate build_sequences' indexing logic specifically
    df["fake_feature"] = 1.0
    import src.data.dataset as ds_module
    original_features = list(__import__("src.data.features", fromlist=["FEATURE_COLUMNS"]).FEATURE_COLUMNS)
    ds_module.FEATURE_COLUMNS = ["fake_feature"]
    try:
        X, y, tickers, target_dates, last_close = build_sequences(df, seq_len=3)
    finally:
        ds_module.FEATURE_COLUMNS = original_features

    # window 0 covers rows [100, 102, 101] (dates[0:3]), last window day = row 2 (Close=101, date=dates[2])
    # target = return from day D2 to D3 = log(105/101), last_close = 101, target_date = dates[3]
    assert last_close[0] == 101.0
    assert abs(y[0] - np.log(105.0 / 101.0)) < 1e-6
    assert target_dates[0] == np.datetime64(dates[3])

    # window 1 covers rows [102, 101, 105] (dates[1:4]), last window day = row 3 (Close=105, date=dates[3])
    # target = return from D3 to D4 = log(103/105), last_close = 105, target_date = dates[4]
    assert last_close[1] == 105.0
    assert abs(y[1] - np.log(103.0 / 105.0)) < 1e-6
    assert target_dates[1] == np.datetime64(dates[4])


def test_build_sequences_raises_on_insufficient_history():
    tiny_universe = make_synthetic_universe(["AAPL"], n_days=15)
    featured = build_features_for_universe(tiny_universe)
    # after dropping ~21-day warmup + last row, almost nothing is left
    with pytest.raises(ValueError):
        build_sequences(featured, seq_len=10)
