"""
Captures PSI reference distributions at training time. Deliberately kept
in its own module, separate from train.py: this logic has no dependency
on torch at all (it's pure pandas/numpy + Postgres writes), but train.py
imports torch at module level -- keeping this here means it can be
imported and tested without dragging torch in, and it's a more honest
home for it anyway (this is monitoring logic, not training logic; train.py
just happens to be the caller).
"""
from src.data.features import FEATURE_COLUMNS
from src.monitoring.drift import build_reference_distribution
from src.storage.database import SessionLocal, init_db
from src.storage import crud


def capture_reference_distributions(train_df, run_id: str) -> None:
    """
    Builds and stores PSI reference distributions (bin edges + reference
    proportions) per (ticker, feature), from the RAW (unscaled) training
    feature values -- not the scaled ones. Scaled features are forced to
    mean~=0/std~=1 by construction, so a PSI reference built from them
    would be somewhat redundant; the raw ratio/return features (macd_pct,
    close_to_ma21, etc.) are the natural, interpretable quantities to
    monitor for drift, and this decouples drift monitoring from whichever
    scaler happens to be attached to the current production model.
    """
    init_db()  # idempotent -- ensures the schema exists even if train.py
    # runs before the API has ever started (e.g. a fresh Postgres volume)
    db = SessionLocal()
    try:
        for ticker, group in train_df.groupby("ticker", sort=False):
            for feature in FEATURE_COLUMNS:
                values = group[feature].dropna().values
                if len(values) < 10:
                    continue
                bin_edges, reference_proportions = build_reference_distribution(values)
                crud.upsert_reference_distribution(
                    db,
                    ticker=ticker,
                    feature_name=feature,
                    training_run_id=run_id,
                    bin_edges=bin_edges,
                    reference_proportions=reference_proportions,
                )
    finally:
        db.close()
