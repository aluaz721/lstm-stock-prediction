"""
Storage schema for everything MLflow itself doesn't track: a log of every
live prediction served (what the dashboard's "live prediction updates"
will read from), drift check results over time, and the reference feature
distributions each drift check gets compared against.

Drift detection uses PSI (Population Stability Index) rather than a
two-sample KS-test, deliberately: PSI only needs a binned reference
distribution (edges + proportions) rather than the full raw reference
sample, which is far cheaper to store and query repeatedly than keeping
raw training-set arrays around indefinitely. It's also the more common
choice in production ML monitoring specifically (vs. research contexts,
where KS-test is more common) for exactly this reason.

The actual PSI computation lives in src/monitoring/drift.py (next
increment) -- this file only defines the schema its results get written
into, so the storage contract is locked in before the computation logic
is built against it.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    JSON,
    Index,
)

from src.storage.database import Base


class PredictionLog(Base):
    """One row per prediction actually served -- the dashboard's live-prediction feed."""

    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    model_type = Column(String(20), nullable=False)  # "lstm" | "qlstm" | "tcn" | "qtcn"
    model_version = Column(Integer, nullable=False)  # MLflow registry version number
    predicted_close = Column(Float, nullable=False)
    prediction_for_date = Column(DateTime, nullable=False)  # the date being predicted
    last_known_close = Column(Float, nullable=False)  # close price the prediction was made from
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        Index("ix_prediction_log_ticker_created", "ticker", "created_at"),
    )


class DriftCheck(Base):
    """One row per (ticker, feature) pair per scheduled drift check run."""

    __tablename__ = "drift_check"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    feature_name = Column(String(30), nullable=False)
    psi_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    is_drifted = Column(Boolean, nullable=False)
    checked_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (
        Index("ix_drift_check_ticker_checked", "ticker", "checked_at"),
    )


class FeatureReferenceDistribution(Base):
    """
    The baseline distribution each drift check compares against: bin edges
    and reference proportions per (ticker, feature), captured from the
    training set at the time a model was trained. bin_edges and
    reference_proportions are stored as JSON arrays rather than a
    normalized bins table -- there's no query pattern that needs
    individual bins addressable, so normalizing would only add joins for
    no benefit.
    """

    __tablename__ = "feature_reference_distribution"

    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    feature_name = Column(String(30), nullable=False)
    training_run_id = Column(String(50), nullable=False)  # the MLflow run this was captured from
    bin_edges = Column(JSON, nullable=False)
    reference_proportions = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_feature_ref_ticker_feature", "ticker", "feature_name"),
    )
