"""
Runs against a real Postgres instance via MONITORING_DATABASE_URL (or the
localhost default in database.py). Not mocked -- these are integration
tests for the actual schema and queries.
"""
from datetime import datetime

import pytest

from src.storage.database import engine, Base, SessionLocal
from src.storage import crud


@pytest.fixture(autouse=True)
def clean_tables():
    """Recreate all tables before each test for isolation."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def db():
    session = SessionLocal()
    yield session
    session.close()


def test_log_and_retrieve_prediction(db):
    crud.log_prediction(
        db, ticker="NVDA", model_type="lstm", model_version=1,
        predicted_close=132.50, prediction_for_date=datetime(2026, 7, 20),
        last_known_close=130.00,
    )
    crud.log_prediction(
        db, ticker="NVDA", model_type="qlstm", model_version=1,
        predicted_close=131.00, prediction_for_date=datetime(2026, 7, 20),
        last_known_close=130.00,
    )

    recent = crud.get_recent_predictions(db, ticker="NVDA")
    assert len(recent) == 2
    assert {r.model_type for r in recent} == {"lstm", "qlstm"}


def test_recent_predictions_ordered_newest_first(db):
    import time

    crud.log_prediction(
        db, ticker="AAPL", model_type="lstm", model_version=1,
        predicted_close=190.0, prediction_for_date=datetime(2026, 7, 20),
        last_known_close=189.0,
    )
    time.sleep(0.01)
    crud.log_prediction(
        db, ticker="AAPL", model_type="lstm", model_version=2,
        predicted_close=191.0, prediction_for_date=datetime(2026, 7, 21),
        last_known_close=190.0,
    )

    recent = crud.get_recent_predictions(db, ticker="AAPL")
    assert recent[0].model_version == 2  # newest first
    assert recent[1].model_version == 1


def test_recent_predictions_respects_limit(db):
    for i in range(5):
        crud.log_prediction(
            db, ticker="MSFT", model_type="lstm", model_version=1,
            predicted_close=400.0 + i, prediction_for_date=datetime(2026, 7, 20),
            last_known_close=399.0,
        )
    recent = crud.get_recent_predictions(db, ticker="MSFT", limit=3)
    assert len(recent) == 3


def test_log_drift_check_flags_drift_above_threshold(db):
    drifted = crud.log_drift_check(db, ticker="TSLA", feature_name="Close", psi_score=0.35, threshold=0.2)
    not_drifted = crud.log_drift_check(db, ticker="TSLA", feature_name="macd", psi_score=0.05, threshold=0.2)

    assert drifted.is_drifted is True
    assert not_drifted.is_drifted is False


def test_latest_drift_status_returns_one_per_feature(db):
    # two checks for the same feature at different times -- only the newer should surface
    crud.log_drift_check(db, ticker="AMD", feature_name="Close", psi_score=0.1, threshold=0.2)
    crud.log_drift_check(db, ticker="AMD", feature_name="Close", psi_score=0.4, threshold=0.2)
    crud.log_drift_check(db, ticker="AMD", feature_name="macd", psi_score=0.05, threshold=0.2)

    status = crud.get_latest_drift_status(db, ticker="AMD")
    by_feature = {s.feature_name: s for s in status}

    assert set(by_feature.keys()) == {"Close", "macd"}
    assert by_feature["Close"].psi_score == 0.4  # the more recent one


def test_upsert_reference_distribution_creates_then_updates(db):
    first = crud.upsert_reference_distribution(
        db, ticker="INTC", feature_name="Close",
        training_run_id="run-1", bin_edges=[0, 1, 2], reference_proportions=[0.5, 0.5],
    )
    assert first.training_run_id == "run-1"

    updated = crud.upsert_reference_distribution(
        db, ticker="INTC", feature_name="Close",
        training_run_id="run-2", bin_edges=[0, 1, 2, 3], reference_proportions=[0.3, 0.3, 0.4],
    )
    assert updated.id == first.id  # same row, not a duplicate
    assert updated.training_run_id == "run-2"

    fetched = crud.get_reference_distribution(db, ticker="INTC", feature_name="Close")
    assert fetched.training_run_id == "run-2"
    assert fetched.bin_edges == [0, 1, 2, 3]


def test_get_reference_distribution_returns_none_when_missing(db):
    result = crud.get_reference_distribution(db, ticker="CRM", feature_name="Close")
    assert result is None
