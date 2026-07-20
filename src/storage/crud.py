from datetime import datetime, timezone

from sqlalchemy import desc
from sqlalchemy.orm import Session

from src.storage.models import PredictionLog, DriftCheck, FeatureReferenceDistribution


def log_prediction(
    db: Session,
    ticker: str,
    model_type: str,
    model_version: int,
    predicted_close: float,
    prediction_for_date: datetime,
    last_known_close: float,
) -> PredictionLog:
    row = PredictionLog(
        ticker=ticker,
        model_type=model_type,
        model_version=model_version,
        predicted_close=predicted_close,
        prediction_for_date=prediction_for_date,
        last_known_close=last_known_close,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_recent_predictions(db: Session, ticker: str, limit: int = 50) -> list[PredictionLog]:
    return (
        db.query(PredictionLog)
        .filter_by(ticker=ticker)
        .order_by(desc(PredictionLog.created_at))
        .limit(limit)
        .all()
    )


def log_drift_check(
    db: Session,
    ticker: str,
    feature_name: str,
    psi_score: float,
    threshold: float,
) -> DriftCheck:
    row = DriftCheck(
        ticker=ticker,
        feature_name=feature_name,
        psi_score=psi_score,
        threshold=threshold,
        is_drifted=psi_score >= threshold,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_latest_drift_status(db: Session, ticker: str) -> list[DriftCheck]:
    """
    Returns the most recent drift check per feature for a ticker -- i.e.
    the current drift status, not the full history. Uses a subquery-free
    approach (fine at this data volume; revisit with a window function if
    the table grows large).
    """
    all_checks = (
        db.query(DriftCheck)
        .filter_by(ticker=ticker)
        .order_by(desc(DriftCheck.checked_at))
        .all()
    )
    latest_per_feature: dict[str, DriftCheck] = {}
    for check in all_checks:
        if check.feature_name not in latest_per_feature:
            latest_per_feature[check.feature_name] = check
    return list(latest_per_feature.values())


def upsert_reference_distribution(
    db: Session,
    ticker: str,
    feature_name: str,
    training_run_id: str,
    bin_edges: list[float],
    reference_proportions: list[float],
) -> FeatureReferenceDistribution:
    existing = (
        db.query(FeatureReferenceDistribution)
        .filter_by(ticker=ticker, feature_name=feature_name)
        .one_or_none()
    )
    if existing:
        existing.training_run_id = training_run_id
        existing.bin_edges = bin_edges
        existing.reference_proportions = reference_proportions
        existing.created_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(existing)
        return existing

    row = FeatureReferenceDistribution(
        ticker=ticker,
        feature_name=feature_name,
        training_run_id=training_run_id,
        bin_edges=bin_edges,
        reference_proportions=reference_proportions,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_reference_distribution(
    db: Session, ticker: str, feature_name: str
) -> FeatureReferenceDistribution | None:
    return (
        db.query(FeatureReferenceDistribution)
        .filter_by(ticker=ticker, feature_name=feature_name)
        .one_or_none()
    )
